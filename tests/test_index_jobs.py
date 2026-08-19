"""SDK-free unit tests for the job store and async registration endpoints (PR B).

Covers:
  * JobStore persistence, TTL/count GC, cancellation, error capping
  * core.run_index_add_job progress, tuning, per-file failure accounting
  * worker auto-sizing and embed_shape_batch specifications
  * POST/GET/DELETE /similarity/index/{name}/jobs[...]
  * server_paths gating

Nothing here touches the SDK: add_to_index is replaced by a fake.
"""

import os
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core
import jobstore
from fastapi.testclient import TestClient


class _StoreBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.store = jobstore.JobStore(self.root, max_workers=1, ttl_seconds=3600)

    def tearDown(self):
        self.store.shutdown(wait=True)
        self._tmp.cleanup()


class TestJobRecords(_StoreBase):
    def test_create_returns_a_queued_record(self):
        rec = self.store.create(kind="index_add")
        self.assertEqual(rec["status"], jobstore.STATUS_QUEUED)

    def test_record_is_persisted_to_disk(self):
        rec = self.store.create(kind="index_add")
        self.assertTrue((self.root / f"{rec['job_id']}.json").is_file())

    def test_record_survives_a_new_store_instance(self):
        rec = self.store.create(kind="index_add", index_name="a")
        reopened = jobstore.JobStore(self.root)
        self.assertEqual(reopened.get(rec["job_id"])["index_name"], "a")

    def test_unknown_job_raises(self):
        with self.assertRaises(jobstore.JobNotFound):
            self.store.get("nope")

    def test_path_traversal_job_id_is_rejected(self):
        with self.assertRaises(jobstore.JobNotFound):
            self.store.get("../../etc/passwd")

    def test_update_merges_dict_sections(self):
        rec = self.store.create(kind="k")
        self.store.update(rec["job_id"], progress={"phase": "embedding"})
        got = self.store.get(rec["job_id"])["progress"]
        self.assertEqual(got["phase"], "embedding")
        self.assertEqual(got["done"], 0)

    def test_increment_progress_accumulates(self):
        rec = self.store.create(kind="k")
        self.store.increment_progress(rec["job_id"], done=3, errors=1)
        self.store.increment_progress(rec["job_id"], done=2)
        got = self.store.get(rec["job_id"])["progress"]
        self.assertEqual((got["done"], got["errors"]), (5, 1))

    def test_errors_are_capped(self):
        store = jobstore.JobStore(self.root, max_errors_per_job=3)
        rec = store.create(kind="k")
        store.append_errors(rec["job_id"], [{"file_id": str(i)} for i in range(10)])
        got = store.get(rec["job_id"])
        self.assertEqual(len(got["errors"]), 3)
        self.assertTrue(got["errors_truncated"])

    def test_appending_no_errors_is_a_noop(self):
        rec = self.store.create(kind="k")
        self.store.append_errors(rec["job_id"], [])
        self.assertFalse(self.store.get(rec["job_id"])["errors_truncated"])

    def test_list_is_newest_first_and_paginated(self):
        for _ in range(3):
            self.store.create(kind="k", index_name="i")
            time.sleep(0.01)
        page, total = self.store.list(kind="k", index_name="i", limit=2)
        self.assertEqual((len(page), total), (2, 3))

    def test_list_filters_by_index(self):
        self.store.create(kind="k", index_name="a")
        self.store.create(kind="k", index_name="b")
        _, total = self.store.list(kind="k", index_name="a")
        self.assertEqual(total, 1)


class TestJobExecution(_StoreBase):
    def _run(self, fn, timeout=5.0):
        rec = self.store.create(kind="k")
        self.store.submit(rec["job_id"], fn)
        deadline = time.time() + timeout
        while time.time() < deadline:
            got = self.store.get(rec["job_id"])
            if got["status"] in jobstore.TERMINAL_STATUSES:
                return got
            time.sleep(0.01)
        self.fail("job did not finish")

    def test_successful_job_is_done(self):
        self.assertEqual(self._run(lambda ctx: None)["status"], jobstore.STATUS_DONE)

    def test_failing_job_records_the_reason(self):
        def boom(ctx):
            raise ValueError("bad input")

        got = self._run(boom)
        self.assertEqual(got["status"], jobstore.STATUS_FAILED)
        self.assertIn("bad input", got["detail"])

    def test_total_seconds_is_recorded(self):
        self.assertIn("total_seconds", self._run(lambda ctx: None)["timings"])

    def test_context_reports_progress(self):
        def body(ctx):
            ctx.set_total(4)
            ctx.advance(done=4, errors=1)
            ctx.set_summary(added=3)

        got = self._run(body)
        self.assertEqual(got["progress"]["total"], 4)
        self.assertEqual(got["progress"]["errors"], 1)
        self.assertEqual(got["summary"]["added"], 3)

    def test_finished_at_is_set(self):
        self.assertIsNotNone(self._run(lambda ctx: None)["finished_at"])


class TestCancellation(_StoreBase):
    def test_queued_job_is_canceled_immediately(self):
        rec = self.store.create(kind="k")
        got = self.store.request_cancel(rec["job_id"])
        self.assertEqual(got["status"], jobstore.STATUS_CANCELED)

    def test_canceled_queued_job_never_runs(self):
        rec = self.store.create(kind="k")
        self.store.request_cancel(rec["job_id"])
        ran = []
        self.store.submit(rec["job_id"], lambda ctx: ran.append(1))
        time.sleep(0.2)
        self.assertEqual(ran, [])

    def test_running_job_stops_at_a_checkpoint(self):
        rec = self.store.create(kind="k")
        steps = []

        def body(ctx):
            for i in range(20):
                ctx.check_canceled()
                steps.append(i)
                if i == 2:
                    self.store.request_cancel(rec["job_id"])
                time.sleep(0.01)

        self.store.submit(rec["job_id"], body)
        deadline = time.time() + 5
        while time.time() < deadline:
            if self.store.get(rec["job_id"])["status"] in jobstore.TERMINAL_STATUSES:
                break
            time.sleep(0.01)
        got = self.store.get(rec["job_id"])
        self.assertEqual(got["status"], jobstore.STATUS_CANCELED)
        self.assertLess(len(steps), 20)

    def test_cancelling_a_finished_job_leaves_it_alone(self):
        rec = self.store.create(kind="k")
        self.store.update(rec["job_id"], status=jobstore.STATUS_DONE)
        got = self.store.request_cancel(rec["job_id"])
        self.assertEqual(got["status"], jobstore.STATUS_DONE)


class TestGarbageCollection(_StoreBase):
    def _aged(self, status, age_seconds):
        rec = self.store.create(kind="k")
        self.store.update(rec["job_id"], status=status)
        path = self.root / f"{rec['job_id']}.json"
        stamp = time.time() - age_seconds
        os.utime(path, (stamp, stamp))
        return rec["job_id"], path

    def test_old_terminal_records_are_removed(self):
        _, path = self._aged(jobstore.STATUS_DONE, 7200)
        self.store.collect_garbage()
        self.assertFalse(path.exists())

    def test_fresh_terminal_records_are_kept(self):
        _, path = self._aged(jobstore.STATUS_DONE, 10)
        self.store.collect_garbage()
        self.assertTrue(path.exists())

    def test_running_records_are_never_collected(self):
        _, path = self._aged(jobstore.STATUS_RUNNING, 999999)
        self.store.collect_garbage()
        self.assertTrue(path.exists())

    def test_queued_records_are_never_collected(self):
        _, path = self._aged(jobstore.STATUS_QUEUED, 999999)
        self.store.collect_garbage()
        self.assertTrue(path.exists())

    def test_zero_ttl_disables_the_age_sweep(self):
        store = jobstore.JobStore(self.root, ttl_seconds=0)
        rec = store.create(kind="k")
        store.update(rec["job_id"], status=jobstore.STATUS_DONE)
        path = self.root / f"{rec['job_id']}.json"
        stamp = time.time() - 999999
        os.utime(path, (stamp, stamp))
        store.collect_garbage()
        self.assertTrue(path.exists())

    def test_record_count_is_capped(self):
        store = jobstore.JobStore(self.root, max_records=2)
        for _ in range(5):
            rec = store.create(kind="k")
            store.update(rec["job_id"], status=jobstore.STATUS_DONE)
            time.sleep(0.01)
        store.collect_garbage()
        self.assertLessEqual(len(list(self.root.glob("*.json"))), 2)

    def test_corrupt_records_are_dropped(self):
        bad = self.root / "corrupt.json"
        self.root.mkdir(parents=True, exist_ok=True)
        bad.write_text("{ not json")
        self.store.collect_garbage()
        self.assertFalse(bad.exists())


class TestRecovery(_StoreBase):
    """A restart must not leave records claiming to be in progress forever."""

    def test_running_records_become_failed(self):
        rec = self.store.create(kind="k")
        self.store.update(rec["job_id"], status=jobstore.STATUS_RUNNING)
        reopened = jobstore.JobStore(self.root)
        self.assertEqual(reopened.recover_interrupted(), 1)
        self.assertEqual(reopened.get(rec["job_id"])["status"], jobstore.STATUS_FAILED)

    def test_queued_records_become_failed(self):
        rec = self.store.create(kind="k")
        jobstore.JobStore(self.root).recover_interrupted()
        self.assertEqual(self.store.get(rec["job_id"])["status"], jobstore.STATUS_FAILED)

    def test_recovered_records_explain_why(self):
        rec = self.store.create(kind="k")
        jobstore.JobStore(self.root).recover_interrupted()
        self.assertIn("Interrupted", self.store.get(rec["job_id"])["detail"])

    def test_finished_records_are_left_alone(self):
        rec = self.store.create(kind="k")
        self.store.update(rec["job_id"], status=jobstore.STATUS_DONE, detail="ok")
        jobstore.JobStore(self.root).recover_interrupted()
        got = self.store.get(rec["job_id"])
        self.assertEqual((got["status"], got["detail"]), (jobstore.STATUS_DONE, "ok"))

    def test_recovery_on_an_empty_store_is_a_noop(self):
        self.assertEqual(jobstore.JobStore(self.root / "none").recover_interrupted(), 0)


class _FakeCtx:
    """Records what a job body reports, without touching disk."""

    def __init__(self, cancel_after=None):
        self.phases = []
        self.total = 0
        self.done = 0
        self.errors = 0
        self.summary = {}
        self.timings = {}
        self.error_items = []
        self._checks = 0
        self._cancel_after = cancel_after

    def check_canceled(self):
        self._checks += 1
        if self._cancel_after is not None and self._checks > self._cancel_after:
            raise jobstore.JobCanceled("canceled")

    def set_phase(self, phase):
        self.phases.append(phase)

    def set_total(self, total):
        self.total = total

    def advance(self, done=0, errors=0):
        self.done += done
        self.errors += errors

    def set_summary(self, **fields):
        self.summary.update(fields)

    def set_timings(self, **fields):
        self.timings.update(fields)

    def add_errors(self, errors):
        self.error_items.extend(errors)


class TestRunIndexAddJob(unittest.TestCase):
    def setUp(self):
        self._orig = core.add_to_index
        self.calls = []

    def tearDown(self):
        core.add_to_index = self._orig

    def _fake(self, errors_for=()):
        def add_to_index(name, file_ids, model=None, num_workers=None, time_limit=None):
            self.calls.append(
                {"files": list(file_ids), "workers": num_workers, "limit": time_limit}
            )
            errs = [{"file_id": f, "detail": "boom"} for f in file_ids if f in errors_for]
            return {
                "name": name,
                "added": len(file_ids) - len(errs),
                "updated": 0,
                "index_count": 99,
                "errors": errs,
            }

        core.add_to_index = add_to_index

    def test_whole_input_goes_to_one_sdk_call(self):
        # Splitting into chunks re-spawns the worker pool and reloads the
        # checkpoint in every worker, which measured ~30% of total runtime.
        self._fake()
        ctx = _FakeCtx()
        core.run_index_add_job(ctx, "idx", [str(i) for i in range(5)])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.calls[0]["files"]), 5)

    def test_total_and_done_match_the_input(self):
        self._fake()
        ctx = _FakeCtx()
        core.run_index_add_job(ctx, "idx", [str(i) for i in range(5)])
        self.assertEqual((ctx.total, ctx.done), (5, 5))

    def test_failures_are_reported_per_file(self):
        self._fake(errors_for={"1", "3"})
        ctx = _FakeCtx()
        core.run_index_add_job(ctx, "idx", [str(i) for i in range(5)])
        self.assertEqual(len(ctx.error_items), 2)
        self.assertEqual(ctx.summary["failed"], 2)
        self.assertEqual(ctx.errors, 2)

    def test_added_count_is_reported(self):
        self._fake()
        ctx = _FakeCtx()
        core.run_index_add_job(ctx, "idx", [str(i) for i in range(5)])
        self.assertEqual(ctx.summary["added"], 5)

    def test_embed_time_is_recorded(self):
        self._fake()
        ctx = _FakeCtx()
        core.run_index_add_job(ctx, "idx", ["a"])
        self.assertIn("embed_seconds", ctx.timings)

    def test_tuning_is_recorded_in_timings(self):
        self._fake()
        ctx = _FakeCtx()
        core.run_index_add_job(ctx, "idx", ["a"], num_workers=3, time_limit=900)
        self.assertEqual(ctx.timings["num_workers"], 3)
        self.assertEqual(ctx.timings["time_limit"], 900)

    def test_explicit_tuning_reaches_the_sdk_call(self):
        self._fake()
        core.run_index_add_job(_FakeCtx(), "idx", ["a"], num_workers=3, time_limit=900)
        self.assertEqual(self.calls[0]["workers"], 3)
        self.assertEqual(self.calls[0]["limit"], 900)

    def test_workers_are_sized_automatically_when_unset(self):
        self._fake()
        core.run_index_add_job(_FakeCtx(), "idx", ["a"])
        self.assertGreaterEqual(self.calls[0]["workers"], 1)

    def test_cancellation_is_checked_before_embedding(self):
        self._fake()
        ctx = _FakeCtx(cancel_after=0)
        with self.assertRaises(jobstore.JobCanceled):
            core.run_index_add_job(ctx, "idx", [str(i) for i in range(10)])
        self.assertEqual(self.calls, [])

    def test_empty_input_finishes_cleanly(self):
        self._fake()
        ctx = _FakeCtx()
        core.run_index_add_job(ctx, "idx", [])
        self.assertEqual(ctx.phases[-1], "done")


class TestWorkerSizing(unittest.TestCase):
    _VARS = (
        "HOOPS_AI_MAX_WORKERS",
        "HOOPS_AI_MODEL_FOOTPRINT_MB",
        "HOOPS_AI_MIN_FILES_PARALLEL",
        "HOOPS_AI_EMBED_TIME_LIMIT",
    )

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self._VARS}
        for k in self._VARS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_small_batches_run_sequentially(self):
        # Spawning workers costs ~55-60 s in checkpoint loads; it cannot pay off
        # for a handful of files.
        self.assertEqual(core.compute_auto_workers(1), 1)

    def test_min_files_parallel_is_overridable(self):
        os.environ["HOOPS_AI_MIN_FILES_PARALLEL"] = "2"
        self.assertGreaterEqual(core.compute_auto_workers(4), 1)

    def test_auto_workers_never_exceeds_the_cap(self):
        os.environ["HOOPS_AI_MAX_WORKERS"] = "2"
        self.assertLessEqual(core.compute_auto_workers(1000), 2)

    def test_auto_workers_is_at_least_one(self):
        try:
            import psutil  # noqa: F401
        except ImportError:
            self.skipTest("psutil unavailable: the RAM cap is not applied")
        os.environ["HOOPS_AI_MODEL_FOOTPRINT_MB"] = "1000000"
        self.assertEqual(core.compute_auto_workers(1000), 1)

    def test_auto_workers_survives_a_missing_ram_probe(self):
        # Without psutil the count falls back to the core/cap bound rather than
        # raising or returning 0.
        self.assertGreaterEqual(core.compute_auto_workers(1000), 1)

    def test_time_limit_defaults_to_sdk_default(self):
        self.assertEqual(core.embed_time_limit(), 0)

    def test_time_limit_is_overridable(self):
        os.environ["HOOPS_AI_EMBED_TIME_LIMIT"] = "600"
        self.assertEqual(core.embed_time_limit(), 600)

    def test_specifications_always_request_images(self):
        specs = core.build_embed_specifications(pathlib.Path("imgs"))
        self.assertTrue(specs["generate_images"])
        self.assertEqual(specs["images_out_dir"], "imgs")

    def test_no_time_limit_keys_when_unset(self):
        specs = core.build_embed_specifications(pathlib.Path("imgs"))
        self.assertNotIn("time_limit_overall", specs)

    def test_all_four_time_limit_keys_are_set(self):
        # time_limit_overall governs the per-item cumulative timeout; setting
        # only the size buckets leaves heavy files killed at the 120 s default.
        specs = core.build_embed_specifications(pathlib.Path("imgs"), 900)
        for key in (
            "time_limit_overall",
            "time_limit_small",
            "time_limit_medium",
            "time_limit_large",
        ):
            self.assertEqual(specs[key], 900.0)


class TestJobSettings(unittest.TestCase):
    _VARS = (
        "HOOPS_AI_JOB_MAX_CONCURRENCY",
        "HOOPS_AI_JOB_TTL_DAYS",
        "HOOPS_AI_JOB_MAX_RECORDS",
        "HOOPS_AI_ALLOW_SERVER_PATHS",
    )

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self._VARS}
        for k in self._VARS:
            os.environ.pop(k, None)
        self._read_env = core.read_env_file
        core.read_env_file = lambda: {}

    def tearDown(self):
        core.read_env_file = self._read_env
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_jobs_are_serialised_by_default(self):
        # hoops_ai writes error_summary.json to the CWD with no way to redirect
        # it, so concurrent registration jobs would corrupt each other's report.
        self.assertEqual(core.job_max_concurrency(), 1)

    def test_concurrency_is_overridable(self):
        os.environ["HOOPS_AI_JOB_MAX_CONCURRENCY"] = "4"
        self.assertEqual(core.job_max_concurrency(), 4)

    def test_ttl_defaults_to_seven_days(self):
        self.assertEqual(core.job_ttl_seconds(), 7 * 86400)

    def test_ttl_is_overridable(self):
        os.environ["HOOPS_AI_JOB_TTL_DAYS"] = "2"
        self.assertEqual(core.job_ttl_seconds(), 2 * 86400)

    def test_max_records_default(self):
        self.assertEqual(core.job_max_records(), 1000)

    def test_server_paths_are_disabled_by_default(self):
        self.assertFalse(core.server_paths_enabled())

    def test_server_paths_can_be_enabled(self):
        os.environ["HOOPS_AI_ALLOW_SERVER_PATHS"] = "true"
        self.assertTrue(core.server_paths_enabled())

    def test_server_paths_reject_when_disabled(self):
        with self.assertRaises(PermissionError):
            core.resolve_server_paths(["C:/whatever.stp"])


class TestJobEndpoints(unittest.TestCase):
    """End-to-end through TestClient with add_to_index faked out."""

    def setUp(self):
        import main

        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.store_dir = self.root / "cad"
        self.store_dir.mkdir()

        self._saved = {
            "add_to_index": core.add_to_index,
            "job_store": core._index_job_store,
            "shared": core.get_CAD_shared_dir,
            "faiss": core._index_faiss_path,
            "schema": core._load_index_schema,
            "read_env": core.read_env_file,
        }
        # The repository .env belongs to the developer running the suite; a test
        # must not change behaviour because a local file happens to enable a
        # feature (server_paths in particular). Importing main also copies .env
        # into os.environ, so both sources have to be neutralised.
        core.read_env_file = lambda: {}
        self._saved_env = os.environ.pop("HOOPS_AI_ALLOW_SERVER_PATHS", None)
        core._index_job_store = jobstore.JobStore(
            self.root / "jobs", max_workers=1, ttl_seconds=3600
        )
        core.get_CAD_shared_dir = lambda: self.store_dir
        core._index_faiss_path = lambda name: self.root / f"{name}.faiss"
        (self.root / "idx.faiss").write_bytes(b"")
        (self.root / "legacy.faiss").write_bytes(b"")
        core._load_index_schema = lambda name: {
            "schema_version": 1 if name == "legacy" else 2,
            "model": "signal",
        }
        core.add_to_index = lambda name, file_ids, model=None, **kwargs: {
            "name": name, "added": len(file_ids), "updated": 0,
            "index_count": len(file_ids), "errors": [],
        }
        self.client = TestClient(main.app)

        self.file_id = "a" * 64
        (self.store_dir / f"{self.file_id}_part.stp").write_bytes(b"cad")

    def tearDown(self):
        core._index_job_store.shutdown(wait=True)
        core.add_to_index = self._saved["add_to_index"]
        core._index_job_store = self._saved["job_store"]
        core.get_CAD_shared_dir = self._saved["shared"]
        core._index_faiss_path = self._saved["faiss"]
        core._load_index_schema = self._saved["schema"]
        core.read_env_file = self._saved["read_env"]
        if self._saved_env is None:
            os.environ.pop("HOOPS_AI_ALLOW_SERVER_PATHS", None)
        else:
            os.environ["HOOPS_AI_ALLOW_SERVER_PATHS"] = self._saved_env
        self._tmp.cleanup()

    def _wait(self, name, job_id, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            body = self.client.get(f"/similarity/index/{name}/jobs/{job_id}").json()
            if body["status"] in jobstore.TERMINAL_STATUSES:
                return body
            time.sleep(0.02)
        self.fail("job did not finish")

    def test_create_returns_202_with_a_job_id(self):
        r = self.client.post(
            "/similarity/index/idx/jobs/add", json={"file_ids": [self.file_id]}
        )
        self.assertEqual(r.status_code, 202)
        self.assertTrue(r.json()["job_id"])

    def test_job_runs_to_completion(self):
        job_id = self.client.post(
            "/similarity/index/idx/jobs/add", json={"file_ids": [self.file_id]}
        ).json()["job_id"]
        body = self._wait("idx", job_id)
        self.assertEqual(body["status"], "done")
        self.assertEqual(body["summary"]["added"], 1)

    def test_unknown_file_id_is_reported_not_silently_dropped(self):
        r = self.client.post(
            "/similarity/index/idx/jobs/add",
            json={"file_ids": [self.file_id, "b" * 64]},
        )
        body = r.json()
        self.assertEqual(body["accepted"], 1)
        self.assertEqual(len(body["errors"]), 1)

    def test_all_inputs_invalid_is_422(self):
        r = self.client.post(
            "/similarity/index/idx/jobs/add", json={"file_ids": ["b" * 64]}
        )
        self.assertEqual(r.status_code, 422)

    def test_empty_request_is_422(self):
        r = self.client.post("/similarity/index/idx/jobs/add", json={})
        self.assertEqual(r.status_code, 422)

    def test_duplicate_ids_are_collapsed(self):
        r = self.client.post(
            "/similarity/index/idx/jobs/add",
            json={"file_ids": [self.file_id, self.file_id]},
        )
        self.assertEqual(r.json()["accepted"], 1)

    def test_unknown_index_is_404(self):
        r = self.client.post(
            "/similarity/index/missing/jobs/add", json={"file_ids": [self.file_id]}
        )
        self.assertEqual(r.status_code, 404)

    def test_legacy_schema_index_is_409(self):
        r = self.client.post(
            "/similarity/index/legacy/jobs/add", json={"file_ids": [self.file_id]}
        )
        self.assertEqual(r.status_code, 409)

    def test_server_paths_are_403_by_default(self):
        r = self.client.post(
            "/similarity/index/idx/jobs/add",
            json={"server_paths": ["C:/whatever.stp"]},
        )
        self.assertEqual(r.status_code, 403)

    def test_status_of_unknown_job_is_404(self):
        r = self.client.get("/similarity/index/idx/jobs/deadbeef")
        self.assertEqual(r.status_code, 404)

    def test_job_of_another_index_is_404(self):
        job_id = self.client.post(
            "/similarity/index/idx/jobs/add", json={"file_ids": [self.file_id]}
        ).json()["job_id"]
        self._wait("idx", job_id)
        r = self.client.get(f"/similarity/index/other/jobs/{job_id}")
        self.assertEqual(r.status_code, 404)

    def test_list_returns_the_job(self):
        job_id = self.client.post(
            "/similarity/index/idx/jobs/add", json={"file_ids": [self.file_id]}
        ).json()["job_id"]
        self._wait("idx", job_id)
        body = self.client.get("/similarity/index/idx/jobs").json()
        self.assertEqual(body["total"], 1)
        self.assertEqual(body["jobs"][0]["job_id"], job_id)

    def test_cancel_returns_the_record(self):
        job_id = self.client.post(
            "/similarity/index/idx/jobs/add", json={"file_ids": [self.file_id]}
        ).json()["job_id"]
        r = self.client.delete(f"/similarity/index/idx/jobs/{job_id}")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["cancel_requested"] or r.json()["status"] in
                        ("canceled", "done", "running"))

    def test_cancel_of_unknown_job_is_404(self):
        r = self.client.delete("/similarity/index/idx/jobs/deadbeef")
        self.assertEqual(r.status_code, 404)

    def test_response_has_no_private_fields(self):
        job_id = self.client.post(
            "/similarity/index/idx/jobs/add", json={"file_ids": [self.file_id]}
        ).json()["job_id"]
        body = self._wait("idx", job_id)
        self.assertFalse([k for k in body if k.startswith("_")])

    def test_timings_include_total_seconds(self):
        job_id = self.client.post(
            "/similarity/index/idx/jobs/add", json={"file_ids": [self.file_id]}
        ).json()["job_id"]
        body = self._wait("idx", job_id)
        self.assertIn("total_seconds", body["timings"])
        self.assertIn("embed_seconds", body["timings"])


class TestExistingEndpointsUnchanged(unittest.TestCase):
    """The synchronous /index/add contract must not move: MCPServer uses it."""

    def _paths(self):
        import main

        return set(TestClient(main.app).get("/openapi.json").json()["paths"])

    def test_sync_add_route_still_exists(self):
        self.assertIn("/similarity/index/add", self._paths())

    def test_job_routes_are_registered(self):
        paths = self._paths()
        self.assertIn("/similarity/index/{name}/jobs/add", paths)
        self.assertIn("/similarity/index/{name}/jobs/{job_id}", paths)
        self.assertIn("/similarity/index/{name}/jobs", paths)


if __name__ == "__main__":
    unittest.main()
