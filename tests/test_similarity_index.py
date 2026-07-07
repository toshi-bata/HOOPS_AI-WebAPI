"""Tests for the named index management API (POST/GET/DELETE /similarity/index/*).

Tests that require a HOOPS AI license are decorated with ``@unittest.skip`` to match the
project convention.  All validation-layer and mock-based tests run without a license.
"""

import io
import os
import pathlib
import sys
import tempfile
import threading
import unittest
import zipfile
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_app():
    """Return a minimal FastAPI app with only the similarity router mounted."""
    from fastapi import FastAPI
    from routers.similarity import router

    app = FastAPI()
    app.include_router(router)
    return app


def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _fake_vs(ids: list[str] | None = None):
    """Build a minimal FaissVectorStore mock."""
    ids = ids or []
    vs = MagicMock()
    vs.get_ids.return_value = list(ids)
    vs.count.return_value = len(ids)
    vs.query.return_value = []
    return vs


def _fake_embedding(file_id: str, dim: int = 4, model: str = "legacy"):
    import numpy as np

    v = np.array([1.0, 0.0, 0.0, 0.0][:dim], dtype=np.float32)
    return {
        "file_id": file_id,
        "vector": v,
        "dim": dim,
        "model_name": "hoops_embeddings_model",
        "num_bodies": 1,
        "filename": f"{file_id[:8]}.step",
        "cached": False,
    }


# ---------------------------------------------------------------------------
# Lifecycle test: create → list → add → search → remove → delete
# ---------------------------------------------------------------------------


class TestIndexLifecycle(unittest.TestCase):
    """Full lifecycle with mocked FaissVectorStore and compute_embedding."""

    def setUp(self):
        import core
        from fastapi.testclient import TestClient

        self.client = TestClient(_make_app())
        self.core = core

        # Each test gets a fresh temp dir for INDEXES_DIR
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_indexes_dir = core.INDEXES_DIR
        core.INDEXES_DIR = pathlib.Path(self._tmpdir.name)
        # Clear the in-memory cache between tests
        core._named_indexes.clear()
        core._index_locks.clear()

    def tearDown(self):
        import core

        core.INDEXES_DIR = self._orig_indexes_dir
        core._named_indexes.clear()
        core._index_locks.clear()
        self._tmpdir.cleanup()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _mock_faiss(self, dim: int = 4):
        """Return a context-manager that patches FaissVectorStore and related imports."""
        import numpy as np
        from unittest.mock import MagicMock, patch

        # We need to patch inside hoops_ai.ml.embeddings AND inside core's local import
        vs_instance = MagicMock()
        id_store: list[str] = []

        def upsert(records):
            for r in records:
                if r.id not in id_store:
                    id_store.append(r.id)

        def delete(ids):
            for i in ids:
                if i in id_store:
                    id_store.remove(i)

        vs_instance.get_ids.side_effect = lambda: list(id_store)
        vs_instance.count.side_effect = lambda: len(id_store)
        vs_instance.upsert.side_effect = upsert
        vs_instance.delete.side_effect = delete
        vs_instance.save = MagicMock()

        FaissVectorStoreMock = MagicMock()
        FaissVectorStoreMock.return_value = vs_instance
        FaissVectorStoreMock.load.return_value = vs_instance

        EmbeddingMock = MagicMock()
        VectorRecordMock = MagicMock(side_effect=lambda id, embedding, metadata: MagicMock(id=id))

        embedder_mock = MagicMock()
        embedder_mock.embedding_dim = dim

        patches = [
            patch("core.FaissVectorStore", FaissVectorStoreMock, create=True),
        ]

        # We also need to mock the import inside the function bodies
        import_patch = patch.dict(
            "sys.modules",
            {
                "hoops_ai": MagicMock(),
                "hoops_ai.ml": MagicMock(),
                "hoops_ai.ml.embeddings": MagicMock(
                    FaissVectorStore=FaissVectorStoreMock,
                    Embedding=EmbeddingMock,
                    VectorRecord=VectorRecordMock,
                ),
            },
        )
        return import_patch, FaissVectorStoreMock, vs_instance, embedder_mock, EmbeddingMock, VectorRecordMock

    def test_create_then_list(self):
        """create_index creates a .faiss file; list_indexes reports it."""
        import core

        fid = "a" * 64
        mock_embedder = MagicMock()
        mock_embedder.embedding_dim = 4

        FaissVS = MagicMock()
        vs_inst = MagicMock()
        vs_inst.get_ids.return_value = []
        vs_inst.count.return_value = 0
        vs_inst.save.side_effect = lambda path: (
            pathlib.Path(path + ".faiss").write_bytes(b"fake"),
            pathlib.Path(path + ".meta").write_bytes(b"fake"),
        )
        FaissVS.return_value = vs_inst
        FaissVS.load.return_value = vs_inst

        with (
            patch("core.get_embedder", return_value=mock_embedder),
            patch.dict(
                "sys.modules",
                {"hoops_ai.ml.embeddings": MagicMock(
                    FaissVectorStore=FaissVS,
                    Embedding=MagicMock(),
                    VectorRecord=MagicMock(),
                )},
            ),
        ):
            result = core.create_index("demo")
            self.assertEqual(result["name"], "demo")
            self.assertEqual(result["count"], 0)
            self.assertEqual(result["dim"], 4)

            indexes = core.list_indexes()
            names = [i["name"] for i in indexes]
            self.assertIn("demo", names)

    def test_add_increments_count(self):
        """add_to_index upserts and increments the count."""
        import core
        import numpy as np

        vs_inst = _fake_vs([])
        id_store: list[str] = []

        def upsert_impl(records):
            for r in records:
                _id = getattr(r, "id", None) or getattr(r, "_mock_name", None)
                # extract id via how VectorRecord mock is set up
                if hasattr(r, "id") and isinstance(r.id, str):
                    if r.id not in id_store:
                        id_store.append(r.id)

        vs_inst.upsert.side_effect = upsert_impl
        vs_inst.get_ids.side_effect = lambda: list(id_store)

        fid = "a" * 64
        FaissVS = MagicMock()
        FaissVS.load.return_value = vs_inst

        # Pre-create the index.faiss file under the new per-index subdirectory
        (core.INDEXES_DIR / "demo").mkdir(parents=True, exist_ok=True)
        (core.INDEXES_DIR / "demo" / "index.faiss").write_bytes(b"fake")
        (core.INDEXES_DIR / "demo" / "index.meta").write_bytes(b"fake")
        core._named_indexes["demo"] = vs_inst

        EmbeddingMock = MagicMock()
        VRMock = MagicMock(side_effect=lambda id, embedding, metadata: MagicMock(id=id))

        with (
            patch.dict(
                "sys.modules",
                {"hoops_ai.ml.embeddings": MagicMock(
                    FaissVectorStore=FaissVS,
                    Embedding=EmbeddingMock,
                    VectorRecord=VRMock,
                )},
            ),
            patch.object(core, "compute_embedding", side_effect=_fake_embedding),
            patch.object(core, "_save_named_index_atomic"),
        ):
            result = core.add_to_index("demo", [fid])

        self.assertEqual(result["added"], 1)
        self.assertEqual(result["updated"], 0)
        self.assertEqual(len(result["errors"]), 0)

    def test_readd_counts_as_updated(self):
        """Re-adding an existing file_id is counted as 'updated', not 'added'."""
        import core

        fid = "b" * 64
        vs_inst = _fake_vs([fid])  # already contains fid

        (core.INDEXES_DIR / "myidx").mkdir(parents=True, exist_ok=True)
        (core.INDEXES_DIR / "myidx" / "index.faiss").write_bytes(b"fake")
        (core.INDEXES_DIR / "myidx" / "index.meta").write_bytes(b"fake")
        core._named_indexes["myidx"] = vs_inst

        FaissVS = MagicMock()
        EmbeddingMock = MagicMock()
        VRMock = MagicMock(side_effect=lambda id, embedding, metadata: MagicMock(id=id))

        with (
            patch.dict(
                "sys.modules",
                {"hoops_ai.ml.embeddings": MagicMock(
                    FaissVectorStore=FaissVS,
                    Embedding=EmbeddingMock,
                    VectorRecord=VRMock,
                )},
            ),
            patch.object(core, "compute_embedding", side_effect=_fake_embedding),
            patch.object(core, "_save_named_index_atomic"),
        ):
            result = core.add_to_index("myidx", [fid])

        self.assertEqual(result["added"], 0)
        self.assertEqual(result["updated"], 1)
        # delete was called once for the existing id
        vs_inst.delete.assert_called_once_with([fid])

    def test_search_returns_empty_for_empty_index(self):
        """search_index on a 0-entry index returns empty hits without error."""
        import core

        vs_inst = _fake_vs([])
        (core.INDEXES_DIR / "empty").mkdir(parents=True, exist_ok=True)
        (core.INDEXES_DIR / "empty" / "index.faiss").write_bytes(b"fake")
        (core.INDEXES_DIR / "empty" / "index.meta").write_bytes(b"fake")
        core._named_indexes["empty"] = vs_inst

        with patch.object(core, "compute_embedding", side_effect=_fake_embedding):
            result = core.search_index("empty", "a" * 64, top_k=5)

        self.assertEqual(result["hits"], [])
        self.assertEqual(result["count"], 0)
        vs_inst.query.assert_not_called()

    def test_remove_decrements_count(self):
        """remove_from_index calls delete and saves."""
        import core

        fid = "c" * 64
        vs_inst = _fake_vs([fid])
        deleted: list[str] = []

        def fake_delete(ids):
            for i in ids:
                if i in vs_inst.get_ids.return_value:
                    vs_inst.get_ids.return_value = [x for x in vs_inst.get_ids.return_value if x != i]
                deleted.extend(ids)

        vs_inst.delete.side_effect = fake_delete

        (core.INDEXES_DIR / "idx").mkdir(parents=True, exist_ok=True)
        (core.INDEXES_DIR / "idx" / "index.faiss").write_bytes(b"fake")
        (core.INDEXES_DIR / "idx" / "index.meta").write_bytes(b"fake")
        core._named_indexes["idx"] = vs_inst

        FaissVS = MagicMock()
        with (
            patch.dict(
                "sys.modules",
                {"hoops_ai.ml.embeddings": MagicMock(FaissVectorStore=FaissVS)},
            ),
            patch.object(core, "_save_named_index_atomic"),
        ):
            result = core.remove_from_index("idx", [fid])

        self.assertEqual(result["removed"], 1)
        self.assertIn(fid, deleted)

    def test_delete_index_removes_files(self):
        """delete_index removes the entire index directory from disk."""
        import core

        (core.INDEXES_DIR / "todel").mkdir(parents=True, exist_ok=True)
        (core.INDEXES_DIR / "todel" / "index.faiss").write_bytes(b"data")
        (core.INDEXES_DIR / "todel" / "index.meta").write_bytes(b"data")

        result = core.delete_index("todel")
        self.assertTrue(result["deleted"])
        self.assertFalse((core.INDEXES_DIR / "todel").exists())
        self.assertNotIn("todel", core._named_indexes)


# ---------------------------------------------------------------------------
# Reserved name protection
# ---------------------------------------------------------------------------


class TestReservedNameProtection(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(_make_app())

    def test_create_default_returns_422(self):
        response = self.client.post("/similarity/index/create?name=default")
        self.assertEqual(response.status_code, 422)
        self.assertIn("reserved", response.json()["detail"].lower())

    def test_add_to_default_returns_403(self):
        response = self.client.post("/similarity/index/add?name=default")
        self.assertEqual(response.status_code, 403)

    def test_delete_default_returns_403(self):
        response = self.client.delete("/similarity/index/default?confirm=true")
        self.assertEqual(response.status_code, 403)

    def test_remove_parts_from_default_returns_403(self):
        response = self.client.delete(
            "/similarity/index/default/parts?part_ids=abc"
        )
        self.assertEqual(response.status_code, 403)

    def test_search_default_returns_403(self):
        """Search on 'default' is reserved and should return 403."""
        response = self.client.post(
            "/similarity/index/default/search?file_id=abc"
        )
        self.assertEqual(response.status_code, 403)


# ---------------------------------------------------------------------------
# Invalid index name validation
# ---------------------------------------------------------------------------


class TestInvalidIndexNames(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(_make_app())

    def _assert_422_on_create(self, name: str):
        resp = self.client.post(f"/similarity/index/create?name={name}")
        self.assertIn(
            resp.status_code,
            (422,),
            msg=f"Expected 422 for name={name!r}, got {resp.status_code}",
        )

    def test_uppercase_name_rejected(self):
        self._assert_422_on_create("MyIndex")

    def test_space_in_name_rejected(self):
        # Spaces cause FastAPI to reject or core validates
        resp = self.client.post("/similarity/index/create", params={"name": "my index"})
        self.assertIn(resp.status_code, (422,))

    def test_path_traversal_rejected(self):
        resp = self.client.post("/similarity/index/create", params={"name": "../evil"})
        self.assertIn(resp.status_code, (422,))

    def test_too_long_name_rejected(self):
        long_name = "a" * 65
        self._assert_422_on_create(long_name)

    def test_empty_name_rejected(self):
        # FastAPI requires non-empty query param
        resp = self.client.post("/similarity/index/create?name=")
        self.assertIn(resp.status_code, (422,))


# ---------------------------------------------------------------------------
# Duplicate create returns 409
# ---------------------------------------------------------------------------


class TestDuplicateCreate(unittest.TestCase):
    def setUp(self):
        import core
        from fastapi.testclient import TestClient

        self.client = TestClient(_make_app())
        self.core = core
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_indexes_dir = core.INDEXES_DIR
        core.INDEXES_DIR = pathlib.Path(self._tmpdir.name)
        core._named_indexes.clear()
        core._index_locks.clear()

    def tearDown(self):
        import core

        core.INDEXES_DIR = self._orig_indexes_dir
        core._named_indexes.clear()
        core._index_locks.clear()
        self._tmpdir.cleanup()

    def test_duplicate_create_returns_409(self):
        """Creating an index that already exists must return 409."""
        import core

        # Pre-create the index directory with the new structure
        (core.INDEXES_DIR / "exists").mkdir(parents=True, exist_ok=True)
        (core.INDEXES_DIR / "exists" / "index.faiss").write_bytes(b"data")
        (core.INDEXES_DIR / "exists" / "index.meta").write_bytes(b"data")

        with patch.object(core, "get_embedder", return_value=MagicMock(embedding_dim=4)):
            try:
                core.create_index("exists")
                self.fail("Expected FileExistsError")
            except FileExistsError:
                pass  # expected


# ---------------------------------------------------------------------------
# Delete without confirm returns 409
# ---------------------------------------------------------------------------


class TestDeleteConfirmRequired(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(_make_app())

    def test_delete_without_confirm_returns_409(self):
        """DELETE /similarity/index/{name} without ?confirm=true must return 409."""
        response = self.client.delete("/similarity/index/demo")
        self.assertEqual(response.status_code, 409)
        self.assertIn("confirm=true", response.json()["detail"])

    def test_delete_with_confirm_false_returns_409(self):
        response = self.client.delete("/similarity/index/demo?confirm=false")
        self.assertEqual(response.status_code, 409)


# ---------------------------------------------------------------------------
# Atomic save unit test
# ---------------------------------------------------------------------------


class TestAtomicSave(unittest.TestCase):
    """Verify that _save_named_index_atomic writes temp files then renames them."""

    def setUp(self):
        import core
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_indexes_dir = core.INDEXES_DIR
        core.INDEXES_DIR = pathlib.Path(self._tmpdir.name)

    def tearDown(self):
        import core
        core.INDEXES_DIR = self._orig_indexes_dir
        self._tmpdir.cleanup()

    def test_atomic_save_creates_target_files_via_rename(self):
        """save() must write temp files and replace() them into the final paths."""
        import core

        written_paths: list[str] = []

        def fake_save(path: str):
            pathlib.Path(path + ".faiss").write_bytes(b"faiss_data")
            pathlib.Path(path + ".meta").write_bytes(b"meta_data")
            written_paths.append(path)

        vs = MagicMock()
        vs.save.side_effect = fake_save

        core._save_named_index_atomic("mytest", vs)

        dest_faiss = core.INDEXES_DIR / "mytest" / "index.faiss"
        dest_meta = core.INDEXES_DIR / "mytest" / "index.meta"
        self.assertTrue(dest_faiss.exists(), "Final index.faiss must exist after atomic save")
        self.assertTrue(dest_meta.exists(), "Final index.meta must exist after atomic save")
        self.assertEqual(dest_faiss.read_bytes(), b"faiss_data")
        self.assertEqual(dest_meta.read_bytes(), b"meta_data")

        self.assertEqual(len(written_paths), 1)
        tmp_base = written_paths[0]
        self.assertFalse(
            pathlib.Path(tmp_base + ".faiss").exists(),
            "Temp .faiss must not remain after successful replace()",
        )

    def test_atomic_save_cleans_up_temp_on_failure(self):
        """If replace() fails, temp files must be cleaned up."""
        import core

        def bad_save(path: str):
            pathlib.Path(path + ".faiss").write_bytes(b"x")
            pathlib.Path(path + ".meta").write_bytes(b"y")
            raise RuntimeError("disk full")

        vs = MagicMock()
        vs.save.side_effect = bad_save

        with self.assertRaises(RuntimeError):
            core._save_named_index_atomic("fail_idx", vs)

        # No temp files should remain
        remaining = list(core.INDEXES_DIR.glob("_tmp_*.faiss"))
        self.assertEqual(remaining, [], "Temp .faiss must be cleaned up after failure")


# ---------------------------------------------------------------------------
# Router-level validation tests (no HOOPS AI license needed)
# ---------------------------------------------------------------------------


class TestRouterValidation(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(_make_app())

    def test_add_with_no_input_returns_422(self):
        """POST /similarity/index/add without any input must return 422."""
        response = self.client.post("/similarity/index/add?name=demo")
        self.assertEqual(response.status_code, 422)

    def test_search_with_no_file_or_id_returns_422(self):
        """POST /similarity/index/{name}/search without file or file_id must return 422."""
        response = self.client.post("/similarity/index/demo/search")
        self.assertEqual(response.status_code, 422)

    def test_remove_parts_with_no_ids_returns_422(self):
        """DELETE /similarity/index/{name}/parts without part_ids must return 422 (missing required query)."""
        response = self.client.delete("/similarity/index/demo/parts")
        # FastAPI returns 422 for missing required query params
        self.assertIn(response.status_code, (422, 400))

    def test_list_indexes_returns_200(self):
        """GET /similarity/index/list must return 200 even when no indexes exist."""
        import core
        with patch.object(core, "list_indexes", return_value=[]):
            response = self.client.get("/similarity/index/list")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])


# ---------------------------------------------------------------------------
# HOOPS AI dependent integration tests – skipped without runtime
# ---------------------------------------------------------------------------


@unittest.skip(
    "Requires HOOPS AI license and trained model checkpoint. "
    "Run manually in a configured HOOPS AI environment."
)
class TestIndexManagementIntegration(unittest.TestCase):
    """End-to-end tests — require HOOPS AI runtime."""

    def test_full_lifecycle(self):
        """create → add → search → remove → delete lifecycle with real CAD files."""
        raise NotImplementedError("Provide real .step file paths to run this test.")


if __name__ == "__main__":
    unittest.main()
