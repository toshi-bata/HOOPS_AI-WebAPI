"""SDK-free unit tests for the body-level named-index migration.

Covers the pure/testable pieces of the schema-v2 change:
  * per-file_id search dedup (keep best score, rank, truncate)
  * ``list_index_parts`` paging bounds (offset past end, limit clamp)
  * schema fallback (index.json / model.json / neither)
  * OBB serialisation guard (drop when > 512 bytes or not JSON-serialisable)
  * per-part asset resolution path-traversal protection

FaissVectorStore is faked; no HOOPS AI license is required.
"""

import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core


def _hit(id_, score):
    return SimpleNamespace(id=id_, score=score, metadata={"file_id": id_})


class TestDedupHitsByFile(unittest.TestCase):
    def test_collapses_to_best_score_per_id(self):
        hits = [
            _hit("a", 0.9),
            _hit("b", 0.8),
            _hit("a", 0.95),  # better duplicate of a
            _hit("b", 0.5),
            _hit("c", 0.1),
        ]
        out = core._dedup_hits_by_file(hits, top_k=10)
        ids = [h.id for h in out]
        self.assertEqual(ids, ["a", "b", "c"])  # sorted by best score desc
        best_a = next(h for h in out if h.id == "a")
        self.assertEqual(best_a.score, 0.95)

    def test_truncates_to_top_k(self):
        hits = [_hit(str(i), 1.0 - i * 0.01) for i in range(20)]
        out = core._dedup_hits_by_file(hits, top_k=5)
        self.assertEqual(len(out), 5)
        self.assertEqual([h.id for h in out], ["0", "1", "2", "3", "4"])

    def test_ignores_hits_without_id(self):
        hits = [SimpleNamespace(score=0.9), _hit("a", 0.5)]
        out = core._dedup_hits_by_file(hits, top_k=5)
        self.assertEqual([h.id for h in out], ["a"])


class TestObbSerializationGuard(unittest.TestCase):
    def test_small_value_kept(self):
        val = {"min": [0.0, 0.0, 0.0], "max": [1.0, 2.0, 3.0]}
        out = core._obb_meta_value(val)
        self.assertIsInstance(out, str)
        self.assertIn("min", out)

    def test_oversize_value_dropped(self):
        val = {"data": list(range(500))}  # serialises to well over 512 bytes
        self.assertIsNone(core._obb_meta_value(val))

    def test_non_serializable_dropped(self):
        self.assertIsNone(core._obb_meta_value({1, 2, 3}))  # set is not JSON

    def test_boundary_512(self):
        # Construct a value whose JSON is exactly <= 512 bytes and just over.
        ok = {"s": "x" * 400}
        self.assertIsNotNone(core._obb_meta_value(ok))
        too_big = {"s": "x" * 520}
        self.assertIsNone(core._obb_meta_value(too_big))


class TestSchemaFallback(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = core.INDEXES_DIR
        core.INDEXES_DIR = pathlib.Path(self._tmp.name)

    def tearDown(self):
        core.INDEXES_DIR = self._orig
        self._tmp.cleanup()

    def _mk(self, name):
        d = core.INDEXES_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_index_json_takes_precedence(self):
        d = self._mk("a")
        (d / "index.json").write_text('{"model": "signal", "schema_version": 2}')
        (d / "model.json").write_text('{"model": "legacy"}')
        schema = core._load_index_schema("a")
        self.assertEqual(schema, {"model": "signal", "schema_version": 2})

    def test_model_json_only_is_schema_1(self):
        d = self._mk("b")
        (d / "model.json").write_text('{"model": "signal"}')
        schema = core._load_index_schema("b")
        self.assertEqual(schema, {"model": "signal", "schema_version": 1})

    def test_neither_is_legacy_schema_1(self):
        self._mk("c")
        schema = core._load_index_schema("c")
        self.assertEqual(schema, {"model": "legacy", "schema_version": 1})

    def test_load_index_model_uses_schema(self):
        d = self._mk("d")
        (d / "index.json").write_text('{"model": "signal", "schema_version": 2}')
        self.assertEqual(core._load_index_model("d"), "signal")


class TestResolveIndexAssetTraversal(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = core.INDEXES_DIR
        core.INDEXES_DIR = pathlib.Path(self._tmp.name)
        # Create a valid thumbnail and a secret file outside the index dir.
        self.idx = core.INDEXES_DIR / "idx"
        (self.idx / "thumbnails").mkdir(parents=True, exist_ok=True)
        self.part_id = "a" * 64
        (self.idx / "thumbnails" / f"{self.part_id}.png").write_bytes(b"png")
        (core.INDEXES_DIR / "secret.png").write_bytes(b"secret")

    def tearDown(self):
        core.INDEXES_DIR = self._orig
        self._tmp.cleanup()

    def test_valid_part_resolves(self):
        path = core._resolve_index_asset("idx", self.part_id, "thumbnail")
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())

    def test_missing_part_returns_none(self):
        self.assertIsNone(core._resolve_index_asset("idx", "b" * 64, "thumbnail"))

    def test_traversal_part_id_returns_none(self):
        # Attempt to escape the index directory via a crafted part_id.
        malicious = os.path.join("..", "..", "secret")
        self.assertIsNone(core._resolve_index_asset("idx", malicious, "thumbnail"))

    def test_unknown_asset_returns_none(self):
        self.assertIsNone(core._resolve_index_asset("idx", self.part_id, "bogus"))


class TestListIndexPartsPaging(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = core.INDEXES_DIR
        core.INDEXES_DIR = pathlib.Path(self._tmp.name)
        (core.INDEXES_DIR / "idx").mkdir(parents=True, exist_ok=True)
        core._save_index_schema("idx", "signal", 2)

        # Fake vs: 5 files, each with a couple of body rows (metadata repeats).
        self._metas = []
        for i in range(5):
            fid = f"{i:064d}"
            kind = "assembly" if i % 2 == 0 else "part"
            bodies = 3 if kind == "assembly" else 1
            for _ in range(bodies):
                self._metas.append(
                    {
                        "file_id": fid,
                        "filename": f"file{i}.step",
                        "kind": kind,
                        "bodies": bodies,
                        "registered_at": "2026-01-01T00:00:00Z",
                    }
                )
        vs = MagicMock()
        vs.iter_metadata.side_effect = lambda: iter(self._metas)
        core._named_indexes["idx"] = vs

    def tearDown(self):
        core._named_indexes.pop("idx", None)
        core.INDEXES_DIR = self._orig
        self._tmp.cleanup()

    def test_total_is_distinct_files(self):
        result = core.list_index_parts("idx", offset=0, limit=100)
        self.assertEqual(result["total"], 5)
        self.assertEqual(len(result["items"]), 5)

    def test_offset_past_end_returns_empty(self):
        result = core.list_index_parts("idx", offset=99, limit=10)
        self.assertEqual(result["total"], 5)
        self.assertEqual(result["items"], [])
        self.assertEqual(result["offset"], 99)

    def test_limit_clamped_low(self):
        result = core.list_index_parts("idx", offset=0, limit=0)
        self.assertEqual(result["limit"], 1)
        self.assertEqual(len(result["items"]), 1)

    def test_limit_clamped_high(self):
        result = core.list_index_parts("idx", offset=0, limit=999999)
        self.assertEqual(result["limit"], 2000)

    def test_negative_offset_clamped(self):
        result = core.list_index_parts("idx", offset=-5, limit=10)
        self.assertEqual(result["offset"], 0)

    def test_kind_filter(self):
        parts = core.list_index_parts("idx", offset=0, limit=100, kind="part")
        self.assertTrue(all(it["kind"] == "part" for it in parts["items"]))
        self.assertEqual(parts["total"], 2)  # i=1,3 are parts

        asms = core.list_index_parts("idx", offset=0, limit=100, kind="assembly")
        self.assertEqual(asms["total"], 3)  # i=0,2,4

    def test_item_shape(self):
        result = core.list_index_parts("idx", offset=0, limit=1)
        item = result["items"][0]
        self.assertIn("has_thumbnail", item)
        self.assertIn("has_scs", item)
        self.assertFalse(item["has_thumbnail"])  # no asset files created
        self.assertFalse(item["has_scs"])


if __name__ == "__main__":
    unittest.main()
