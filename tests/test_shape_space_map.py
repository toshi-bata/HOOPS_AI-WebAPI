"""Unit tests for the Shape Space Map MDS implementation and endpoint."""

import math
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _pairwise(coords):
    """Return the list of all i<j Euclidean distances between rows of coords."""
    n = coords.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(float(np.linalg.norm(coords[i] - coords[j])))
    return out


class MDSTests(unittest.TestCase):
    """Exercise core._classical_mds directly with known geometries."""

    def _compute_mds_coords(self, dist_matrix):
        import core

        coords, stress = core._classical_mds(np.asarray(dist_matrix, dtype=float))
        return coords, stress

    def test_equilateral_triangle(self):
        """N=3 all pairwise distances 1.0 → coords form an equilateral triangle."""
        D = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        coords, stress = self._compute_mds_coords(D)

        self.assertEqual(coords.shape, (3, 3))
        for d in _pairwise(coords):
            self.assertAlmostEqual(d, 1.0, delta=1e-6)
        self.assertAlmostEqual(stress, 0.0, delta=1e-6)

    def test_regular_tetrahedron(self):
        """N=4 all pairwise distances 1.0 → regular tetrahedron; all 6 dists ~1.0."""
        D = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]
        coords, stress = self._compute_mds_coords(D)

        self.assertEqual(coords.shape, (4, 3))
        dists = _pairwise(coords)
        self.assertEqual(len(dists), 6)
        for d in dists:
            self.assertAlmostEqual(d, 1.0, delta=1e-6)
        self.assertAlmostEqual(stress, 0.0, delta=1e-6)

    def test_coords_are_mean_centred(self):
        """MDS coordinates are centred at the origin."""
        D = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        coords, _ = self._compute_mds_coords(D)
        mean = coords.mean(axis=0)
        for c in mean:
            self.assertAlmostEqual(c, 0.0, delta=1e-9)

    def test_stress_computed_for_n6(self):
        """N=6 random symmetric distance matrix → stress is a float in [0, 1]."""
        rng = np.random.default_rng(42)
        n = 6
        m = rng.random((n, n))
        D = (m + m.T) / 2.0
        np.fill_diagonal(D, 0.0)

        coords, stress = self._compute_mds_coords(D)
        self.assertEqual(coords.shape, (6, 3))
        self.assertIsInstance(stress, float)
        self.assertGreaterEqual(stress, 0.0)
        self.assertLessEqual(stress, 1.0)

    def test_all_identical_similarity_zero_stress(self):
        """All distances 0 (identical parts) → stress is exactly 0.0."""
        n = 4
        D = np.zeros((n, n))
        _, stress = self._compute_mds_coords(D)
        self.assertEqual(stress, 0.0)


class ComputeShapeMapDataTests(unittest.TestCase):
    """Exercise the full pipeline with HOOPS/embedding dependencies mocked."""

    def test_stress_computed_for_n6(self):
        """compute_shape_map_data returns stress as a float in [0, 1] for N=6."""
        import core

        n = 6
        # A plausible similarity matrix (symmetric, diagonal 1.0).
        rng = np.random.default_rng(7)
        s = rng.random((n, n)) * 0.5 + 0.25
        s = (s + s.T) / 2.0
        np.fill_diagonal(s, 1.0)
        matrix = [[float(s[i][j]) for j in range(n)] for i in range(n)]

        fake_compare = {
            "count": n,
            "model_name": "hoops_embeddings_model",
            "matrix": matrix,
            "files": [
                {"index": i, "file_id": f"id_{i}", "filename": f"part_{i}.step", "num_bodies": 1}
                for i in range(n)
            ],
            "pairs": [],
        }

        with (
            patch.object(core, "compare_embeddings", return_value=fake_compare),
            patch.object(core, "export_scs_for_part", side_effect=lambda fid: f"{fid}.scs"),
            patch("builtins.open"),
            patch("json.dump"),
        ):
            result = core.compute_shape_map_data([f"id_{i}" for i in range(n)])

        self.assertEqual(result["count"], n)
        self.assertEqual(len(result["parts"]), n)
        self.assertIsInstance(result["stress"], float)
        self.assertGreaterEqual(result["stress"], 0.0)
        self.assertLessEqual(result["stress"], 1.0)
        for i, part in enumerate(result["parts"]):
            self.assertEqual(len(part["position"]), 3)
            self.assertEqual(part["scs_url"], f"/out/id_{i}.scs")
        self.assertEqual(result["errors"], [])

    def test_scs_failure_is_non_fatal(self):
        """SCS conversion failures are collected in errors, not raised."""
        import core

        n = 3
        matrix = [[1.0 if i == j else 0.5 for j in range(n)] for i in range(n)]
        fake_compare = {
            "count": n,
            "model_name": "m",
            "matrix": matrix,
            "files": [
                {"index": i, "file_id": f"id_{i}", "filename": f"part_{i}.step", "num_bodies": 1}
                for i in range(n)
            ],
            "pairs": [],
        }

        def flaky_export(fid):
            if fid == "id_1":
                raise RuntimeError("boom")
            return f"{fid}.scs"

        with (
            patch.object(core, "compare_embeddings", return_value=fake_compare),
            patch.object(core, "export_scs_for_part", side_effect=flaky_export),
            patch("builtins.open"),
            patch("json.dump"),
        ):
            result = core.compute_shape_map_data(["id_0", "id_1", "id_2"])

        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(result["errors"][0]["filename"], "part_1.step")
        self.assertIsNone(result["parts"][1]["scs_url"])

    @unittest.skip("Requires HOOPS license")
    def test_export_scs_for_part_real(self):
        """export_scs_for_part converts a real CAD file (needs HOOPS license)."""
        import core

        core.export_scs_for_part("some_file_id")


class ShapeMapEndpointValidationTests(unittest.TestCase):
    """FastAPI TestClient validation for POST /similarity/map."""

    def _client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.similarity import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_single_file_returns_422(self):
        """POST /similarity/map with only 1 file_id → 422."""
        import core
        import pathlib

        client = self._client()
        with patch.object(core, "find_persistent_CAD_file",
                          return_value=pathlib.Path("/fake/a.step")):
            resp = client.post("/similarity/map?file_ids=only_one")
        self.assertEqual(resp.status_code, 422)

    def test_no_files_returns_422(self):
        """POST /similarity/map with no input → 422."""
        client = self._client()
        resp = client.post("/similarity/map")
        self.assertEqual(resp.status_code, 422)

    def test_two_files_calls_compute(self):
        """POST /similarity/map with 2 valid file_ids returns 200 and calls core."""
        import core
        import pathlib

        client = self._client()

        fake_result = {
            "map_id": "abcd1234",
            "viewer_url": "/similarity/map/show?map=abcd1234",
            "count": 2,
            "parts": [
                {"index": 0, "file_id": "a", "filename": "a.step",
                 "scs_url": "/out/a.scs", "position": [0.0, 0.0, 0.0]},
                {"index": 1, "file_id": "b", "filename": "b.step",
                 "scs_url": "/out/b.scs", "position": [1.0, 0.0, 0.0]},
            ],
            "matrix": [[1.0, 0.5], [0.5, 1.0]],
            "stress": 0.0,
            "errors": [],
        }

        with (
            patch.object(core, "find_persistent_CAD_file", side_effect=lambda fid: pathlib.Path(f"/fake/{fid}.step")),
            patch.object(core, "compute_embedding", return_value={"file_id": "x"}),
            patch.object(core, "compute_shape_map_data", return_value=fake_result),
        ):
            resp = client.post("/similarity/map?file_ids=a,b")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["map_id"], "abcd1234")
        self.assertEqual(body["count"], 2)
        self.assertTrue(body["viewer_url"].endswith("/similarity/map/show?map=abcd1234"))
        self.assertTrue(body["parts"][0]["scs_url"].endswith("/out/a.scs"))
        self.assertTrue(body["parts"][0]["scs_url"].startswith("http"))


class ProjectOosMdsTests(unittest.TestCase):
    """Exercise core._project_oos_mds directly."""

    def test_known_collinear_projection(self):
        """Query equidistant from two endpoints → projected to their midpoint (origin)."""
        import core

        # Mean-centred MDS coords for 2 points 2 units apart: [-1, 0, 0] and [1, 0, 0]
        coords = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        dist_matrix = np.array([[0.0, 2.0], [2.0, 0.0]])
        # Query equidistant from both endpoints
        query_dist = np.array([math.sqrt(3), math.sqrt(3)])
        pos = core._project_oos_mds(coords, dist_matrix, query_dist)
        self.assertEqual(len(pos), 3)
        # Equidistant from both → projects to midpoint at origin (x ≈ 0)
        self.assertAlmostEqual(pos[0], 0.0, delta=0.1)

    def test_output_length_is_always_3(self):
        """Return value always has exactly 3 components."""
        import core

        coords = np.array([[0.0, 0.0, 0.0]])
        dist_matrix = np.array([[0.0]])
        query_dist = np.array([1.0])
        pos = core._project_oos_mds(coords, dist_matrix, query_dist)
        self.assertEqual(len(pos), 3)

    def test_empty_coords_returns_zeros(self):
        """Empty existing map → returns zero vector."""
        import core

        import numpy as _np

        pos = core._project_oos_mds(
            _np.zeros((0, 3)), _np.zeros((0, 0)), _np.zeros(0)
        )
        self.assertEqual(list(pos), [0.0, 0.0, 0.0])

    def test_identical_query_projects_near_existing(self):
        """Query identical to one existing part (distance 0) should land near it."""
        import core

        # Mean-centred MDS coords for 3 collinear parts with distances [[0,1,2],[1,0,1],[2,1,0]]
        coords = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        dist_matrix = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ])
        # Query is identical to the middle part (distance 0 from it, 1 from neighbours)
        query_dist = np.array([1.0, 0.0, 1.0])
        pos = core._project_oos_mds(coords, dist_matrix, query_dist)
        # Middle part is at origin (0, 0, 0) in mean-centred coords
        self.assertAlmostEqual(pos[0], 0.0, delta=0.2)
        self.assertAlmostEqual(pos[1], 0.0, delta=0.2)


class QueryShapeMapCoreTests(unittest.TestCase):
    """Exercise core.query_shape_map with mocked dependencies."""

    def _make_map_file(self, tmp_dir, map_id, n=3):
        """Write a minimal shape_map_{map_id}.json to tmp_dir and return its path."""
        import json, pathlib

        rng = np.random.default_rng(42)
        sim = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                v = round(float(rng.random() * 0.4 + 0.3), 6)
                sim[i, j] = v
                sim[j, i] = v
        matrix = sim.tolist()
        parts = [
            {
                "index": i,
                "file_id": f"id_{i}",
                "filename": f"part_{i}.step",
                "scs_url": f"/out/id_{i}.scs",
                "position": [float(i), 0.0, 0.0],
            }
            for i in range(n)
        ]
        data = {
            "map_id": map_id,
            "viewer_url": f"/similarity/map/show?map={map_id}",
            "count": n,
            "parts": parts,
            "matrix": matrix,
            "stress": 0.0,
            "errors": [],
        }
        p = pathlib.Path(tmp_dir) / f"shape_map_{map_id}.json"
        p.write_text(__import__("json").dumps(data), encoding="utf-8")
        return p

    def test_overlay_saved_and_returned(self):
        """query_shape_map saves a new overlay JSON and returns expected keys."""
        import core, json, pathlib, tempfile

        with tempfile.TemporaryDirectory() as tmp:
            orig_out = core.CAD_VIEWER_OUTPUT_DIR
            core.CAD_VIEWER_OUTPUT_DIR = pathlib.Path(tmp)
            try:
                self._make_map_file(tmp, "aabb1122")

                fake_emb = {
                    "file_id": "q0",
                    "vector": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    "dim": 4,
                    "model_name": "m",
                    "num_bodies": 1,
                    "filename": "query.step",
                    "cached": False,
                }

                def fake_compute(fid, **kwargs):
                    return {**fake_emb, "file_id": fid}

                with (
                    patch.object(core, "compute_embedding", side_effect=fake_compute),
                    patch.object(core, "export_scs_for_part", side_effect=lambda fid: f"{fid}.scs"),
                ):
                    result = core.query_shape_map("aabb1122", "q0")

                self.assertIn("overlay_map_id", result)
                self.assertIn("viewer_url", result)
                self.assertEqual(result["query_part"]["file_id"], "q0")
                self.assertTrue(result["query_part"]["is_query"])
                self.assertEqual(len(result["query_part"]["position"]), 3)
                self.assertFalse(result["persisted"])

                # Overlay file must exist
                overlay_path = pathlib.Path(tmp) / f"shape_map_{result['overlay_map_id']}.json"
                self.assertTrue(overlay_path.exists())
                overlay = json.loads(overlay_path.read_text())
                self.assertEqual(overlay["count"], 4)  # 3 existing + 1 query
                # Last part is the query
                self.assertTrue(overlay["parts"][-1]["is_query"])
                # Matrix is (4x4)
                self.assertEqual(len(overlay["matrix"]), 4)
                self.assertEqual(len(overlay["matrix"][0]), 4)
            finally:
                core.CAD_VIEWER_OUTPUT_DIR = orig_out

    def test_map_not_found_raises_key_error(self):
        """query_shape_map raises KeyError when the map JSON does not exist."""
        import core, pathlib, tempfile

        with tempfile.TemporaryDirectory() as tmp:
            orig_out = core.CAD_VIEWER_OUTPUT_DIR
            core.CAD_VIEWER_OUTPUT_DIR = pathlib.Path(tmp)
            try:
                with self.assertRaises(KeyError):
                    core.query_shape_map("nonexistent", "q0")
            finally:
                core.CAD_VIEWER_OUTPUT_DIR = orig_out

    def test_persist_updates_original_map(self):
        """persist=True adds the query part to the original map JSON."""
        import core, json, pathlib, tempfile

        with tempfile.TemporaryDirectory() as tmp:
            orig_out = core.CAD_VIEWER_OUTPUT_DIR
            core.CAD_VIEWER_OUTPUT_DIR = pathlib.Path(tmp)
            try:
                self._make_map_file(tmp, "ccdd5566")

                fake_emb = {
                    "file_id": "qp",
                    "vector": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                    "dim": 4,
                    "model_name": "m",
                    "num_bodies": 1,
                    "filename": "persist_query.step",
                    "cached": False,
                }

                def fake_compute(fid, **kwargs):
                    return {**fake_emb, "file_id": fid}

                with (
                    patch.object(core, "compute_embedding", side_effect=fake_compute),
                    patch.object(core, "export_scs_for_part", side_effect=lambda fid: f"{fid}.scs"),
                ):
                    result = core.query_shape_map("ccdd5566", "qp", persist=True)

                self.assertTrue(result["persisted"])
                orig = json.loads(
                    (pathlib.Path(tmp) / "shape_map_ccdd5566.json").read_text()
                )
                self.assertEqual(orig["count"], 4)
                self.assertEqual(orig["parts"][-1]["file_id"], "qp")
            finally:
                core.CAD_VIEWER_OUTPUT_DIR = orig_out

    def test_scs_failure_is_non_fatal(self):
        """SCS export failure for the query part is collected in errors, not raised."""
        import core, pathlib, tempfile

        with tempfile.TemporaryDirectory() as tmp:
            orig_out = core.CAD_VIEWER_OUTPUT_DIR
            core.CAD_VIEWER_OUTPUT_DIR = pathlib.Path(tmp)
            try:
                self._make_map_file(tmp, "eeff7788")

                fake_emb = {
                    "file_id": "qf",
                    "vector": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    "dim": 4,
                    "model_name": "m",
                    "num_bodies": 1,
                    "filename": "fail_query.step",
                    "cached": False,
                }

                def fake_compute(fid, **kwargs):
                    return {**fake_emb, "file_id": fid}

                with (
                    patch.object(core, "compute_embedding", side_effect=fake_compute),
                    patch.object(core, "export_scs_for_part", side_effect=RuntimeError("scs boom")),
                ):
                    result = core.query_shape_map("eeff7788", "qf")

                self.assertIsNone(result["query_part"]["scs_url"])
                self.assertEqual(len(result["errors"]), 1)
                self.assertIn("scs boom", result["errors"][0]["detail"])
            finally:
                core.CAD_VIEWER_OUTPUT_DIR = orig_out

    def test_nearest_parts_sorted_by_score(self):
        """nearest_parts is sorted by descending similarity score."""
        import core, pathlib, tempfile

        with tempfile.TemporaryDirectory() as tmp:
            orig_out = core.CAD_VIEWER_OUTPUT_DIR
            core.CAD_VIEWER_OUTPUT_DIR = pathlib.Path(tmp)
            try:
                self._make_map_file(tmp, "11223344", n=5)

                # Give the query a specific vector so dot products are predictable
                query_vec = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)
                # Make each part's vector distinct so similarities differ
                part_vecs = {
                    "id_0": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    "id_1": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                    "id_2": np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    "id_3": np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32),
                    "id_4": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                }
                dummy = {
                    "dim": 4, "model_name": "m", "num_bodies": 1,
                    "filename": "x.step", "cached": False,
                }

                def fake_compute(fid, **kwargs):
                    vec = part_vecs.get(fid, query_vec)
                    return {**dummy, "file_id": fid, "vector": vec}

                with (
                    patch.object(core, "compute_embedding", side_effect=fake_compute),
                    patch.object(core, "export_scs_for_part", side_effect=lambda fid: f"{fid}.scs"),
                ):
                    result = core.query_shape_map("11223344", "q_sort")

                scores = [p["score"] for p in result["nearest_parts"]]
                self.assertEqual(scores, sorted(scores, reverse=True))
            finally:
                core.CAD_VIEWER_OUTPUT_DIR = orig_out


class MapQueryEndpointTests(unittest.TestCase):
    """FastAPI TestClient tests for POST /similarity/map/{map_id}/query."""

    def _client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.similarity import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def _fake_query_result(self, overlay_id="ov123456"):
        return {
            "overlay_map_id": overlay_id,
            "viewer_url": f"/similarity/map/show?map={overlay_id}",
            "query_part": {
                "index": 2,
                "file_id": "qid",
                "filename": "query.step",
                "scs_url": "/out/qid.scs",
                "position": [0.5, 0.1, 0.0],
                "is_query": True,
            },
            "nearest_parts": [
                {"index": 0, "file_id": "a", "filename": "a.step", "score": 0.9},
                {"index": 1, "file_id": "b", "filename": "b.step", "score": 0.7},
            ],
            "persisted": False,
            "errors": [],
        }

    def test_no_input_returns_422(self):
        """POST with no file/file_id → 422."""
        client = self._client()
        resp = client.post("/similarity/map/abcd1234/query")
        self.assertEqual(resp.status_code, 422)

    def test_map_not_found_returns_404(self):
        """POST with a valid file_id but missing map → 404."""
        import core, pathlib

        client = self._client()
        with (
            patch.object(core, "find_persistent_CAD_file",
                         return_value=pathlib.Path("/fake/q.step")),
            patch.object(core, "query_shape_map",
                         side_effect=KeyError("Shape map 'missing' not found.")),
        ):
            resp = client.post("/similarity/map/missing/query?file_id=some_id")
        self.assertEqual(resp.status_code, 404)

    def test_success_returns_200_with_expected_shape(self):
        """POST with valid file_id and existing map → 200 with expected response."""
        import core

        client = self._client()
        fake_result = self._fake_query_result()

        with patch.object(core, "query_shape_map", return_value=fake_result):
            resp = client.post("/similarity/map/abcd1234/query?file_id=qid")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["overlay_map_id"], "ov123456")
        self.assertTrue(body["viewer_url"].endswith("/similarity/map/show?map=ov123456"))
        self.assertTrue(body["query_part"]["is_query"])
        self.assertTrue(body["query_part"]["scs_url"].startswith("http"))
        self.assertEqual(len(body["nearest_parts"]), 2)
        self.assertAlmostEqual(body["nearest_parts"][0]["score"], 0.9)
        self.assertFalse(body["persisted"])

    def test_persist_flag_forwarded(self):
        """persist=true is forwarded to core.query_shape_map."""
        import core

        client = self._client()
        fake_result = {**self._fake_query_result(), "persisted": True}

        with patch.object(core, "query_shape_map", return_value=fake_result) as mock_fn:
            resp = client.post("/similarity/map/abcd1234/query?file_id=qid&persist=true")

        self.assertEqual(resp.status_code, 200)
        _, kwargs = mock_fn.call_args
        self.assertTrue(kwargs.get("persist") or mock_fn.call_args[0][2])
        self.assertTrue(resp.json()["persisted"])


if __name__ == "__main__":
    unittest.main()
