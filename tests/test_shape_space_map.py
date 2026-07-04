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


if __name__ == "__main__":
    unittest.main()
