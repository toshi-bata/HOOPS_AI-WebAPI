"""Tests for the HOOPS_AI_ENABLE_DEMO_FEATURES gate (core.require_demo_enabled).

Demo-only endpoints (named similarity index management, embedding-model
switching, Shape Space Map, MFR/Part-Classification dataset browsing,
context-layer prediction) must be hidden (404) unless the flag is enabled,
and reachable when it is enabled. conftest.py enables the flag for the rest
of the suite, so this module manages the env var itself per test.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import core


def _make_app():
    from routers import context_layer, mfr, part_classification, similarity

    app = FastAPI()
    app.include_router(mfr.router)
    app.include_router(similarity.router)
    app.include_router(part_classification.router)
    app.include_router(context_layer.router)
    return app


class DemoFeatureGateTests(unittest.TestCase):
    def setUp(self):
        self._orig = os.environ.get("HOOPS_AI_ENABLE_DEMO_FEATURES")
        self.client = TestClient(_make_app())

    def tearDown(self):
        if self._orig is None:
            os.environ.pop("HOOPS_AI_ENABLE_DEMO_FEATURES", None)
        else:
            os.environ["HOOPS_AI_ENABLE_DEMO_FEATURES"] = self._orig

    def test_gated_endpoints_return_404_when_disabled(self):
        os.environ["HOOPS_AI_ENABLE_DEMO_FEATURES"] = "false"
        for method, path in (
            ("get", "/MFR/dataset/table-of-contents"),
            ("get", "/similarity/default-model/setting"),
            ("get", "/similarity/index/list"),
            ("post", "/similarity/map"),
            ("get", "/part-classification/dataset/label-distribution"),
            ("post", "/context/predict"),
        ):
            with self.subTest(path=path):
                response = getattr(self.client, method)(path)
                self.assertEqual(response.status_code, 404)

    def test_gated_endpoint_reachable_when_enabled(self):
        os.environ["HOOPS_AI_ENABLE_DEMO_FEATURES"] = "true"
        # list_indexes has no other prerequisites (license/model), so it is a
        # good smoke test that the dependency no longer blocks the request.
        from unittest.mock import patch

        with patch.object(core, "list_indexes", return_value=[]):
            response = self.client.get("/similarity/index/list")
        self.assertEqual(response.status_code, 200)

    def test_non_gated_endpoint_unaffected_by_flag(self):
        os.environ["HOOPS_AI_ENABLE_DEMO_FEATURES"] = "false"
        # search_similarity_index (POST /similarity/index/{name}/search) stays
        # public; a missing file/file_id must still yield 422, not 404.
        response = self.client.post("/similarity/index/some-index/search")
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
