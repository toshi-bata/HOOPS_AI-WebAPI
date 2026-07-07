"""Unit tests for core.get_similar_search_index_info."""

import unittest
from unittest.mock import patch


class FakeEmbedder:
    model_name = "hoops_embeddings_model"


class FakeCADSearch:
    def __init__(self, embedder=None):
        self._shape_model = embedder or FakeEmbedder()


class FakeEmbeddingBatch:
    def __init__(self, ids=None, model=None, metadata=None, dim=512):
        self.ids = ids if ids is not None else [f"part_{i:03d}" for i in range(5)]
        self.model = model or "hoops_embeddings_model"
        self.metadata = metadata
        self.dim = dim


# Helpers to populate the preset dicts for a given preset key
def _patch_preset(searcher, batch, preset="signal"):
    """Return context managers that populate the preset caches."""
    import core
    searchers = dict(core._default_index_searchers)
    shapes = dict(core._default_index_shapes)
    if searcher is not None:
        searchers[preset] = searcher
    if batch is not None:
        shapes[preset] = batch
    return (
        patch.object(core, "_default_index_searchers", searchers),
        patch.object(core, "_default_index_shapes", shapes),
    )


class SimilarSearchIndexInfoTests(unittest.TestCase):

    # ------------------------------------------------------------------
    # not_loaded state
    # ------------------------------------------------------------------

    def test_returns_not_loaded_when_neither_searcher_nor_index_is_loaded(self):
        """When caches are empty and get_shape_index_for raises → status not_loaded."""
        import core

        with (
            patch.object(core, "_default_index_searchers", {}),
            patch.object(core, "_default_index_shapes", {}),
            patch.object(core, "get_shape_index_for", side_effect=RuntimeError("no index")),
            patch.object(core, "_resolve_default_index_paths", side_effect=RuntimeError("no env")),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = core.get_similar_search_index_info()

        self.assertEqual(result["status"], "not_loaded")
        self.assertIsNone(result["index_count"])
        self.assertIsNone(result["model_name"])
        self.assertIsNone(result["embedding_dim"])
        self.assertIsNone(result["metadata"])

    def test_not_loaded_includes_index_path_from_env(self):
        """Even when not loaded, index_path is populated from _resolve_default_index_paths."""
        import pathlib
        import core

        fake_path = pathlib.Path("/fake/notebooks/my_index.faiss")

        with (
            patch.object(core, "_default_index_searchers", {}),
            patch.object(core, "_default_index_shapes", {}),
            patch.object(core, "get_shape_index_for", side_effect=RuntimeError("no index")),
            patch.object(core, "_resolve_default_index_paths",
                         return_value={"faiss_path": fake_path, "images_dir": pathlib.Path("/fake/images"), "model_key": "legacy"}),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = core.get_similar_search_index_info()

        self.assertEqual(result["status"], "not_loaded")
        self.assertIn("my_index.faiss", result["index_path"])

    # ------------------------------------------------------------------
    # loaded state — embedder attributes
    # ------------------------------------------------------------------

    def test_returns_loaded_with_model_name_and_dim_from_embedder(self):
        """model_name and embedding_dim come from the preset's CADSearch and EmbeddingBatch."""
        import pathlib
        import core

        preset = core.get_active_default_index()
        searcher = FakeCADSearch()
        batch = FakeEmbeddingBatch(ids=list(range(100)))
        fake_path = pathlib.Path("/fake/index.faiss")

        cm1, cm2 = _patch_preset(searcher, batch, preset)
        with (
            cm1, cm2,
            patch.object(core, "_resolve_default_index_paths",
                         return_value={"faiss_path": fake_path, "images_dir": pathlib.Path("/fake/images"), "model_key": preset}),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = core.get_similar_search_index_info()

        self.assertEqual(result["status"], "loaded")
        self.assertEqual(result["model_name"], "hoops_embeddings_model")
        self.assertEqual(result["embedding_dim"], 512)
        self.assertEqual(result["index_count"], 100)

    def test_returns_loaded_when_only_searcher_is_set(self):
        """Works even when shape_index cache is absent but searcher is present."""
        import pathlib
        import core

        preset = core.get_active_default_index()
        searcher = FakeCADSearch()
        fake_path = pathlib.Path("/fake/index.faiss")

        cm1, cm2 = _patch_preset(searcher, None, preset)
        with (
            cm1, cm2,
            patch.object(core, "_resolve_default_index_paths",
                         return_value={"faiss_path": fake_path, "images_dir": pathlib.Path("/fake/images"), "model_key": preset}),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = core.get_similar_search_index_info()

        self.assertEqual(result["status"], "loaded")
        self.assertEqual(result["model_name"], "hoops_embeddings_model")
        self.assertIsNone(result["index_count"])

    # ------------------------------------------------------------------
    # loaded state — EmbeddingBatch attributes
    # ------------------------------------------------------------------

    def test_index_count_reflects_number_of_ids_in_batch(self):
        import pathlib
        import core

        preset = core.get_active_default_index()
        ids = [f"part_{i}" for i in range(42)]
        batch = FakeEmbeddingBatch(ids=ids)
        searcher = FakeCADSearch()
        fake_path = pathlib.Path("/fake/index.faiss")

        cm1, cm2 = _patch_preset(searcher, batch, preset)
        with (
            cm1, cm2,
            patch.object(core, "_resolve_default_index_paths",
                         return_value={"faiss_path": fake_path, "images_dir": pathlib.Path("/fake/images"), "model_key": preset}),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = core.get_similar_search_index_info()

        self.assertEqual(result["index_count"], 42)

    def test_metadata_is_returned_when_present(self):
        import pathlib
        import core

        preset = core.get_active_default_index()
        batch = FakeEmbeddingBatch(metadata={"failed_count": 3, "source": "fabwave"})
        searcher = FakeCADSearch()
        fake_path = pathlib.Path("/fake/index.faiss")

        cm1, cm2 = _patch_preset(searcher, batch, preset)
        with (
            cm1, cm2,
            patch.object(core, "_resolve_default_index_paths",
                         return_value={"faiss_path": fake_path, "images_dir": pathlib.Path("/fake/images"), "model_key": preset}),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = core.get_similar_search_index_info()

        self.assertIsNotNone(result["metadata"])
        self.assertEqual(result["metadata"]["failed_count"], 3)
        self.assertEqual(result["metadata"]["source"], "fabwave")

    def test_metadata_is_none_when_absent(self):
        import pathlib
        import core

        preset = core.get_active_default_index()
        batch = FakeEmbeddingBatch(metadata=None)
        searcher = FakeCADSearch()
        fake_path = pathlib.Path("/fake/index.faiss")

        cm1, cm2 = _patch_preset(searcher, batch, preset)
        with (
            cm1, cm2,
            patch.object(core, "_resolve_default_index_paths",
                         return_value={"faiss_path": fake_path, "images_dir": pathlib.Path("/fake/images"), "model_key": preset}),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = core.get_similar_search_index_info()

        self.assertIsNone(result["metadata"])

    def test_model_name_falls_back_to_embedding_batch_model(self):
        """When embedder has no model_name, fall back to shape_index.model."""
        import pathlib
        import core

        class EmbedderNoName:
            model_name = None

        preset = core.get_active_default_index()
        searcher = FakeCADSearch(embedder=EmbedderNoName())
        batch = FakeEmbeddingBatch(model="fallback_model_v2")
        fake_path = pathlib.Path("/fake/index.faiss")

        cm1, cm2 = _patch_preset(searcher, batch, preset)
        with (
            cm1, cm2,
            patch.object(core, "_resolve_default_index_paths",
                         return_value={"faiss_path": fake_path, "images_dir": pathlib.Path("/fake/images"), "model_key": preset}),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = core.get_similar_search_index_info()

        self.assertEqual(result["model_name"], "fallback_model_v2")

    # ------------------------------------------------------------------
    # HTTP layer (router)
    # ------------------------------------------------------------------

    def test_router_endpoint_returns_200_with_not_loaded_payload(self):
        """GET /similarity/index-info returns 200 even when index is not loaded."""
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        from fastapi.testclient import TestClient
        import core
        from routers.similarity import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        not_loaded_info = {
            "preset": "signal",
            "status": "not_loaded",
            "index_path": None,
            "index_last_modified": None,
            "index_count": None,
            "model_name": None,
            "embedding_dim": None,
            "metadata": None,
        }
        with patch.object(core, "get_similar_search_index_info", return_value=not_loaded_info):
            response = client.get("/similarity/index-info")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "not_loaded")
        self.assertIsNone(body["index_count"])

    def test_router_endpoint_returns_200_with_loaded_payload(self):
        """GET /similarity/index-info returns full metadata when index is loaded."""
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        from fastapi.testclient import TestClient
        import core
        from routers.similarity import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        loaded_info = {
            "preset": "signal",
            "status": "loaded",
            "index_path": "/fake/notebooks/fabwave_embeddings_store.faiss",
            "index_last_modified": "2025-06-01T12:00:00Z",
            "index_count": 5000,
            "model_name": "hoops_embeddings_model",
            "embedding_dim": 512,
            "metadata": {"failed_count": 0},
        }
        with patch.object(core, "get_similar_search_index_info", return_value=loaded_info):
            response = client.get("/similarity/index-info")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "loaded")
        self.assertEqual(body["index_count"], 5000)
        self.assertEqual(body["model_name"], "hoops_embeddings_model")
        self.assertEqual(body["embedding_dim"], 512)
        self.assertEqual(body["metadata"]["failed_count"], 0)
        self.assertNotIn("index_path", body)


if __name__ == "__main__":
    unittest.main()
