"""SDK-free unit tests for the search result-grid image.

The contact sheet used to cost one PNG decode plus one matplotlib subplot per
hit, which made the search endpoint scale linearly with top_k (39 s of a 40 s
request at top_k=300). These tests pin the two fixes: the sheet is capped at
``_GRID_MAX_TILES`` tiles, and ``include_image=False`` skips image work entirely
without touching the hits.

matplotlib is replaced with a fake, so neither it nor the SDK is required.
"""

import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core


def _hits(n):
    return [
        {"id": f"{i:064d}", "score": 1.0 - i / 1000.0, "metadata": {"filename": f"{i}.stp"}}
        for i in range(n)
    ]


class TestGridTiles(unittest.TestCase):
    def test_caps_at_max_tiles(self):
        shown, caption = core._grid_tiles(_hits(300))
        self.assertEqual(len(shown), core._GRID_MAX_TILES)
        self.assertIn("of 300", caption)
        self.assertEqual(caption, f"top {core._GRID_MAX_TILES} of 300")

    def test_keeps_the_highest_ranked_hits(self):
        hits = _hits(300)
        shown, _ = core._grid_tiles(hits)
        self.assertEqual(shown, hits[: core._GRID_MAX_TILES])

    def test_draws_everything_below_the_cap(self):
        shown, caption = core._grid_tiles(_hits(5))
        self.assertEqual(len(shown), 5)
        self.assertEqual(caption, "top 5 of 5")

    def test_empty_hits(self):
        shown, caption = core._grid_tiles([])
        self.assertEqual(shown, [])
        self.assertEqual(caption, "top 0 of 0")


class _FakeAxes:
    transAxes = object()

    def __init__(self):
        self.titles = []
        self.off = False

    def set_facecolor(self, *_a, **_k):
        pass

    def text(self, *_a, **_k):
        pass

    def imshow(self, *_a, **_k):
        pass

    def set_title(self, title, **_k):
        self.titles.append(title)

    def set_xlabel(self, *_a, **_k):
        pass

    def set_xticks(self, *_a, **_k):
        pass

    def set_yticks(self, *_a, **_k):
        pass

    def axis(self, _mode):
        self.off = True


class _FakeFigure:
    def __init__(self):
        self.texts = []
        self.saved = []

    def text(self, _x, _y, s, **_k):
        self.texts.append(s)

    def savefig(self, path, **_k):
        self.saved.append(path)


class TestBuildSearchGridImage(unittest.TestCase):
    """Exercises the real function with matplotlib faked out."""

    def setUp(self):
        import numpy as np

        self.figures = []
        self.axes_created = []

        def _subplots(rows, cols, **_k):
            fig = _FakeFigure()
            self.figures.append(fig)
            axes = [_FakeAxes() for _ in range(rows * cols)]
            self.axes_created.append(axes)
            return fig, np.array(axes, dtype=object).reshape(rows, cols)

        plt = types.ModuleType("matplotlib.pyplot")
        plt.subplots = _subplots
        plt.tight_layout = lambda **_k: None
        plt.close = lambda _fig: None

        mpimg = types.ModuleType("matplotlib.image")
        mpimg.imread = lambda _p: [[0]]

        mpl = types.ModuleType("matplotlib")
        mpl.use = lambda _b: None
        # `import matplotlib.image as mpimg` resolves the attribute on the parent
        # module, so submodules must be attached, not only put in sys.modules.
        mpl.image = mpimg
        mpl.pyplot = plt

        self._saved_modules = {
            k: sys.modules.get(k)
            for k in ("matplotlib", "matplotlib.pyplot", "matplotlib.image")
        }
        sys.modules["matplotlib"] = mpl
        sys.modules["matplotlib.pyplot"] = plt
        sys.modules["matplotlib.image"] = mpimg

        self._tmp = tempfile.TemporaryDirectory()
        self._orig_out = core.CAD_VIEWER_OUTPUT_DIR
        core.CAD_VIEWER_OUTPUT_DIR = pathlib.Path(self._tmp.name)

    def tearDown(self):
        for k, v in self._saved_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        core.CAD_VIEWER_OUTPUT_DIR = self._orig_out
        self._tmp.cleanup()

    def _drawn_cells(self):
        return sum(1 for ax in self.axes_created[0] if ax.titles)

    def test_large_result_draws_only_capped_tiles(self):
        url = core._build_search_grid_image("q" * 64, _hits(300), "idx")
        self.assertTrue(url.startswith("/out/"))
        # One query cell plus at most _GRID_MAX_TILES hit cells.
        self.assertEqual(self._drawn_cells(), core._GRID_MAX_TILES + 1)

    def test_caption_reports_the_full_total(self):
        core._build_search_grid_image("q" * 64, _hits(300), "idx")
        self.assertIn(f"top {core._GRID_MAX_TILES} of 300", self.figures[0].texts)

    def test_small_result_draws_every_hit(self):
        core._build_search_grid_image("q" * 64, _hits(5), "idx")
        self.assertEqual(self._drawn_cells(), 6)
        self.assertIn("top 5 of 5", self.figures[0].texts)


class TestSearchIndexIncludeImage(unittest.TestCase):
    """include_image must control image work only, never the hits."""

    QUERY_ID = "q" * 64

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_idx_dir = core.INDEXES_DIR
        core.INDEXES_DIR = pathlib.Path(self._tmp.name)

        from types import SimpleNamespace

        results = [
            [SimpleNamespace(id="a" * 64, score=0.9, metadata={"file_id": "a" * 64}),
             SimpleNamespace(id="b" * 64, score=0.8, metadata={"file_id": "b" * 64})],
            [SimpleNamespace(id="c" * 64, score=0.7, metadata={"file_id": "c" * 64})],
        ]
        searcher = SimpleNamespace(search_by_shape=lambda path, **kw: results)

        vs = MagicMock()
        vs.get_ids.side_effect = lambda: ["a" * 64, "b" * 64, "c" * 64]
        vs.iter_metadata.side_effect = lambda: iter([])

        self._saved = {
            "_load_named_index": core._load_named_index,
            "_get_named_searcher": core._get_named_searcher,
            "_load_index_model": core._load_index_model,
            "find_persistent_CAD_file": core.find_persistent_CAD_file,
            "_generate_part_thumbnail": core._generate_part_thumbnail,
            "_build_search_grid_image": core._build_search_grid_image,
            "_resolve_index_asset": core._resolve_index_asset,
        }
        self.thumbnail_calls = []
        self.grid_calls = []
        core._load_named_index = lambda name: vs
        core._get_named_searcher = lambda name, model: searcher
        core._load_index_model = lambda name: "signal"
        core.find_persistent_CAD_file = lambda fid: pathlib.Path(self._tmp.name) / f"{fid}.stp"
        core._generate_part_thumbnail = lambda fid, name: self.thumbnail_calls.append(fid)
        core._resolve_index_asset = lambda name, pid, asset: None

        def _grid(fid, hits, name):
            self.grid_calls.append(len(hits))
            return "/out/fake.png"

        core._build_search_grid_image = _grid

    def tearDown(self):
        for attr, val in self._saved.items():
            setattr(core, attr, val)
        core.INDEXES_DIR = self._orig_idx_dir
        self._tmp.cleanup()

    def test_include_image_false_returns_no_url_and_skips_grid(self):
        out = core.search_index("idx", self.QUERY_ID, 5, include_image=False)
        self.assertIsNone(out["image_url"])
        self.assertEqual(self.grid_calls, [])

    def test_include_image_false_skips_query_thumbnail_generation(self):
        core.search_index("idx", self.QUERY_ID, 5, include_image=False)
        self.assertEqual(self.thumbnail_calls, [])

    def test_include_image_true_is_the_default(self):
        out = core.search_index("idx", self.QUERY_ID, 5)
        self.assertEqual(out["image_url"], "/out/fake.png")
        self.assertEqual(self.grid_calls, [3])
        self.assertEqual(self.thumbnail_calls, [self.QUERY_ID])

    def test_hits_are_identical_either_way(self):
        with_image = core.search_index("idx", self.QUERY_ID, 5, include_image=True)
        without = core.search_index("idx", self.QUERY_ID, 5, include_image=False)
        self.assertEqual(with_image["hits"], without["hits"])
        self.assertEqual(with_image["count"], without["count"])

    def test_no_image_work_when_there_are_no_hits(self):
        core._get_named_searcher = lambda name, model: types.SimpleNamespace(
            search_by_shape=lambda path, **kw: []
        )
        out = core.search_index("idx", self.QUERY_ID, 5)
        self.assertIsNone(out["image_url"])
        self.assertEqual(self.grid_calls, [])
        self.assertEqual(self.thumbnail_calls, [])


if __name__ == "__main__":
    unittest.main()
