"""Tests for core.predict_context and the POST /context/predict router.

All tests mock ``hoops_ai.ml.context_layer`` so that no HOOPS AI installation
or licence is required.  The test style follows the conventions established in
``tests/test_mfr_search.py`` (unittest + unittest.mock.patch).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_context_layer():
    """Return a pre-configured mock for ``hoops_ai.ml.context_layer``.

    ``ContextProvider`` is set to ``object`` so that the inner
    ``_StaticContextProvider(ContextProvider)`` class definition inside
    ``predict_context`` works without metaclass conflicts.
    """
    mock_cl = MagicMock()
    mock_cl.ContextProvider = object  # valid Python base class

    # Default infer return value: one prediction
    pred = MagicMock()
    pred.value = "Steel"
    pred.confidence = 0.9
    pred.status = "ready_to_propose"
    pred.injected_context = None
    mock_cl.ContextPredictor.return_value.infer.return_value = {"Material": pred}

    return mock_cl


def _make_app():
    """Return a minimal FastAPI app with only the context_layer router mounted.

    Registers the same ``EnvConfigError`` / ``PathConfigError`` exception
    handlers as ``main.py`` so that the 503 behaviour can be tested without
    depending on the full application stack.
    """
    import core as _core
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from routers.context_layer import router

    app = FastAPI()

    @app.exception_handler(_core.EnvConfigError)
    async def env_config_error_handler(request: Request, exc: _core.EnvConfigError):
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Service unavailable: server configuration error. "
                "Check the server logs for details."
            },
        )

    @app.exception_handler(_core.PathConfigError)
    async def path_config_error_handler(request: Request, exc: _core.PathConfigError):
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Service unavailable: required server resource not found. "
                "Check the server logs for details."
            },
        )

    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# core.predict_context — unit tests
# ---------------------------------------------------------------------------


class TestPredictContextNormalCase(unittest.TestCase):
    """Normal-path tests for core.predict_context with a mocked context_layer."""

    def _run(self, mock_cl, **kwargs):
        import core

        modules = {
            "hoops_ai": MagicMock(),
            "hoops_ai.ml": MagicMock(),
            "hoops_ai.ml.context_layer": mock_cl,
        }
        with patch.dict(sys.modules, modules):
            return core.predict_context(**kwargs)

    # ------------------------------------------------------------------
    # Basic happy path
    # ------------------------------------------------------------------

    def test_returns_expected_keys_and_structure(self):
        """predict_context returns a dict keyed by predicted metadata keys."""
        mock_cl = _make_mock_context_layer()

        result = self._run(
            mock_cl,
            hits=[{"id": "part-a", "score": 0.95}, {"id": "part-b", "score": 0.80}],
            contexts={
                "part-a": {"Material": "Steel", "CostUSD": 12.5},
                "part-b": {"Material": "Aluminum", "CostUSD": 9.0},
            },
            keys=["Material"],
        )

        self.assertIn("Material", result)
        entry = result["Material"]
        self.assertIn("value", entry)
        self.assertIn("confidence", entry)
        self.assertIn("status", entry)
        self.assertIn("injected_context", entry)

    def test_serialised_values_match_mock_prediction(self):
        """Returned value, confidence, and status match the mocked ContextPrediction."""
        mock_cl = _make_mock_context_layer()

        result = self._run(
            mock_cl,
            hits=[{"id": "part-a", "score": 0.95}],
            contexts={"part-a": {"Material": "Steel"}},
            keys=["Material"],
        )

        self.assertEqual(result["Material"]["value"], "Steel")
        self.assertAlmostEqual(result["Material"]["confidence"], 0.9)
        self.assertEqual(result["Material"]["status"], "ready_to_propose")
        self.assertIsNone(result["Material"]["injected_context"])

    # ------------------------------------------------------------------
    # ContextPredictor.infer receives correct hit objects
    # ------------------------------------------------------------------

    def test_infer_receives_hit_objects_with_id_and_score(self):
        """ContextPredictor.infer is called with _Hit objects having correct id/score."""
        mock_cl = _make_mock_context_layer()

        self._run(
            mock_cl,
            hits=[
                {"id": "part-x", "score": 0.88},
                {"id": "part-y", "score": 0.72},
                {"id": "part-z", "score": 0.61},
            ],
            contexts={
                "part-x": {"Material": "Steel"},
                "part-y": {"Material": "Iron"},
                "part-z": {"Material": "Aluminum"},
            },
            keys=["Material"],
        )

        predictor_instance = mock_cl.ContextPredictor.return_value
        infer_call_args = predictor_instance.infer.call_args
        hit_objs = infer_call_args[0][0]  # first positional arg

        self.assertEqual(len(hit_objs), 3)
        self.assertEqual(hit_objs[0].id, "part-x")
        self.assertAlmostEqual(hit_objs[0].score, 0.88)
        self.assertEqual(hit_objs[1].id, "part-y")
        self.assertAlmostEqual(hit_objs[2].score, 0.61)

    # ------------------------------------------------------------------
    # _StaticContextProvider.get_contexts filters to requested ids
    # ------------------------------------------------------------------

    def test_provider_get_contexts_returns_only_requested_ids(self):
        """The _StaticContextProvider passed to ContextPredictor only returns
        data for ids explicitly requested via get_contexts."""
        mock_cl = _make_mock_context_layer()
        captured = []

        original_predictor = mock_cl.ContextPredictor

        def capturing_predictor(context_provider, **kwargs):
            captured.append(context_provider)
            return original_predictor.return_value

        mock_cl.ContextPredictor.side_effect = capturing_predictor

        self._run(
            mock_cl,
            hits=[{"id": "id1", "score": 0.9}, {"id": "id2", "score": 0.7}],
            contexts={
                "id1": {"Material": "Steel"},
                "id2": {"Material": "Iron"},
                "id3": {"Material": "Copper"},  # extra entry, not in hits
            },
            keys=["Material"],
        )

        self.assertEqual(len(captured), 1)
        provider = captured[0]

        # Requesting id1 and id2 → both returned
        ctx = provider.get_contexts(["id1", "id2"])
        self.assertEqual(ctx["id1"], {"Material": "Steel"})
        self.assertEqual(ctx["id2"], {"Material": "Iron"})

        # Requesting an unknown id → empty dict
        ctx2 = provider.get_contexts(["unknown-id"])
        self.assertEqual(ctx2["unknown-id"], {})

    def test_provider_list_numeric_keys_returns_supplied_keys(self):
        """The _StaticContextProvider's list_numeric_keys returns numeric_keys arg."""
        mock_cl = _make_mock_context_layer()
        captured = []

        def capturing_predictor(context_provider, **kwargs):
            captured.append(context_provider)
            return mock_cl.ContextPredictor.return_value

        mock_cl.ContextPredictor.side_effect = capturing_predictor

        self._run(
            mock_cl,
            hits=[{"id": "p1", "score": 0.9}],
            contexts={"p1": {"CostUSD": 10.0}},
            keys=["CostUSD"],
            numeric_keys=["CostUSD"],
        )

        provider = captured[0]
        self.assertEqual(provider.list_numeric_keys(), ["CostUSD"])


# ---------------------------------------------------------------------------
# per_key_rules — rule construction tests
# ---------------------------------------------------------------------------


class TestBuildAggregationRule(unittest.TestCase):
    """Verify that _build_aggregation_rule constructs the right rule instances."""

    def _call_predict_context(self, mock_cl, per_key_rules):
        import core

        modules = {
            "hoops_ai": MagicMock(),
            "hoops_ai.ml": MagicMock(),
            "hoops_ai.ml.context_layer": mock_cl,
        }
        with patch.dict(sys.modules, modules):
            core.predict_context(
                hits=[{"id": "p", "score": 0.9}],
                contexts={"p": {"CostUSD": 10.0}},
                keys=list(per_key_rules.keys()),
                per_key_rules=per_key_rules,
            )

    def test_numeric_weighted_rule_constructed_with_correct_params(self):
        """NumericWeightedRule is built with log_scale and auto_relevance_weight."""
        mock_cl = _make_mock_context_layer()
        mock_numeric_rule = MagicMock()
        mock_cl.NumericWeightedRule.return_value = mock_numeric_rule

        self._call_predict_context(
            mock_cl,
            per_key_rules={
                "CostUSD": {
                    "type": "numeric_weighted",
                    "log_scale": True,
                    "auto_relevance_weight": False,
                    "nearest_neighbor_threshold": None,
                    "score_temperature": None,
                }
            },
        )

        mock_cl.NumericWeightedRule.assert_called_once_with(
            log_scale=True,
            auto_relevance_weight=False,
        )

    def test_numeric_weighted_rule_with_optional_params(self):
        """NumericWeightedRule includes nearest_neighbor_threshold and score_temperature
        when set."""
        mock_cl = _make_mock_context_layer()

        self._call_predict_context(
            mock_cl,
            per_key_rules={
                "CostUSD": {
                    "type": "numeric_weighted",
                    "log_scale": False,
                    "auto_relevance_weight": True,
                    "nearest_neighbor_threshold": 0.85,
                    "score_temperature": 8.0,
                }
            },
        )

        mock_cl.NumericWeightedRule.assert_called_once_with(
            log_scale=False,
            auto_relevance_weight=True,
            nearest_neighbor_threshold=0.85,
            score_temperature=8.0,
        )

    def test_nearest_neighbor_rule_constructed_with_threshold(self):
        """NearestNeighborRule is built with threshold when provided."""
        mock_cl = _make_mock_context_layer()
        mock_nn_rule = MagicMock()
        mock_cl.NearestNeighborRule.return_value = mock_nn_rule

        self._call_predict_context(
            mock_cl,
            per_key_rules={
                "Material": {
                    "type": "nearest_neighbor",
                    "log_scale": False,
                    "auto_relevance_weight": False,
                    "nearest_neighbor_threshold": None,
                    "score_temperature": None,
                    "threshold": 0.75,
                }
            },
        )

        mock_cl.NearestNeighborRule.assert_called_once_with(threshold=0.75)

    def test_nearest_neighbor_rule_without_threshold(self):
        """NearestNeighborRule is built with no args when threshold is None."""
        mock_cl = _make_mock_context_layer()

        self._call_predict_context(
            mock_cl,
            per_key_rules={
                "Material": {
                    "type": "nearest_neighbor",
                    "log_scale": False,
                    "auto_relevance_weight": False,
                    "nearest_neighbor_threshold": None,
                    "score_temperature": None,
                    "threshold": None,
                }
            },
        )

        mock_cl.NearestNeighborRule.assert_called_once_with()

    def test_unknown_rule_type_raises_value_error(self):
        """An unknown rule type raises ValueError (propagated from _build_aggregation_rule)."""
        import core

        mock_cl = _make_mock_context_layer()
        modules = {
            "hoops_ai": MagicMock(),
            "hoops_ai.ml": MagicMock(),
            "hoops_ai.ml.context_layer": mock_cl,
        }
        with patch.dict(sys.modules, modules):
            with self.assertRaises(ValueError) as ctx:
                core.predict_context(
                    hits=[{"id": "p", "score": 0.9}],
                    contexts={"p": {}},
                    keys=["X"],
                    per_key_rules={
                        "X": {
                            "type": "bogus_rule",
                            "log_scale": False,
                            "auto_relevance_weight": False,
                            "nearest_neighbor_threshold": None,
                            "score_temperature": None,
                            "threshold": None,
                        }
                    },
                )
        self.assertIn("bogus_rule", str(ctx.exception))

    def test_mixed_rule_types_in_per_key_rules(self):
        """NumericWeightedRule and NearestNeighborRule can coexist in per_key_rules."""
        mock_cl = _make_mock_context_layer()

        # Two predictions in infer result
        pred_cost = MagicMock()
        pred_cost.value = 15.0
        pred_cost.confidence = 0.8
        pred_cost.status = "ready_to_propose"
        pred_cost.injected_context = None

        pred_material = MagicMock()
        pred_material.value = "Steel"
        pred_material.confidence = 0.95
        pred_material.status = "ready_to_propose"
        pred_material.injected_context = None

        mock_cl.ContextPredictor.return_value.infer.return_value = {
            "CostUSD": pred_cost,
            "Material": pred_material,
        }

        import core

        modules = {
            "hoops_ai": MagicMock(),
            "hoops_ai.ml": MagicMock(),
            "hoops_ai.ml.context_layer": mock_cl,
        }
        with patch.dict(sys.modules, modules):
            result = core.predict_context(
                hits=[{"id": "p", "score": 0.9}],
                contexts={"p": {"CostUSD": 10.0, "Material": "Iron"}},
                keys=["CostUSD", "Material"],
                per_key_rules={
                    "CostUSD": {
                        "type": "numeric_weighted",
                        "log_scale": True,
                        "auto_relevance_weight": True,
                        "nearest_neighbor_threshold": None,
                        "score_temperature": None,
                    },
                    "Material": {
                        "type": "nearest_neighbor",
                        "log_scale": False,
                        "auto_relevance_weight": False,
                        "nearest_neighbor_threshold": None,
                        "score_temperature": None,
                        "threshold": 0.8,
                    },
                },
            )

        self.assertIn("CostUSD", result)
        self.assertIn("Material", result)
        mock_cl.NumericWeightedRule.assert_called_once()
        mock_cl.NearestNeighborRule.assert_called_once_with(threshold=0.8)


# ---------------------------------------------------------------------------
# Return value serialisation
# ---------------------------------------------------------------------------


class TestPredictContextReturnFormat(unittest.TestCase):
    """Verify that predict_context converts ContextPrediction to plain dicts."""

    def test_injected_context_included_when_present(self):
        """injected_context is serialised when non-None."""
        mock_cl = _make_mock_context_layer()

        pred = MagicMock()
        pred.value = "Aluminum"
        pred.confidence = 0.75
        pred.status = "needs_review"
        pred.injected_context = {"Material": "Iron"}  # known query context
        mock_cl.ContextPredictor.return_value.infer.return_value = {"Material": pred}

        import core

        modules = {
            "hoops_ai": MagicMock(),
            "hoops_ai.ml": MagicMock(),
            "hoops_ai.ml.context_layer": mock_cl,
        }
        with patch.dict(sys.modules, modules):
            result = core.predict_context(
                hits=[{"id": "p", "score": 0.9}],
                contexts={"p": {"Material": "Iron"}},
                keys=["Material"],
                query_context={"Material": "Iron"},
            )

        self.assertEqual(result["Material"]["injected_context"], {"Material": "Iron"})
        self.assertEqual(result["Material"]["status"], "needs_review")

    def test_confidence_is_float(self):
        """confidence is always a Python float."""
        mock_cl = _make_mock_context_layer()

        import core

        modules = {
            "hoops_ai": MagicMock(),
            "hoops_ai.ml": MagicMock(),
            "hoops_ai.ml.context_layer": mock_cl,
        }
        with patch.dict(sys.modules, modules):
            result = core.predict_context(
                hits=[{"id": "p", "score": 0.9}],
                contexts={"p": {"Material": "Steel"}},
                keys=["Material"],
            )

        self.assertIsInstance(result["Material"]["confidence"], float)

    def test_status_is_string(self):
        """status is always a str."""
        mock_cl = _make_mock_context_layer()

        import core

        modules = {
            "hoops_ai": MagicMock(),
            "hoops_ai.ml": MagicMock(),
            "hoops_ai.ml.context_layer": mock_cl,
        }
        with patch.dict(sys.modules, modules):
            result = core.predict_context(
                hits=[{"id": "p", "score": 0.9}],
                contexts={"p": {"Material": "Steel"}},
                keys=["Material"],
            )

        self.assertIsInstance(result["Material"]["status"], str)


# ---------------------------------------------------------------------------
# Router-layer tests
# ---------------------------------------------------------------------------


class TestContextPredictRouterValidation(unittest.TestCase):
    """Router-level tests using FastAPI TestClient with mocked core."""

    def setUp(self):
        from fastapi.testclient import TestClient

        self.client = TestClient(_make_app())

    def _valid_payload(self, **overrides):
        payload = {
            "hits": [{"id": "part-a", "score": 0.9}],
            "contexts": {"part-a": {"Material": "Steel"}},
            "keys": ["Material"],
        }
        payload.update(overrides)
        return payload

    def test_valid_request_returns_200(self):
        """A well-formed request with mocked core returns 200."""
        import core

        mock_result = {
            "Material": {
                "value": "Steel",
                "confidence": 0.9,
                "status": "ready_to_propose",
                "injected_context": None,
            }
        }
        with patch.object(core, "predict_context", return_value=mock_result):
            response = self.client.post("/context/predict", json=self._valid_payload())

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("predictions", body)
        self.assertIn("Material", body["predictions"])

    def test_invalid_rule_type_returns_422(self):
        """An unknown rule type in per_key_rules must return 422."""
        import core

        with patch.object(
            core,
            "predict_context",
            side_effect=ValueError("Unknown rule type 'bogus_rule'."),
        ):
            payload = self._valid_payload(
                per_key_rules={
                    "Material": {
                        "type": "bogus_rule",
                        "log_scale": False,
                        "auto_relevance_weight": False,
                    }
                }
            )
            response = self.client.post("/context/predict", json=payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("bogus_rule", response.json().get("detail", ""))

    def test_env_config_error_returns_503(self):
        """EnvConfigError (e.g. hoops_ai not installed) results in 503."""
        import core

        with patch.object(
            core,
            "predict_context",
            side_effect=core.EnvConfigError("hoops_ai.ml.context_layer not importable"),
        ):
            response = self.client.post("/context/predict", json=self._valid_payload())

        self.assertEqual(response.status_code, 503)

    def test_missing_required_fields_returns_422(self):
        """A request body missing required fields returns 422 from Pydantic."""
        response = self.client.post("/context/predict", json={})
        self.assertEqual(response.status_code, 422)

    def test_response_schema_matches_context_predict_response(self):
        """Response body matches the ContextPredictResponse schema."""
        import core

        mock_result = {
            "CostUSD": {
                "value": 14.5,
                "confidence": 0.82,
                "status": "ready_to_propose",
                "injected_context": None,
            },
            "Material": {
                "value": "Steel",
                "confidence": 0.91,
                "status": "ready_to_propose",
                "injected_context": {"Material": "Steel"},
            },
        }
        with patch.object(core, "predict_context", return_value=mock_result):
            payload = self._valid_payload(
                keys=["CostUSD", "Material"],
                contexts={"part-a": {"CostUSD": 10.0, "Material": "Iron"}},
            )
            response = self.client.post("/context/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        preds = response.json()["predictions"]
        self.assertAlmostEqual(preds["CostUSD"]["confidence"], 0.82)
        self.assertEqual(preds["Material"]["injected_context"], {"Material": "Steel"})


if __name__ == "__main__":
    unittest.main()
