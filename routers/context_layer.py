"""Router for POST /context/predict — stateless context-layer inference.

Delegates all business logic to :func:`core.predict_context`.  The endpoint
accepts caller-supplied metadata (*contexts*) alongside similarity-search
*hits* and returns predicted values for each requested metadata key.

No PLM/ERP connectivity is performed here; the caller (e.g. a Claude agent
via MCP) is responsible for fetching *contexts* from the appropriate data
store and including them in the request body.

Error codes
-----------
* **422** — ``ValueError``: invalid rule ``type`` in ``per_key_rules``, or
  other request-validation failure.
* **500** — ``RuntimeError``: unexpected error during inference.
* **503** — ``core.EnvConfigError`` / ``core.PathConfigError``: the
  ``hoops_ai.ml.context_layer`` module could not be imported (e.g. the
  hoops_ai package is absent or the licence is not configured).  Other
  endpoints remain fully operational.
"""

from typing import Any, Optional

import core
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/context", tags=["Context Layer Prediction"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ContextHit(BaseModel):
    """A single similarity-search hit carrying a part id and its match score."""

    id: str
    score: float


class NumericRuleSpec(BaseModel):
    """Rule specification for numeric metadata keys.

    Set ``type`` to ``"numeric_weighted"`` for weighted-average inference or
    ``"nearest_neighbor"`` to adopt the closest neighbour's value verbatim.

    Fields that are irrelevant for the chosen ``type`` are ignored:
    ``log_scale``, ``auto_relevance_weight``, ``nearest_neighbor_threshold``,
    and ``score_temperature`` apply only to ``"numeric_weighted"``;
    ``threshold`` applies only to ``"nearest_neighbor"``.
    """

    type: str  # "numeric_weighted" | "nearest_neighbor"
    log_scale: bool = False
    auto_relevance_weight: bool = False
    nearest_neighbor_threshold: Optional[float] = None
    score_temperature: Optional[float] = None
    threshold: Optional[float] = None  # nearest_neighbor only


class CategoricalRuleSpec(BaseModel):
    """Rule configuration for categorical metadata keys (softmax-weighted voting).

    Applied as the *default* rule for keys not listed in ``per_key_rules``.
    """

    temperature: float = 12.0
    min_margin: float = 0.05


class ContextPredictRequest(BaseModel):
    """Request body for ``POST /context/predict``.

    ``hits`` and ``contexts`` are the two required inputs; all other fields
    have sensible defaults and can be omitted.
    """

    hits: list[ContextHit]
    contexts: dict[str, dict[str, Any]]
    keys: list[str]
    numeric_keys: list[str] = []
    query_context: dict[str, Any] = {}
    default_categorical_rule: CategoricalRuleSpec = CategoricalRuleSpec()
    per_key_rules: dict[str, NumericRuleSpec] = {}
    status_policy: Optional[dict[str, float]] = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ContextPredictionOut(BaseModel):
    """Predicted value and metadata for a single key."""

    value: Any
    confidence: float
    status: str
    injected_context: Optional[dict[str, Any]] = None


class ContextPredictResponse(BaseModel):
    """Response for ``POST /context/predict``."""

    predictions: dict[str, ContextPredictionOut]


# ---------------------------------------------------------------------------
# POST /context/predict
# ---------------------------------------------------------------------------


@router.post("/predict", response_model=ContextPredictResponse, dependencies=[Depends(core.require_demo_enabled)])
def context_predict(request: ContextPredictRequest):
    """Predict missing metadata values from similarity-search hits.

    Given a set of *hits* (part ids + similarity scores returned by a prior
    shape search) and their associated *contexts* (metadata fetched from a
    PLM/ERP system by the caller), this endpoint infers likely values for each
    key listed in ``keys`` using ``hoops_ai.ml.context_layer.ContextPredictor``.

    The endpoint is **stateless** — it holds no database connection.  The caller
    must supply *contexts* directly in the request body.

    Rule selection:

    * Keys not listed in ``per_key_rules`` use the ``default_categorical_rule``
      (softmax-weighted voting over neighbour categories).
    * Keys listed in ``per_key_rules`` use either ``NumericWeightedRule``
      (``type: "numeric_weighted"``) or ``NearestNeighborRule``
      (``type: "nearest_neighbor"``).

    Returns **422** when an unknown rule ``type`` is supplied.
    Returns **503** when the ``hoops_ai.ml.context_layer`` module is unavailable.
    """
    try:
        raw_hits = [{"id": h.id, "score": h.score} for h in request.hits]
        raw_per_key = {k: v.model_dump() for k, v in request.per_key_rules.items()}

        predictions = core.predict_context(
            hits=raw_hits,
            contexts=request.contexts,
            keys=request.keys,
            numeric_keys=request.numeric_keys,
            query_context=request.query_context or None,
            default_categorical_rule=request.default_categorical_rule.model_dump(),
            per_key_rules=raw_per_key,
            status_policy=request.status_policy,
        )
        return ContextPredictResponse(
            predictions={
                key: ContextPredictionOut(**pred)
                for key, pred in predictions.items()
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Context prediction failed: {exc}"
        ) from exc
