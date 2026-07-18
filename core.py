import ast
import datetime
import hashlib
import io
import logging
import os
import pathlib
import re
import shutil
import ssl
import tempfile
import threading
import uuid
import zipfile
from contextlib import redirect_stdout
from typing import Any, Optional

logger = logging.getLogger(__name__)

from fastapi import UploadFile

APP_ROOT = pathlib.Path(__file__).resolve().parent
ENV_FILE_PATH = pathlib.Path(__file__).with_name(".env")
CAD_UPLOAD_DIR = APP_ROOT.joinpath("uploads")
CAD_VIEWER_OUTPUT_DIR = APP_ROOT.joinpath("out")
EMBEDDINGS_CACHE_DIR = APP_ROOT.joinpath("embeddings_cache")
INDEXES_DIR = APP_ROOT.joinpath("indexes")

# Allowed CAD file extensions for upload/ZIP extraction
CAD_ALLOWED_EXTENSIONS = frozenset({
    ".step", ".stp", ".iges", ".igs", ".x_t", ".x_b",
    ".sat", ".ipt", ".prt", ".sldprt", ".catpart",
})
MFR_LABELS_DESCRIPTION_ENV_NAMES = ("HOOPS_AI_MFR_LABELS_DESCRIPTION", "labels_description")

MFR_dataset_explorer = None
MFR_inference_model = None
CAD_viewers: dict[str, dict[str, Any]] = {}  # session_id -> {file_key -> viewer_info}
CAD_face_colors: dict[str, list] = {}  # scs_filename -> [[r,g,b], ...] indexed by face_id
CAD_color_maps: dict[str, dict] = {}  # scs_filename -> {label_id: {name, color_rgb}}
PART_CLASS_inference_model = None
PART_CLASS_dataset_explorer = None
_embedder = None
_embedder_signal = None
_embedding_memory_cache: dict[str, dict] = {}  # cache_key -> embedding entry

# ---------------------------------------------------------------------------
# Embedding model key constants
# "legacy" = 1M model (HOOPS_AI_EMBEDDINGS_MODEL_NAME)
# "signal" = SIGNAL model (HOOPS_AI_EMBEDDINGS_MODEL_NAME_SIGNAL)
# ---------------------------------------------------------------------------
_EMBEDDER_MODEL_LEGACY = "legacy"
_EMBEDDER_MODEL_SIGNAL = "signal"
_EMBEDDER_MODELS = frozenset({_EMBEDDER_MODEL_LEGACY, _EMBEDDER_MODEL_SIGNAL})

# Server-wide active embedding model (used by /compare, /map, /index/create).
# Use get_active_embedding_model() / set_active_embedding_model() to read/write.
_active_embedding_model: str = _EMBEDDER_MODEL_SIGNAL


def get_active_embedding_model() -> str:
    """Return the server-wide active embedding model key ('legacy' or 'signal')."""
    return _active_embedding_model


def set_active_embedding_model(model: str) -> None:
    """Set the server-wide active embedding model.

    Raises ``ValueError`` if *model* is not a recognised key.
    """
    global _active_embedding_model
    if model not in _EMBEDDER_MODELS:
        raise ValueError(
            f"Invalid model '{model}'. Must be one of: {sorted(_EMBEDDER_MODELS)}."
        )
    _active_embedding_model = model


# ---------------------------------------------------------------------------
# Default index preset management
# "signal" = HOOPS_AI_FAISS_INDEX_PATH_SIGNAL (SIGNAL model)  Edefault
# "legacy" = HOOPS_AI_FAISS_INDEX_PATH (1M model)
# ---------------------------------------------------------------------------

# Per-preset caches: keyed by preset name ("legacy" / "signal")
_default_index_searchers: dict[str, Any] = {}
_default_index_shapes: dict[str, Any] = {}

# Active preset for /similarity/search, /similarity/part-image, /similarity/index-info
_active_default_index: str = _EMBEDDER_MODEL_SIGNAL  # "signal"

_DEFAULT_INDEX_PRESETS = frozenset({_EMBEDDER_MODEL_LEGACY, _EMBEDDER_MODEL_SIGNAL})


def get_active_default_index() -> str:
    """Return the active default-index preset key ('legacy' or 'signal')."""
    return _active_default_index


def set_active_default_index(preset: str) -> None:
    """Set the active default-index preset.

    Raises ``ValueError`` if *preset* is not a recognised key.
    """
    global _active_default_index
    if preset not in _DEFAULT_INDEX_PRESETS:
        raise ValueError(
            f"Invalid index preset '{preset}'. Must be one of: {sorted(_DEFAULT_INDEX_PRESETS)}."
        )
    _active_default_index = preset


# ---------------------------------------------------------------------------
# Named index registry (incremental index management)
# ---------------------------------------------------------------------------

# SDK verification notes (verified against hoops_ai.ml.embeddings):
#   - FaissVectorStore.load(path) is a CLASSMETHOD returning a new store instance.
#     Calling vs_instance.load(path) does NOT load data into the existing instance.
#   - upsert() accepts VectorRecord(id, embedding, metadata) where `embedding` must
#     be an Embedding(values=np.ndarray, model=str, dim=int) object, not a raw array.
#   - upsert() with a duplicate ID inserts a second entry in the FAISS index;
#     get_ids() deduplicates IDs but count() reflects raw FAISS entries.
#     Solution: always delete(id) before upserting an existing ID to avoid duplicates.
#   - Empty-index query() returns [] without error.
#   - HOOPSEmbeddings exposes an `embedding_dim` attribute for dimension discovery.

_INDEX_NAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
_RESERVED_INDEX_NAMES = frozenset({"default"})

# name -> FaissVectorStore (in-memory cache, updated on every write)
_named_indexes: dict[str, Any] = {}

# per-index write lock; also held during searches to prevent torn reads
_index_locks: dict[str, threading.Lock] = {}
_index_locks_mutex = threading.Lock()


def _get_index_lock(name: str) -> threading.Lock:
    with _index_locks_mutex:
        if name not in _index_locks:
            _index_locks[name] = threading.Lock()
        return _index_locks[name]


def _validate_index_name(name: str) -> None:
    """Raise ValueError for invalid names, PermissionError for reserved names."""
    if not _INDEX_NAME_RE.match(name):
        raise ValueError(
            f"Invalid index name '{name}'. "
            r"Must match ^[a-z0-9_-]{1,64}$ (lowercase alphanumerics, hyphens, underscores)."
        )
    if name in _RESERVED_INDEX_NAMES:
        raise PermissionError(
            f"Index name '{name}' is reserved and cannot be created, modified, or deleted."
        )


def _index_base_path(name: str) -> pathlib.Path:
    """Base path (no extension) passed to FaissVectorStore.save/load.
    Generates indexes/{name}/index.faiss and indexes/{name}/index.meta.
    """
    return INDEXES_DIR / name / "index"


def _index_faiss_path(name: str) -> pathlib.Path:
    return INDEXES_DIR / name / "index.faiss"


def _get_embedder_dim(model: str = _EMBEDDER_MODEL_LEGACY) -> int:
    """Return the embedding dimension from the lazy-loaded HOOPSEmbeddings model."""
    embedder = get_embedder(model)
    dim = getattr(embedder, "embedding_dim", None)
    if dim is not None:
        return int(dim)
    raise RuntimeError(
        "Cannot determine embedding dimension from HOOPSEmbeddings. "
        "Ensure the embeddings model is correctly configured."
    )


def _load_named_index(name: str) -> Any:
    """Return the in-memory FaissVectorStore for `name`, loading from disk if needed.

    Must be called while holding the per-index lock.
    Raises KeyError if the index does not exist on disk.
    """
    if name in _named_indexes:
        return _named_indexes[name]

    faiss_file = _index_faiss_path(name)
    if not faiss_file.exists():
        raise KeyError(f"Index '{name}' does not exist.")

    from hoops_ai.ml.embeddings import FaissVectorStore

    vs = FaissVectorStore.load(str(_index_base_path(name)))
    _named_indexes[name] = vs
    return vs


def _save_named_index_atomic(name: str, vs: Any) -> None:
    """Persist *vs* to disk atomically using temp-file + replace pattern.

    Writes to ``_tmp_{uuid}.faiss`` / ``.meta`` in INDEXES_DIR and then calls
    ``Path.replace()`` to swap them in place.  replace() is atomic on POSIX and
    as close to atomic as Windows allows (non-atomic but crash-safe for our use case).
    """
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    index_dir = INDEXES_DIR / name
    index_dir.mkdir(parents=True, exist_ok=True)
    tmp_base = index_dir / f"_tmp_{uuid.uuid4().hex}"
    try:
        vs.save(str(tmp_base))
        pathlib.Path(str(tmp_base) + ".faiss").replace(_index_faiss_path(name))
        pathlib.Path(str(tmp_base) + ".meta").replace(INDEXES_DIR / name / "index.meta")
    except Exception:
        for suffix in (".faiss", ".meta"):
            tmp = pathlib.Path(str(tmp_base) + suffix)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Named index public API
# ---------------------------------------------------------------------------


def create_index(name: str, model: str = _EMBEDDER_MODEL_LEGACY) -> dict[str, Any]:
    """Create a new empty named index.

    *model* selects which embeddings model is used for this index: ``'default'``
    (the 1M model set by ``HOOPS_AI_EMBEDDINGS_MODEL_NAME``) or ``'signal'``
    (the SIGNAL model set by ``HOOPS_AI_EMBEDDINGS_MODEL_NAME_SIGNAL``).

    Raises ValueError for invalid names/models, PermissionError for reserved names,
    and FileExistsError if an index with that name already exists.
    """
    if model not in _EMBEDDER_MODELS:
        raise ValueError(
            f"Invalid model '{model}'. Must be one of: {sorted(_EMBEDDER_MODELS)}."
        )
    _validate_index_name(name)
    lock = _get_index_lock(name)
    with lock:
        if _index_faiss_path(name).exists():
            raise FileExistsError(f"Index '{name}' already exists.")

        from hoops_ai.ml.embeddings import FaissVectorStore

        dim = _get_embedder_dim(model)
        vs = FaissVectorStore(dim)
        _save_named_index_atomic(name, vs)
        _save_index_model(name, model)
        _named_indexes[name] = vs
        return {"name": name, "count": 0, "dim": dim, "model": model}


def list_indexes() -> list[dict[str, Any]]:
    """Return metadata for all known indexes, including the read-only ``default`` index."""
    result: list[dict[str, Any]] = []

    # "default" index  Eread-only, backed by the env-configured FAISS file
    load_env_file()
    faiss_name = os.environ.get("HOOPS_AI_FAISS_INDEX_PATH")
    if faiss_name:
        sdk_dir_str = os.environ.get("HOOPS_AI_SDK_DIR")
        if sdk_dir_str:
            default_path = pathlib.Path(sdk_dir_str) / "notebooks" / faiss_name
            last_modified: Optional[str] = None
            if default_path.exists():
                mtime = default_path.stat().st_mtime
                last_modified = (
                    datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ")
                )
            default_count: Optional[int] = None
            _legacy_idx = _default_index_shapes.get(_EMBEDDER_MODEL_LEGACY)
            if _legacy_idx is not None:
                ids = getattr(_legacy_idx, "ids", None)
                if ids is not None:
                    try:
                        default_count = int(len(ids))
                    except (TypeError, ValueError):
                        pass
            result.append(
                {
                    "name": "default",
                    "count": default_count,
                    "last_modified": last_modified,
                    "is_readonly": True,
                }
            )

    # Named indexes from INDEXES_DIR  Eeach lives in its own subdirectory
    if INDEXES_DIR.exists():
        for faiss_file in sorted(INDEXES_DIR.glob("*/index.faiss")):
            idx_name = faiss_file.parent.name
            last_modified_idx: Optional[str] = None
            mtime = faiss_file.stat().st_mtime
            last_modified_idx = (
                datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            idx_count: Optional[int] = None
            lock = _get_index_lock(idx_name)
            with lock:
                try:
                    vs = _load_named_index(idx_name)
                    idx_count = len(vs.get_ids())
                except Exception:
                    pass
            result.append(
                {
                    "name": idx_name,
                    "count": idx_count,
                    "last_modified": last_modified_idx,
                    "is_readonly": False,
                    "model": _load_index_model(idx_name),
                }
            )

    return result


def add_to_index(
    name: str,
    file_ids: list[str],
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Compute embeddings for *file_ids* and upsert them into the named index.

    Re-uses the ``compute_embedding()`` disk cache.  Re-inserting the same
    file_id overwrites the existing entry (delete-then-upsert to avoid FAISS
    duplicate entries).

    *model* overrides the embedder used for this batch.  When ``None`` (default)
    the model recorded in the index's ``model.json`` sidecar is used so that all
    entries in an index always use the same model.

    Returns ``added``, ``updated``, ``index_count``, and per-file ``errors``.
    """
    _validate_index_name(name)
    # Resolve model: explicit param > sidecar > default
    effective_model = model if model is not None else _load_index_model(name)
    if effective_model not in _EMBEDDER_MODELS:
        raise ValueError(
            f"Invalid model '{effective_model}'. Must be one of: {sorted(_EMBEDDER_MODELS)}."
        )
    lock = _get_index_lock(name)
    with lock:
        vs = _load_named_index(name)

        from hoops_ai.ml.embeddings import Embedding, FaissVectorStore, VectorRecord

        existing_ids: set[str] = set(vs.get_ids())
        added = 0
        updated = 0
        errors: list[dict[str, Any]] = []

        for fid in file_ids:
            try:
                emb_result = compute_embedding(fid, model=effective_model)
                v = emb_result["vector"]
                emb_obj = Embedding(values=v, model=emb_result["model_name"], dim=emb_result["dim"])
                meta: dict[str, Any] = {
                    "file_id": fid,
                    "filename": emb_result["filename"],
                    "registered_at": datetime.datetime.now(datetime.timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
                rec = VectorRecord(id=fid, embedding=emb_obj, metadata=meta)

                if fid in existing_ids:
                    # delete first to prevent FAISS duplicate entries on re-insert
                    vs.delete([fid])
                    updated += 1
                else:
                    added += 1
                    existing_ids.add(fid)

                vs.upsert([rec])
            except Exception as exc:
                errors.append({"file_id": fid, "detail": str(exc)})

        _save_named_index_atomic(name, vs)
        result = {
            "name": name,
            "added": added,
            "updated": updated,
            "index_count": len(vs.get_ids()),
            "errors": errors,
        }

    # Generate thumbnails outside the lock (non-fatal, can be slow)
    for fid in [f for f in file_ids if f not in [e["file_id"] for e in errors]]:
        _generate_part_thumbnail(fid, name)

    return result


def save_map_cluster_tags(map_id: str, tags: dict[str, str]) -> dict[str, Any]:
    """Write ``cluster_tag`` fields into a shape map JSON.

    *tags* is a ``{file_id: tag_name}`` mapping.  Only parts whose ``file_id``
    appears in *tags* are updated; other parts are left unchanged.

    Returns ``{"map_id": ..., "tagged": <count of parts updated>}``.
    Raises ``KeyError`` when the map does not exist.
    """
    import json

    map_path = CAD_VIEWER_OUTPUT_DIR / f"shape_map_{map_id}.json"
    if not map_path.exists():
        raise KeyError(f"Map '{map_id}' not found.")

    with open(map_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    applied = 0
    for part in data.get("parts", []):
        fid = part.get("file_id", "")
        if fid in tags:
            part["cluster_tag"] = tags[fid]
            applied += 1

    with open(map_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)

    return {"map_id": map_id, "tagged": applied}


def add_map_parts_to_index(map_id: str, index_name: str) -> dict[str, Any]:
    """Register all parts from a shape map into a named similarity index.

    If the named index does not yet exist it is created automatically.

    Raises ``KeyError`` only for an unknown *map_id*.
    """
    import json

    map_path = CAD_VIEWER_OUTPUT_DIR / f"shape_map_{map_id}.json"
    if not map_path.exists():
        raise KeyError(f"Map '{map_id}' not found.")

    with open(map_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    parts = [p for p in data.get("parts", []) if p.get("file_id")]
    file_ids = [p["file_id"] for p in parts]

    # Use the model recorded in the map JSON so the index matches the map's embeddings.
    map_model: str = data.get("model", _EMBEDDER_MODEL_LEGACY)

    # Auto-create the index if it doesn't exist yet
    if not _index_faiss_path(index_name).exists():
        create_index(index_name, model=map_model)

    return add_to_index(index_name, file_ids, model=map_model)


def remove_from_index(name: str, part_ids: list[str]) -> dict[str, Any]:
    """Delete *part_ids* from the named index, persist, and remove their thumbnails."""
    _validate_index_name(name)
    lock = _get_index_lock(name)
    with lock:
        vs = _load_named_index(name)
        vs.delete(part_ids)
        _save_named_index_atomic(name, vs)

    # Remove thumbnails outside the lock (non-fatal)
    thumb_dir = _index_thumbnails_dir(name)
    for pid in part_ids:
        png = thumb_dir / f"{pid}.png"
        png.unlink(missing_ok=True)

    return {"name": name, "removed": len(part_ids), "index_count": len(vs.get_ids())}


def search_index(name: str, file_id: str, top_k: int) -> dict[str, Any]:
    """Search the named index for the top-k most similar parts to *file_id*.

    The query embedding is computed using the same model that was selected when
    the index was created (recorded in ``model.json``).

    Returns an empty hits list (and no image_url) when the index has no entries.
    Generates a result-grid PNG in out/ and returns its relative URL as ``image_url``.
    """
    _validate_index_name(name)
    index_model = _load_index_model(name)
    lock = _get_index_lock(name)

    # Phase 1: vector store operations (locked)
    with lock:
        vs = _load_named_index(name)
        if len(vs.get_ids()) == 0:
            return {"hits": [], "count": 0, "image_url": None}
        emb_result = compute_embedding(file_id, model=index_model)
        raw_hits = vs.query(emb_result["vector"], top_k=top_k)

    hit_dicts = [
        {"id": h.id, "score": round(float(h.score), 6), "metadata": h.metadata}
        for h in raw_hits
    ]

    # Phase 2: thumbnail + grid generation (outside lock, non-fatal)
    _generate_part_thumbnail(file_id, name)
    image_url = _build_search_grid_image(file_id, hit_dicts, name) if hit_dicts else None

    return {"hits": hit_dicts, "count": len(hit_dicts), "image_url": image_url}


def delete_index(name: str) -> dict[str, Any]:
    """Delete the named index, its on-disk FAISS files, and all stored thumbnails.

    Raises PermissionError for reserved names, KeyError if the index does not exist.
    """
    _validate_index_name(name)
    lock = _get_index_lock(name)
    with lock:
        faiss_file = _index_faiss_path(name)
        if not faiss_file.exists():
            raise KeyError(f"Index '{name}' does not exist.")
        _named_indexes.pop(name, None)

    # Remove the entire index directory (faiss + meta + thumbnails)
    index_dir = INDEXES_DIR / name
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)

    return {"name": name, "deleted": True}


def _index_thumbnails_dir(name: str) -> pathlib.Path:
    """Return the per-index thumbnail image directory path."""
    return INDEXES_DIR / name / "thumbnails"


def _index_model_sidecar_path(name: str) -> pathlib.Path:
    """Return the per-index model sidecar JSON path (indexes/{name}/model.json)."""
    return INDEXES_DIR / name / "model.json"


def _load_index_model(name: str) -> str:
    """Return the model key ('default' or 'signal') recorded for the named index.

    Falls back to 'default' for indexes created before model selection was added.
    """
    import json

    path = _index_model_sidecar_path(name)
    if not path.exists():
        return _EMBEDDER_MODEL_LEGACY
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("model", _EMBEDDER_MODEL_LEGACY)
    except Exception:
        return _EMBEDDER_MODEL_LEGACY


def _save_index_model(name: str, model: str) -> None:
    """Persist the model key for the named index."""
    import json

    path = _index_model_sidecar_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"model": model}, fh)


def _generate_part_thumbnail(file_id: str, index_name: str) -> Optional[pathlib.Path]:
    """Render a CAD part as a white-background PNG and store it in the index thumbnails dir.

    Idempotent: skips generation when the PNG already exists.
    Non-fatal: returns None on any failure so callers are never interrupted.
    The SCS intermediate file is deleted after the PNG is extracted.
    """
    thumb_dir = _index_thumbnails_dir(index_name)
    dest_png = thumb_dir / f"{file_id}.png"

    if dest_png.exists():
        return dest_png

    try:
        from hoops_ai.cadaccess import HOOPSLoader, HOOPSTools

        cad_path = find_persistent_CAD_file(file_id)
        thumb_dir.mkdir(parents=True, exist_ok=True)

        # exportStreamCache writes {stem}.scs and {stem}.png next to the given filename.
        tmp_scs = thumb_dir / f"_tmp_{uuid.uuid4().hex}.scs"
        cad_loader = HOOPSLoader()
        model = cad_loader.create_from_file(str(cad_path))
        tools = HOOPSTools()
        png_result, scs_result = tools.exportStreamCache(
            model,
            filename=str(tmp_scs),
            is_white_background=True,
            overwrite=True,
        )

        # Move the generated PNG to the final destination
        png_file = pathlib.Path(png_result) if png_result else None
        if png_file and png_file.exists():
            png_file.replace(dest_png)

        # Remove the SCS file  Ewe only need the PNG for thumbnails
        for p_str in (scs_result, str(tmp_scs)):
            if p_str:
                p = pathlib.Path(p_str)
                if p.exists():
                    p.unlink(missing_ok=True)

        return dest_png if dest_png.exists() else None
    except Exception:
        return None


def _build_search_grid_image(
    query_file_id: str,
    hits: list[dict[str, Any]],
    index_name: str,
) -> Optional[str]:
    """Generate a matplotlib result-grid PNG (query + top-k hits) and save to out/.

    Returns a ``/out/{uuid}.png`` relative URL, or None on failure.
    Thumbnails are looked up from the index thumbnails directory.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        thumb_dir = _index_thumbnails_dir(index_name)
        n_total = len(hits) + 1  # query cell + one cell per hit
        cols = min(4, n_total)
        rows = (n_total + cols - 1) // cols

        fig, raw_axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3 + 0.4))
        # Normalise axes to a flat list regardless of shape
        import numpy as np

        axes: list = list(np.array(raw_axes).flatten())

        def _show_cell(ax, png_path: Optional[pathlib.Path], title: str, subtitle: str = "") -> None:
            if png_path and png_path.exists():
                try:
                    img = mpimg.imread(str(png_path))
                    ax.imshow(img)
                except Exception:
                    ax.set_facecolor("#e8e8e8")
            else:
                ax.set_facecolor("#e8e8e8")
                ax.text(
                    0.5, 0.5, title, ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, wrap=True,
                )
            ax.set_title(title[:30], fontsize=7, pad=2)
            if subtitle:
                ax.set_xlabel(subtitle, fontsize=6, labelpad=2)
            ax.set_xticks([])
            ax.set_yticks([])

        # Query cell  Elook up embedding cache for filename
        query_name = "query"
        cached = _embedding_memory_cache.get(f"hoops_embeddings_model__{query_file_id}")
        if cached:
            query_name = cached.get("filename", query_file_id[:12])
        query_thumb = thumb_dir / f"{query_file_id}.png"
        _show_cell(axes[0], query_thumb, f"Query\n{pathlib.Path(query_name).name}")

        # Hit cells
        for i, hit in enumerate(hits):
            if i + 1 >= len(axes):
                break
            hit_thumb = thumb_dir / f"{hit['id']}.png"
            meta = hit.get("metadata") or {}
            hit_name = pathlib.Path(meta.get("filename", hit["id"][:12])).name
            _show_cell(axes[i + 1], hit_thumb, hit_name, f"score: {hit['score']:.4f}")

        # Hide surplus axes
        for i in range(n_total, len(axes)):
            axes[i].axis("off")

        plt.tight_layout(pad=0.5)

        image_filename = f"{uuid.uuid4()}.png"
        image_path = CAD_VIEWER_OUTPUT_DIR / image_filename
        CAD_VIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(image_path), format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        return f"/out/{image_filename}"
    except Exception:
        return None


def get_MFR_dataset_explorer():
    global MFR_dataset_explorer
    if MFR_dataset_explorer is None:
        MFR_dataset_explorer = create_MFR_dataset_explorer()
    return MFR_dataset_explorer


def get_MFR_inference_model():
    global MFR_inference_model
    if MFR_inference_model is None:
        MFR_inference_model = create_MFR_inference_model()
    return MFR_inference_model


def get_cad_searcher_for(preset: str):
    """Return (lazy-init) the CADSearch searcher for the given preset."""
    if preset not in _default_index_searchers:
        _default_index_searchers[preset] = create_cad_searcher_for(preset)
    return _default_index_searchers[preset]


def get_shape_index_for(preset: str):
    """Return (lazy-init) the loaded FAISS index for the given preset."""
    if preset not in _default_index_shapes:
        _default_index_shapes[preset] = load_shape_index_for(preset)
    return _default_index_shapes[preset]


def get_cad_searcher():
    """Return the CADSearch searcher for the currently active default-index preset."""
    return get_cad_searcher_for(get_active_default_index())


def get_shape_index():
    """Return the FAISS index for the currently active default-index preset."""
    return get_shape_index_for(get_active_default_index())


def get_part_class_inference_model():
    global PART_CLASS_inference_model
    if PART_CLASS_inference_model is None:
        PART_CLASS_inference_model = create_part_class_inference_model()
    return PART_CLASS_inference_model


def get_part_class_dataset_explorer():
    global PART_CLASS_dataset_explorer
    if PART_CLASS_dataset_explorer is None:
        PART_CLASS_dataset_explorer = create_part_class_dataset_explorer()
    return PART_CLASS_dataset_explorer


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def read_env_file(path: pathlib.Path = ENV_FILE_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            values[key] = value
    return values


def load_env_file(path: pathlib.Path = ENV_FILE_PATH) -> None:
    for key, value in read_env_file(path).items():
        os.environ.setdefault(key, value)


class EnvConfigError(Exception):
    """Raised when a required environment variable is missing or empty."""


class PathConfigError(Exception):
    """Raised when a path derived from an environment variable does not exist."""


class ZipSlipError(RuntimeError):
    """Raised when a ZIP archive contains a Zip Slip path traversal attempt."""


class ZipSizeLimitError(RuntimeError):
    """Raised when a ZIP archive's uncompressed size exceeds the configured limit."""


class ZipFileLimitError(RuntimeError):
    """Raised when a ZIP archive contains more files than the configured limit."""


def get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        msg = f"[CONFIG] Required environment variable '{name}' is not set. Set it in {ENV_FILE_PATH.name} or the environment."
        logger.error(msg)
        print(msg, flush=True)
        raise EnvConfigError(msg)
    return value


def get_required_file_env(name: str) -> str:
    value = read_env_file().get(name)
    if not value:
        msg = f"[CONFIG] Required environment variable '{name}' is not set in {ENV_FILE_PATH.name}."
        logger.error(msg)
        print(msg, flush=True)
        raise EnvConfigError(msg)
    return value


def require_path(path: pathlib.Path, *, env_name: str = "", label: str = "") -> pathlib.Path:
    """Return *path* unchanged, or raise PathConfigError if it does not exist.

    Logs the missing path to the console before raising so that operators can
    immediately see which file or directory is absent.
    """
    if not path.exists():
        origin = f" (from {env_name})" if env_name else (f" ({label})" if label else "")
        msg = f"[CONFIG] Path not found{origin}: {path}"
        logger.error(msg)
        print(msg, flush=True)
        raise PathConfigError(msg)
    return path


def get_sdk_dir() -> pathlib.Path:
    """Return the HOOPS AI SDK install directory (``HOOPS_AI_SDK_DIR``).

    This directory must contain the ``notebooks/`` and ``packages/`` subdirectories.
    """
    return require_path(
        pathlib.Path(get_required_env("HOOPS_AI_SDK_DIR")),
        env_name="HOOPS_AI_SDK_DIR",
    )


def get_notebooks_dir() -> pathlib.Path:
    """Return ``<HOOPS_AI_SDK_DIR>/notebooks``."""
    return require_path(
        get_sdk_dir() / "notebooks",
        env_name="HOOPS_AI_SDK_DIR",
        label="notebooks directory",
    )


def get_packages_dir() -> pathlib.Path:
    """Return ``<HOOPS_AI_SDK_DIR>/packages``."""
    return require_path(
        get_sdk_dir() / "packages",
        env_name="HOOPS_AI_SDK_DIR",
        label="packages directory",
    )


def read_env_literal_assignment(names: tuple[str, ...]) -> Optional[str]:
    if not ENV_FILE_PATH.exists():
        return None

    lines = ENV_FILE_PATH.read_text(encoding="utf-8").splitlines()
    for line_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        if key.strip() not in names:
            continue

        value_lines = [value.strip().strip('"').strip("'")]
        brace_depth = value_lines[0].count("{") - value_lines[0].count("}")
        next_line_index = line_index + 1
        while brace_depth > 0 and next_line_index < len(lines):
            next_value_line = lines[next_line_index].rstrip()
            value_lines.append(next_value_line)
            brace_depth += next_value_line.count("{") - next_value_line.count("}")
            next_line_index += 1
        return "\n".join(value_lines).strip()

    return None


def _load_mfr_labels_file() -> dict[int, dict[str, str]]:
    """Load the default MFR labels from mfr_labels.py at the repository root."""
    import importlib.util

    labels_file = APP_ROOT / "mfr_labels.py"
    if not labels_file.exists():
        raise RuntimeError(
            f"MFR labels file not found: {labels_file}. "
            "Expected mfr_labels.py at the repository root."
        )
    spec = importlib.util.spec_from_file_location("mfr_labels", str(labels_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.labels_description


def get_MFR_labels_description() -> dict[int, dict[str, str]]:
    raw_value = next(
        (os.environ[name] for name in MFR_LABELS_DESCRIPTION_ENV_NAMES if os.environ.get(name)),
        None,
    ) or read_env_literal_assignment(MFR_LABELS_DESCRIPTION_ENV_NAMES)
    if not raw_value:
        return _load_mfr_labels_file()

    try:
        labels_description = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        raise RuntimeError(
            f"{' or '.join(MFR_LABELS_DESCRIPTION_ENV_NAMES)} must be a Python dictionary literal."
        ) from exc

    if not isinstance(labels_description, dict):
        raise RuntimeError(f"{' or '.join(MFR_LABELS_DESCRIPTION_ENV_NAMES)} must be a dictionary.")

    normalized_labels_description: dict[int, dict[str, str]] = {}
    for raw_label_id, raw_label_info in labels_description.items():
        try:
            label_id = int(raw_label_id)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{' or '.join(MFR_LABELS_DESCRIPTION_ENV_NAMES)} label IDs must be integers."
            ) from exc

        if not isinstance(raw_label_info, dict) or not isinstance(raw_label_info.get("name"), str):
            raise RuntimeError(
                f'{" or ".join(MFR_LABELS_DESCRIPTION_ENV_NAMES)} entries must include a string "name" field.'
            )

        normalized_labels_description[label_id] = {
            "name": raw_label_info["name"],
            "description": str(raw_label_info.get("description", "")),
        }
    return normalized_labels_description


def get_MFR_face_labels(feature_name: str) -> int:
    normalized_feature_name = feature_name.strip().casefold()
    labels_description = get_MFR_labels_description()
    for label_id, label_info in labels_description.items():
        if label_info["name"].strip().casefold() == normalized_feature_name:
            return label_id

    available_feature_names = sorted(
        label_info["name"] for label_info in labels_description.values()
    )
    raise RuntimeError(
        f"Unknown MFR feature name: {feature_name}. Available feature names: "
        f"{', '.join(available_feature_names)}"
    )


def get_CAD_shared_dir() -> pathlib.Path:
    value = os.environ.get("HOOPS_AI_CAD_SHARED_DIR") or read_env_file().get(
        "HOOPS_AI_CAD_SHARED_DIR"
    )
    if value:
        return pathlib.Path(value).expanduser().resolve()
    return CAD_UPLOAD_DIR.resolve()


def import_MFR_dataset_explorer():
    if os.name != "nt":
        from hoops_ai.dataset import DatasetExplorer

        return DatasetExplorer

    original_load_default_certs = ssl.SSLContext.load_default_certs
    ssl.SSLContext.load_default_certs = lambda self, purpose=ssl.Purpose.SERVER_AUTH: None
    try:
        from hoops_ai.dataset import DatasetExplorer

        return DatasetExplorer
    finally:
        ssl.SSLContext.load_default_certs = original_load_default_certs


def init_hoops_license() -> None:
    import hoops_ai

    load_env_file()
    license_key = get_required_file_env("HOOPS_AI_LICENSE")
    hoops_ai.set_license(license_key, validate=True)


def create_MFR_dataset_explorer():
    load_env_file()

    DatasetExplorer = import_MFR_dataset_explorer()

    notebooks_dir = get_notebooks_dir()
    MFR_flow_name = get_required_env("HOOPS_AI_MFR_FLOW_NAME")

    # Dataset files are produced by running the ETL tutorial notebook:
    #   notebooks/3b_workflow_for_MFR_cadsynth.ipynb
    # Output is written to: <HOOPS_AI_SDK_DIR>/notebooks/out/flows/<HOOPS_AI_MFR_FLOW_NAME>/
    flow_root_dir = require_path(
        notebooks_dir / "out" / "flows" / MFR_flow_name,
        env_name="HOOPS_AI_MFR_FLOW_NAME",
        label=f"MFR flow directory for '{MFR_flow_name}'",
    )

    return DatasetExplorer(
        merged_store_path=str(flow_root_dir.joinpath(f"{MFR_flow_name}.dataset")),
        parquet_file_path=str(flow_root_dir.joinpath(f"{MFR_flow_name}.infoset")),
        parquet_file_attribs=str(flow_root_dir.joinpath(f"{MFR_flow_name}.attribset")),
        dask_client_params={"processes": False},
    )


def create_MFR_inference_model():
    from hoops_ai.cadaccess import HOOPSLoader
    from hoops_ai.ml.EXPERIMENTAL import FlowInference, GraphNodeClassification

    load_env_file()

    notebooks_dir = get_notebooks_dir()
    model_name = get_required_env("HOOPS_AI_MFR_MODEL_NAME")
    trained_model = require_path(
        get_packages_dir().joinpath("trained_ml_models", model_name),
        env_name="HOOPS_AI_MFR_MODEL_NAME",
    )
    output_dir = notebooks_dir.joinpath("out")
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = HOOPSLoader()
    inference_model = FlowInference(
        cad_loader=loader,
        flowmodel=GraphNodeClassification(result_dir=str(output_dir)),
    )
    inference_model.load_from_checkpoint(trained_model)
    return inference_model


def _resolve_default_index_paths(preset: str) -> dict:
    """Return the resolved paths for the given default-index preset.

    Returns a dict with keys:
      ``faiss_path`` (Path), ``images_dir`` (Path), ``model_key`` (str)
    """
    load_env_file()
    notebooks_dir = get_notebooks_dir()
    if preset == _EMBEDDER_MODEL_LEGACY:
        faiss_file_name = get_required_env("HOOPS_AI_FAISS_INDEX_PATH")
        faiss_path = require_path(
            notebooks_dir.joinpath(faiss_file_name),
            env_name="HOOPS_AI_FAISS_INDEX_PATH",
        )
        images_dir = pathlib.Path(
            os.environ.get("HOOPS_AI_EMBEDDINGS_IMAGES_DIR")
            or notebooks_dir / "out" / "images"
        )
        model_key = _EMBEDDER_MODEL_LEGACY
    elif preset == _EMBEDDER_MODEL_SIGNAL:
        faiss_file_name_signal = os.environ.get("HOOPS_AI_FAISS_INDEX_PATH_SIGNAL") or "TMCAD_SIGNAL.faiss"
        faiss_path = require_path(
            get_packages_dir() / "vectorstores" / "tmcad" / faiss_file_name_signal,
            env_name="HOOPS_AI_FAISS_INDEX_PATH_SIGNAL",
        )
        images_dir = get_packages_dir() / "vectorstores" / "tmcad" / "images_tmcad"
        model_key = _EMBEDDER_MODEL_SIGNAL
    else:
        raise ValueError(f"Unknown default-index preset: '{preset}'")
    return {"faiss_path": faiss_path, "images_dir": images_dir, "model_key": model_key}


def create_cad_searcher_for(preset: str):
    """Create a CADSearch instance loaded with the embedder for *preset*."""
    from hoops_ai.ml import CADSearch

    embedder = get_embedder(_resolve_default_index_paths(preset)["model_key"])
    return CADSearch(shape_model=embedder)


def load_shape_index_for(preset: str):
    """Load the FAISS index for *preset* into its CADSearch searcher."""
    paths = _resolve_default_index_paths(preset)
    faiss_index_path = paths["faiss_path"]
    searcher = get_cad_searcher_for(preset)
    # Cross-OS pickle compatibility: Windows-pickled files contain WindowsPath objects.
    if not hasattr(pathlib, "WindowsPath") or not issubclass(pathlib.WindowsPath, pathlib.Path):
        pathlib.WindowsPath = pathlib.PurePosixPath  # type: ignore[attr-defined]
        return searcher.load_shape_index(path=str(faiss_index_path))
    else:
        _orig = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc]
            return searcher.load_shape_index(path=str(faiss_index_path))
        finally:
            pathlib.PosixPath = _orig


def search_by_shape(cad_file_path: pathlib.Path, top_k: int = 10) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from hoops_ai.insights import DatasetViewer

    preset = get_active_default_index()
    get_shape_index_for(preset)  # ensure FAISS index is loaded into the searcher
    searcher = get_cad_searcher_for(preset)
    hits = searcher.search_by_shape(str(cad_file_path), top_k=top_k)
    results = [
        {"id": _json_safe(hit.id), "score": _json_safe(hit.score)}
        for hit in hits[0]
    ]

    images_dir = _resolve_default_index_paths(preset)["images_dir"]
    ds_viewer = DatasetViewer([], [], [], reference_dir=images_dir)
    fig = ds_viewer.show_search_results(hits, query_file=str(cad_file_path), grid_cols=3)
    if fig is None:
        fig = plt.gcf()
    image_filename = f"{uuid.uuid4()}.png"
    image_path = CAD_VIEWER_OUTPUT_DIR / image_filename
    CAD_VIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(image_path), format="png", bbox_inches="tight")
    plt.close(fig)

    return {"hits": results, "count": len(results), "image_url": f"/out/{image_filename}"}


def get_similar_search_index_info() -> dict[str, Any]:
    """Return metadata about the currently loaded FAISS similarity search index.

    Always succeeds  Ereturns ``status: "not_loaded"`` when neither the searcher
    nor the index has been initialised yet, so callers never need to treat an
    unloaded state as an error.
    """
    import datetime

    load_env_file()
    preset = get_active_default_index()

    # Resolve index file path for the active preset (best-effort, no error if missing).
    index_path: Optional[str] = None
    index_last_modified: Optional[str] = None
    try:
        paths = _resolve_default_index_paths(preset)
        faiss_index_path = paths["faiss_path"]
        index_path = str(faiss_index_path)
        if faiss_index_path.exists():
            mtime = faiss_index_path.stat().st_mtime
            index_last_modified = (
                datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            )
    except Exception:
        pass

    # Trigger lazy loading if not yet initialised.
    searcher = _default_index_searchers.get(preset)
    idx = _default_index_shapes.get(preset)
    if searcher is None or idx is None:
        try:
            get_shape_index_for(preset)
            searcher = _default_index_searchers.get(preset)
            idx = _default_index_shapes.get(preset)
        except Exception:
            pass

    if searcher is None and idx is None:
        return {
            "preset": preset,
            "status": "not_loaded",
            "index_path": index_path,
            "index_last_modified": index_last_modified,
            "index_count": None,
            "model_name": None,
            "embedding_dim": None,
            "metadata": None,
        }

    info: dict[str, Any] = {
        "preset": preset,
        "status": "loaded",
        "index_path": index_path,
        "index_last_modified": index_last_modified,
        "index_count": None,
        "model_name": None,
        "embedding_dim": None,
        "metadata": None,
    }

    # Extract embedder attributes from the CADSearch object.
    if searcher is not None:
        embedder = (
            getattr(searcher, "_shape_model", None)
            or getattr(searcher, "shape_model", None)
        )
        if embedder is not None:
            info["model_name"] = (
                getattr(embedder, "model_name", None)
                or getattr(embedder, "model", None)
            )

    # Extract counts, dim, and metadata from the loaded EmbeddingBatch object.
    if idx is not None:
        ids = getattr(idx, "ids", None)
        if ids is not None:
            try:
                info["index_count"] = int(len(ids))
            except (TypeError, ValueError):
                pass
        dim = getattr(idx, "dim", None)
        if dim is not None:
            try:
                info["embedding_dim"] = int(dim)
            except (TypeError, ValueError):
                info["embedding_dim"] = _json_safe(dim)
        if info["model_name"] is None:
            info["model_name"] = getattr(idx, "model", None)
        metadata = getattr(idx, "metadata", None)
        if metadata is not None:
            info["metadata"] = _json_safe(dict(metadata) if hasattr(metadata, "items") else metadata)

    return info


# ---------------------------------------------------------------------------
# Embedding-only helpers (no FAISS index required)
# ---------------------------------------------------------------------------


def get_embedder(model: str = _EMBEDDER_MODEL_LEGACY):
    """Lazy-initialise HOOPSEmbeddings without loading a FAISS index.

    *model* selects which checkpoint to load:

    * ``'legacy'``  E``HOOPS_AI_EMBEDDINGS_MODEL_NAME`` (1M model), registered
      as ``"hoops_embeddings_model"``.
    * ``'signal'``  E``HOOPS_AI_EMBEDDINGS_MODEL_NAME_SIGNAL`` (SIGNAL model),
      registered as ``"hoops_embeddings_signal"``.

    Safe to call alongside (or after) ``create_cad_searcher_for()``  Ethe model
    registration is guarded so it is never performed twice.
    """
    global _embedder, _embedder_signal

    if model == _EMBEDDER_MODEL_SIGNAL:
        if _embedder_signal is not None:
            return _embedder_signal

        from hoops_ai.ml.embeddings import HOOPSEmbeddings

        load_env_file()

        ckpt_name = get_required_env("HOOPS_AI_EMBEDDINGS_MODEL_NAME_SIGNAL")
        trained_model = require_path(
            get_packages_dir().joinpath("trained_ml_models", ckpt_name),
            env_name="HOOPS_AI_EMBEDDINGS_MODEL_NAME_SIGNAL",
        )

        try:
            HOOPSEmbeddings.register_model(
                model_name="hoops_embeddings_signal",
                checkpoint_path=str(trained_model),
            )
        except Exception:
            pass  # already registered

        _embedder_signal = HOOPSEmbeddings(model="hoops_embeddings_signal")
        return _embedder_signal

    # Default model
    if _embedder is not None:
        return _embedder

    from hoops_ai.ml.embeddings import HOOPSEmbeddings

    load_env_file()

    ckpt_name = get_required_env("HOOPS_AI_EMBEDDINGS_MODEL_NAME")
    trained_model = require_path(
        get_packages_dir().joinpath("trained_ml_models", ckpt_name),
        env_name="HOOPS_AI_EMBEDDINGS_MODEL_NAME",
    )

    try:
        HOOPSEmbeddings.register_model(
            model_name="hoops_embeddings_model",
            checkpoint_path=str(trained_model),
        )
    except Exception:
        pass  # already registered (e.g. by create_cad_searcher)

    _embedder = HOOPSEmbeddings(model="hoops_embeddings_model")
    return _embedder


def _l2_normalize(v):
    """L2-normalise a 1-D numpy float32 array. Returns a zero vector unchanged."""
    import numpy as np

    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v.astype(np.float32)
    return (v / norm).astype(np.float32)


def compute_embedding(file_id: str, model: str = _EMBEDDER_MODEL_LEGACY) -> dict[str, Any]:
    """Compute (or retrieve from cache) the shape embedding for a CAD file.

    *model* selects the embedder: ``'legacy'`` (1M model) or ``'signal'``
    (SIGNAL model).  The cache key includes the model so that embeddings from
    different models are stored and retrieved independently.

    The result vector is a single L2-normalised float32 array representing the
    whole part.  For multi-body models the per-body vectors are individually
    L2-normalised, averaged, then re-normalised.

    Returns a dict with keys:
      ``file_id``, ``vector`` (np.ndarray), ``dim``, ``model_name``,
      ``num_bodies``, ``filename``, ``cached`` (bool).
    """
    import numpy as np

    embedder = get_embedder(model)
    model_name = (
        getattr(embedder, "model_name", None)
        or getattr(embedder, "model", None)
        or "hoops_embeddings_model"
    )

    # Build a filesystem-safe cache key that includes the model name.
    safe_model = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_name)
    cache_key = f"{safe_model}__{file_id}"

    # 1. Memory cache (fastest)
    if cache_key in _embedding_memory_cache:
        entry = _embedding_memory_cache[cache_key]
        return {**entry, "cached": True}

    # 2. Disk cache (survives server restart)
    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    disk_vec_path = EMBEDDINGS_CACHE_DIR / f"{cache_key}.npy"
    disk_meta_path = EMBEDDINGS_CACHE_DIR / f"{cache_key}.meta.npy"

    if disk_vec_path.exists() and disk_meta_path.exists():
        try:
            vector = np.load(str(disk_vec_path))
            meta = np.load(str(disk_meta_path), allow_pickle=True).item()
            entry: dict[str, Any] = {
                "file_id": file_id,
                "vector": vector,
                "dim": int(vector.shape[0]),
                "model_name": model_name,
                "num_bodies": int(meta.get("num_bodies", 1)),
                "filename": str(meta.get("filename", "")),
            }
            _embedding_memory_cache[cache_key] = entry
            return {**entry, "cached": True}
        except Exception:
            pass  # corrupted cache  Efall through to recompute

    # 3. Compute via HOOPS embedder
    cad_path = find_persistent_CAD_file(file_id)
    # The stored file name pattern is "{file_id}_{original_filename}".
    parts = cad_path.name.split("_", 1)
    filename = parts[1] if len(parts) == 2 and len(parts[0]) == 64 else cad_path.name

    raw_embeddings = embedder.embed_shape(str(cad_path))

    body_vectors: list = []
    for emb in raw_embeddings:
        v = getattr(emb, "values", None)
        if v is None:
            v = np.array(emb)
        v = np.array(v, dtype=np.float32).flatten()
        body_vectors.append(_l2_normalize(v))

    if not body_vectors:
        raise RuntimeError(f"embed_shape() returned no embeddings for file_id '{file_id}'.")

    if len(body_vectors) == 1:
        vector = body_vectors[0]
    else:
        stacked = np.stack(body_vectors, axis=0)  # (N, D)
        vector = _l2_normalize(stacked.mean(axis=0))

    num_bodies = len(body_vectors)

    entry = {
        "file_id": file_id,
        "vector": vector,
        "dim": int(vector.shape[0]),
        "model_name": model_name,
        "num_bodies": num_bodies,
        "filename": filename,
    }

    # Persist to disk (non-fatal on failure)
    try:
        np.save(str(disk_vec_path), vector)
        np.save(str(disk_meta_path), {"num_bodies": num_bodies, "filename": filename})
    except Exception:
        pass

    _embedding_memory_cache[cache_key] = entry
    return {**entry, "cached": False}


def compare_embeddings(file_ids: list[str], model: str = _EMBEDDER_MODEL_LEGACY) -> dict[str, Any]:
    """Compute an N×N cosine similarity matrix for the given file_ids.

    All embedding vectors are L2-normalised, so cosine similarity equals their
    dot product.  Diagonal entries are forced to exactly ``1.0``.

    *model* selects the embedder: ``'legacy'`` (1M model) or ``'signal'``
    (SIGNAL model).

    Returns a dict with keys:
      ``count``, ``model_name``, ``files``, ``matrix`` (N×N list of lists),
      ``pairs`` (all i<j combos sorted by score descending).
    """
    import numpy as np

    embeddings = [compute_embedding(fid, model=model) for fid in file_ids]
    vectors = np.stack([e["vector"] for e in embeddings], axis=0)  # (N, D)

    # Cosine similarity via dot product (vectors are already L2-normalised)
    raw_matrix = (vectors @ vectors.T).tolist()
    n = len(embeddings)
    matrix = [[round(float(raw_matrix[i][j]), 6) for j in range(n)] for i in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0  # enforce exact diagonal

    files = [
        {
            "index": i,
            "file_id": e["file_id"],
            "filename": e["filename"],
            "num_bodies": e["num_bodies"],
        }
        for i, e in enumerate(embeddings)
    ]

    pairs = [
        {"a": i, "b": j, "score": matrix[i][j]}
        for i in range(n)
        for j in range(i + 1, n)
    ]
    pairs.sort(key=lambda p: p["score"], reverse=True)

    model_name = embeddings[0]["model_name"] if embeddings else "hoops_embeddings_model"
    return {
        "count": n,
        "model_name": model_name,
        "files": files,
        "matrix": matrix,
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Shape Space Map (classical MDS layout of CAD parts)
# ---------------------------------------------------------------------------


def _classical_mds(dist_matrix: "Any") -> tuple:
    """Classical (Torgerson) multidimensional scaling into 3 dimensions.

    Given an ``N×N`` symmetric distance matrix, returns ``(coords, stress)``
    where ``coords`` has shape ``(N, 3)`` (mean-centred) and ``stress`` is the
    Kruskal stress-1 goodness-of-fit value in ``[0, 1]``.

    Implemented with numpy only (no sklearn/scipy).
    """
    import numpy as np

    D = np.asarray(dist_matrix, dtype=float)
    n = D.shape[0]

    # Double-centering: B = -0.5 * H @ D^2 @ H,  H = I - (1/n) * ones
    H = np.eye(n) - np.ones((n, n)) / n
    D_squared = D ** 2
    B = -0.5 * H @ D_squared @ H

    # eigh returns eigenvalues in ascending order ↁEtake the largest 3
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    eigenvalues = np.maximum(eigenvalues, 0.0)  # clamp negatives to 0

    top_vals = eigenvalues[-3:]
    top_vecs = eigenvectors[:, -3:]
    coords = top_vecs @ np.diag(np.sqrt(top_vals))  # (N, up-to-3)

    # Pad to 3 columns when N < 3
    if coords.shape[1] < 3:
        pad = np.zeros((n, 3 - coords.shape[1]))
        coords = np.hstack([coords, pad])

    # Reverse so columns are in descending eigenvalue order
    coords = coords[:, ::-1]

    # Mean-centre
    coords = coords - coords.mean(axis=0)

    # Kruskal stress-1 over all i<j pairs
    iu = np.triu_indices(n, k=1)
    d_target = D[iu]
    diff = coords[iu[0]] - coords[iu[1]]
    d_hat = np.sqrt((diff ** 2).sum(axis=1))
    denom = float((d_target ** 2).sum())
    if denom == 0.0:
        stress = 0.0
    else:
        stress = float(np.sqrt(((d_hat - d_target) ** 2).sum() / denom))

    return coords, stress


def export_scs_for_part(file_id: str) -> str:
    """Convert a persistent CAD file to an SCS stream cache and return its filename.

    A thin wrapper around the conversion logic in ``create_CAD_viewer``  Eit does
    NOT touch the per-session viewer cache.  Returns just the SCS filename (served
    under ``/out/``), not a full path or URL.
    """
    from hoops_ai.cadaccess import HOOPSLoader, HOOPSTools

    cad_file_path = find_persistent_CAD_file(file_id)
    CAD_VIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    unique_id = uuid.uuid4().hex[:12]
    scs_name = f"{unique_id}_{cad_file_path.stem}.scs"
    scs_path = CAD_VIEWER_OUTPUT_DIR / scs_name

    cad_loader = HOOPSLoader()
    model = cad_loader.create_from_file(str(cad_file_path))

    tools = HOOPSTools()
    _png_path, scs_path = tools.exportStreamCache(
        model,
        filename=str(scs_path),
        is_white_background=True,
        overwrite=True,
    )
    return pathlib.Path(scs_path).name


def compute_shape_map_data(
    file_ids: list[str], model: str = _EMBEDDER_MODEL_LEGACY
) -> dict[str, Any]:
    """Compute a 3D "Shape Space Map" layout for a set of CAD parts.

    Steps:
      1. Compute the N×N cosine-similarity matrix via ``compare_embeddings``.
      2. Convert each part to an SCS stream cache (failures are non-fatal and
         collected in ``errors``).
      3. Lay the parts out in 3D with classical MDS so that similar parts sit
         closer together (distance ``d_ij = 1 - similarity_ij``).
      4. Persist the result to ``out/shape_map_{map_id}.json`` and return it.

    *model* selects the embedder: ``'default'`` (1M model) or ``'signal'``
    (SIGNAL model).  The model identifier is stored in the map JSON so that
    subsequent overlay queries can use the same model automatically.
    """
    import json

    import numpy as np

    compare = compare_embeddings(file_ids, model=model)
    matrix = compare["matrix"]
    files = compare["files"]
    n = compare["count"]

    sim = np.asarray(matrix, dtype=float)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    coords, stress = _classical_mds(dist)

    errors: list[dict] = []
    parts: list[dict] = []
    for i, finfo in enumerate(files):
        fid = finfo["file_id"]
        filename = finfo["filename"]
        scs_url = None
        try:
            scs_filename = export_scs_for_part(fid)
            scs_url = f"/out/{scs_filename}"
        except Exception as exc:  # SCS conversion failure is non-fatal
            errors.append({"filename": filename, "detail": str(exc)})

        parts.append(
            {
                "index": i,
                "file_id": fid,
                "filename": filename,
                "scs_url": scs_url,
                "position": [
                    float(coords[i, 0]),
                    float(coords[i, 1]),
                    float(coords[i, 2]),
                ],
            }
        )

    map_id = uuid.uuid4().hex[:8]
    result = {
        "map_id": map_id,
        "viewer_url": f"/similarity/map/show?map={map_id}",
        "count": n,
        "parts": parts,
        "matrix": matrix,
        "stress": float(stress),
        "errors": errors,
        "model": model,
    }

    CAD_VIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CAD_VIEWER_OUTPUT_DIR / f"shape_map_{map_id}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)

    return result


def _project_oos_mds(
    coords: "Any",
    dist_matrix: "Any",
    query_dist: "Any",
) -> "Any":
    """Project a new point into an existing classical-MDS coordinate space.

    Uses the out-of-sample extension formula (Bengio et al., 2004).  Given the
    ``NÁE`` coordinate matrix ``coords`` (mean-centred) and the ``N×N`` distance
    matrix ``dist_matrix`` used to produce it, place the new point whose
    distances to the N existing points are ``query_dist`` (length-N array) into
    the same space.

    Returns a length-3 numpy array.  Falls back gracefully when N < 3 or the
    system is rank-deficient.
    """
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    query_dist = np.asarray(query_dist, dtype=float)

    n = coords.shape[0]
    if n == 0:
        return np.zeros(3)

    d_sq = dist_matrix ** 2
    d_q_sq = query_dist ** 2

    # Row/column means of D² (symmetric, so equal)
    row_means = d_sq.mean(axis=0)      # (N,)  E(1/N) Σ_i D²_ij for each j
    grand_mean = float(d_sq.mean())
    q_mean = float(d_q_sq.mean())

    # Out-of-sample centering: b_j = <x_query, x_j> approximation
    b = -0.5 * (d_q_sq - q_mean - row_means + grand_mean)  # (N,)

    # Solve coords @ x_query ≁Eb  (least-squares, handles rank-deficiency)
    x_q, _, _, _ = np.linalg.lstsq(coords, b, rcond=None)

    # Ensure output is exactly length 3
    result = np.zeros(3)
    result[: len(x_q)] = x_q[:3]
    return result


def query_shape_map(map_id: str, query_file_id: str, persist: bool = False) -> dict[str, Any]:
    """Overlay a query CAD part on an existing shape-space map.

    Steps:
      1. Load the persisted map JSON for *map_id*.
      2. Compute the query part's shape embedding.
      3. Compute cosine similarities between the query and every existing part.
      4. Project the query into the existing 3D coordinate space via the
         out-of-sample MDS extension formula.
      5. Export an SCS stream cache for the query part (non-fatal on failure).
      6. Save a new *overlay* map JSON (``shape_map_{overlay_id}.json``) that
         contains all original parts plus the query part tagged with
         ``is_query=True`` for magenta highlighting in the viewer.
      7. Optionally persist the query part into the original map JSON when
         *persist* is ``True``.

    Returns a dict with keys: ``overlay_map_id``, ``viewer_url``,
    ``query_part``, ``nearest_parts`` (top-5), ``persisted``, ``errors``.

    Raises:
      ``KeyError``     Emap *map_id* does not exist.
      ``RuntimeError`` Eembedding computation fails for the query part.
    """
    import json

    import numpy as np

    map_path = CAD_VIEWER_OUTPUT_DIR / f"shape_map_{map_id}.json"
    if not map_path.exists():
        raise KeyError(f"Shape map '{map_id}' not found.")

    with open(map_path, encoding="utf-8") as fh:
        map_data = json.load(fh)

    existing_parts: list[dict] = map_data["parts"]
    existing_matrix: list[list[float]] = map_data["matrix"]
    n = len(existing_parts)
    model: str = map_data.get("model", _EMBEDDER_MODEL_LEGACY)

    # Compute query embedding (raises on failure  Elet caller handle)
    query_emb = compute_embedding(query_file_id, model=model)
    query_vec = query_emb["vector"]
    query_filename = query_emb["filename"]

    # Cosine similarities: query vs. each existing part (vectors are L2-normalised)
    query_sims: list[float] = []
    for part in existing_parts:
        part_emb = compute_embedding(part["file_id"], model=model)
        sim = float(np.dot(query_vec, part_emb["vector"]))
        query_sims.append(round(sim, 6))

    # Project query into existing coordinate space via out-of-sample MDS
    coords = np.array([p["position"] for p in existing_parts], dtype=float)  # (N, 3)
    sim_mat = np.array(existing_matrix, dtype=float)
    dist_mat = 1.0 - sim_mat
    np.fill_diagonal(dist_mat, 0.0)
    query_dist = 1.0 - np.array(query_sims)
    query_pos = _project_oos_mds(coords, dist_mat, query_dist)

    # Export SCS for the query part (non-fatal)
    errors: list[dict] = []
    query_scs_url: Optional[str] = None
    try:
        scs_name = export_scs_for_part(query_file_id)
        query_scs_url = f"/out/{scs_name}"
    except Exception as exc:
        errors.append({"filename": query_filename, "detail": str(exc)})

    query_part_info: dict[str, Any] = {
        "index": n,
        "file_id": query_file_id,
        "filename": query_filename,
        "scs_url": query_scs_url,
        "position": [float(query_pos[0]), float(query_pos[1]), float(query_pos[2])],
        "is_query": True,
    }

    # Build (N+1)ÁEN+1) extended similarity matrix
    ext_matrix = [row + [query_sims[i]] for i, row in enumerate(existing_matrix)]
    ext_matrix.append(query_sims + [1.0])

    # Save overlay map JSON (new map_id so the original is untouched)
    overlay_map_id = uuid.uuid4().hex[:8]
    overlay = {
        "map_id": overlay_map_id,
        "viewer_url": f"/similarity/map/show?map={overlay_map_id}",
        "count": n + 1,
        "parts": list(existing_parts) + [query_part_info],
        "matrix": ext_matrix,
        "stress": map_data.get("stress", 0.0),
        "errors": errors,
        "base_map_id": map_id,
        "query_file_id": query_file_id,
    }
    CAD_VIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = CAD_VIEWER_OUTPUT_DIR / f"shape_map_{overlay_map_id}.json"
    with open(overlay_path, "w", encoding="utf-8") as fh:
        json.dump(overlay, fh)

    # Optional: persist query part into the original map
    if persist:
        existing_ids = {p["file_id"] for p in existing_parts}
        if query_file_id not in existing_ids:
            map_data["parts"].append(query_part_info)
            map_data["matrix"] = ext_matrix
            map_data["count"] = n + 1
            with open(map_path, "w", encoding="utf-8") as fh:
                json.dump(map_data, fh)

    # Top-5 nearest existing parts by similarity
    nearest_parts = sorted(
        [
            {
                "index": i,
                "file_id": existing_parts[i]["file_id"],
                "filename": existing_parts[i]["filename"],
                "score": query_sims[i],
            }
            for i in range(n)
        ],
        key=lambda x: x["score"],
        reverse=True,
    )[:5]

    return {
        "overlay_map_id": overlay_map_id,
        "viewer_url": f"/similarity/map/show?map={overlay_map_id}",
        "query_part": query_part_info,
        "nearest_parts": nearest_parts,
        "persisted": persist,
        "errors": errors,
    }


def search_MFR_files(feature_name: str) -> dict[str, Any]:
    explorer = get_MFR_dataset_explorer()

    face_label = get_MFR_face_labels(feature_name)
    label_matches = lambda ds: ds["face_labels"] == face_label
    file_ids = explorer.get_file_list(group="Labels", where=label_matches)
    file_info = explorer.get_file_info_all()
    file_names_by_id = dict(zip(file_info["id"].astype(str), file_info["description"]))

    matched: list[tuple[int, str]] = [
        (int(_json_safe(file_id)), _json_safe(file_names_by_id[str(_json_safe(file_id))]))
        for file_id in file_ids
        if str(_json_safe(file_id)) in file_names_by_id
    ]
    return {
        "file_names": [name for _, name in matched],
        "file_list": [fid for fid, _ in matched],
    }


def run_MFR_inference(cad_file_path: pathlib.Path, session_id: Optional[str] = None) -> dict[str, Any]:
    from hoops_ai.insights.utils import ColorPalette

    inference_model = get_MFR_inference_model()
    ml_input = inference_model.preprocess(str(cad_file_path))
    predictions, probabilities = inference_model.predict_and_postprocess(ml_input)

    session_preds = _json_safe(predictions)

    viewer_url = None
    image_url = None
    scs_filename = None
    try:
        viewer_result = create_CAD_viewer(cad_file_path, session_id)
        viewer_url = viewer_result.get("viewer_url")
        image_url = viewer_result.get("image_url")
        scs_filename = viewer_result.get("_scs_filename")
    except Exception:
        pass

    # Colorize viewer faces based on predictions
    labels_description = get_MFR_labels_description()
    color_palette = ColorPalette.from_labels(
        labels_description,
        reserved_colors={0: (200, 200, 200)},
    )

    face_colors: list[list[int]] = []
    for label_id in session_preds:
        rgb = color_palette.get_color(int(label_id))
        face_colors.append([int(rgb[0]), int(rgb[1]), int(rgb[2])])

    if scs_filename:
        CAD_face_colors[scs_filename] = face_colors

    # Build color legend for only the labels present in this model
    present_label_ids = set(int(lid) for lid in session_preds)
    color_map = {
        str(label_id): {
            "name": info["name"],
            "color_rgb": list(color_palette.get_color(label_id)),
        }
        for label_id, info in labels_description.items()
        if int(label_id) in present_label_ids
    }

    if scs_filename:
        CAD_color_maps[scs_filename] = color_map

    return {
        "predictions": _json_safe(predictions),
        "probabilities": _json_safe(probabilities),
        "viewer_url": viewer_url,
        "image_url": image_url,
        "color_map": color_map,
    }


def get_MFR_file_thumbnail(file_id: int) -> bytes:
    from hoops_ai.insights import DatasetViewer

    dataset_viewer = DatasetViewer.from_explorer(get_MFR_dataset_explorer())
    fig = dataset_viewer.show_preview_as_image(
        [file_id],
        k=1,
        grid_cols=1,
        label_format="id",
        figsize=(3, 3),
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def get_MFR_table_of_contents() -> dict[str, Any]:
    explorer = get_MFR_dataset_explorer()

    output = io.StringIO()
    with redirect_stdout(output):
        result = explorer.print_table_of_contents()

    response: dict[str, Any] = {"table_of_contents": output.getvalue()}
    if result is not None:
        response["result"] = _json_safe(result)
    return response


def save_uploaded_CAD_file(upload_file: UploadFile) -> pathlib.Path:
    if not upload_file.filename:
        raise RuntimeError("Uploaded CAD file must have a filename.")

    CAD_shared_dir = get_CAD_shared_dir()
    CAD_shared_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = pathlib.PurePath(upload_file.filename).name
    cad_file_path = CAD_shared_dir.joinpath(f"{uuid.uuid4().hex}_{safe_filename}")
    with cad_file_path.open("wb") as file_object:
        shutil.copyfileobj(upload_file.file, file_object)
    return cad_file_path


def upload_CAD_file_persistent(upload_file: UploadFile) -> tuple[str, pathlib.Path, bool]:
    """Upload a CAD file and store it persistently using SHA-256 content hash as file_id.

    Returns (file_id, path, already_existed). Uploading the same file twice is idempotent.
    """
    if not upload_file.filename:
        raise RuntimeError("Uploaded CAD file must have a filename.")

    data = upload_file.file.read()
    file_id = hashlib.sha256(data).hexdigest()
    safe_filename = pathlib.PurePath(upload_file.filename).name

    CAD_shared_dir = get_CAD_shared_dir()
    CAD_shared_dir.mkdir(parents=True, exist_ok=True)
    dest_path = CAD_shared_dir / f"{file_id}_{safe_filename}"
    existed = dest_path.exists()
    if not existed:
        dest_path.write_bytes(data)
    return file_id, dest_path, existed


def find_persistent_CAD_file(file_id: str) -> pathlib.Path:
    """Look up a previously uploaded CAD file by its file_id (SHA-256 hash).

    Raises RuntimeError if no matching file is found.
    """
    CAD_shared_dir = get_CAD_shared_dir()
    matches = list(CAD_shared_dir.glob(f"{file_id}_*"))
    if not matches:
        raise RuntimeError(
            f"No uploaded file found for file_id '{file_id}'. "
            "Upload the file first via POST /files/upload."
        )
    return matches[0]


def delete_persistent_CAD_file(file_id: str) -> bool:
    """Delete a persistent CAD file by file_id. Returns True if a file was deleted."""
    CAD_shared_dir = get_CAD_shared_dir()
    matches = list(CAD_shared_dir.glob(f"{file_id}_*"))
    for path in matches:
        path.unlink(missing_ok=True)
    return len(matches) > 0


def list_persistent_CAD_files() -> list[dict[str, Any]]:
    """Return metadata for all persistently uploaded CAD files."""
    CAD_shared_dir = get_CAD_shared_dir()
    if not CAD_shared_dir.exists():
        return []
    result = []
    for path in sorted(CAD_shared_dir.iterdir()):
        if not path.is_file():
            continue
        parts = path.name.split("_", 1)
        if len(parts) == 2 and len(parts[0]) == 64:  # SHA-256 hex = 64 chars
            result.append(
                {
                    "file_id": parts[0],
                    "filename": parts[1],
                    "size": path.stat().st_size,
                }
            )
    return result


# Limits for ZIP extraction (also used by routers)
ZIP_MAX_TOTAL_BYTES = 500 * 1024 * 1024  # 500 MB uncompressed
ZIP_MAX_FILES = 50


class _BytesUploadFile:
    """Minimal duck-typed UploadFile used to pass in-memory bytes to core helpers."""

    def __init__(self, data: bytes, filename: str) -> None:
        self.filename = filename
        self.file = io.BytesIO(data)


def extract_zip_cad_files(
    zip_data: bytes,
    max_total_bytes: int = ZIP_MAX_TOTAL_BYTES,
    max_files: int = ZIP_MAX_FILES,
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Extract a ZIP archive and upload each valid CAD file persistently.

    Returns ``(resolved, errors)`` where *resolved* is a list of
    ``(file_id, display_name)`` tuples for successfully processed files, and
    *errors* is a list of ``{"filename": ..., "detail": ...}`` dicts for files
    that could not be processed.

    Raises:
        ZipSlipError: if any member path resolves outside the extraction directory.
        ZipSizeLimitError: if the total uncompressed size exceeds *max_total_bytes*.
        ZipFileLimitError: if the number of CAD files exceeds *max_files*.
    """
    resolved: list[tuple[str, str]] = []
    errors: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir).resolve()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            total_size = 0
            file_count = 0

            for member in zf.infolist():
                if member.is_dir():
                    continue

                # Zip Slip guard: resolve the destination and verify it stays
                # inside the temporary directory.
                member_dest = (tmp_path / member.filename).resolve()
                try:
                    member_dest.relative_to(tmp_path)
                except ValueError:
                    raise ZipSlipError(
                        f"Zip Slip detected: '{member.filename}' "
                        "resolves outside the extraction directory."
                    )

                suffix = pathlib.Path(member.filename).suffix.lower()
                if suffix not in CAD_ALLOWED_EXTENSIONS:
                    continue  # skip non-CAD entries silently

                total_size += member.file_size
                if total_size > max_total_bytes:
                    raise ZipSizeLimitError(
                        f"ZIP archive total uncompressed size exceeds the "
                        f"{max_total_bytes // (1024 * 1024)} MB limit."
                    )

                file_count += 1
                if file_count > max_files:
                    raise ZipFileLimitError(
                        f"ZIP archive contains more than {max_files} CAD files."
                    )

                member_dest.parent.mkdir(parents=True, exist_ok=True)
                member_dest.write_bytes(zf.read(member.filename))

                display_name = pathlib.Path(member.filename).name
                try:
                    fake = _BytesUploadFile(member_dest.read_bytes(), display_name)
                    fid, _, _ = upload_CAD_file_persistent(fake)
                    resolved.append((fid, display_name))
                except Exception as exc:
                    errors.append({"filename": display_name, "detail": str(exc)})

    return resolved, errors


def get_shared_CAD_file(cad_file_path: str) -> pathlib.Path:
    requested_path = pathlib.Path(cad_file_path).expanduser()

    if requested_path.is_absolute():
        resolved_path = requested_path.resolve()
    else:
        CAD_shared_dir = get_CAD_shared_dir()
        resolved_path = CAD_shared_dir.joinpath(requested_path).resolve()
        try:
            resolved_path.relative_to(CAD_shared_dir)
        except ValueError as exc:
            raise RuntimeError(f"CAD file must be under shared folder: {CAD_shared_dir}") from exc

    if not resolved_path.exists() or not resolved_path.is_file():
        raise RuntimeError(f"CAD file not found: {resolved_path}")
    return resolved_path


def create_CAD_viewer(cad_file_path: pathlib.Path, session_id: Optional[str] = None) -> dict[str, Any]:
    from hoops_ai.cadaccess import HOOPSLoader, HOOPSTools

    CAD_VIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_key = str(cad_file_path.resolve())

    # Always track under a session key (fall back to "" when no session ID is provided)
    # so that terminate_CAD_viewer can find and delete the files.
    # Reuse existing SCS only when a stable session_id is provided.
    track_key = session_id or ""
    session_viewers = CAD_viewers.setdefault(track_key, {})
    if session_id:
        existing = session_viewers.get(file_key)
        if existing:
            scs_path = CAD_VIEWER_OUTPUT_DIR / existing["scs_filename"]
            if scs_path.exists():
                png_url = f"/out/{existing['png_filename']}" if existing.get("png_filename") else None
                return {"viewer_url": f"/CAD/viewer/show?scs={scs_path.name}", "image_url": png_url, "_scs_filename": scs_path.name}

    # Always generate a UUID-based SCS filename so that different clients (or
    # different sessions) never collide even when opening the same source file.
    unique_id = uuid.uuid4().hex[:12]
    scs_name = f"{unique_id}_{cad_file_path.stem}.scs"
    scs_path = CAD_VIEWER_OUTPUT_DIR / scs_name

    cad_loader = HOOPSLoader()
    model = cad_loader.create_from_file(str(cad_file_path))

    tools = HOOPSTools()
    png_path, scs_path = tools.exportStreamCache(
        model,
        filename=str(scs_path),
        is_white_background=True,
        overwrite=True,
    )
    scs_path = pathlib.Path(scs_path)
    png_path = pathlib.Path(png_path) if png_path else None

    png_filename = png_path.name if png_path and png_path.exists() else None
    session_viewers[file_key] = {"scs_filename": scs_path.name, "png_filename": png_filename}

    png_url = f"/out/{png_filename}" if png_filename else None
    return {"viewer_url": f"/CAD/viewer/show?scs={scs_path.name}", "image_url": png_url, "_scs_filename": scs_path.name}


def terminate_CAD_viewer(session_id: Optional[str] = None, terminate_all: bool = False) -> dict[str, Any]:
    session_viewers = CAD_viewers.get(session_id or "", {})

    def _delete_viewer_files(info: dict) -> None:
        for key in ("scs_filename", "png_filename"):
            fname = info.get(key)
            if fname:
                fpath = CAD_VIEWER_OUTPUT_DIR / fname
                fpath.unlink(missing_ok=True)
        scs = info.get("scs_filename")
        if scs:
            CAD_face_colors.pop(scs, None)
            CAD_color_maps.pop(scs, None)

    if terminate_all:
        count = len(session_viewers)
        for info in session_viewers.values():
            _delete_viewer_files(info)
        session_viewers.clear()
        return {"terminated": count}

    if not session_viewers:
        raise RuntimeError("No active CAD viewer.")

    file_key = next(reversed(session_viewers))
    _delete_viewer_files(session_viewers[file_key])
    del session_viewers[file_key]
    return {"terminated": 1}


def build_brep_adjacency_graph(cad_file_path: pathlib.Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from hoops_ai.cadaccess import HOOPSLoader, HOOPSTools
    from hoops_ai.cadencoder import BrepEncoder

    cad_loader = HOOPSLoader()
    cad_model = cad_loader.create_from_file(str(cad_file_path))

    hoopstools = HOOPSTools()
    hoopstools.adapt_brep(cad_model)

    brep_encoder = BrepEncoder(cad_model.get_brep())
    adj_graph = brep_encoder.push_face_adjacency_graph()

    # Graph data as JSON
    graph_data = {
        "nodes": list(adj_graph.nodes()),
        "edges": [list(e) for e in adj_graph.edges()],
        "num_nodes": adj_graph.number_of_nodes(),
        "num_edges": adj_graph.number_of_edges(),
    }

    # Graph image saved to /out and returned as URL
    CAD_VIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_filename = f"{uuid.uuid4()}.png"
    image_path = CAD_VIEWER_OUTPUT_DIR / image_filename
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.kamada_kawai_layout(adj_graph)
    nx.draw_networkx(adj_graph, pos, arrows=False, ax=ax)
    ax.axis("off")
    fig.savefig(str(image_path), format="png", bbox_inches="tight")
    plt.close(fig)

    return {
        "graph": graph_data,
        "image_url": f"/out/{image_filename}",
    }


def get_brep_attributes(cad_file_path: pathlib.Path) -> dict[str, Any]:
    from hoops_ai.cadaccess import HOOPSLoader, HOOPSTools
    from hoops_ai.cadencoder import BrepEncoder

    cad_loader = HOOPSLoader()
    cad_model = cad_loader.create_from_file(str(cad_file_path))

    hoopstools = HOOPSTools()
    hoopstools.adapt_brep(cad_model)

    brep_encoder = BrepEncoder(cad_model.get_brep())

    [face_types, face_areas, face_centroids, face_loops], face_types_descr = brep_encoder.push_face_attributes()
    [edge_types, edge_lengths, edge_dihedrals, edge_convexities], edge_types_descr = brep_encoder.push_edge_attributes()

    return {
        "faces": {
            "types": _json_safe(face_types),
            "areas": _json_safe(face_areas),
            "centroids": _json_safe(face_centroids),
            "loops": _json_safe(face_loops),
            "types_description": _json_safe(face_types_descr),
        },
        "edges": {
            "types": _json_safe(edge_types),
            "lengths": _json_safe(edge_lengths),
            "dihedrals": _json_safe(edge_dihedrals),
            "convexities": _json_safe(edge_convexities),
            "types_description": _json_safe(edge_types_descr),
        },
    }


# ---------------------------------------------------------------------------
# Part Classification
# ---------------------------------------------------------------------------

def get_part_class_labels_description() -> dict[int, dict[str, str]]:
    """Return the 45-class part label dict from the shared labels module."""
    import importlib.util

    labels_file = APP_ROOT / "part_classification_labels.py"
    if not labels_file.exists():
        raise RuntimeError(
            f"Part classification labels file not found: {labels_file}. "
            "Expected part_classification_labels.py at the repository root."
        )
    spec = importlib.util.spec_from_file_location("part_classification_labels", str(labels_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.labels_description


def _get_part_class_flow_root_dir() -> pathlib.Path:
    """Resolve the flow root directory for the part classification dataset.

    Priority (same convention as MFR):
    1. <HOOPS_AI_SDK_DIR>/notebooks/out/flows/<FLOW_NAME>  Enotebook-generated output (has stream_cache)
    2. <HOOPS_AI_SDK_DIR>/packages/flows/<FLOW_NAME>  Epre-packaged dataset
    """
    notebooks_dir = get_notebooks_dir()
    flow_name = get_required_env("HOOPS_AI_PART_CLASS_FLOW_NAME")
    notebook_out = notebooks_dir / "out" / "flows" / flow_name
    if notebook_out.exists():
        return notebook_out.resolve()
    packages_flow = (get_packages_dir() / "flows" / flow_name).resolve()
    return require_path(
        packages_flow,
        env_name="HOOPS_AI_PART_CLASS_FLOW_NAME",
        label=f"Part classification flow directory for '{flow_name}'",
    )


def create_part_class_inference_model():
    """Load the GraphClassification checkpoint once and return the FlowInference instance."""
    import torch.nn as nn
    from hoops_ai.cadaccess import HOOPSLoader
    from hoops_ai.ml.EXPERIMENTAL import FlowInference, GraphClassification

    load_env_file()

    notebooks_dir = get_notebooks_dir()
    model_name = get_required_env("HOOPS_AI_PART_CLASS_MODEL_NAME")
    ckpt_path = require_path(
        get_packages_dir() / "trained_ml_models" / model_name,
        env_name="HOOPS_AI_PART_CLASS_MODEL_NAME",
    )

    use_gnn = os.environ.get("HOOPS_AI_PART_CLASS_USE_GNN_SURFACE_ENCODER", "false").lower() != "false"

    output_dir = notebooks_dir / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoints saved with older PyG store NNConv's edge MLP as 'edge_func';
    # current PyG renamed it to 'nn'.  Patch Module._load_from_state_dict so
    # every submodule's load transparently remaps the old key name.
    _orig = nn.Module._load_from_state_dict

    def _remap_edge_func(self, state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
        old_prefix = prefix + "edge_func."
        new_prefix = prefix + "nn."
        for k in [k for k in list(state_dict.keys()) if k.startswith(old_prefix)]:
            state_dict[new_prefix + k[len(old_prefix):]] = state_dict.pop(k)
        return _orig(self, state_dict, prefix, local_metadata, strict,
                     missing_keys, unexpected_keys, error_msgs)

    nn.Module._load_from_state_dict = _remap_edge_func
    try:
        solid_classification = GraphClassification(
            num_classes=45,
            use_gnn_surface_encoder=use_gnn,
            result_dir=str(output_dir),
        )
        inference_model = FlowInference(
            cad_loader=HOOPSLoader(),
            flowmodel=solid_classification,
        )
        inference_model.load_from_checkpoint(str(ckpt_path))
    finally:
        nn.Module._load_from_state_dict = _orig  # always restore

    return inference_model


def create_part_class_dataset_explorer():
    """Open the part classification DatasetExplorer (dataset / infoset / attribset)."""
    load_env_file()

    DatasetExplorer = import_MFR_dataset_explorer()  # reuse the same SSL workaround

    flow_root_dir = _get_part_class_flow_root_dir()
    # Derive flow name: prefer env var, fall back to directory name
    flow_name_val = os.environ.get("HOOPS_AI_PART_CLASS_FLOW_NAME") or flow_root_dir.name

    return DatasetExplorer(
        merged_store_path=str(flow_root_dir / f"{flow_name_val}.dataset"),
        parquet_file_path=str(flow_root_dir / f"{flow_name_val}.infoset"),
        parquet_file_attribs=str(flow_root_dir / f"{flow_name_val}.attribset"),
        dask_client_params={"processes": False},
    )


def run_part_classification_inference(cad_file_path: pathlib.Path, top_k: int = 5) -> dict[str, Any]:
    """Preprocess a CAD file and run part classification. Returns top-k predictions."""
    labels = get_part_class_labels_description()
    inference_model = get_part_class_inference_model()

    ml_input = inference_model.preprocess(str(cad_file_path))
    predictions = inference_model.predict_and_postprocess(ml_input)

    # predictions shape: (batch=1, 2, num_classes)
    #   axis-1[0]: class indices sorted by confidence (descending)
    #   axis-1[1]: probability percentages (int)
    n_classes = predictions.shape[2]
    top_k_actual = min(top_k, n_classes)

    top_predictions = []
    for i in range(top_k_actual):
        class_id = int(_json_safe(predictions[0, 0, i]))
        confidence = int(_json_safe(predictions[0, 1, i]))
        part_name = labels.get(class_id, {}).get("name", f"class_{class_id}")
        top_predictions.append(
            {"rank": i + 1, "class_id": class_id, "part_name": part_name, "confidence": confidence}
        )

    return {
        "predicted_class_id": top_predictions[0]["class_id"] if top_predictions else None,
        "predicted_part_name": top_predictions[0]["part_name"] if top_predictions else None,
        "top_predictions": top_predictions,
    }


def get_part_class_table_of_contents() -> dict[str, Any]:
    explorer = get_part_class_dataset_explorer()

    output = io.StringIO()
    with redirect_stdout(output):
        result = explorer.print_table_of_contents()

    response: dict[str, Any] = {
        "table_of_contents": output.getvalue(),
        "available_groups": _json_safe(explorer.available_groups()),
    }
    if result is not None:
        response["result"] = _json_safe(result)
    return response


def get_part_class_label_distribution() -> dict[str, Any]:
    explorer = get_part_class_dataset_explorer()
    label_key = os.environ.get("HOOPS_AI_PART_CLASS_LABEL_KEY", "part_label")

    dist = explorer.create_distribution(key=label_key, bins=None, group="Labels")
    labels = get_part_class_labels_description()

    bins = []
    for i, file_ids_in_bin in enumerate(dist["file_id_codes_in_bins"]):
        bin_start = float(dist["bin_edges"][i])
        bin_end = float(dist["bin_edges"][i + 1])
        class_id = int(bin_start + 0.5)  # bin is centered on the integer label
        part_name = labels.get(class_id, {}).get("name", f"class_{class_id}")
        safe_ids = _json_safe(file_ids_in_bin)
        file_count = len(safe_ids) if isinstance(safe_ids, list) else int(file_ids_in_bin.size)
        bins.append(
            {
                "class_id": class_id,
                "part_name": part_name,
                "file_count": file_count,
            }
        )

    return {"label_key": label_key, "bins": bins}


def get_part_class_file_list(label_id: int) -> dict[str, Any]:
    explorer = get_part_class_dataset_explorer()
    label_key = os.environ.get("HOOPS_AI_PART_CLASS_LABEL_KEY", "part_label")

    label_matches = lambda ds: ds[label_key] == label_id  # noqa: E731
    file_ids = explorer.get_file_list(group="Labels", where=label_matches)

    labels = get_part_class_labels_description()
    part_name = labels.get(label_id, {}).get("name", f"class_{label_id}")

    safe_ids = _json_safe(file_ids)
    return {
        "label_id": label_id,
        "part_name": part_name,
        "file_ids": safe_ids,
        "count": len(safe_ids) if isinstance(safe_ids, list) else int(file_ids.shape[0]),
    }


def get_part_class_preview_image(file_ids: list, k: int = 25, grid_cols: int = 8) -> str:
    """Render a thumbnail grid for the given file IDs. Saves to /out/ and returns the filename."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hoops_ai.insights import DatasetViewer

    explorer = get_part_class_dataset_explorer()
    dataset_viewer = DatasetViewer.from_explorer(explorer)

    fig = dataset_viewer.show_preview_as_image(
        file_ids,
        k=k,
        grid_cols=grid_cols,
        label_format="id",
        figsize=(15, 5),
    )

    image_filename = f"{uuid.uuid4()}.png"
    image_path = CAD_VIEWER_OUTPUT_DIR / image_filename
    CAD_VIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(image_path), format="png", bbox_inches="tight")
    plt.close(fig)
    return image_filename


# ---------------------------------------------------------------------------
# Context Layer prediction
# ---------------------------------------------------------------------------


class _Hit:
    """Lightweight hit proxy exposing ``.id`` and ``.score`` for ContextPredictor.

    Wraps a single ``{"id": str, "score": float}`` entry returned by a prior
    similarity search so that it satisfies the duck-typed contract expected by
    ``ContextPredictor.infer()``.
    """

    __slots__ = ("id", "score")

    def __init__(self, id: str, score: float) -> None:  # noqa: A002
        self.id = id
        self.score = score


def _build_aggregation_rule(spec: dict[str, Any]) -> Any:
    """Build a ``NumericWeightedRule`` or ``NearestNeighborRule`` from a spec dict.

    *spec* must contain a ``"type"`` key with value ``"numeric_weighted"`` or
    ``"nearest_neighbor"``.  Remaining keys are forwarded as constructor kwargs.

    Raises
    ------
    ValueError
        When ``"type"`` is missing or not a recognised rule-type string.

    Notes
    -----
    This helper performs its own lazy import and therefore relies on Python's
    import cache; it should only be called after the ``hoops_ai.ml.context_layer``
    guard in :func:`predict_context` has already run successfully.
    """
    from hoops_ai.ml.context_layer import NearestNeighborRule, NumericWeightedRule

    rule_type = spec.get("type")
    if rule_type == "numeric_weighted":
        kwargs: dict[str, Any] = {
            "log_scale": spec["log_scale"],
            "auto_relevance_weight": spec["auto_relevance_weight"],
        }
        if spec.get("nearest_neighbor_threshold") is not None:
            kwargs["nearest_neighbor_threshold"] = spec["nearest_neighbor_threshold"]
        if spec.get("score_temperature") is not None:
            kwargs["score_temperature"] = spec["score_temperature"]
        return NumericWeightedRule(**kwargs)
    elif rule_type == "nearest_neighbor":
        kwargs = {}
        if spec.get("threshold") is not None:
            kwargs["threshold"] = spec["threshold"]
        return NearestNeighborRule(**kwargs)
    else:
        raise ValueError(
            f"Unknown rule type '{rule_type}'. "
            "Must be 'numeric_weighted' or 'nearest_neighbor'."
        )


def predict_context(
    hits: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    keys: list[str],
    numeric_keys: Optional[list[str]] = None,
    query_context: Optional[dict[str, Any]] = None,
    default_categorical_rule: Optional[dict[str, Any]] = None,
    per_key_rules: Optional[dict[str, dict[str, Any]]] = None,
    status_policy: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Predict missing metadata values from similarity-search hits.

    Builds an in-memory ``_StaticContextProvider`` from *contexts*, wraps each
    entry in *hits* as a :class:`_Hit`, constructs a ``ContextPredictor`` with
    the supplied rule configuration, calls ``infer()``, and converts the result
    to a JSON-serialisable dict.

    This function is **stateless** — it performs no network I/O and holds no
    reference to PLM/ERP systems.  The caller is responsible for populating
    *contexts* from whatever metadata store it has access to.

    Parameters
    ----------
    hits:
        List of ``{"id": str, "score": float}`` dicts from a prior similarity
        search.  Each entry is wrapped in a :class:`_Hit`.
    contexts:
        Caller-supplied metadata keyed by part id.  Only ids present here
        are visible to the predictor; missing ids get an empty dict.
    keys:
        Metadata keys to predict (e.g. ``["Material", "Process", "CostUSD"]``).
    numeric_keys:
        Keys in *contexts* that should be treated as numeric by the predictor.
        Forwarded to ``ContextProvider.list_numeric_keys()``.
    query_context:
        Known metadata for the query part (e.g. already-known ``Material``)
        that the predictor can use as a boosting signal when computing
        ``NumericWeightedRule`` or ``NearestNeighborRule`` scores.
    default_categorical_rule:
        Dict with ``temperature`` and/or ``min_margin`` for the default
        ``CategoricalRule``.  ``None`` uses the hoops_ai built-in defaults.
    per_key_rules:
        Mapping from key name to a rule-spec dict as returned by
        ``NumericRuleSpec.model_dump()``.  Keys not listed here fall back to
        *default_categorical_rule*.
    status_policy:
        Optional ``{status_label: threshold}`` mapping forwarded verbatim to
        ``ContextPredictor.infer()``.

    Returns
    -------
    dict
        ``{key: {"value": ..., "confidence": float, "status": str,
        "injected_context": dict | None}}`` for every predicted key.

    Raises
    ------
    EnvConfigError
        When ``hoops_ai.ml.context_layer`` cannot be imported (e.g. the
        hoops_ai package is not installed or the licence is not configured).
        Other endpoints are unaffected.
    ValueError
        When an unknown rule type is supplied in *per_key_rules*.
    """
    try:
        from hoops_ai.ml.context_layer import (
            CategoricalRule,
            ContextPredictor,
            ContextProvider,
        )
    except ImportError as exc:
        msg = (
            "[CONFIG] hoops_ai.ml.context_layer could not be imported. "
            "Ensure the hoops_ai package is installed and the licence is configured: "
            f"{exc}"
        )
        logger.error(msg)
        print(msg, flush=True)
        raise EnvConfigError(msg) from exc

    class _StaticContextProvider(ContextProvider):
        """In-request ContextProvider that serves pre-supplied metadata dicts.

        Instantiated once per request from caller-supplied *contexts* so that
        ``ContextPredictor`` can retrieve metadata without any network I/O.

        Parameters
        ----------
        contexts:
            Part id → metadata dict.  Only ids present here are returned by
            ``get_contexts``; unknown ids yield an empty dict.
        numeric_keys:
            Keys that ``ContextPredictor`` should treat as numeric.
        """

        def __init__(
            self,
            ctx: dict[str, dict[str, Any]],
            nkeys: list[str],
        ) -> None:
            self._ctx = ctx
            self._nkeys = list(nkeys)

        def get_contexts(self, ids: list[str]) -> dict[str, dict[str, Any]]:
            return {id_: self._ctx.get(id_, {}) for id_ in ids}

        def list_numeric_keys(self) -> list[str]:
            return self._nkeys

    provider = _StaticContextProvider(
        ctx=contexts,
        nkeys=numeric_keys or [],
    )

    cat_rule_kwargs: dict[str, Any] = {}
    if default_categorical_rule:
        if "temperature" in default_categorical_rule:
            cat_rule_kwargs["temperature"] = default_categorical_rule["temperature"]
        if "min_margin" in default_categorical_rule:
            cat_rule_kwargs["min_margin"] = default_categorical_rule["min_margin"]
    default_cat = CategoricalRule(**cat_rule_kwargs)

    built_per_key: dict[str, Any] = {}
    if per_key_rules:
        for key, spec in per_key_rules.items():
            built_per_key[key] = _build_aggregation_rule(spec)

    predictor = ContextPredictor(
        context_provider=provider,
        default_categorical_rule=default_cat,
        per_key_rules=built_per_key,
    )

    hit_objs = [_Hit(id=h["id"], score=h["score"]) for h in hits]

    infer_kwargs: dict[str, Any] = {"keys": keys}
    if query_context is not None:
        infer_kwargs["query_context"] = query_context
    if status_policy is not None:
        infer_kwargs["status_policy"] = status_policy

    results = predictor.infer(hit_objs, **infer_kwargs)

    return {
        key: {
            "value": _json_safe(pred.value),
            "confidence": float(pred.confidence),
            "status": str(pred.status),
            "injected_context": (
                _json_safe(pred.injected_context) if pred.injected_context else None
            ),
        }
        for key, pred in results.items()
    }

