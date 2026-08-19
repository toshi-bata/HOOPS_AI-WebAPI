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
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from contextlib import contextmanager, redirect_stdout
from typing import Any, Optional

logger = logging.getLogger(__name__)

from fastapi import UploadFile

APP_ROOT = pathlib.Path(__file__).resolve().parent
ENV_FILE_PATH = pathlib.Path(__file__).with_name(".env")
CAD_UPLOAD_DIR = APP_ROOT.joinpath("uploads")
CAD_VIEWER_OUTPUT_DIR = APP_ROOT.joinpath("out")
EMBEDDINGS_CACHE_DIR = APP_ROOT.joinpath("embeddings_cache")
INDEXES_DIR = APP_ROOT.joinpath("indexes")
JOBS_DIR = APP_ROOT.joinpath("jobs")

# Named-index on-disk schema version.
#   1 = legacy: one averaged vector row per file (created before this migration)
#   2 = body-level: one vector row per body, with kind/bodies/thumbnail/scs/obb
# schema_version is recorded in indexes/<name>/index.json.
_INDEX_SCHEMA_VERSION = 2

# Allowed CAD file extensions for upload/ZIP extraction.
#
# Mirrors cadNameFilters() in hoops_ai_qt_sandbox/SimilarityIndexPanel.cpp, which
# in turn mirrors the "All Supported Files" list of HOOPS Exchange. Kept lower
# case; every comparison lower-cases the candidate suffix first, so files saved
# with upper-case extensions (part.SLDPRT is common with SolidWorks exports) are
# accepted on case-sensitive filesystems too.
CAD_ALLOWED_EXTENSIONS = frozenset({
    ".3ds", ".3mf", ".3dxml", ".sat", ".sab", ".dgn",
    ".dwg", ".dxf", ".dwf", ".dwfx", ".model", ".dlv",
    ".exp", ".session", ".catpart", ".catproduct", ".catshape",
    ".catdrawing", ".cgr", ".dae", ".prt", ".neu",
    ".asm", ".xas", ".xpr", ".fbx", ".gltf", ".glb",
    ".arc", ".unv", ".mf1", ".pkg", ".ifc", ".ifczip",
    ".igs", ".iges", ".ipt", ".iam", ".jt", ".kmz",
    ".nwd", ".prc", ".x_t", ".xmt", ".x_b", ".xmt_txt",
    ".rvt", ".3dm", ".stp", ".step", ".stpz", ".stl",
    ".par", ".pwd", ".psm", ".sldprt", ".sldasm", ".sldfpp",
    ".u3d", ".vda", ".wrl", ".vrml", ".obj", ".xv3",
    ".xv0", ".hsf",
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

# Cache of CADSearch instances for named indexes, keyed by (faiss path, mtime_ns).
# The mtime is part of the key so a rebuilt index is picked up automatically on
# the next request. Each instance loads the whole index into its own memory, so
# the cache is deliberately tiny; two entries let one index be searched while a
# second one is in use, without holding several copies of large indexes.
_NAMED_SEARCHER_CACHE_MAX = 2
_named_searchers: "OrderedDict[tuple[str, int], Any]" = OrderedDict()

# Accepted values for the named-index search `kind` filter. "any" means no
# filter is passed to CADSearch at all (preserves pre-filter behaviour).
_SEARCH_KINDS = frozenset({"any", "part", "assembly"})

# --- Assembly-to-assembly search (vendor/assembly_matcher.py) --------------
# Accepted coverage denominators. "symmetric" = matched / larger side,
# "containment" = matched / query size, "jaccard" = matched / union.
_ASSEMBLY_COVERAGE_MODES = frozenset({"symmetric", "containment", "jaccard"})

# An id with at least this many body rows counts as an assembly. Matches both
# hoops_ai_native_bridge and _compute_faiss_stats().
_MIN_BODIES_FOR_ASSEMBLY = 2

# Cache of AssemblyMatcher instances, keyed by (faiss path, mtime_ns) exactly
# like _named_searchers, so an index write invalidates it automatically.
# Only ONE entry is kept (hoops_ai_native_bridge keeps one too): a matcher owns
# a CADSearch *plus* an L2-normalised copy of every body vector, roughly twice
# the on-disk index size, which is far too large to hold several of.
_ASSEMBLY_MATCHER_CACHE_MAX = 1
_assembly_matchers: "OrderedDict[tuple[str, int], Any]" = OrderedDict()

# Thread-pool size for the matcher's stage-2 scoring. The bridge hard-codes 8;
# here it is configurable because those threads compete with the server's own
# request workers.
_ASSEMBLY_SEARCH_JOBS_DEFAULT = 8

# Bounds for the tunable scoring parameters, enforced before reaching the SDK.
_ASSEMBLY_CANDIDATE_K_MAX = 1000

# per-index write lock; also held while resolving the cached searcher
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
        _save_index_schema(name, model, _INDEX_SCHEMA_VERSION)
        _named_indexes[name] = vs
        return {"name": name, "count": 0, "dim": dim, "model": model}


def list_indexes() -> list[dict[str, Any]]:
    """Return metadata for all known indexes, including the read-only ``default`` index."""
    result: list[dict[str, Any]] = []

    # "default" index 窶・read-only, backed by the env-configured FAISS file
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

    # Named indexes from INDEXES_DIR 窶・each lives in its own subdirectory
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


_EMBED_ERROR_DEFAULT = "embedding produced no rows (file skipped by embed_shape_batch)"


def _embed_error_detail(
    embed_errors: Any,
    fid: str,
    filename: str,
    norm_path: str,
) -> str:
    """Best-effort failure reason for *fid* from ``batch.metadata['errors']``.

    Observed on hoops_ai 1.1.0: a list of plain strings shaped like
    ``"Failed to compute embedding for <abs path>: <reason>"``. Other shapes
    (list of dicts, dict keyed by path) are still handled defensively because
    the contract is not documented. Always returns a non-empty string.

    Matching is path-first and filename matching is boundary-aware, so a file
    named ``0.stp`` never picks up the error belonging to ``100.stp``. Matched
    messages are returned verbatim -- they are short and already name the file.
    """
    default = _EMBED_ERROR_DEFAULT

    def _norm(p: str) -> str:
        try:
            return os.path.normcase(os.path.normpath(p))
        except Exception:
            return os.path.normcase(p)

    want_path = _norm(norm_path) if norm_path else ""
    want_file = os.path.normcase(filename) if filename else ""

    def _mentions(text: str) -> bool:
        if not text:
            return False
        low = _norm(text)
        if want_path and want_path in low:
            return True
        if fid and fid in text:
            return True
        if not want_file:
            return False
        # Boundary-aware filename search: reject "0.stp" inside "100.stp".
        start = 0
        while True:
            k = low.find(want_file, start)
            if k < 0:
                return False
            before_ok = k == 0 or low[k - 1] in "\\/\"' \t([<,"
            end = k + len(want_file)
            after_ok = end >= len(low) or not (
                low[end].isalnum() or low[end] in "._-"
            )
            if before_ok and after_ok:
                return True
            start = k + 1

    if isinstance(embed_errors, dict):
        for key, val in embed_errors.items():
            if _mentions(str(key)) and val:
                return str(val).strip() or default
    elif isinstance(embed_errors, (list, tuple)):
        for item in embed_errors:
            if isinstance(item, str):
                if _mentions(item):
                    return item.strip() or default
            elif isinstance(item, dict):
                text = " ".join(str(v) for v in item.values())
                if _mentions(text):
                    for k in ("detail", "message", "error", "reason"):
                        if item.get(k):
                            return str(item[k]).strip() or default
                    return text.strip() or default
    return default


# --- Embedding tuning: worker count and per-file time budget -----------------
#
# Ported from hoops_ai_native_bridge's ComputeAutoWorkers. Every worker is a
# spawned interpreter holding its own copy of the ~2 GB checkpoint, so peak RSS
# grows roughly linearly with the worker count while throughput reaches a
# plateau and then declines once RAM-bound. Measured on the bridge: light
# single-body parts plateau near the physical core count (12 was the peak on a
# 14-core machine, 8..cores within +/-5%); heavy assemblies plateau lower (12
# peaked on a 16-core machine, and 16/20/24 were each slower than 8). So aim at
# the plateau rather than a single "best" value -- run-to-run noise exceeds the
# difference between neighbouring counts.
#
# The default cap of 8 is a conservative floor chosen for an 8-core laptop with
# heavy files. Raise HOOPS_AI_MAX_WORKERS on a large machine with light CAD.
# The three variable names below are the ones hoops_ai and the bridge already
# use, deliberately kept identical so one environment configures both.
#
# HOOPS_AI_MIN_FILES_PARALLEL defaults to 1 here, i.e. disabled, unlike the
# bridge's 32. That threshold judges by file count and ignores how much work
# each file is: six heavy assemblies are measurably faster on three workers,
# and forcing them onto one would be a large regression. Its stated rationale
# -- that pool startup does not pay off for a few files -- is also weak, since
# the ~55-60 s startup measured here was for a 20-worker pool and scales with
# the worker count. Set it if your corpus is uniformly light.
_MAX_WORKERS_DEFAULT = 8
_MODEL_FOOTPRINT_MB_DEFAULT = 2048
_MIN_FILES_PARALLEL_DEFAULT = 1
_HOST_RAM_RESERVE_BYTES = 2 * 1024 * 1024 * 1024


def compute_auto_workers(file_count: int) -> int:
    """Choose a worker count for *file_count* files.

    ``min(HOOPS_AI_MAX_WORKERS, logical_cpus / 2, usable_RAM / model_footprint,
    file_count)``. Half the logical CPUs approximates the physical/performance
    cores and avoids E-core and SMT oversubscription; never exceeding the file
    count keeps idle workers from paying the checkpoint load for nothing.

    This is only consulted when the caller passes no explicit worker count. An
    explicit value is forwarded to the SDK unchanged, cap included -- the caps
    here exist to pick a safe default, not to overrule a deliberate choice.

    Deliberately never returns ``None`` (which would let the SDK auto-detect
    across all logical cores): that oversubscribes RAM, and on this corpus a
    20-worker run was ~2x slower per file than the bridge's 12.
    """
    max_workers = _env_int("HOOPS_AI_MAX_WORKERS", _MAX_WORKERS_DEFAULT, minimum=1)
    model_mb = _env_int(
        "HOOPS_AI_MODEL_FOOTPRINT_MB", _MODEL_FOOTPRINT_MB_DEFAULT, minimum=1
    )
    min_parallel = _env_int(
        "HOOPS_AI_MIN_FILES_PARALLEL", _MIN_FILES_PARALLEL_DEFAULT, minimum=1
    )
    if file_count < min_parallel:
        return 1

    logical = os.cpu_count() or 2
    core_cap = max(1, logical // 2)

    ram_cap = max_workers
    ram_bound = False
    available_gb = None
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
        available_gb = available / (1024 ** 3)
        usable = max(0, available - _HOST_RAM_RESERVE_BYTES)
        ram_cap = usable // (model_mb * 1024 * 1024)
        ram_bound = True
    except Exception:
        # No psutil (or an unsupported platform): fall back to the core cap
        # alone rather than guessing at memory.
        logger.debug("[EMBED] RAM probe unavailable; sizing workers by cores only")

    chosen = max(1, min(max_workers, core_cap, ram_cap, max(1, file_count)))

    # Free RAM is sampled at job start and fluctuates -- a preceding job's
    # workers may not have been reclaimed yet -- so a surprisingly low count
    # must be explainable from the log rather than looking arbitrary.
    logger.info(
        "[EMBED] auto workers=%d (cap=%d, cores=%d, ram=%s, files=%d)",
        chosen, max_workers, core_cap, ram_cap if ram_bound else "n/a", file_count,
    )
    if ram_bound and ram_cap <= 1 and min(max_workers, core_cap, file_count) > 1:
        logger.warning(
            "[EMBED] falling back to sequential embedding: only %.1f GB free, "
            "and each worker needs %d MB on top of a %.1f GB host reserve. "
            "Cores would allow %d. Wait for the previous job's workers to be "
            "reclaimed, or pass an explicit 'workers' to override.",
            available_gb, model_mb, _HOST_RAM_RESERVE_BYTES / (1024 ** 3),
            min(max_workers, core_cap),
        )
    return chosen


def embed_time_limit() -> int:
    """Per-file embedding budget in seconds, from HOOPS_AI_EMBED_TIME_LIMIT.

    ``0`` (the default) leaves the SDK's own 120 s in place.
    """
    return _env_int("HOOPS_AI_EMBED_TIME_LIMIT", 0, minimum=0)


def read_too_heavy_files(
    directory: Optional["pathlib.Path"] = None,
) -> list[str]:
    """Paths hoops_ai deferred to its built-in single-worker RAM fallback.

    ``too_heavy_files.log`` lists them: ``#`` comment lines, then one path per
    line. It is informational only -- these files are usually embedded
    successfully, just slowly -- but it is what tells an operator which parts
    made the run expensive.

    *directory* is the pass's ``log_dir``. The process working directory is
    tried as well, because ``log_dir`` support was established by inspecting the
    compiled SDK rather than from documentation; if it is ever ignored, the file
    still lands in the CWD and is still found.
    """
    candidates: list[pathlib.Path] = []
    if directory is not None:
        candidates.append(pathlib.Path(directory))
    candidates.append(pathlib.Path.cwd())

    for base in candidates:
        path = base / "too_heavy_files.log"
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        return [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
    return []


def read_error_summary_reasons(
    directory: Optional["pathlib.Path"] = None,
    newer_than: Optional[float] = None,
) -> dict[str, str]:
    """Per-file failure reasons from the SDK's ``error_summary.json``.

    This is the same source ``hoops_ai_qt_sandbox`` reads
    (``readErrorSummaryReasons`` in SimilarityIndexPanel.cpp): a JSON array of
    ``{"item_index", "item", "error"}`` objects where ``item`` is the absolute
    input path. Only the first line of ``error`` is kept.

    It is needed because ``batch.metadata['errors']`` is not always attributable.
    Measured on hoops_ai 1.1.0: a parallel run that times out reports bare reason
    strings with no path in them, so there is nothing to match a file against --
    and the list is in completion order, which is not the submission order, so
    zipping it positionally would mis-attribute reasons. ``error_summary.json``
    carries the path explicitly, so it is exact.

    The SDK logs ``Undefined flowdir ... Using path [.]`` and writes the file to
    the process working directory, overwriting it on every ``embed_shape_batch``
    call. *directory* is tried first in case a future SDK honours ``log_dir``.
    *newer_than* is a ``time.time()`` stamp taken before the batch started: an
    older file belongs to a previous run and is ignored rather than reported as
    if it described this one.

    Returns ``{normalised path: reason}``; empty when the file is absent,
    unreadable, stale or not in the expected shape.
    """
    import json

    candidates: list[pathlib.Path] = []
    if directory is not None:
        candidates.append(pathlib.Path(directory))
    candidates.append(pathlib.Path.cwd())

    for base in candidates:
        path = base / "error_summary.json"
        try:
            if newer_than is not None and path.stat().st_mtime < newer_than:
                logger.info(
                    "ignoring stale %s (written before this batch started)", path
                )
                continue
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                doc = json.load(fh)
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue

        if not isinstance(doc, list):
            logger.warning(
                "%s has unexpected shape %s (expected a list)",
                path,
                type(doc).__name__,
            )
            continue

        reasons: dict[str, str] = {}
        for entry in doc:
            if not isinstance(entry, dict):
                continue
            item = entry.get("item")
            if not item:
                continue
            reason = str(entry.get("error") or "").splitlines()
            reasons[_norm_path_key(str(item))] = reason[0].strip() if reason else ""
        return reasons
    return {}


def _norm_path_key(path: str) -> str:
    """Normalise a path for matching against the SDK's own log files.

    The SDK writes paths with either slash direction and, on Windows, in a case
    that need not match ours. Mirrors ``normPathKey`` in qt_sandbox.
    """
    try:
        return os.path.normcase(os.path.normpath(path))
    except Exception:
        return os.path.normcase(path)


def build_embed_specifications(
    images_dir: "pathlib.Path",
    time_limit: Optional[int] = None,
    log_dir: Optional["pathlib.Path"] = None,
) -> dict[str, Any]:
    """Build the ``specifications`` dict for ``embed_shape_batch``.

    All four time-limit keys are set to the same value. ``time_limit_overall``
    is the one that governs the per-item "Timeout (CUMULATIVE)" that drops heavy
    assemblies; setting only the size buckets leaves it at the 120 s default, so
    a nominally longer pass would still be killed at 120 s. This was verified
    empirically in hoops_ai_native_bridge, and a large per-item value does not
    abort the whole batch.
    """
    specs: dict[str, Any] = {
        "generate_images": True,
        "images_out_dir": str(images_dir),
    }
    if log_dir is not None:
        # Sends too_heavy_files.log to a per-pass directory so pass 2 does not
        # overwrite pass 1's list. Established by inspecting the compiled SDK
        # (ParallelExecutor.write_too_heavy_log), not from documentation, so
        # readers must tolerate the file landing in the CWD instead.
        specs["log_dir"] = str(log_dir)
    limit = time_limit if time_limit is not None else embed_time_limit()
    if limit and limit > 0:
        value = float(limit)
        specs["time_limit_overall"] = value
        specs["time_limit_small"] = value
        specs["time_limit_medium"] = value
        specs["time_limit_large"] = value
    return specs


_PROGRESS_STDERR_LOCK = threading.Lock()


class _TqdmProgressStderr:
    """``sys.stderr`` replacement that parses hoops_ai's tqdm bar.

    ``embed_shape_batch`` renders a tqdm bar on stderr that already carries
    everything a caller needs::

        Computing embeddings:  70%|####| 28/40 [01:06<00:22, 1.84s/it, pool=1, errors=18, heavy=0]

    Every write is mirrored to the real stderr and then parsed, so the server log
    is unchanged and the job record gains live counters. Ported from
    ``_BridgeProgressStderr`` in hoops_ai_native_bridge (hoops_ai_bridge.cpp),
    including ``isatty() -> True``: tqdm only emits incremental updates when it
    believes it is writing to a terminal, and the server's stderr may be a pipe
    or a service log.

    Parsing is strictly best-effort. Any failure is swallowed, because dropping a
    progress update must never break the embedding run.
    """

    _re_frac = re.compile(r"(\d+)\s*/\s*(\d+)")
    _re_err = re.compile(r"errors=(\d+)")
    _re_heavy = re.compile(r"heavy=(\d+)")

    def __init__(self, real: Any, report: Any) -> None:
        self._real = real
        self._report = report
        self._last: Optional[tuple] = None

    def write(self, s: str) -> int:
        try:
            if self._real is not None:
                self._real.write(s)
        except Exception:
            pass
        try:
            if s and ("Computing embeddings" in s or "Heavy Files" in s):
                phase = "heavy" if "Heavy Files" in s else "embedding"
                m = self._re_frac.search(s)
                if m:
                    done, total = int(m.group(1)), int(m.group(2))
                    me = self._re_err.search(s)
                    mh = self._re_heavy.search(s)
                    key = (
                        phase,
                        done,
                        total,
                        int(me.group(1)) if me else -1,
                        int(mh.group(1)) if mh else -1,
                    )
                    # tqdm redraws the same line many times a second; only report
                    # when a counter actually moved, so the job record is not
                    # rewritten on every repaint.
                    if key != self._last:
                        self._last = key
                        self._report(*key)
        except Exception:
            pass
        return len(s) if s else 0

    def flush(self) -> None:
        try:
            if self._real is not None:
                self._real.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        if self._real is not None:
            return self._real.fileno()
        raise OSError("no fileno")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@contextmanager
def capture_embed_progress(report: Optional[Any]):
    """Install the tqdm-parsing stderr shim for the duration of the block.

    Yields ``True`` when the shim is active. Yields ``False`` when there is
    nothing to report to, or when another embedding run already holds the shim:
    ``sys.stderr`` is process-global, so two concurrent runs would interleave
    into one bar and each would report the other's counters. Registration jobs
    are serialised by default (see :func:`job_max_concurrency`), so the second
    case only arises if an operator raised the concurrency; progress is then
    dropped rather than reported wrongly.

    The original stderr is always restored, including on exceptions.
    """
    if report is None:
        yield False
        return
    if not _PROGRESS_STDERR_LOCK.acquire(blocking=False):
        logger.warning(
            "another embedding run holds the progress shim; "
            "live progress is disabled for this one"
        )
        yield False
        return
    real = sys.stderr
    try:
        sys.stderr = _TqdmProgressStderr(real, report)
        yield True
    finally:
        sys.stderr = real
        _PROGRESS_STDERR_LOCK.release()


def _embed_bodies_for_index(
    file_ids: list[str],
    index_name: str,
    model: str,
    num_workers: Optional[int] = None,
    time_limit: Optional[int] = None,
    log_dir: Optional["pathlib.Path"] = None,
    progress_cb: Optional[Any] = None,
) -> tuple[list, list[dict[str, Any]]]:
    """Embed *file_ids* at body granularity and build one VectorRecord per body.

    Calls ``embedder.embed_shape_batch()`` exactly once for the whole batch
    (performance-critical: never loop per file). Thumbnails (PNG) and stream
    caches (SCS) are produced by the same CAD load via ``generate_images`` and
    relocated to the canonical ``thumbnails/<file_id>.png`` / ``scs/<file_id>.scs``
    locations. Per-body vectors are L2-normalised individually (never averaged).

    Returns ``(records, errors)`` where *records* is a flat list of VectorRecord
    (multiple rows share the same file_id) and *errors* is a per-file error list.

    *num_workers* and *time_limit* are forwarded to the SDK; ``None`` means
    "decide here" (see ``compute_auto_workers`` and ``embed_time_limit``).
    *log_dir* redirects the SDK's ``too_heavy_files.log`` (see
    :func:`build_embed_specifications`).

    *progress_cb* is called as ``cb(phase, done, total, errors, heavy)`` while
    the batch runs, parsed from the SDK's tqdm bar (see
    :func:`capture_embed_progress`). ``None`` keeps the bar suppressed.
    """
    import json
    import shutil
    from collections import OrderedDict

    import numpy as np

    from hoops_ai.ml.embeddings import Embedding, VectorRecord

    embedder = get_embedder(model)
    model_name = (
        getattr(embedder, "model_name", None)
        or getattr(embedder, "model", None)
        or "hoops_embeddings_model"
    )

    errors: list[dict[str, Any]] = []

    # Resolve real CAD paths; keep bidirectional maps keyed by file_id and by
    # the resolved path string (embed_shape_batch echoes input paths in .ids).
    fid_by_norm_path: dict[str, str] = {}
    norm_by_fid: dict[str, str] = {}
    filename_by_fid: dict[str, str] = {}
    ordered_paths: list[str] = []
    for fid in file_ids:
        try:
            cad_path = find_persistent_CAD_file(fid)
        except Exception as exc:
            errors.append({"file_id": fid, "detail": str(exc)})
            continue
        norm = str(cad_path.resolve())
        fid_by_norm_path[norm] = fid
        norm_by_fid[fid] = norm
        parts = cad_path.name.split("_", 1)
        filename_by_fid[fid] = (
            parts[1] if len(parts) == 2 and len(parts[0]) == 64 else cad_path.name
        )
        ordered_paths.append(str(cad_path))

    if not ordered_paths:
        return [], errors

    def _fid_for_row(raw_id: str) -> Optional[str]:
        norm = str(pathlib.Path(str(raw_id)).resolve())
        if norm in fid_by_norm_path:
            return fid_by_norm_path[norm]
        # Single-input fallback: every row belongs to the only file.
        if len(fid_by_norm_path) == 1:
            return next(iter(fid_by_norm_path.values()))
        return None

    tmp_images = INDEXES_DIR / index_name / f"_tmp_images_{uuid.uuid4().hex}"
    tmp_images.mkdir(parents=True, exist_ok=True)

    records: list = []
    try:
        specs = build_embed_specifications(tmp_images, time_limit, log_dir=log_dir)
        workers = num_workers if num_workers is not None else compute_auto_workers(
            len(ordered_paths)
        )
        kwargs: dict[str, Any] = {"show_progress": False, "specifications": specs}
        if workers and workers > 0:
            kwargs["num_workers"] = int(workers)
        logger.info(
            "[EMBED] index '%s': %d file(s), num_workers=%s, time_limit=%s",
            index_name, len(ordered_paths), kwargs.get("num_workers", "sdk-default"),
            specs.get("time_limit_overall", "sdk-default"),
        )
        # Stamp taken before the batch so a leftover error_summary.json from an
        # earlier run is recognised as stale.
        batch_started = time.time()
        with capture_embed_progress(progress_cb) as live:
            # show_progress is deliberately left False. The bridge sets it True
            # to make the bar appear, but measured here on hoops_ai 1.1.0 the
            # parallel executor draws "Computing embeddings: 0/40 ..." on stderr
            # even with it False, so turning it on risks a second, outer bar
            # whose counters would interleave with the one we parse. If a future
            # SDK stops drawing it, progress simply stops advancing -- the run
            # itself is unaffected.
            _ = live
            batch = embedder.embed_shape_batch(ordered_paths, **kwargs)

        ids = list(getattr(batch, "ids", []) or [])
        values = np.asarray(getattr(batch, "values", []), dtype=np.float32)
        if values.ndim == 1 and values.size:
            values = values.reshape(1, -1)
        meta = getattr(batch, "metadata", None) or {}
        png_paths = meta.get("png_paths") if hasattr(meta, "get") else None
        kinds = meta.get("kind") if hasattr(meta, "get") else None
        obbs = meta.get("oriented_bounding_box") if hasattr(meta, "get") else None

        # Group row indices by file_id, preserving first-seen order.
        rows_by_fid: "OrderedDict[str, list[int]]" = OrderedDict()
        unmatched_rows = 0
        for i, raw_id in enumerate(ids):
            fid = _fid_for_row(raw_id)
            if fid is None:
                unmatched_rows += 1
                continue
            rows_by_fid.setdefault(fid, []).append(i)

        if unmatched_rows:
            logger.warning(
                "%d embedded row(s) could not be matched to an input file "
                "in index '%s'",
                unmatched_rows,
                index_name,
            )

        # Surface files that embed_shape_batch silently skipped (CAD load error,
        # timeout, etc.): they never appear in batch.ids, so they are missing
        # from rows_by_fid. Report each one instead of dropping it silently.
        embed_errors = meta.get("errors") if hasattr(meta, "get") else None
        if embed_errors:
            try:
                if isinstance(embed_errors, dict):
                    _first = next(iter(embed_errors.items()))
                else:
                    _first = embed_errors[0]
                print(
                    f"[INDEX] metadata['errors'] type={type(embed_errors).__name__} "
                    f"first={_first!r}",
                    flush=True,
                )
            except Exception:
                pass

        # A later phase routes retries on the reason text (e.g. "Timeout"), so a
        # silent fallback to the generic message must never go unnoticed. Warn
        # at most once per batch when the metadata stops matching the observed
        # contract (list[str] as of hoops_ai 1.1.0).
        if embed_errors is not None and not isinstance(embed_errors, list):
            logger.warning(
                "index '%s': batch metadata['errors'] has unexpected type %s "
                "(expected list[str]); embedding failure reasons may be unreliable",
                index_name,
                type(embed_errors).__name__,
            )

        submitted_fids = set(fid_by_norm_path.values())
        missing_fids = [
            fid for fid in file_ids if fid in submitted_fids and fid not in rows_by_fid
        ]
        # Fallback source for reasons the batch metadata cannot attribute. Read
        # once per batch, and only when there is something to explain.
        summary_reasons: dict[str, str] = {}
        if missing_fids:
            summary_reasons = read_error_summary_reasons(
                log_dir, newer_than=batch_started
            )
        matched_any_reason = False
        summary_hits = 0
        for fid in missing_fids:
            detail = _embed_error_detail(
                embed_errors,
                fid,
                filename_by_fid.get(fid, ""),
                norm_by_fid.get(fid, ""),
            )
            if detail != _EMBED_ERROR_DEFAULT:
                matched_any_reason = True
            elif summary_reasons:
                reason = summary_reasons.get(_norm_path_key(norm_by_fid.get(fid, "")))
                if reason:
                    detail = reason
                    summary_hits += 1
            errors.append({"file_id": fid, "detail": detail})

        if summary_hits:
            logger.info(
                "index '%s': recovered %d of %d failure reason(s) from "
                "error_summary.json",
                index_name,
                summary_hits,
                len(missing_fids),
            )

        if (
            missing_fids
            and isinstance(embed_errors, list)
            and embed_errors
            and not matched_any_reason
            and not summary_hits
        ):
            logger.warning(
                "index '%s': %d file(s) produced no rows but none matched any of "
                "the %d entries in batch metadata['errors'], and error_summary.json "
                "did not cover them either; embedding failure reasons are unreliable "
                "for this batch",
                index_name,
                len(missing_fids),
                len(embed_errors),
            )

        failed_count = meta.get("failed_count") if hasattr(meta, "get") else None
        logger.info(
            "index '%s' batch: input=%d rows=%d failed=%s",
            index_name,
            len(ordered_paths),
            len(ids),
            failed_count,
        )

        # Relocate PNG + SCS once per file (png_paths repeats per body row).
        thumb_dir = _index_thumbnails_dir(index_name)
        scs_dir = _index_scs_dir(index_name)
        thumbnail_rel: dict[str, str] = {}
        scs_rel: dict[str, str] = {}
        for fid, row_idx in rows_by_fid.items():
            src_png = None
            if png_paths:
                for i in row_idx:
                    if i < len(png_paths) and png_paths[i]:
                        p = pathlib.Path(png_paths[i])
                        # embed_shape_batch returns paths relative to images_out_dir.
                        if not p.is_absolute():
                            p = tmp_images / p
                        if p.exists():
                            src_png = p
                            break
            if src_png is None:
                continue
            try:
                thumb_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_png), str(thumb_dir / f"{fid}.png"))
                thumbnail_rel[fid] = f"thumbnails/{fid}.png"
                # SCS sibling shares the stem minus the "_white" suffix.
                stem = src_png.stem
                base_stem = stem[: -len("_white")] if stem.endswith("_white") else stem
                scs_src = src_png.parent / f"{base_stem}.scs"
                if scs_src.exists():
                    scs_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(scs_src), str(scs_dir / f"{fid}.scs"))
                    scs_rel[fid] = f"scs/{fid}.scs"
            except Exception as exc:  # non-fatal: search still works without assets
                logger.warning("Failed to relocate thumbnail/scs for %s: %s", fid, exc)

        # Build one record per body row.
        registered_at = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        obb_sample_logged = False
        obb_dropped_logged = False
        for fid, row_idx in rows_by_fid.items():
            bodies = len(row_idx)
            row_kind = None
            if kinds and row_idx[0] < len(kinds):
                row_kind = kinds[row_idx[0]]
            kind = row_kind or ("assembly" if bodies >= 2 else "part")

            base_meta: dict[str, Any] = {
                "file_id": fid,
                "filename": filename_by_fid.get(fid, ""),
                "registered_at": registered_at,
                "kind": kind,
                "bodies": bodies,
            }
            if fid in thumbnail_rel:
                base_meta["thumbnail"] = thumbnail_rel[fid]
            if fid in scs_rel:
                base_meta["scs"] = scs_rel[fid]

            # oriented_bounding_box is file-level: hoops_ai returns it as a dict
            # keyed by the input path (== batch.ids entry), or as a per-row list.
            # It is the same for every body row of a file, so resolve it once.
            raw_id = ids[row_idx[0]]
            obb_val = None
            if isinstance(obbs, dict):
                obb_val = obbs.get(raw_id)
            elif isinstance(obbs, (list, tuple)) and row_idx[0] < len(obbs):
                obb_val = obbs[row_idx[0]]

            if obb_val is not None:
                # Print the first OBB's type + serialised size so the human
                # reviewer can judge whether to keep storing it.
                if not obb_sample_logged:
                    try:
                        _n = len(json.dumps(obb_val).encode("utf-8"))
                    except (TypeError, ValueError):
                        _n = -1
                    print(
                        f"[INDEX] obb sample: type={type(obb_val).__name__} "
                        f"serialized_bytes={_n}",
                        flush=True,
                    )
                    obb_sample_logged = True
                obb_str = _obb_meta_value(obb_val)
                if obb_str is not None:
                    base_meta["obb"] = obb_str
                elif not obb_dropped_logged:
                    logger.warning(
                        "obb dropped for index '%s' (not serialisable or > 512 bytes)",
                        index_name,
                    )
                    obb_dropped_logged = True

            for i in row_idx:
                vec = _l2_normalize(values[i].astype(np.float32).flatten())
                emb_obj = Embedding(values=vec, model=model_name, dim=int(vec.shape[0]))
                records.append(
                    VectorRecord(id=fid, embedding=emb_obj, metadata=dict(base_meta))
                )
    finally:
        shutil.rmtree(tmp_images, ignore_errors=True)

    return records, errors


def add_to_index(
    name: str,
    file_ids: list[str],
    model: Optional[str] = None,
    num_workers: Optional[int] = None,
    time_limit: Optional[int] = None,
    log_dir: Optional["pathlib.Path"] = None,
    progress_cb: Optional[Any] = None,
) -> dict[str, Any]:
    """Embed *file_ids* at body granularity and upsert one row per body.

    Re-inserting an existing file_id overwrites all of its previous body rows
    (delete-then-upsert). *model* overrides the embedder used for this batch;
    when ``None`` the model recorded in the index's ``index.json`` is used.

    Raises ``SchemaVersionError`` for legacy (schema v1) indexes, which must be
    rebuilt before body-level writes are allowed.

    *num_workers* and *time_limit* tune the SDK call; ``None`` keeps this
    module's defaults. *log_dir* redirects the SDK's ``too_heavy_files.log``.
    *progress_cb* receives live counters parsed from the SDK's progress bar.
    Existing callers that pass none of them are unaffected.

    Returns ``added``, ``updated``, ``index_count``, and per-file ``errors``.
    ``added``/``updated`` count distinct files (not body rows).
    """
    _validate_index_name(name)
    schema = _load_index_schema(name)
    effective_model = model if model is not None else schema["model"]
    if effective_model not in _EMBEDDER_MODELS:
        raise ValueError(
            f"Invalid model '{effective_model}'. Must be one of: {sorted(_EMBEDDER_MODELS)}."
        )
    lock = _get_index_lock(name)
    with lock:
        vs = _load_named_index(name)

        if schema["schema_version"] < _INDEX_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Index '{name}' uses legacy schema v{schema['schema_version']} "
                f"and must be rebuilt as schema v{_INDEX_SCHEMA_VERSION} "
                f"before adding parts."
            )

        records, errors = _embed_bodies_for_index(
            file_ids,
            name,
            effective_model,
            num_workers=num_workers,
            time_limit=time_limit,
            log_dir=log_dir,
            progress_cb=progress_cb,
        )

        existing_ids: set[str] = set(vs.get_ids())
        distinct_fids: list[str] = []
        seen: set[str] = set()
        for rec in records:
            if rec.id not in seen:
                seen.add(rec.id)
                distinct_fids.append(rec.id)

        added = 0
        updated = 0
        for fid in distinct_fids:
            if fid in existing_ids:
                # delete first to prevent FAISS duplicate entries on re-insert
                vs.delete([fid])
                updated += 1
            else:
                added += 1

        if records:
            vs.upsert(records)

        _save_named_index_atomic(name, vs)
        result = {
            "name": name,
            "added": added,
            "updated": updated,
            "index_count": len(vs.get_ids()),
            "errors": errors,
        }

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
    """Delete *part_ids* from the named index, persist, and remove their assets.

    Raises ``SchemaVersionError`` for legacy (schema v1) indexes.
    """
    _validate_index_name(name)
    schema = _load_index_schema(name)
    lock = _get_index_lock(name)
    with lock:
        vs = _load_named_index(name)
        if schema["schema_version"] < _INDEX_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Index '{name}' uses legacy schema v{schema['schema_version']} "
                f"and must be rebuilt as schema v{_INDEX_SCHEMA_VERSION} "
                f"before removing parts."
            )
        vs.delete(part_ids)
        _save_named_index_atomic(name, vs)

    # Remove thumbnails + stream caches outside the lock (non-fatal)
    thumb_dir = _index_thumbnails_dir(name)
    scs_dir = _index_scs_dir(name)
    for pid in part_ids:
        (thumb_dir / f"{pid}.png").unlink(missing_ok=True)
        (scs_dir / f"{pid}.scs").unlink(missing_ok=True)

    return {"name": name, "removed": len(part_ids), "index_count": len(vs.get_ids())}


def _load_shape_index_pickle_compat(searcher: Any, faiss_index_path: pathlib.Path) -> Any:
    """Call ``searcher.load_shape_index()`` with cross-OS pickle compatibility.

    Index sidecars pickled on Windows contain WindowsPath objects (and vice
    versa), which fail to unpickle on the other platform. Temporarily aliasing
    the path classes lets the same index file be read anywhere.
    """
    if not hasattr(pathlib, "WindowsPath") or not issubclass(pathlib.WindowsPath, pathlib.Path):
        pathlib.WindowsPath = pathlib.PurePosixPath  # type: ignore[attr-defined]
        return searcher.load_shape_index(path=str(faiss_index_path))
    _orig = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc]
        return searcher.load_shape_index(path=str(faiss_index_path))
    finally:
        pathlib.PosixPath = _orig


def _evict_named_searchers(name: str) -> None:
    """Drop every cached CADSearch instance belonging to *name*."""
    path_key = str(_index_faiss_path(name))
    for stale in [k for k in _named_searchers if k[0] == path_key]:
        _named_searchers.pop(stale, None)


def _get_named_searcher(name: str, model: str) -> Any:
    """Return a CADSearch instance with the named index loaded, from cache.

    Creating a CADSearch and calling ``load_shape_index()`` is expensive, so
    instances are cached by (faiss path, mtime). A write to the index changes
    the mtime, which invalidates the entry automatically.

    Must be called while holding the per-index lock. The returned instance is
    safe to use *outside* the lock: ``load_shape_index()`` reads the index into
    the instance's own memory, so a concurrent ``_save_named_index_atomic()``
    replacing the file on disk cannot corrupt an already-loaded searcher.

    Raises ``KeyError`` when the index does not exist.
    """
    faiss_path = _index_faiss_path(name)
    if not faiss_path.exists():
        raise KeyError(f"Index '{name}' does not exist.")

    path_key = str(faiss_path)
    key = (path_key, faiss_path.stat().st_mtime_ns)
    cached = _named_searchers.get(key)
    if cached is not None:
        _named_searchers.move_to_end(key)
        return cached

    try:
        from hoops_ai.ml.embeddings import CADSearch
    except ImportError:  # older/newer layouts re-export it one level up
        from hoops_ai.ml import CADSearch

    searcher = CADSearch(shape_model=get_embedder(model))
    _load_shape_index_pickle_compat(searcher, faiss_path)

    # Drop the previous generation of this index before inserting the new one.
    _evict_named_searchers(name)
    _named_searchers[key] = searcher
    while len(_named_searchers) > _NAMED_SEARCHER_CACHE_MAX:
        _named_searchers.popitem(last=False)
    return searcher


def _public_metadata(meta: Any) -> dict[str, Any]:
    """Return a copy of *meta* without SDK-internal keys.

    ``CADSearch.search_by_shape()`` decorates every hit with undocumented
    underscore-prefixed fields (``_query_body_rep_count``,
    ``_query_body_rep_indices``, ...). They are implementation details that may
    change between SDK releases, so they are stripped from the public response.
    The input dict is never mutated: hits may reference the store's own
    metadata objects.
    """
    if not isinstance(meta, dict):
        return {}
    return {k: v for k, v in meta.items() if not str(k).startswith("_")}


def _flatten_body_hits(results: Any, top_k: int) -> list[dict[str, Any]]:
    """Flatten ``CADSearch.search_by_shape()`` output, matching hoops_ai_native_bridge.

    *results* is ``List[List[VectorHit]]``: one already-ranked list per unique
    query body. Body lists are concatenated in order, de-duplicated by hit id,
    and capped at *top_k*.

    Scores are deliberately **not** re-sorted. The per-body ranking produced by
    CADSearch (embedding score blended with the geometric reranker) is
    authoritative, and re-sorting on the raw score would diverge from the
    bridge implementation.
    """
    k = int(top_k) if int(top_k) > 0 else 1
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for body_hits in results or []:
        for h in body_hits or []:
            hid = str(getattr(h, "id", "") or "")
            if not hid or hid in seen:
                continue
            seen.add(hid)
            out.append(
                {
                    "id": hid,
                    "score": round(float(getattr(h, "score", 0.0) or 0.0), 6),
                    "metadata": _public_metadata(getattr(h, "metadata", None)),
                }
            )
            if len(out) >= k:
                return out
    return out


def search_index(
    name: str,
    file_id: str,
    top_k: int,
    kind: str = "any",
    include_self: bool = False,
    include_image: bool = True,
) -> dict[str, Any]:
    """Search the named index for the top-k most similar parts to *file_id*.

    Runs through ``CADSearch.search_by_shape()`` so that queries are handled at
    body granularity and benefit from the geometric reranker, matching both the
    preset search and hoops_ai_native_bridge. Results are flattened with
    :func:`_flatten_body_hits` (concatenate per-body rankings, de-duplicate by
    file_id, cap at *top_k*) without re-sorting.

    *kind* is ``"any"`` (no filter), ``"part"`` or ``"assembly"``; anything else
    raises ``ValueError``.

    ``CADSearch.search_by_shape()`` does *not* drop the query from its own
    candidates here: records are keyed by SHA-256 while the query is passed as
    an ``uploads/<sha256>_<name>`` path, so the ids never match. The self hit is
    therefore removed explicitly unless *include_self* is set, in which case it
    is promoted to the front with score 1.0 instead. Either way the result never
    exceeds *top_k*.

    Returns an empty hits list (and no image_url) when the index has no entries.
    Generates a result-grid PNG in out/ and returns its relative URL as
    ``image_url`` unless *include_image* is False, in which case ``image_url`` is
    None and no CAD is reloaded for the query thumbnail. Hits are identical
    either way: the image is a display-only by-product.
    """
    _validate_index_name(name)
    if kind not in _SEARCH_KINDS:
        raise ValueError(
            f"Invalid kind '{kind}'. Must be one of: {sorted(_SEARCH_KINDS)}."
        )
    index_model = _load_index_model(name)
    lock = _get_index_lock(name)

    # Phase 1 (locked): resolve the cached searcher only. The expensive part --
    # embedding the query and searching -- runs unlocked below so that indexing
    # a batch no longer blocks every search against the same index.
    with lock:
        vs = _load_named_index(name)
        registered_ids = set(vs.get_ids())
        if not registered_ids:
            return {"hits": [], "count": 0, "image_url": None}
        self_meta: Optional[dict[str, Any]] = None
        if include_self and file_id in registered_ids:
            for meta in vs.iter_metadata():
                if meta.get("file_id") == file_id:
                    self_meta = dict(meta)
                    break
        searcher = _get_named_searcher(name, index_model)

    query_path = find_persistent_CAD_file(file_id)

    # Phase 2 (unlocked): search_by_shape() embeds the query internally, so the
    # embedding cannot be hoisted out of this call. Ask for one extra candidate
    # because the query itself usually occupies a slot and is dropped below.
    k = int(top_k) if int(top_k) > 0 else 1
    search_kwargs: dict[str, Any] = {"top_k": k + 1}
    if kind != "any":
        search_kwargs["filters"] = {"kind": kind}
    results = searcher.search_by_shape(str(query_path), **search_kwargs)

    hit_dicts = _flatten_body_hits(results, k + 1)
    hit_dicts = [h for h in hit_dicts if h["id"] != file_id]

    if include_self and file_id in registered_ids:
        # Put the query back at the front so clients can tag the whole displayed
        # cluster. It was removed above, so this cannot produce a duplicate.
        hit_dicts.insert(
            0,
            {
                "id": file_id,
                "score": 1.0,
                "metadata": _public_metadata(self_meta) or {"file_id": file_id},
            },
        )
    hit_dicts = hit_dicts[:k]

    # Phase 3: thumbnail + grid generation (outside lock, non-fatal). Both are
    # display-only, so include_image=False skips them entirely -- including the
    # CAD reload behind _generate_part_thumbnail(). Registered queries already
    # have a thumbnail, so skip the extra CAD load in that case too.
    image_url: Optional[str] = None
    if include_image and hit_dicts:
        if _resolve_index_asset(name, file_id, "thumbnail") is None:
            _generate_part_thumbnail(file_id, name)
        image_url = _build_search_grid_image(file_id, hit_dicts, name)

    return {"hits": hit_dicts, "count": len(hit_dicts), "image_url": image_url}


# ---------------------------------------------------------------------------
# Assembly-to-assembly search
# ---------------------------------------------------------------------------


def _assembly_search_jobs() -> int:
    """Thread-pool size for AssemblyMatcher stage-2 scoring.

    Read from HOOPS_AI_ASSEMBLY_SEARCH_JOBS on every call so the value can be
    tuned without restarting. Invalid or non-positive values fall back to the
    default rather than failing a search.
    """
    raw = os.getenv("HOOPS_AI_ASSEMBLY_SEARCH_JOBS", "").strip()
    if not raw:
        return _ASSEMBLY_SEARCH_JOBS_DEFAULT
    try:
        jobs = int(raw)
    except ValueError:
        logger.warning(
            "Invalid HOOPS_AI_ASSEMBLY_SEARCH_JOBS=%r; using %d.",
            raw, _ASSEMBLY_SEARCH_JOBS_DEFAULT,
        )
        return _ASSEMBLY_SEARCH_JOBS_DEFAULT
    if jobs < 1:
        logger.warning(
            "HOOPS_AI_ASSEMBLY_SEARCH_JOBS=%d is out of range; using %d.",
            jobs, _ASSEMBLY_SEARCH_JOBS_DEFAULT,
        )
        return _ASSEMBLY_SEARCH_JOBS_DEFAULT
    return jobs


def _evict_assembly_matchers(name: str) -> None:
    """Drop every cached AssemblyMatcher belonging to *name*."""
    path_key = str(_index_faiss_path(name))
    for stale in [k for k in _assembly_matchers if k[0] == path_key]:
        _assembly_matchers.pop(stale, None)


def _get_assembly_matcher(name: str, model: str) -> Any:
    """Return an AssemblyMatcher for the named index, from cache.

    Building one is far more expensive than a plain searcher: on top of loading
    the index it runs ``build_part_rarity_weights()``, a FAISS k-means over
    every body vector in the corpus. The construction time is logged so the
    real cost stays visible.

    Must be called while holding the per-index lock. The returned instance is
    safe to use *outside* the lock for the same reason as
    :func:`_get_named_searcher`: it holds its own copy of the corpus, so a
    concurrent ``_save_named_index_atomic()`` cannot disturb an in-flight
    search. The next request sees a new mtime and rebuilds.

    Raises ``KeyError`` when the index does not exist.
    """
    faiss_path = _index_faiss_path(name)
    if not faiss_path.exists():
        raise KeyError(f"Index '{name}' does not exist.")

    key = (str(faiss_path), faiss_path.stat().st_mtime_ns)
    cached = _assembly_matchers.get(key)
    if cached is not None:
        _assembly_matchers.move_to_end(key)
        return cached

    try:
        from hoops_ai.ml.embeddings import CADSearch
    except ImportError:  # older/newer layouts re-export it one level up
        from hoops_ai.ml import CADSearch

    from vendor.assembly_matcher import AssemblyMatcher

    started = time.perf_counter()
    searcher = CADSearch(shape_model=get_embedder(model))
    batch = _load_shape_index_pickle_compat(searcher, faiss_path)
    matcher = AssemblyMatcher(
        searcher=searcher,
        embedding_batch=batch,
        min_bodies_for_assembly=_MIN_BODIES_FOR_ASSEMBLY,
    )
    matcher.build_part_rarity_weights()
    logger.info(
        "[ASSEMBLY] built matcher for index '%s' in %.2fs",
        name, time.perf_counter() - started,
    )

    _evict_assembly_matchers(name)
    _assembly_matchers[key] = matcher
    while len(_assembly_matchers) > _ASSEMBLY_MATCHER_CACHE_MAX:
        _assembly_matchers.popitem(last=False)
    return matcher


def _assembly_metadata_by_id(vs: Any) -> dict[str, dict[str, Any]]:
    """Map file_id -> stored metadata in a single pass over the index.

    ``iter_metadata()`` yields one entry per body row, so the first entry per
    file wins; per-file fields (filename, kind, bodies, ...) are identical
    across the rows of a file.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for meta in vs.iter_metadata():
        if not isinstance(meta, dict):
            continue
        fid = meta.get("file_id")
        if fid and fid not in by_id:
            by_id[str(fid)] = dict(meta)
    return by_id


def _assembly_hit(result: Any, meta_by_id: dict[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Convert one AssemblyMatcher result dict into a response hit.

    The matcher's key names are kept verbatim on its side and renamed here:
    ``matched`` -> matched_parts, ``n_parts`` -> candidate_parts (bodies in the
    candidate) and ``M`` -> query_parts (bodies in the query). Returns None for
    anything unusable so one odd row cannot fail the whole search.
    """
    if not isinstance(result, dict):
        return None
    hid = str(result.get("assembly") or "")
    if not hid:
        return None

    def _num(key: str) -> float:
        try:
            return float(result.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _count(key: str) -> int:
        try:
            return int(result.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "id": hid,
        "score": round(_num("score"), 6),
        "geom_score": round(_num("geom_score"), 6),
        "coverage": round(_num("coverage"), 6),
        "matched_parts": _count("matched"),
        "candidate_parts": _count("n_parts"),
        "query_parts": _count("M"),
        "metadata": _public_metadata(meta_by_id.get(hid)) or {"file_id": hid},
    }


def _assembly_self_hit(file_id: str, meta: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Build the synthetic self hit inserted when include_self is set.

    An assembly is a perfect match for itself, so the scores are 1.0 and every
    body counts as matched. The values are synthetic, never produced by the
    matcher, because the query is excluded from scoring (see
    :func:`search_assembly_index`).
    """
    public = _public_metadata(meta)
    try:
        bodies = int(public.get("bodies") or 0)
    except (TypeError, ValueError):
        bodies = 0
    return {
        "id": file_id,
        "score": 1.0,
        "geom_score": 1.0,
        "coverage": 1.0,
        "matched_parts": bodies,
        "candidate_parts": bodies,
        "query_parts": bodies,
        "metadata": public or {"file_id": file_id},
    }


def search_assembly_index(
    name: str,
    file_id: str,
    top_k: int,
    candidate_k: int = 30,
    sim_thresh: float = 0.80,
    bop_weight: float = 0.30,
    coverage_mode: str = "symmetric",
    use_idf: bool = True,
    include_self: bool = False,
    include_image: bool = True,
) -> dict[str, Any]:
    """Search the named index for assemblies similar to the assembly *file_id*.

    Runs the vendored ``AssemblyMatcher`` (see vendor/assembly_matcher.py) with
    exactly the arguments hoops_ai_native_bridge uses, so both projects rank the
    same corpus identically: a per-body Hungarian matching, blended with a
    bag-of-parts composition score, over an index-backed candidate shortlist.

    *coverage_mode* is ``"symmetric"`` (matched mass over the larger side),
    ``"containment"`` (over the query, i.e. "which assemblies contain this
    one?") or ``"jaccard"``. Out-of-range parameters raise ``ValueError``.

    Like :func:`search_index`, the query cannot be dropped by the SDK itself --
    ``AssemblyMatcher.search()`` discards candidates equal to the query *path*
    while records are keyed by SHA-256 -- so the self hit is removed explicitly
    here and only re-inserted, at the front, when *include_self* is set. One
    extra result is requested to compensate, and the total never exceeds
    *top_k*.

    A query that is already indexed is passed to the matcher by its record id
    (the file_id) rather than by path, so ``reuse_index_vectors=True`` reuses
    the stored per-body vectors instead of re-embedding the CAD file.

    Returns ``{"hits": [...], "count": n, "image_url": str | None}``; the hits
    carry the matcher's diagnostics (geom_score, coverage, matched_parts,
    candidate_parts, query_parts) alongside the stored metadata. As in
    :func:`search_index`, ``include_image`` only controls the display-only
    contact sheet and never affects the hits.

    Raises ``KeyError`` when the index does not exist.
    """
    _validate_index_name(name)
    if coverage_mode not in _ASSEMBLY_COVERAGE_MODES:
        raise ValueError(
            f"Invalid coverage_mode '{coverage_mode}'. "
            f"Must be one of: {sorted(_ASSEMBLY_COVERAGE_MODES)}."
        )
    k = int(top_k) if int(top_k) > 0 else 1
    candidate_k = int(candidate_k)
    if not 1 <= candidate_k <= _ASSEMBLY_CANDIDATE_K_MAX:
        raise ValueError(
            f"Invalid candidate_k {candidate_k}. Must be between 1 and "
            f"{_ASSEMBLY_CANDIDATE_K_MAX}."
        )
    sim_thresh = float(sim_thresh)
    if not 0.0 <= sim_thresh <= 1.0:
        raise ValueError(f"Invalid sim_thresh {sim_thresh}. Must be between 0.0 and 1.0.")
    bop_weight = float(bop_weight)
    if not 0.0 <= bop_weight <= 1.0:
        raise ValueError(f"Invalid bop_weight {bop_weight}. Must be between 0.0 and 1.0.")

    schema = _load_index_schema(name)
    if schema["schema_version"] < _INDEX_SCHEMA_VERSION:
        # A legacy index stores one averaged vector per file, so every entry
        # looks like a single-body part: assemblies_only would discard the whole
        # corpus and the search would silently return nothing useful.
        raise SchemaVersionError(
            f"Index '{name}' uses legacy schema v{schema['schema_version']} and "
            f"has no per-body vectors. Rebuild it as schema v{_INDEX_SCHEMA_VERSION} "
            f"to use assembly search."
        )
    index_model = schema["model"]
    lock = _get_index_lock(name)

    # Phase 1 (locked): resolve the cached matcher and read the metadata map.
    # Building the matcher is the expensive part; the search itself runs
    # unlocked below so that indexing does not block concurrent searches.
    with lock:
        vs = _load_named_index(name)
        registered_ids = set(vs.get_ids())
        if not registered_ids:
            return {"hits": [], "count": 0, "image_url": None}
        meta_by_id = _assembly_metadata_by_id(vs)
        matcher = _get_assembly_matcher(name, index_model)

    if file_id in registered_ids:
        # AssemblyMatcher keys its per-file rows by record id, which here is the
        # SHA-256 file_id. Passing the id lets reuse_index_vectors=True hit the
        # stored per-body vectors instead of re-reading and re-embedding the CAD
        # file (~1.16s -> ~0.16s warm). It also makes the matcher's own
        # candidates.discard(query_path) work, which it cannot do with a path
        # because our records are keyed by hash rather than by location.
        matcher_query = file_id
    else:
        # Not indexed: hand over the real path so the matcher falls back to
        # embedder.embed_shape().
        matcher_query = str(find_persistent_CAD_file(file_id))

    # Phase 2 (unlocked): the fixed arguments mirror hoops_ai_native_bridge.
    results = matcher.search(
        query_path=matcher_query,
        top_k=k + 1,
        candidate_k=candidate_k,
        sim_thresh=sim_thresh,
        method="hungarian",
        use_idf=bool(use_idf),
        candidate_mode="search",
        assemblies_only=True,
        reuse_index_vectors=True,
        coverage_mode=coverage_mode,
        bop_weight=bop_weight,
        n_jobs=_assembly_search_jobs(),
    )

    hit_dicts: list[dict[str, Any]] = []
    for result in results or []:
        hit = _assembly_hit(result, meta_by_id)
        if hit is not None and hit["id"] != file_id:
            hit_dicts.append(hit)

    if include_self and file_id in registered_ids:
        # The query was filtered out above, so this cannot duplicate a hit.
        hit_dicts.insert(0, _assembly_self_hit(file_id, meta_by_id.get(file_id)))
    hit_dicts = hit_dicts[:k]

    # Phase 3: display-only contact sheet, identical policy to search_index().
    image_url: Optional[str] = None
    if include_image and hit_dicts:
        if _resolve_index_asset(name, file_id, "thumbnail") is None:
            _generate_part_thumbnail(file_id, name)
        image_url = _build_search_grid_image(file_id, hit_dicts, name)

    return {"hits": hit_dicts, "count": len(hit_dicts), "image_url": image_url}


def get_index_stats(name: str) -> dict[str, Any]:
    """Return body-level statistics for the named index.

    Keys: ``name, files, bodies, assemblies, single_part, dim, model,
    schema_version, last_modified``. Works for both schema versions (a legacy
    index simply reports one body per file).

    Raises ``KeyError`` when the index does not exist.
    """
    _validate_index_name(name)
    schema = _load_index_schema(name)
    lock = _get_index_lock(name)
    with lock:
        vs = _load_named_index(name)
        stats = _compute_faiss_stats(vs)
        dim = getattr(vs, "dim", None)

    last_modified: Optional[str] = None
    faiss_file = _index_faiss_path(name)
    if faiss_file.exists():
        last_modified = (
            datetime.datetime.fromtimestamp(
                faiss_file.stat().st_mtime, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        )

    return {
        "name": name,
        "files": stats["files"],
        "bodies": stats["bodies"],
        "assemblies": stats["assemblies"],
        "single_part": stats["single_part"],
        "dim": int(dim) if dim is not None else None,
        "model": schema["model"],
        "schema_version": schema["schema_version"],
        "last_modified": last_modified,
    }


def list_index_parts(
    name: str,
    offset: int = 0,
    limit: int = 100,
    kind: Optional[str] = None,
) -> dict[str, Any]:
    """Return a paginated, file-level listing of the named index.

    ``iter_metadata()`` is walked exactly once and collapsed to one item per
    file_id. *limit* is clamped to ``[1, 2000]`` (default 100) and *offset* to
    ``>= 0``. *kind* (``"part"``/``"assembly"``) filters when supplied.

    Each item: ``id, filename, kind, bodies, registered_at, has_thumbnail,
    has_scs`` (the router turns the boolean flags into absolute URLs).

    Raises ``KeyError`` when the index does not exist.
    """
    _validate_index_name(name)
    limit = max(1, min(int(limit), 2000))
    offset = max(0, int(offset))

    lock = _get_index_lock(name)
    with lock:
        vs = _load_named_index(name)
        order: list[str] = []
        by_fid: dict[str, dict[str, Any]] = {}
        for raw in vs.iter_metadata():
            md = raw if isinstance(raw, dict) else (getattr(raw, "metadata", None) or {})
            fid = md.get("file_id")
            if fid is None:
                continue
            if fid not in by_fid:
                order.append(fid)
                by_fid[fid] = {
                    "id": fid,
                    "filename": md.get("filename"),
                    "kind": md.get("kind") or "part",
                    "bodies": int(md.get("bodies") or 1),
                    "registered_at": md.get("registered_at"),
                    "thumbnail": md.get("thumbnail"),
                    "scs": md.get("scs"),
                }

    items = [by_fid[f] for f in order]
    if kind:
        items = [it for it in items if it["kind"] == kind]
    total = len(items)
    page = items[offset : offset + limit]

    # Resolve asset existence (path-traversal safe) and drop the raw rel paths.
    result_items: list[dict[str, Any]] = []
    for it in page:
        has_thumbnail = (
            it.get("thumbnail") is not None
            and _resolve_index_asset(name, it["id"], "thumbnail") is not None
        )
        has_scs = (
            it.get("scs") is not None
            and _resolve_index_asset(name, it["id"], "scs") is not None
        )
        result_items.append(
            {
                "id": it["id"],
                "filename": it["filename"],
                "kind": it["kind"],
                "bodies": it["bodies"],
                "registered_at": it["registered_at"],
                "has_thumbnail": has_thumbnail,
                "has_scs": has_scs,
            }
        )

    return {"total": total, "offset": offset, "limit": limit, "items": result_items}


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
        # Drop cached searchers so a deleted index stops holding memory.
        _evict_named_searchers(name)
        _evict_assembly_matchers(name)

    # Remove the entire index directory (faiss + meta + thumbnails)
    index_dir = INDEXES_DIR / name
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)

    return {"name": name, "deleted": True}


def _index_thumbnails_dir(name: str) -> pathlib.Path:
    """Return the per-index thumbnail image directory path."""
    return INDEXES_DIR / name / "thumbnails"


def _index_model_sidecar_path(name: str) -> pathlib.Path:
    """Return the legacy per-index model sidecar JSON path (indexes/{name}/model.json).

    Only read for backward compatibility with schema-v1 indexes. New indexes
    record their model + schema_version in ``index.json`` (see _index_json_path).
    """
    return INDEXES_DIR / name / "model.json"


def _index_json_path(name: str) -> pathlib.Path:
    """Return the per-index metadata JSON path (indexes/{name}/index.json)."""
    return INDEXES_DIR / name / "index.json"


def _index_scs_dir(name: str) -> pathlib.Path:
    """Return the per-index SCS stream-cache directory path."""
    return INDEXES_DIR / name / "scs"


def _load_index_schema(name: str) -> dict[str, Any]:
    """Return ``{"model": <key>, "schema_version": <int>}`` for the named index.

    Fallback order:
      1. ``index.json`` present  -> use it (body-level, schema 2 by default)
      2. only ``model.json``     -> ``{"model": <that>, "schema_version": 1}``
      3. neither present         -> ``{"model": "legacy", "schema_version": 1}``
    """
    import json

    json_path = _index_json_path(name)
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return {
                "model": data.get("model", _EMBEDDER_MODEL_LEGACY),
                "schema_version": int(data.get("schema_version", _INDEX_SCHEMA_VERSION)),
            }
        except Exception:
            # Corrupt index.json: treat as body-level but unknown model.
            return {"model": _EMBEDDER_MODEL_LEGACY, "schema_version": _INDEX_SCHEMA_VERSION}

    model_path = _index_model_sidecar_path(name)
    if model_path.exists():
        try:
            with open(model_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return {"model": data.get("model", _EMBEDDER_MODEL_LEGACY), "schema_version": 1}
        except Exception:
            return {"model": _EMBEDDER_MODEL_LEGACY, "schema_version": 1}

    return {"model": _EMBEDDER_MODEL_LEGACY, "schema_version": 1}


def _save_index_schema(name: str, model: str, schema_version: int = _INDEX_SCHEMA_VERSION) -> None:
    """Persist ``index.json`` for the named index."""
    import json

    path = _index_json_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"model": model, "schema_version": schema_version}, fh)


def _load_index_model(name: str) -> str:
    """Return the embedding model key recorded for the named index."""
    return _load_index_schema(name)["model"]


def _obb_meta_value(value: Any) -> Optional[str]:
    """Return a JSON string for *value* suitable for storing in record metadata.

    Returns ``None`` (drop the key) when the value is not JSON-serialisable or
    when the serialised form exceeds 512 bytes, keeping index.meta compact.
    """
    import json

    try:
        serialized = json.dumps(value)
    except (TypeError, ValueError):
        return None
    if len(serialized.encode("utf-8")) > 512:
        return None
    return serialized


def _resolve_index_asset(name: str, part_id: str, asset: str) -> Optional[pathlib.Path]:
    """Resolve a per-part thumbnail/scs file, guarding against path traversal.

    *asset* is ``"thumbnail"`` (thumbnails/<id>.png) or ``"scs"`` (scs/<id>.scs).
    Returns the file Path only when it exists AND resolves inside indexes/<name>/;
    otherwise returns ``None``.
    """
    _validate_index_name(name)
    index_dir = (INDEXES_DIR / name).resolve()
    if asset == "thumbnail":
        candidate = (INDEXES_DIR / name / "thumbnails" / f"{part_id}.png").resolve()
    elif asset == "scs":
        candidate = (INDEXES_DIR / name / "scs" / f"{part_id}.scs").resolve()
    else:
        return None
    try:
        candidate.relative_to(index_dir)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _compute_faiss_stats(vs: Any) -> dict[str, int]:
    """Derive file/body/assembly/single-part counts from a FaissVectorStore.

    Reads the underlying IndexIDMap directly (verified against hoops_ai 1.1.0).
    """
    import faiss
    from collections import Counter

    index = vs._index
    idm = faiss.vector_to_array(index.id_map)
    bodies = int(index.ntotal)
    counts = Counter(int(x) for x in idm.tolist())
    files = len(vs._id_to_int)
    assemblies = sum(1 for v in counts.values() if v >= 2)
    single = sum(1 for v in counts.values() if v == 1)
    return {"files": files, "bodies": bodies, "assemblies": assemblies, "single_part": single}


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

        # Remove the SCS file 窶・we only need the PNG for thumbnails
        for p_str in (scs_result, str(tmp_scs)):
            if p_str:
                p = pathlib.Path(p_str)
                if p.exists():
                    p.unlink(missing_ok=True)

        return dest_png if dest_png.exists() else None
    except Exception:
        return None


# A contact sheet stops being readable long before it stops being expensive: each
# tile costs one PNG decode plus one matplotlib subplot, so the image grew
# linearly with top_k (top_k=300 produced a 1200x22840 px, 4.2 MB strip and
# accounted for 39 s of a 40 s request). Cap the sheet so its cost is fixed and
# small regardless of top_k -- the JSON response still carries every hit.
_GRID_MAX_TILES = 12


def _grid_tiles(hits: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Return the hits to draw on the contact sheet plus its caption.

    The caption always states the truncation ("top 12 of 300") so a viewer can
    tell that the image is a preview of a longer ranking.
    """
    shown = hits[:_GRID_MAX_TILES]
    return shown, f"top {len(shown)} of {len(hits)}"


def _build_search_grid_image(
    query_file_id: str,
    hits: list[dict[str, Any]],
    index_name: str,
) -> Optional[str]:
    """Generate a matplotlib result-grid PNG (query + top hits) and save to out/.

    Returns a ``/out/{uuid}.png`` relative URL, or None on failure.
    Thumbnails are looked up from the index thumbnails directory.

    Only the first :data:`_GRID_MAX_TILES` hits are drawn; the caption states how
    many of the total are shown. The JSON response always carries the full
    ranking, so nothing is lost.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        thumb_dir = _index_thumbnails_dir(index_name)
        shown, caption = _grid_tiles(hits)
        n_total = len(shown) + 1  # query cell + one cell per drawn hit
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

        # Query cell 窶・look up embedding cache for filename
        query_name = "query"
        cached = _embedding_memory_cache.get(f"hoops_embeddings_model__{query_file_id}")
        if cached:
            query_name = cached.get("filename", query_file_id[:12])
        query_thumb = thumb_dir / f"{query_file_id}.png"
        _show_cell(axes[0], query_thumb, f"Query\n{pathlib.Path(query_name).name}")

        # Hit cells
        for i, hit in enumerate(shown):
            if i + 1 >= len(axes):
                break
            hit_thumb = thumb_dir / f"{hit['id']}.png"
            meta = hit.get("metadata") or {}
            hit_name = pathlib.Path(meta.get("filename", hit["id"][:12])).name
            _show_cell(axes[i + 1], hit_thumb, hit_name, f"score: {hit['score']:.4f}")

        # Hide surplus axes
        for i in range(n_total, len(axes)):
            axes[i].axis("off")

        fig.text(0.5, 0.005, caption, ha="center", fontsize=8)
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


class UploadSizeLimitError(RuntimeError):
    """Raised when a single uploaded file exceeds the configured size limit."""


class SchemaVersionError(RuntimeError):
    """Raised when a legacy (schema v1) index cannot serve the request.

    Mapped to HTTP 409 by the router. Legacy indexes store one averaged vector
    per file, so body-level writes and assembly search both require a rebuild.
    """


def get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        msg = f"[CONFIG] Required environment variable '{name}' is not set. Set it in {ENV_FILE_PATH.name} or the environment."
        logger.error(msg)
        print(msg, flush=True)
        raise EnvConfigError(msg)
    return value


def demo_features_enabled() -> bool:
    """Return True when demo-only endpoints (named similarity index management,
    embedding-model switching, Shape Space Map generation/query, MFR and Part
    Classification dataset browsing, context-layer prediction) should be served.

    Controlled by HOOPS_AI_ENABLE_DEMO_FEATURES in .env / the environment.
    Defaults to disabled so a freshly cloned public deployment never exposes
    these routes unless explicitly opted in.
    """
    return os.environ.get("HOOPS_AI_ENABLE_DEMO_FEATURES", "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


def require_demo_enabled() -> None:
    """FastAPI dependency that hides demo-only endpoints unless enabled.

    Raises a 404 (rather than 403) so a disabled endpoint is indistinguishable
    from one that does not exist on this deployment.
    """
    from fastapi import HTTPException

    if not demo_features_enabled():
        raise HTTPException(status_code=404, detail="Not Found")


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
    """Return the CAD store: the single permanent home of every uploaded file.

    Index records carry only the SHA-256 file_id, so this directory holds the
    only copy of the CAD payload behind every registered index. It ranks with
    ``indexes/`` as permanent data and is never cleared by the server.
    ``CAD_UPLOAD_DIR`` is merely the default location; HOOPS_AI_CAD_SHARED_DIR
    overrides it. Every reader and writer must go through this function so the
    configured location and the location actually used can never diverge.
    """
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
    searcher = get_cad_searcher_for(preset)
    return _load_shape_index_pickle_compat(searcher, paths["faiss_path"])


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

    Always succeeds 窶・returns ``status: "not_loaded"`` when neither the searcher
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

    * ``'legacy'`` 窶・``HOOPS_AI_EMBEDDINGS_MODEL_NAME`` (1M model), registered
      as ``"hoops_embeddings_model"``.
    * ``'signal'`` 窶・``HOOPS_AI_EMBEDDINGS_MODEL_NAME_SIGNAL`` (SIGNAL model),
      registered as ``"hoops_embeddings_signal"``.

    Safe to call alongside (or after) ``create_cad_searcher_for()`` 窶・the model
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
            pass  # corrupted cache 窶・fall through to recompute

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
    """Compute an Nﾃ湧 cosine similarity matrix for the given file_ids.

    All embedding vectors are L2-normalised, so cosine similarity equals their
    dot product.  Diagonal entries are forced to exactly ``1.0``.

    *model* selects the embedder: ``'legacy'`` (1M model) or ``'signal'``
    (SIGNAL model).

    Returns a dict with keys:
      ``count``, ``model_name``, ``files``, ``matrix`` (Nﾃ湧 list of lists),
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

    Given an ``Nﾃ湧`` symmetric distance matrix, returns ``(coords, stress)``
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

    # eigh returns eigenvalues in ascending order 竊・take the largest 3
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

    A thin wrapper around the conversion logic in ``create_CAD_viewer`` 窶・it does
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
      1. Compute the Nﾃ湧 cosine-similarity matrix via ``compare_embeddings``.
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
    ``Nﾃ・`` coordinate matrix ``coords`` (mean-centred) and the ``Nﾃ湧`` distance
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

    # Row/column means of Dﾂｲ (symmetric, so equal)
    row_means = d_sq.mean(axis=0)      # (N,) 窶・(1/N) ﾎ｣_i Dﾂｲ_ij for each j
    grand_mean = float(d_sq.mean())
    q_mean = float(d_q_sq.mean())

    # Out-of-sample centering: b_j = <x_query, x_j> approximation
    b = -0.5 * (d_q_sq - q_mean - row_means + grand_mean)  # (N,)

    # Solve coords @ x_query 竕・b  (least-squares, handles rank-deficiency)
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
      ``KeyError``    窶・map *map_id* does not exist.
      ``RuntimeError``窶・embedding computation fails for the query part.
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

    # Compute query embedding (raises on failure 窶・let caller handle)
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

    # Build (N+1)ﾃ・N+1) extended similarity matrix
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
    scs_filename = None
    try:
        viewer_result = create_CAD_viewer(cad_file_path, session_id)
        viewer_url = viewer_result.get("viewer_url")
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


def _env_int(name: str, default: int, minimum: int = 1, maximum: Optional[int] = None) -> int:
    """Read a positive integer setting from the environment.

    Read on every call so limits can be tuned without a restart. A missing,
    malformed or out-of-range value falls back to *default* with a warning
    rather than failing the request: an operator typo must not take the server
    down, but it must not be silent either.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d.", name, raw, default)
        return default
    if value < minimum or (maximum is not None and value > maximum):
        logger.warning(
            "%s=%d is out of range [%s, %s]; using default %d.",
            name, value, minimum, maximum if maximum is not None else "inf", default,
        )
        return default
    return value


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


_UPLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_UPLOAD_BYTES_DEFAULT = 4 * 1024 * 1024 * 1024  # 4 GB
_FILE_ID_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_EXISTS_MAX_IDS_DEFAULT = 1000


def _max_upload_bytes() -> int:
    """Per-file upload cap, from HOOPS_AI_MAX_UPLOAD_BYTES."""
    return _env_int("HOOPS_AI_MAX_UPLOAD_BYTES", _MAX_UPLOAD_BYTES_DEFAULT, minimum=1)


def exists_max_ids() -> int:
    """Maximum number of ids accepted by a single existence check."""
    return _env_int("HOOPS_AI_EXISTS_MAX_IDS", _EXISTS_MAX_IDS_DEFAULT, minimum=1)


def upload_CAD_file_persistent(upload_file: UploadFile) -> tuple[str, pathlib.Path, bool]:
    """Upload a CAD file and store it persistently using SHA-256 content hash as file_id.

    Returns (file_id, path, already_existed). Uploading the same file twice is
    idempotent.

    The body is streamed to disk in chunks and hashed as it goes, so a large
    assembly never has to fit in memory and is read only once. The bytes land in
    a temporary ``.part`` file that is renamed into place only after the whole
    stream has been consumed, so an aborted upload can never leave a truncated
    file that would later be served under a hash it does not match.

    Raises ``UploadSizeLimitError`` when the body exceeds
    HOOPS_AI_MAX_UPLOAD_BYTES, and ``RuntimeError`` when the filename is missing.
    """
    if not upload_file.filename:
        raise RuntimeError("Uploaded CAD file must have a filename.")

    safe_filename = pathlib.PurePath(upload_file.filename).name
    CAD_shared_dir = get_CAD_shared_dir()
    CAD_shared_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = _max_upload_bytes()
    digest = hashlib.sha256()
    total = 0
    tmp_path = CAD_shared_dir / f".upload_{uuid.uuid4().hex}.part"
    try:
        with tmp_path.open("wb") as handle:
            while True:
                chunk = upload_file.file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise UploadSizeLimitError(
                        f"Uploaded file '{safe_filename}' exceeds the "
                        f"{max_bytes // (1024 * 1024)} MB per-file limit."
                    )
                digest.update(chunk)
                handle.write(chunk)

        file_id = digest.hexdigest()
        dest_path = CAD_shared_dir / f"{file_id}_{safe_filename}"
        existed = dest_path.exists()
        if not existed:
            os.replace(tmp_path, dest_path)
        return file_id, dest_path, existed
    finally:
        # No-op once the rename succeeded; cleans up on error or duplicate.
        tmp_path.unlink(missing_ok=True)


def known_persistent_CAD_file_ids() -> set[str]:
    """Return the set of file_ids currently held in the CAD store.

    Scans the directory once. Callers checking many ids must use this rather
    than calling find_persistent_CAD_file() in a loop, which globs per id.
    """
    CAD_shared_dir = get_CAD_shared_dir()
    if not CAD_shared_dir.exists():
        return set()
    known: set[str] = set()
    for path in CAD_shared_dir.iterdir():
        if not path.is_file():
            continue
        head, sep, _ = path.name.partition("_")
        if sep and _FILE_ID_RE.match(head):
            known.add(head)
    return known


def check_persistent_CAD_files(file_ids: list[str]) -> dict[str, list[str]]:
    """Split *file_ids* into known / unknown / invalid, preserving input order.

    Lets a client skip uploading files the server already has. Malformed ids are
    reported separately instead of being lumped in with "unknown", so a client
    that hashes incorrectly sees a distinct signal rather than silently
    re-uploading everything on every run.
    """
    known_ids = known_persistent_CAD_file_ids()
    known: list[str] = []
    unknown: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for raw in file_ids:
        candidate = str(raw).strip()
        if candidate in seen:
            continue
        seen.add(candidate)
        if not _FILE_ID_RE.match(candidate.lower()) or candidate != candidate.lower():
            invalid.append(candidate)
        elif candidate in known_ids:
            known.append(candidate)
        else:
            unknown.append(candidate)
    return {"known": known, "unknown": unknown, "invalid": invalid}


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


# --- Runtime directory lifecycle --------------------------------------------
#
# The CAD store (``uploads/``, or HOOPS_AI_CAD_SHARED_DIR when set) holds the
# only copy of every file registered in an index: index records store the
# SHA-256 file_id, never the bytes. It is therefore a permanent data directory
# on the same footing as ``indexes/`` and is never swept here.
#
# ``out/`` is the opposite: throw-away viewer artefacts (SCS streams, result
# images, shape maps) served over ``/out/`` and referenced only by URLs handed
# to a client moments earlier. It grows without bound, so it gets a TTL sweep.
# ``embeddings_cache/`` is a cache, but recomputing one entry costs a full CAD
# load, so it is kept indefinitely unless an operator opts into a TTL.
#
# Nothing is wiped at startup any more. The previous rmtree destroyed the CAD
# store -- taking the payload of every registered index with it -- and, when a
# second instance was started on another port, deleted the working files of the
# instance already running.

_OUT_TTL_HOURS_DEFAULT = 24
_EMBEDDINGS_CACHE_TTL_DAYS_DEFAULT = 0  # 0 = keep forever


def viewer_output_ttl_seconds() -> int:
    """TTL for ``out/``, from HOOPS_AI_OUT_TTL_HOURS. 0 disables the sweep."""
    return _env_int("HOOPS_AI_OUT_TTL_HOURS", _OUT_TTL_HOURS_DEFAULT, minimum=0) * 3600


def embeddings_cache_ttl_seconds() -> int:
    """TTL for ``embeddings_cache/``, from HOOPS_AI_EMBEDDINGS_CACHE_TTL_DAYS.

    0 (the default) disables the sweep.
    """
    return _env_int(
        "HOOPS_AI_EMBEDDINGS_CACHE_TTL_DAYS",
        _EMBEDDINGS_CACHE_TTL_DAYS_DEFAULT,
        minimum=0,
    ) * 86400


def sweep_directory(
    directory: pathlib.Path,
    max_age_seconds: int,
    now: Optional[float] = None,
) -> dict[str, int]:
    """Delete files under *directory* whose mtime is older than *max_age_seconds*.

    Returns ``{"removed": n, "kept": n, "failed": n}``. A *max_age_seconds* of 0
    or less disables the sweep and reports zeroes. Individual failures are
    logged and counted but never raised: a file another request is still
    streaming must not be able to take startup down.
    """
    stats = {"removed": 0, "kept": 0, "failed": 0}
    if max_age_seconds <= 0 or not directory.exists():
        return stats

    cutoff = (time.time() if now is None else now) - max_age_seconds
    for path in directory.rglob("*"):
        try:
            if not path.is_file():
                continue
            if path.stat().st_mtime >= cutoff:
                stats["kept"] += 1
                continue
            path.unlink()
            stats["removed"] += 1
        except OSError as exc:
            stats["failed"] += 1
            logger.warning("Failed to sweep %s: %s", path, exc)

    # Drop directories the sweep emptied, deepest first. Best effort only.
    for path in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except OSError:
            pass
    return stats


def ensure_runtime_dirs() -> None:
    """Create the directories the server writes to, without deleting anything."""
    for folder in (
        get_CAD_shared_dir(),
        CAD_VIEWER_OUTPUT_DIR,
        EMBEDDINGS_CACHE_DIR,
        INDEXES_DIR,
        JOBS_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)


def run_startup_maintenance() -> dict[str, dict[str, int]]:
    """Create runtime directories and TTL-sweep the transient ones."""
    ensure_runtime_dirs()
    results = {
        "out": sweep_directory(CAD_VIEWER_OUTPUT_DIR, viewer_output_ttl_seconds()),
        "embeddings_cache": sweep_directory(
            EMBEDDINGS_CACHE_DIR, embeddings_cache_ttl_seconds()
        ),
    }
    for label, stats in results.items():
        if stats["removed"] or stats["failed"]:
            logger.info(
                "[GC] %s: removed %d, kept %d, failed %d",
                label, stats["removed"], stats["kept"], stats["failed"],
            )
    get_index_job_store().recover_interrupted()
    get_index_job_store().collect_garbage()
    return results


# --- Background registration jobs -------------------------------------------

JOB_KIND_INDEX_ADD = "index_add"

_JOB_MAX_CONCURRENCY_DEFAULT = 1
_JOB_TTL_DAYS_DEFAULT = 7
_JOB_MAX_RECORDS_DEFAULT = 1000

_index_job_store: Any = None
_index_job_store_lock = threading.Lock()


def job_max_concurrency() -> int:
    """How many jobs may run at once, from HOOPS_AI_JOB_MAX_CONCURRENCY.

    Defaults to 1, i.e. registration jobs are serialised across all indexes.
    Each job spawns its own pool of embedding workers, every one of which loads
    the ~2 GB model checkpoint, so two concurrent jobs multiply peak memory and
    push the SDK into its single-worker RAM fallback -- slower in total than
    running them one after the other. hoops_ai also writes ``too_heavy_files.log``
    next to its work; ``specifications['log_dir']`` keeps each job's copy apart,
    but that redirection was established by inspecting the compiled SDK rather
    than from documentation, so concurrency would rest on it holding.
    """
    return _env_int("HOOPS_AI_JOB_MAX_CONCURRENCY", _JOB_MAX_CONCURRENCY_DEFAULT, minimum=1)


def job_ttl_seconds() -> int:
    """Retention of finished job records, from HOOPS_AI_JOB_TTL_DAYS."""
    return _env_int("HOOPS_AI_JOB_TTL_DAYS", _JOB_TTL_DAYS_DEFAULT, minimum=0) * 86400


def job_max_records() -> int:
    """Hard cap on retained job records, from HOOPS_AI_JOB_MAX_RECORDS."""
    return _env_int("HOOPS_AI_JOB_MAX_RECORDS", _JOB_MAX_RECORDS_DEFAULT, minimum=1)


def server_paths_enabled() -> bool:
    """Whether jobs may read CAD from arbitrary server-side paths.

    Off by default: it lets any client with API access read any file the server
    process can read, and it bypasses the upload path's size and extension
    checks. It exists only for an administrator seeding a fresh installation
    from a local corpus, so it must be switched on deliberately.
    """
    value = os.environ.get("HOOPS_AI_ALLOW_SERVER_PATHS") or read_env_file().get(
        "HOOPS_AI_ALLOW_SERVER_PATHS", ""
    )
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def get_index_job_store():
    """Process-wide job store for index registration jobs."""
    global _index_job_store
    with _index_job_store_lock:
        if _index_job_store is None:
            import jobstore

            _index_job_store = jobstore.JobStore(
                root=JOBS_DIR,
                max_workers=job_max_concurrency(),
                ttl_seconds=job_ttl_seconds(),
                max_records=job_max_records(),
            )
        return _index_job_store


def resolve_server_paths(paths: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    """Import CAD files from server-side *paths* into the CAD store.

    Returns ``(file_ids, errors)``. Every rejected path yields an error entry,
    so a caller can always account for each input. Raises ``PermissionError``
    when the feature is disabled.
    """
    if not server_paths_enabled():
        raise PermissionError(
            "Server-side paths are disabled. Set HOOPS_AI_ALLOW_SERVER_PATHS=true "
            "to enable them (administrator seeding only)."
        )
    file_ids: list[str] = []
    errors: list[dict[str, Any]] = []
    for raw in paths:
        path = pathlib.Path(raw).expanduser()
        try:
            if not path.is_file():
                raise RuntimeError("not a file")
            if path.suffix.lower() not in CAD_ALLOWED_EXTENSIONS:
                raise RuntimeError(f"unsupported CAD extension '{path.suffix}'")
            with path.open("rb") as fh:
                fid, _, _ = upload_CAD_file_persistent(
                    _BytesUploadFile(fh.read(), path.name)
                )
            file_ids.append(fid)
        except Exception as exc:
            errors.append({"filename": str(raw), "detail": str(exc)})
    return file_ids, errors


_RETRY_REASON_MARKERS = ("timeout", "timed out")


def annotate_failures(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach a ``retryable`` flag to each embedding failure.

    A file is marked retryable when it timed out, or when no reason could be
    determined at all. Everything else -- a corrupt file, an unsupported format,
    a CAD kernel error -- will fail again however long the budget, so retrying it
    only wastes the budget. Treating an unknown reason as retryable is the safe
    direction. This is the rule hoops_ai_native_bridge's Qt front end applies.

    The reason itself is already resolved into each error's ``detail`` from
    ``batch.metadata['errors']``. That front end additionally parses
    ``error_summary.json`` out of its working directory, but only because the
    bridge's C ABI drops the reason on the way out; running the SDK in-process,
    this server gets it directly and needs no such fallback.

    The flag exists because only the server sees the reason. **The decision to
    retry stays with the client**, which resubmits the retryable file_ids as a
    second job with a longer ``time_limit`` and fewer ``workers``.

    Returns a new list; the input is not modified.
    """
    annotated: list[dict[str, Any]] = []
    for err in errors:
        item = dict(err)
        detail = str(item.get("detail") or "").strip()
        unknown = not detail or detail == _EMBED_ERROR_DEFAULT
        low = detail.lower()
        item["detail"] = detail or _EMBED_ERROR_DEFAULT
        item["retryable"] = bool(unknown or any(m in low for m in _RETRY_REASON_MARKERS))
        annotated.append(item)
    return annotated


def _write_add_job_report(
    path: "pathlib.Path",
    name: str,
    added: list[str],
    failed: list[dict[str, Any]],
    heavy_flagged: list[str],
    timings: dict[str, Any],
) -> None:
    """Write the three-group registration report (ADDED / FAILED / HEAVY-FLAGGED).

    The groups answer the operator's question directly: which parts are in the
    index, which are not and why (and whether a longer budget could still save
    them), and which files made the run slow. Written to disk rather than into
    the job record because a corpus-sized job would otherwise put tens of
    thousands of paths into a JSON document that is rewritten on every progress
    update.
    """
    def _fmt(seconds: Any) -> str:
        try:
            secs = float(seconds)
        except (TypeError, ValueError):
            return "-"
        total = int(secs + 0.5)
        h, m, s = total // 3600, (total % 3600) // 60, total % 60
        if h:
            hms = f"{h}h {m:02d}m {s:02d}s"
        elif m:
            hms = f"{m}m {s:02d}s"
        else:
            hms = f"{s}s"
        return f"{hms} ({secs:.1f}s)"

    retryable = [e for e in failed if e.get("retryable")]
    lines: list[str] = []
    lines.append(
        "# Index registration report - "
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    lines.append(f"# Index: {name}")
    lines.append(
        f"# Embedding: num_workers={timings.get('num_workers', '-')} "
        f"time_limit={timings.get('time_limit', '-')}s"
    )
    lines.append(f"# Elapsed: {_fmt(timings.get('embed_seconds'))}")
    lines.append(
        f"# Totals: input={len(added) + len(failed)} added={len(added)} "
        f"failed={len(failed)} retryable={len(retryable)} "
        f"heavy_flagged={len(heavy_flagged)}"
    )

    lines.append("")
    lines.append(f"[ADDED] embedded into the index ({len(added)})")
    lines.extend(added)

    lines.append("")
    lines.append(
        f"[FAILED] not in the index ({len(failed)}; {len(retryable)} worth "
        "resubmitting with a longer time_limit)"
    )
    for err in failed:
        mark = "RETRYABLE" if err.get("retryable") else "PERMANENT"
        label = err.get("filename") or err.get("file_id") or ""
        lines.append(f"{mark}\t{label}\t{err.get('detail', '')}")

    lines.append("")
    lines.append(
        "[HEAVY-FLAGGED] deferred by hoops_ai to its 1-worker RAM fallback "
        f"({len(heavy_flagged)})"
    )
    lines.extend(heavy_flagged)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_index_add_job(
    ctx,
    name: str,
    file_ids: list[str],
    model: Optional[str] = None,
    num_workers: Optional[int] = None,
    time_limit: Optional[int] = None,
) -> None:
    """Job body: register *file_ids* into index *name* in a single SDK call.

    One job is one pass. The whole input goes to one ``embed_shape_batch``:
    splitting it into fixed-size chunks was measurably wrong, because each call
    builds a fresh spawn-based process pool whose every worker reloads the ~2 GB
    checkpoint, ~55-60 s per call.

    **The two-pass strategy is the client's to run, not this server's.** A
    caller registers a corpus with many workers and a short ``time_limit``,
    reads the ``retryable`` flag on each reported failure, and resubmits just
    those files as a second job with few workers and a long ``time_limit``.
    Splitting it this way keeps the commit granularity and the retry policy in
    the caller's hands; the server's part is to make the decision possible by
    reporting *why* each file failed (see :func:`annotate_failures`).

    Consequences, both accepted deliberately:
      * Cancellation is only observed before the call starts. ``embed_shape_batch``
        is a single blocking call that cannot be interrupted.
      * A crash mid-job loses that job's work. Callers control the commit
        granularity by splitting a large corpus across several jobs.

    Failures are recorded per file with the reason reported by the SDK, never
    dropped.
    """
    total = len(file_ids)
    ctx.set_total(total)
    ctx.set_phase("embedding")
    ctx.set_summary(added=0, updated=0, failed=0, skipped=0, heavy_flagged=0)
    ctx.check_canceled()

    store = get_index_job_store()
    try:
        artifacts = store.artifacts_dir(ctx.job_id)
    except Exception:  # a job store that predates artefact directories
        artifacts = JOBS_DIR / str(ctx.job_id)

    workers = num_workers if num_workers is not None else compute_auto_workers(total)
    limit = time_limit if time_limit is not None else embed_time_limit()
    timings: dict[str, Any] = {"num_workers": workers, "time_limit": limit}
    ctx.set_timings(**timings)

    # Resolved once, so the report can name files rather than hashes.
    path_by_fid: dict[str, str] = {}
    for fid in file_ids:
        try:
            path_by_fid[fid] = str(find_persistent_CAD_file(fid).resolve())
        except Exception:
            continue

    t0 = time.perf_counter()

    # Live counters parsed from the SDK's own progress bar, so a client polling
    # the job sees it advance instead of 0/N until the batch ends. The bar's own
    # total is ignored: progress.total is the number of files this job submitted,
    # which is the number the client asked about.
    seen_phase = {"value": "embedding"}

    def _on_progress(
        phase: str, done: int, _bar_total: int, err_count: int, heavy: int
    ) -> None:
        fields: dict[str, Any] = {"done": int(done)}
        if err_count >= 0:
            fields["errors"] = int(err_count)
        if heavy >= 0:
            fields["heavy"] = int(heavy)
        if phase != seen_phase["value"]:
            seen_phase["value"] = phase
            fields["phase"] = phase
        try:
            ctx.set_progress(**fields)
        except Exception:
            # Persisting progress is best-effort; never fail a run over it.
            logger.debug("progress update dropped", exc_info=True)

    result = add_to_index(
        name, file_ids, model=model, num_workers=workers, time_limit=limit,
        log_dir=artifacts, progress_cb=_on_progress,
    )
    timings["embed_seconds"] = round(time.perf_counter() - t0, 3)

    # too_heavy_files.log must be read now: the next embedding call overwrites it.
    heavy_flagged = read_too_heavy_files(artifacts)
    errors = annotate_failures(list(result.get("errors") or []))
    for err in errors:
        fid = str(err.get("file_id") or "")
        if fid in path_by_fid:
            err.setdefault("filename", path_by_fid[fid])

    added = int(result.get("added", 0))
    updated = int(result.get("updated", 0))
    index_count = int(result.get("index_count", 0))
    retryable = sum(1 for e in errors if e.get("retryable"))

    # Accounting guard: every submitted file must be added, updated or reported
    # as failed. A silent shortfall means the SDK contract moved.
    accounted = added + updated + len(errors)
    if accounted != total:
        logger.warning(
            "[JOB] index '%s': %d files submitted but only %d accounted for "
            "(added/updated/errors); some inputs were not reported",
            name, total, accounted,
        )
    if retryable:
        logger.info(
            "[JOB] index '%s': %d of %d failure(s) are retryable; resubmit them "
            "with a longer time_limit and fewer workers",
            name, retryable, len(errors),
        )

    ctx.add_errors(errors)
    # Absolute, not incremental: the live callback above has already been moving
    # progress.done, so incrementing here would count the batch twice.
    ctx.set_progress(done=total, errors=len(errors), heavy=len(heavy_flagged))
    ctx.set_summary(
        added=added, updated=updated, failed=len(errors), skipped=0,
        index_count=index_count, heavy_flagged=len(heavy_flagged),
        retryable=retryable,
    )
    ctx.set_timings(**timings)

    failed_ids = {str(e.get("file_id") or "") for e in errors}
    try:
        _write_add_job_report(
            artifacts / "report.txt",
            name,
            [path_by_fid.get(f, f) for f in file_ids if f not in failed_ids],
            errors,
            heavy_flagged,
            timings,
        )
        ctx.set_summary(report=True)
    except Exception as exc:  # a missing report must never fail a good job
        logger.warning("[JOB] could not write report for job %s: %s", ctx.job_id, exc)

    ctx.set_phase("done")


# Default limits for ZIP extraction (also used by routers). Both can be raised
# for bulk registration via HOOPS_AI_ZIP_MAX_TOTAL_BYTES / HOOPS_AI_ZIP_MAX_FILES;
# the module-level values stay as the defaults so existing callers that pass
# nothing keep the historical behaviour.
ZIP_MAX_TOTAL_BYTES = 500 * 1024 * 1024  # 500 MB uncompressed
ZIP_MAX_FILES = 50


def zip_max_total_bytes() -> int:
    """Uncompressed-size cap for a ZIP archive, from HOOPS_AI_ZIP_MAX_TOTAL_BYTES."""
    return _env_int("HOOPS_AI_ZIP_MAX_TOTAL_BYTES", ZIP_MAX_TOTAL_BYTES, minimum=1)


def zip_max_files() -> int:
    """CAD-file-count cap for a ZIP archive, from HOOPS_AI_ZIP_MAX_FILES."""
    return _env_int("HOOPS_AI_ZIP_MAX_FILES", ZIP_MAX_FILES, minimum=1)



class _BytesUploadFile:
    """Minimal duck-typed UploadFile used to pass in-memory bytes to core helpers."""

    def __init__(self, data: bytes, filename: str) -> None:
        self.filename = filename
        self.file = io.BytesIO(data)


def extract_zip_cad_files(
    zip_data: bytes,
    max_total_bytes: Optional[int] = None,
    max_files: Optional[int] = None,
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
    if max_total_bytes is None:
        max_total_bytes = zip_max_total_bytes()
    if max_files is None:
        max_files = zip_max_files()

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


def _create_brep_encoder(cad_file_path: pathlib.Path):
    from hoops_ai.cadaccess import HOOPSLoader, HOOPSTools
    from hoops_ai.cadencoder import BrepEncoder

    cad_loader = HOOPSLoader()
    cad_model = cad_loader.create_from_file(str(cad_file_path))

    hoopstools = HOOPSTools()
    hoopstools.adapt_brep(cad_model)

    return BrepEncoder(cad_model.get_brep())


def get_brep_attributes(cad_file_path: pathlib.Path) -> dict[str, Any]:
    brep_encoder = _create_brep_encoder(cad_file_path)

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


def get_brep_type_counts(cad_file_path: pathlib.Path) -> dict[str, Any]:
    """Return face/edge counts grouped by type, pre-aggregated server-side.

    This is intentionally a separate, narrow endpoint from ``get_brep_attributes``:
    handing an AI agent a raw array of a few hundred face/edge entries and asking it
    to tally counts by type is unreliable (the agent tends to eyeball the array
    instead of counting every entry). Aggregating server-side guarantees a correct,
    reproducible count regardless of model size.
    """
    import numpy as np

    brep_encoder = _create_brep_encoder(cad_file_path)

    [face_types, *_], face_types_descr = brep_encoder.push_face_attributes()
    [edge_types, *_], edge_types_descr = brep_encoder.push_edge_attributes()

    def _counts_by_type(types_array, types_descr: dict) -> dict[str, Any]:
        types_array = np.asarray(types_array)
        unique_codes, counts = np.unique(types_array, return_counts=True)
        by_type = {
            types_descr.get(int(code), str(int(code))): int(count)
            for code, count in zip(unique_codes, counts)
        }
        return {"total": int(types_array.size), "by_type": by_type}

    return {
        "faces": _counts_by_type(face_types, face_types_descr),
        "edges": _counts_by_type(edge_types, edge_types_descr),
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

    This function is **stateless** 窶・it performs no network I/O and holds no
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
            Part id 竊・metadata dict.  Only ids present here are returned by
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

