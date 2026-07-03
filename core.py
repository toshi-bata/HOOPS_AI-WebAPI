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
import threading
import uuid
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
cad_searcher = None
shape_index = None
PART_CLASS_inference_model = None
PART_CLASS_dataset_explorer = None
_embedder = None
_embedding_memory_cache: dict[str, dict] = {}  # cache_key -> embedding entry

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
    """Base path (no extension) for a named index; save() appends .faiss / .meta."""
    return INDEXES_DIR / name


def _index_faiss_path(name: str) -> pathlib.Path:
    return INDEXES_DIR / f"{name}.faiss"


def _get_embedder_dim() -> int:
    """Return the embedding dimension from the lazy-loaded HOOPSEmbeddings model."""
    embedder = get_embedder()
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
    tmp_base = INDEXES_DIR / f"_tmp_{uuid.uuid4().hex}"
    try:
        vs.save(str(tmp_base))
        pathlib.Path(str(tmp_base) + ".faiss").replace(_index_faiss_path(name))
        pathlib.Path(str(tmp_base) + ".meta").replace(INDEXES_DIR / f"{name}.meta")
    except Exception:
        for suffix in (".faiss", ".meta"):
            tmp = pathlib.Path(str(tmp_base) + suffix)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Named index public API
# ---------------------------------------------------------------------------


def create_index(name: str) -> dict[str, Any]:
    """Create a new empty named index.

    Raises ValueError for invalid names, PermissionError for reserved names,
    and FileExistsError if an index with that name already exists.
    """
    _validate_index_name(name)
    lock = _get_index_lock(name)
    with lock:
        if _index_faiss_path(name).exists():
            raise FileExistsError(f"Index '{name}' already exists.")

        from hoops_ai.ml.embeddings import FaissVectorStore

        dim = _get_embedder_dim()
        vs = FaissVectorStore(dim)
        _save_named_index_atomic(name, vs)
        _named_indexes[name] = vs
        return {"name": name, "count": 0, "dim": dim}


def list_indexes() -> list[dict[str, Any]]:
    """Return metadata for all known indexes, including the read-only ``default`` index."""
    result: list[dict[str, Any]] = []

    # "default" index – read-only, backed by the env-configured FAISS file
    load_env_file()
    faiss_name = os.environ.get("HOOPS_AI_FAISS_INDEX_PATH")
    if faiss_name:
        notebooks_dir_str = os.environ.get("HOOPS_AI_NOTEBOOK_DIR")
        if notebooks_dir_str:
            default_path = pathlib.Path(notebooks_dir_str) / faiss_name
            last_modified: Optional[str] = None
            if default_path.exists():
                mtime = default_path.stat().st_mtime
                last_modified = (
                    datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ")
                )
            default_count: Optional[int] = None
            if shape_index is not None:
                ids = getattr(shape_index, "ids", None)
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

    # Named indexes from INDEXES_DIR
    if INDEXES_DIR.exists():
        for faiss_file in sorted(INDEXES_DIR.glob("*.faiss")):
            if faiss_file.name.startswith("_tmp_"):
                continue
            idx_name = faiss_file.stem
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
                }
            )

    return result


def add_to_index(name: str, file_ids: list[str]) -> dict[str, Any]:
    """Compute embeddings for *file_ids* and upsert them into the named index.

    Re-uses the ``compute_embedding()`` disk cache.  Re-inserting the same
    file_id overwrites the existing entry (delete-then-upsert to avoid FAISS
    duplicate entries).

    Returns ``added``, ``updated``, ``index_count``, and per-file ``errors``.
    """
    _validate_index_name(name)
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
                emb_result = compute_embedding(fid)
                v = emb_result["vector"]
                emb_obj = Embedding(values=v, model=emb_result["model_name"], dim=emb_result["dim"])
                meta = {
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
        return {
            "name": name,
            "added": added,
            "updated": updated,
            "index_count": len(vs.get_ids()),
            "errors": errors,
        }


def remove_from_index(name: str, part_ids: list[str]) -> dict[str, Any]:
    """Delete *part_ids* from the named index and persist the change."""
    _validate_index_name(name)
    lock = _get_index_lock(name)
    with lock:
        vs = _load_named_index(name)
        vs.delete(part_ids)
        _save_named_index_atomic(name, vs)
        return {"name": name, "removed": len(part_ids), "index_count": len(vs.get_ids())}


def search_index(name: str, file_id: str, top_k: int) -> dict[str, Any]:
    """Search the named index for the top-k most similar parts to *file_id*.

    Returns an empty hits list when the index contains no entries (no error).
    """
    _validate_index_name(name)
    lock = _get_index_lock(name)
    with lock:
        vs = _load_named_index(name)
        if len(vs.get_ids()) == 0:
            return {"hits": [], "count": 0}
        emb_result = compute_embedding(file_id)
        v = emb_result["vector"]
        hits = vs.query(v, top_k=top_k)
        return {
            "hits": [
                {"id": h.id, "score": round(float(h.score), 6), "metadata": h.metadata}
                for h in hits
            ],
            "count": len(hits),
        }


def delete_index(name: str) -> dict[str, Any]:
    """Delete the named index and its on-disk files.

    Raises PermissionError for reserved names, KeyError if the index does not exist.
    """
    _validate_index_name(name)
    lock = _get_index_lock(name)
    with lock:
        faiss_file = _index_faiss_path(name)
        if not faiss_file.exists():
            raise KeyError(f"Index '{name}' does not exist.")
        faiss_file.unlink(missing_ok=True)
        meta_file = INDEXES_DIR / f"{name}.meta"
        meta_file.unlink(missing_ok=True)
        _named_indexes.pop(name, None)
        return {"name": name, "deleted": True}


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


def get_cad_searcher():
    global cad_searcher
    if cad_searcher is None:
        cad_searcher = create_cad_searcher()
    return cad_searcher


def get_shape_index():
    global shape_index
    if shape_index is None:
        shape_index = load_shape_index()
    return shape_index


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

    notebooks_dir = require_path(
        pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR")),
        env_name="HOOPS_AI_NOTEBOOK_DIR",
    )
    MFR_flow_name = get_required_env("HOOPS_AI_MFR_FLOW_NAME")

    # Dataset files are produced by running the ETL tutorial notebook:
    #   notebooks/3b_workflow_for_MFR_cadsynth.ipynb
    # Output is written to: <HOOPS_AI_NOTEBOOK_DIR>/out/flows/<HOOPS_AI_MFR_FLOW_NAME>/
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

    notebooks_dir = require_path(
        pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR")),
        env_name="HOOPS_AI_NOTEBOOK_DIR",
    )
    model_name = get_required_env("HOOPS_AI_MFR_MODEL_NAME")
    trained_model = require_path(
        notebooks_dir.parent.joinpath("packages", "trained_ml_models", model_name),
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


def create_cad_searcher():
    from hoops_ai.ml import CADSearch
    from hoops_ai.ml.embeddings import HOOPSEmbeddings

    load_env_file()

    notebooks_dir = require_path(
        pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR")),
        env_name="HOOPS_AI_NOTEBOOK_DIR",
    )
    ckpt_name = get_required_env("HOOPS_AI_EMBEDDINGS_MODEL_NAME")
    trained_model = require_path(
        notebooks_dir.parent.joinpath("packages", "trained_ml_models", ckpt_name),
        env_name="HOOPS_AI_EMBEDDINGS_MODEL_NAME",
    )

    HOOPSEmbeddings.register_model(
        model_name="hoops_embeddings_model",
        checkpoint_path=str(trained_model),
    )
    embedder = HOOPSEmbeddings(model="hoops_embeddings_model")
    return CADSearch(shape_model=embedder)


def load_shape_index():
    load_env_file()

    notebooks_dir = require_path(
        pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR")),
        env_name="HOOPS_AI_NOTEBOOK_DIR",
    )
    faiss_file_name = get_required_env("HOOPS_AI_FAISS_INDEX_PATH")
    faiss_index_path = require_path(
        notebooks_dir.joinpath(faiss_file_name),
        env_name="HOOPS_AI_FAISS_INDEX_PATH",
    )
    searcher = get_cad_searcher()
    # The FAISS index may have been pickled on Windows and contain WindowsPath objects.
    if not hasattr(pathlib, "WindowsPath") or not issubclass(pathlib.WindowsPath, pathlib.Path):
        # Linux/Mac: patch WindowsPath → PurePosixPath so Windows-pickled data can be unpickled.
        pathlib.WindowsPath = pathlib.PurePosixPath  # type: ignore[attr-defined]
        return searcher.load_shape_index(path=str(faiss_index_path))
    else:
        # Windows: patch PosixPath → WindowsPath so Linux-pickled data can be unpickled.
        _orig = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc]
            return searcher.load_shape_index(path=str(faiss_index_path))
        finally:
            pathlib.PosixPath = _orig


def search_by_shape(cad_file_path: pathlib.Path, top_k: int = 10) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from hoops_ai.insights import DatasetViewer

    get_shape_index()  # ensure FAISS index is loaded into the searcher
    searcher = get_cad_searcher()
    hits = searcher.search_by_shape(str(cad_file_path), top_k=top_k)
    results = [
        {"id": _json_safe(hit.id), "score": _json_safe(hit.score)}
        for hit in hits[0]
    ]

    notebooks_dir = pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR"))
    embeddings_images_dir = pathlib.Path(
        os.environ.get("HOOPS_AI_EMBEDDINGS_IMAGES_DIR")
        or notebooks_dir / "out" / "images"
    )
    ds_viewer = DatasetViewer([], [], [], reference_dir=embeddings_images_dir)
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

    Always succeeds — returns ``status: "not_loaded"`` when neither the searcher
    nor the index has been initialised yet, so callers never need to treat an
    unloaded state as an error.
    """
    import datetime

    load_env_file()

    # Resolve the index file path from env (best-effort, no error if missing).
    index_path: Optional[str] = None
    index_last_modified: Optional[str] = None
    faiss_file_name = os.environ.get("HOOPS_AI_FAISS_INDEX_PATH")
    if faiss_file_name:
        notebooks_dir_str = os.environ.get("HOOPS_AI_NOTEBOOK_DIR")
        if notebooks_dir_str:
            faiss_index_path = pathlib.Path(notebooks_dir_str) / faiss_file_name
            index_path = str(faiss_index_path)
            if faiss_index_path.exists():
                mtime = faiss_index_path.stat().st_mtime
                index_last_modified = (
                    datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ")
                )
        else:
            index_path = faiss_file_name

    # Trigger lazy loading of the searcher + FAISS index if not yet initialised.
    # This mirrors what search_by_shape() does on first use.  If loading fails
    # (missing env var, file not found, etc.) we fall through to not_loaded.
    if cad_searcher is None or shape_index is None:
        try:
            get_shape_index()  # populates both cad_searcher and shape_index globals
        except Exception:
            pass  # genuine not_loaded situation — reported below

    if cad_searcher is None and shape_index is None:
        return {
            "status": "not_loaded",
            "index_path": index_path,
            "index_last_modified": index_last_modified,
            "index_count": None,
            "model_name": None,
            "embedding_dim": None,
            "metadata": None,
        }

    info: dict[str, Any] = {
        "status": "loaded",
        "index_path": index_path,
        "index_last_modified": index_last_modified,
        "index_count": None,
        "model_name": None,
        "embedding_dim": None,
        "metadata": None,
    }

    # Extract embedder attributes from the CADSearch object.
    # HOOPSEmbeddings does not expose public attributes via dir(), so we try
    # both the public and private attribute names used by CADSearch internals.
    if cad_searcher is not None:
        embedder = (
            getattr(cad_searcher, "_shape_model", None)
            or getattr(cad_searcher, "shape_model", None)
        )
        if embedder is not None:
            info["model_name"] = (
                getattr(embedder, "model_name", None)
                or getattr(embedder, "model", None)
            )

    # Extract counts, dim, and metadata from the loaded EmbeddingBatch object.
    # EmbeddingBatch exposes: ids, dim, model, metadata, values, get.
    if shape_index is not None:
        ids = getattr(shape_index, "ids", None)
        if ids is not None:
            try:
                info["index_count"] = int(len(ids))
            except (TypeError, ValueError):
                pass
        # embedding_dim lives on the batch as .dim
        dim = getattr(shape_index, "dim", None)
        if dim is not None:
            try:
                info["embedding_dim"] = int(dim)
            except (TypeError, ValueError):
                info["embedding_dim"] = _json_safe(dim)
        # Fall back to model name from the batch if not obtained from the embedder.
        if info["model_name"] is None:
            info["model_name"] = getattr(shape_index, "model", None)
        metadata = getattr(shape_index, "metadata", None)
        if metadata is not None:
            info["metadata"] = _json_safe(dict(metadata) if hasattr(metadata, "items") else metadata)

    return info


# ---------------------------------------------------------------------------
# Embedding-only helpers (no FAISS index required)
# ---------------------------------------------------------------------------


def get_embedder():
    """Lazy-initialise HOOPSEmbeddings without loading a FAISS index.

    Safe to call alongside (or after) ``create_cad_searcher()`` — the model
    registration is guarded so it is never performed twice.
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    from hoops_ai.ml.embeddings import HOOPSEmbeddings

    load_env_file()

    notebooks_dir = require_path(
        pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR")),
        env_name="HOOPS_AI_NOTEBOOK_DIR",
    )
    ckpt_name = get_required_env("HOOPS_AI_EMBEDDINGS_MODEL_NAME")
    trained_model = require_path(
        notebooks_dir.parent.joinpath("packages", "trained_ml_models", ckpt_name),
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


def compute_embedding(file_id: str) -> dict[str, Any]:
    """Compute (or retrieve from cache) the shape embedding for a CAD file.

    The result vector is a single L2-normalised float32 array representing the
    whole part.  For multi-body models the per-body vectors are individually
    L2-normalised, averaged, then re-normalised.

    Returns a dict with keys:
      ``file_id``, ``vector`` (np.ndarray), ``dim``, ``model_name``,
      ``num_bodies``, ``filename``, ``cached`` (bool).
    """
    import numpy as np

    embedder = get_embedder()
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
            pass  # corrupted cache — fall through to recompute

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


def compare_embeddings(file_ids: list[str]) -> dict[str, Any]:
    """Compute an N×N cosine similarity matrix for the given file_ids.

    All embedding vectors are L2-normalised, so cosine similarity equals their
    dot product.  Diagonal entries are forced to exactly ``1.0``.

    Returns a dict with keys:
      ``count``, ``model_name``, ``files``, ``matrix`` (N×N list of lists),
      ``pairs`` (all i<j combos sorted by score descending).
    """
    import numpy as np

    embeddings = [compute_embedding(fid) for fid in file_ids]
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

    # Reuse existing SCS only when a stable session_id is provided and the file was
    # already converted for that session.  Without a session_id every call is treated
    # as a fresh request so different clients never share the same SCS file.
    if session_id:
        session_viewers = CAD_viewers.setdefault(session_id, {})
        existing = session_viewers.get(file_key)
        if existing:
            scs_path = CAD_VIEWER_OUTPUT_DIR / existing["scs_filename"]
            if scs_path.exists():
                png_url = f"/out/{existing['png_filename']}" if existing.get("png_filename") else None
                return {"viewer_url": f"/CAD/viewer/show?scs={scs_path.name}", "image_url": png_url, "_scs_filename": scs_path.name}
    else:
        session_viewers = None

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
    if session_viewers is not None:
        session_viewers[file_key] = {"scs_filename": scs_path.name, "png_filename": png_filename}

    png_url = f"/out/{png_filename}" if png_filename else None
    return {"viewer_url": f"/CAD/viewer/show?scs={scs_path.name}", "image_url": png_url, "_scs_filename": scs_path.name}


def terminate_CAD_viewer(session_id: Optional[str] = None, terminate_all: bool = False) -> dict[str, Any]:
    session_viewers = CAD_viewers.get(session_id or "", {})
    if not session_viewers:
        raise RuntimeError("No active CAD viewer.")

    if terminate_all:
        count = len(session_viewers)
        session_viewers.clear()
        return {"terminated": count}
    else:
        file_key = next(reversed(session_viewers))
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
    1. <HOOPS_AI_NOTEBOOK_DIR>/out/flows/<FLOW_NAME>  — notebook-generated output (has stream_cache)
    2. <HOOPS_AI_NOTEBOOK_DIR>/../packages/flows/<FLOW_NAME>  — pre-packaged dataset
    """
    notebooks_dir = require_path(
        pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR")),
        env_name="HOOPS_AI_NOTEBOOK_DIR",
    )
    flow_name = get_required_env("HOOPS_AI_PART_CLASS_FLOW_NAME")
    notebook_out = notebooks_dir / "out" / "flows" / flow_name
    if notebook_out.exists():
        return notebook_out.resolve()
    packages_flow = (notebooks_dir.parent / "packages" / "flows" / flow_name).resolve()
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

    notebooks_dir = require_path(
        pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR")),
        env_name="HOOPS_AI_NOTEBOOK_DIR",
    )
    model_name = get_required_env("HOOPS_AI_PART_CLASS_MODEL_NAME")
    ckpt_path = require_path(
        notebooks_dir.parent / "packages" / "trained_ml_models" / model_name,
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

