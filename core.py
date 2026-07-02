import ast
import hashlib
import io
import os
import pathlib
import shutil
import ssl
import uuid
from contextlib import redirect_stdout
from typing import Any, Optional

from fastapi import UploadFile

APP_ROOT = pathlib.Path(__file__).resolve().parent
ENV_FILE_PATH = pathlib.Path(__file__).with_name(".env")
CAD_UPLOAD_DIR = APP_ROOT.joinpath("uploads")
CAD_VIEWER_OUTPUT_DIR = APP_ROOT.joinpath("out")
MFR_LABELS_DESCRIPTION_ENV_NAMES = ("HOOPS_AI_MFR_LABELS_DESCRIPTION", "labels_description")

MFR_dataset_explorer = None
MFR_inference_model = None
CAD_viewers: dict[str, dict[str, Any]] = {}  # session_id -> {file_key -> viewer_info}
CAD_face_colors: dict[str, list] = {}  # scs_filename -> [[r,g,b], ...] indexed by face_id
cad_searcher = None
shape_index = None
PART_CLASS_inference_model = None
PART_CLASS_dataset_explorer = None


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


def get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} environment variable is required. Set it in {ENV_FILE_PATH.name} or the environment."
        )
    return value


def get_required_file_env(name: str) -> str:
    value = read_env_file().get(name)
    if not value:
        raise RuntimeError(f"{name} is required in {ENV_FILE_PATH.name}.")
    return value


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

    notebooks_dir = pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR"))
    MFR_flow_name = get_required_env("HOOPS_AI_MFR_FLOW_NAME")

    # Dataset files are produced by running the ETL tutorial notebook:
    #   notebooks/3b_workflow_for_MFR_cadsynth.ipynb
    # Output is written to: <HOOPS_AI_NOTEBOOK_DIR>/out/flows/<HOOPS_AI_MFR_FLOW_NAME>/
    flow_root_dir = notebooks_dir / "out" / "flows" / MFR_flow_name

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

    notebooks_dir = pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR"))
    model_name = get_required_env("HOOPS_AI_MFR_MODEL_NAME")
    trained_model = notebooks_dir.parent.joinpath("packages", "trained_ml_models", model_name)
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

    notebooks_dir = pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR"))
    ckpt_name = get_required_env("HOOPS_AI_EMBEDDINGS_MODEL_NAME")
    trained_model = notebooks_dir.parent.joinpath("packages", "trained_ml_models", ckpt_name)

    HOOPSEmbeddings.register_model(
        model_name="hoops_embeddings_model",
        checkpoint_path=str(trained_model),
    )
    embedder = HOOPSEmbeddings(model="hoops_embeddings_model")
    return CADSearch(shape_model=embedder)


def load_shape_index():
    load_env_file()

    notebooks_dir = pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR"))
    faiss_file_name = get_required_env("HOOPS_AI_FAISS_INDEX_PATH")
    faiss_index_path = notebooks_dir.joinpath(faiss_file_name)
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
    notebooks_dir = pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR"))
    flow_name = get_required_env("HOOPS_AI_PART_CLASS_FLOW_NAME")
    notebook_out = notebooks_dir / "out" / "flows" / flow_name
    if notebook_out.exists():
        return notebook_out.resolve()
    return (notebooks_dir.parent / "packages" / "flows" / flow_name).resolve()


def create_part_class_inference_model():
    """Load the GraphClassification checkpoint once and return the FlowInference instance."""
    import torch.nn as nn
    from hoops_ai.cadaccess import HOOPSLoader
    from hoops_ai.ml.EXPERIMENTAL import FlowInference, GraphClassification

    load_env_file()

    notebooks_dir = pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR"))
    model_name = get_required_env("HOOPS_AI_PART_CLASS_MODEL_NAME")
    ckpt_path = notebooks_dir.parent / "packages" / "trained_ml_models" / model_name

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

