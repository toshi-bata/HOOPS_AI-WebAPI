import io
from typing import Optional

import core
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

router = APIRouter(prefix="/part-classification", tags=["Part Classification"])


@router.post("/predict")
def part_classification_predict(
    file: Optional[UploadFile] = File(None),
    file_id: Optional[str] = Query(None, description="file_id returned by POST /files/upload"),
    top_k: int = Query(5, ge=1, le=45, description="Number of top predictions to return (max 45)."),
):
    """Run Part Classification inference on a CAD file.

    Supply **either** a file upload *or* a ``file_id`` from a previous upload.

    Returns the top-k predicted part classes with confidence scores (integer %).
    """
    try:
        if file_id:
            cad_file_path = core.find_persistent_CAD_file(file_id)
        elif file:
            _, cad_file_path, _ = core.upload_CAD_file_persistent(file)
        else:
            raise HTTPException(status_code=422, detail="Either 'file' or 'file_id' is required.")
        return core.run_part_classification_inference(cad_file_path, top_k=top_k)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Part classification failed: {exc}") from exc


@router.get("/labels")
def part_class_labels():
    """Return the full 45-class part label dictionary."""
    try:
        return {"labels_description": core._json_safe(core.get_part_class_labels_description())}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/dataset/table-of-contents")
def part_class_table_of_contents():
    """Return dataset table of contents and available groups."""
    try:
        return core.get_part_class_table_of_contents()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.get("/dataset/label-distribution")
def part_class_label_distribution():
    """Return per-class file count distribution across the training dataset."""
    try:
        return core.get_part_class_label_distribution()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.get("/dataset/files")
def part_class_file_list(
    label_id: int = Query(..., ge=0, le=44, description="Part label ID (0–44)."),
):
    """Return the list of file IDs in the dataset that belong to a given class."""
    try:
        return core.get_part_class_file_list(label_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.get("/dataset/preview")
def part_class_preview_image(
    request: Request,
    label_id: int = Query(..., ge=0, le=44, description="Part label ID (0–44)."),
    k: int = Query(25, ge=1, description="Max number of thumbnails to show."),
    grid_cols: int = Query(8, ge=1, description="Number of columns in the thumbnail grid."),
):
    """Return a URL to a PNG thumbnail grid for the given part class."""
    try:
        file_list_result = core.get_part_class_file_list(label_id)
        file_ids = file_list_result["file_ids"]
        if not file_ids:
            raise HTTPException(
                status_code=404,
                detail=f"No dataset files found for label_id={label_id} "
                f"({file_list_result.get('part_name', '')}).",
            )
        image_filename = core.get_part_class_preview_image(file_ids, k=k, grid_cols=grid_cols)
        image_url = str(request.url_for("out", path=image_filename))
        return {
            "label_id": label_id,
            "part_name": file_list_result["part_name"],
            "image_url": image_url,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
