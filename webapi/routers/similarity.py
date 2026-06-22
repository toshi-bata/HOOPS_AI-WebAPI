import os
import pathlib
from typing import Optional

import core
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

router = APIRouter(prefix="/similarity", tags=["CAD Similarity Search"])


@router.post("/search")
def similarity_search(
    request: Request,
    file: Optional[UploadFile] = File(None),
    file_id: Optional[str] = Query(None, description="file_id returned by POST /files/upload"),
    top_k: int = Query(10, ge=1, description="Number of similar shapes to return."),
):
    """Search for similar CAD shapes.

    Supply **either** a file upload *or* a ``file_id`` from a previous upload.
    """
    try:
        if file_id:
            cad_file_path = core.find_persistent_CAD_file(file_id)
        elif file:
            _, cad_file_path, _ = core.upload_CAD_file_persistent(file)
        else:
            raise HTTPException(status_code=422, detail="Either 'file' or 'file_id' is required.")
        result = core.search_by_shape(cad_file_path, top_k=top_k)
        image_filename = result["image_url"].lstrip("/out/")
        result["image_url"] = str(request.url_for("out", path=image_filename))
        return result
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {exc}") from exc


@router.get("/part-image")
def get_part_image(
    filename: str = Query(..., description="CAD filename (with or without extension) returned by similarity search."),
):
    """Return the pre-generated PNG thumbnail for a trained part as a direct PNG image response."""
    stem = pathlib.Path(filename).stem

    notebooks_dir = pathlib.Path(core.get_required_env("HOOPS_AI_NOTEBOOK_DIR"))
    images_base_dir = pathlib.Path(
        os.environ.get("HOOPS_AI_EMBEDDINGS_IMAGES_DIR")
        or notebooks_dir / "out" / "images"
    )

    # Search order: STEP/_white.png → STEP/.png → base/_white.png → base/.png
    candidates = [
        images_base_dir / "STEP" / f"{stem}_white.png",
        images_base_dir / "STEP" / f"{stem}.png",
        images_base_dir / f"{stem}_white.png",
        images_base_dir / f"{stem}.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return FileResponse(str(candidate), media_type="image/png")

    raise HTTPException(
        status_code=404,
        detail=f"PNG image not found for '{filename}'. Searched: {[str(c) for c in candidates]}",
    )

