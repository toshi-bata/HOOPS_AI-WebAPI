from typing import Optional

import core
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

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

