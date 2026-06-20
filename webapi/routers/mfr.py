import io
from typing import Optional

import core
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/MFR", tags=["MFR"])


def _get_session_id(request: Request) -> Optional[str]:
    return request.headers.get("X-Session-ID") or None


@router.get("/files/search")
def MFR_search_files(feature_name: str = Query(..., description="MFR feature name to search for.")):
    try:
        return core.search_MFR_files(feature_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/labels/description")
def MFR_labels_description():
    try:
        return {"labels_description": core._json_safe(core.get_MFR_labels_description())}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/dataset/table-of-contents")
def MFR_table_of_contents():
    try:
        return core.get_MFR_table_of_contents()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/files/{file_id}/thumbnail")
def MFR_file_thumbnail(file_id: int):
    try:
        png_bytes = core.get_MFR_file_thumbnail(file_id)
        return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/inference")
def MFR_inference(
    request: Request,
    file: Optional[UploadFile] = File(None),
    file_id: Optional[str] = Query(None, description="file_id returned by POST /files/upload"),
):
    """Run MFR inference on a CAD file.

    Supply **either** a file upload *or* a ``file_id`` from a previous upload.
    When ``file_id`` is given the file is reused without re-uploading.
    """
    try:
        if file_id:
            cad_file_path = core.find_persistent_CAD_file(file_id)
        elif file:
            _, cad_file_path, _ = core.upload_CAD_file_persistent(file)
        else:
            raise HTTPException(status_code=422, detail="Either 'file' or 'file_id' is required.")
        result = core.run_MFR_inference(cad_file_path, _get_session_id(request))
        for key in ("viewer_url", "image_url"):
            if result.get(key) and result[key].startswith("/"):
                result[key] = str(request.base_url).rstrip("/") + result[key]
        result.pop("_scs_filename", None)
        return result
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


