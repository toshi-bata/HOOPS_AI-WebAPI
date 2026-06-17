import core
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/files", tags=["File Management"])


@router.post("/upload")
def upload_file(file: UploadFile = File(...)):
    """Upload a CAD file to the server.

    Returns a ``file_id`` derived from the file's SHA-256 hash.
    Uploading the same file again returns the same ``file_id`` without re-storing
    the file (``already_existed`` will be ``true``).
    Pass the ``file_id`` to any processing endpoint instead of re-uploading the file.
    """
    try:
        file_id, path, existed = core.upload_CAD_file_persistent(file)
        return {
            "file_id": file_id,
            "filename": path.name,
            "already_existed": existed,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/")
def list_files():
    """List all CAD files currently stored on the server."""
    return {"files": core.list_persistent_CAD_files()}


@router.delete("/{file_id}")
def delete_file(file_id: str):
    """Delete an uploaded CAD file by its file_id."""
    try:
        deleted = core.delete_persistent_CAD_file(file_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
        return {"deleted": file_id}
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
