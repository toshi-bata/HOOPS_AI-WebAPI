import core
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

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


@router.post("/upload-from-path")
def upload_from_path(
    file_path: str = Query(
        ...,
        description=(
            "Server-side path to a CAD file or ZIP archive. "
            "Absolute paths (e.g. C:\\\\temp\\\\parts.zip) are accepted as-is. "
            "Relative paths are resolved under the configured shared folder "
            "(HOOPS_AI_CAD_SHARED_DIR). "
            "ZIP archives are extracted and every contained CAD file is uploaded "
            "individually. Single CAD files are uploaded directly. "
            "Returns a list of file_ids that can be passed to /similarity/* endpoints."
        ),
    ),
):
    """Upload a CAD file or ZIP archive by server-side file path.

    This endpoint is the recommended way for MCP / scripted clients to register
    files that already exist on the server's filesystem (e.g. a path the user
    dragged into the chat).  Pass the returned ``file_ids`` to
    ``POST /similarity/compare``, ``POST /similarity/map``,
    ``POST /similarity/index/add``, etc.

    * **Single CAD file** – uploaded and a single entry is returned.
    * **ZIP archive** – extracted, every recognised CAD file is uploaded, and all
      resulting ``file_id`` values are returned together.

    Supported CAD extensions: ``.step``, ``.stp``, ``.iges``, ``.igs``,
    ``.x_t``, ``.x_b``, ``.sat``, ``.ipt``, ``.prt``, ``.sldprt``,
    ``.catpart``.

    ZIP limits: 500 MB total uncompressed size, 50 CAD files per archive.
    """
    try:
        resolved_path = core.get_shared_CAD_file(file_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    suffix = resolved_path.suffix.lower()

    if suffix == ".zip":
        try:
            zip_data = resolved_path.read_bytes()
            uploaded, errors = core.extract_zip_cad_files(zip_data)
        except core.ZipSlipError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (core.ZipSizeLimitError, core.ZipFileLimitError) as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to process ZIP archive: {exc}"
            ) from exc

        return {
            "files": [
                {"file_id": fid, "filename": name} for fid, name in uploaded
            ],
            "errors": errors,
        }

    if suffix in core.CAD_ALLOWED_EXTENSIONS:
        try:
            fake = core._BytesUploadFile(resolved_path.read_bytes(), resolved_path.name)
            file_id, dest, existed = core.upload_CAD_file_persistent(fake)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "files": [
                {
                    "file_id": file_id,
                    "filename": resolved_path.name,
                    "already_existed": existed,
                }
            ],
            "errors": [],
        }

    allowed = ", ".join(sorted(core.CAD_ALLOWED_EXTENSIONS | {".zip"}))
    raise HTTPException(
        status_code=422,
        detail=(
            f"Unsupported file type '{suffix}'. "
            f"Allowed extensions: {allowed}"
        ),
    )


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
