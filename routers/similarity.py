import io
import os
import pathlib
import zipfile
from typing import Any, List, Optional

import core
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/similarity", tags=["CAD Similarity Search"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class SimilarSearchIndexInfo(BaseModel):
    """Metadata about the loaded FAISS similarity-search index.

    ``status`` is ``"loaded"`` when the index has been initialised, or
    ``"not_loaded"`` when no index has been loaded yet.
    All other fields are ``null`` / ``None`` when the index is not loaded.
    """

    status: str
    index_path: Optional[str] = None
    index_last_modified: Optional[str] = None
    index_count: Optional[int] = None
    model_name: Optional[str] = None
    embedding_dim: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None


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
    except (core.EnvConfigError, core.PathConfigError):
        raise
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


@router.get("/index-info", response_model=SimilarSearchIndexInfo)
def similarity_index_info():
    """Return metadata about the currently loaded FAISS similarity-search index.

    This is a read-only, inference-side endpoint — it never triggers index
    construction or model training.  When the index has not been loaded yet
    the response contains ``status: "not_loaded"`` and ``null`` values for all
    other fields; no error is raised.
    """
    try:
        return core.get_similar_search_index_info()
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# Response schemas – part similarity compare
# ---------------------------------------------------------------------------


class EmbedResponse(BaseModel):
    """Result of computing a shape embedding for a single CAD part."""

    file_id: str
    filename: str
    dim: int
    model_name: str
    num_bodies: int
    cached: bool
    vector: Optional[list[float]] = None


class CompareFileInfo(BaseModel):
    index: int
    file_id: str
    filename: str
    num_bodies: int


class ComparePair(BaseModel):
    a: int
    b: int
    score: float


class CompareError(BaseModel):
    filename: str
    detail: str


class CompareResponse(BaseModel):
    """Pairwise cosine similarity matrix for a set of CAD parts."""

    count: int
    model_name: str
    files: list[CompareFileInfo]
    matrix: list[list[float]]
    pairs: list[ComparePair]
    errors: list[CompareError]


# ---------------------------------------------------------------------------
# Limits for ZIP extraction
# ---------------------------------------------------------------------------

_ZIP_MAX_TOTAL_BYTES = 500 * 1024 * 1024  # 500 MB uncompressed
_ZIP_MAX_FILES = 50


class _BytesUploadFile:
    """Minimal duck-typed UploadFile used to pass in-memory bytes to core helpers."""

    def __init__(self, data: bytes, filename: str) -> None:
        self.filename = filename
        self.file = io.BytesIO(data)


# ---------------------------------------------------------------------------
# POST /similarity/embed
# ---------------------------------------------------------------------------


@router.post("/embed", response_model=EmbedResponse)
def similarity_embed(
    file: Optional[UploadFile] = File(None),
    file_id: Optional[str] = Query(None, description="file_id returned by POST /files/upload"),
    include_vector: bool = Query(
        False,
        description="When true, include the raw embedding vector in the response. "
        "Omit (default) to save bandwidth.",
    ),
):
    """Compute (or retrieve from cache) the shape embedding for a CAD part.

    Supply **either** a file upload *or* a ``file_id`` from a previous upload.
    Results are cached in memory and on disk — repeated calls for the same file
    are fast.  Set ``include_vector=true`` to include the raw float array.

    This endpoint does **not** require a FAISS index — the embedding model alone
    is sufficient.
    """
    try:
        if file_id:
            resolved_id = file_id
        elif file:
            resolved_id, _, _ = core.upload_CAD_file_persistent(file)
        else:
            raise HTTPException(status_code=422, detail="Either 'file' or 'file_id' is required.")

        result = core.compute_embedding(resolved_id)
        return EmbedResponse(
            file_id=result["file_id"],
            filename=result["filename"],
            dim=result["dim"],
            model_name=result["model_name"],
            num_bodies=result["num_bodies"],
            cached=result["cached"],
            vector=[float(v) for v in result["vector"]] if include_vector else None,
        )
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /similarity/compare
# ---------------------------------------------------------------------------


@router.post("/compare", response_model=CompareResponse)
def similarity_compare(
    file_ids: Optional[str] = Query(
        None,
        description="Comma-separated file_ids of already-uploaded CAD files.",
    ),
    files: Optional[List[UploadFile]] = File(None),
    zip_file: Optional[UploadFile] = File(None),
):
    """Compare multiple CAD parts by cosine similarity of their shape embeddings.

    Input sources can be combined freely:

    * ``file_ids`` – comma-separated ``file_id`` values from previous uploads
    * ``files`` – one or more multipart CAD file uploads
    * ``zip_file`` – a single ZIP archive containing CAD files (auto-extracted)

    At least **two** valid parts are required overall.  Per-file embed failures
    are collected in ``errors`` and do not abort the whole request, unless fewer
    than two parts succeed (returns 422 in that case).

    ZIP archives are extracted safely (Zip Slip paths are rejected) and filtered
    to recognised CAD extensions only.  Uncompressed size is capped at 500 MB
    and file count at 50.

    This endpoint does **not** require a FAISS index.
    """
    errors: list[dict] = []
    # (file_id, display_filename) tuples collected from all input sources
    resolved: list[tuple[str, str]] = []

    # ── 1. file_ids from query parameter ─────────────────────────────────────
    if file_ids:
        for fid in [f.strip() for f in file_ids.split(",") if f.strip()]:
            try:
                path = core.find_persistent_CAD_file(fid)
                parts = path.name.split("_", 1)
                display = parts[1] if len(parts) == 2 and len(parts[0]) == 64 else path.name
                resolved.append((fid, display))
            except RuntimeError as exc:
                errors.append({"filename": fid, "detail": str(exc)})

    # ── 2. Direct file uploads ────────────────────────────────────────────────
    if files:
        for upload in files:
            if not (upload and upload.filename):
                continue
            try:
                fid, _, _ = core.upload_CAD_file_persistent(upload)
                resolved.append((fid, upload.filename))
            except Exception as exc:
                errors.append({"filename": upload.filename or "unknown", "detail": str(exc)})

    # ── 3. ZIP archive ────────────────────────────────────────────────────────
    if zip_file and zip_file.filename:
        try:
            zip_data = zip_file.file.read()
            _process_zip(zip_data, resolved, errors)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to process ZIP archive: {exc}"
            ) from exc

    # ── Require at least 2 inputs ─────────────────────────────────────────────
    if len(resolved) < 2:
        msg = (
            f"At least 2 CAD files are required for comparison. "
            f"Received {len(resolved)} valid input(s)"
            + (f" and {len(errors)} failure(s)." if errors else ".")
        )
        raise HTTPException(status_code=422, detail=msg)

    # ── 4. Compute embeddings (collect per-file errors) ───────────────────────
    valid_ids: list[str] = []
    for fid, display in resolved:
        try:
            core.compute_embedding(fid)
            valid_ids.append(fid)
        except Exception as exc:
            errors.append({"filename": display, "detail": str(exc)})

    if len(valid_ids) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"At least 2 successful embeddings are required. "
            f"{len(errors)} file(s) failed to embed.",
        )

    # ── 5. Compare ────────────────────────────────────────────────────────────
    try:
        result = core.compare_embeddings(valid_ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}") from exc

    return CompareResponse(
        count=result["count"],
        model_name=result["model_name"],
        files=[CompareFileInfo(**f) for f in result["files"]],
        matrix=result["matrix"],
        pairs=[ComparePair(**p) for p in result["pairs"]],
        errors=[CompareError(**e) for e in errors],
    )



# ===========================================================================
# Named index management endpoints
# ===========================================================================

# ---------------------------------------------------------------------------
# Response schemas – index management
# ---------------------------------------------------------------------------


class IndexInfo(BaseModel):
    name: str
    count: Optional[int] = None
    last_modified: Optional[str] = None
    is_readonly: bool = False


class IndexCreateResponse(BaseModel):
    name: str
    count: int
    dim: int


class IndexAddResponse(BaseModel):
    name: str
    added: int
    updated: int
    index_count: int
    errors: list[dict]


class IndexRemoveResponse(BaseModel):
    name: str
    removed: int
    index_count: int


class IndexSearchHit(BaseModel):
    id: str
    score: float
    metadata: Optional[dict] = None


class IndexSearchResponse(BaseModel):
    hits: list[IndexSearchHit]
    count: int


class IndexDeleteResponse(BaseModel):
    name: str
    deleted: bool


# ---------------------------------------------------------------------------
# POST /similarity/index/create
# ---------------------------------------------------------------------------


@router.post("/index/create", response_model=IndexCreateResponse, status_code=201)
def create_index(
    name: str = Query(..., description="Name of the new index (^[a-z0-9_-]{1,64}$)."),
):
    """Create a new empty named similarity index.

    * ``name`` must match ``^[a-z0-9_-]{1,64}$``.
    * ``default`` is reserved for the read-only env-configured index and cannot be created.
    * Returns **409** if an index with that name already exists.
    * Returns **422** if the name is invalid or reserved.
    """
    try:
        result = core.create_index(name)
        return IndexCreateResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create index: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /similarity/index/list
# ---------------------------------------------------------------------------


@router.get("/index/list", response_model=list[IndexInfo])
def list_indexes():
    """Return metadata for all known similarity indexes.

    The built-in ``default`` index (backed by ``HOOPS_AI_FAISS_INDEX_PATH``) is always
    included with ``is_readonly: true``.  All user-created indexes follow with
    ``is_readonly: false``.
    """
    try:
        return [IndexInfo(**entry) for entry in core.list_indexes()]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list indexes: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /similarity/index/add
# ---------------------------------------------------------------------------


@router.post("/index/add", response_model=IndexAddResponse)
def add_to_index(
    name: str = Query(..., description="Target index name."),
    file_ids: Optional[str] = Query(
        None,
        description="Comma-separated file_ids of already-uploaded CAD files.",
    ),
    files: Optional[List[UploadFile]] = File(None),
    zip_file: Optional[UploadFile] = File(None),
):
    """Register CAD parts in the named index.

    Input sources can be combined freely (same rules as ``POST /similarity/compare``):

    * ``file_ids`` – comma-separated ``file_id`` values from previous uploads
    * ``files`` – one or more multipart CAD file uploads
    * ``zip_file`` – a single ZIP archive (auto-extracted, Zip Slip protected)

    Re-registering an existing part ID overwrites the old entry (no duplicates).
    ``added`` counts new parts; ``updated`` counts overwritten parts.
    """
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise

    errors: list[dict] = []
    resolved: list[tuple[str, str]] = []

    if file_ids:
        for fid in [f.strip() for f in file_ids.split(",") if f.strip()]:
            try:
                path = core.find_persistent_CAD_file(fid)
                parts = path.name.split("_", 1)
                display = parts[1] if len(parts) == 2 and len(parts[0]) == 64 else path.name
                resolved.append((fid, display))
            except RuntimeError as exc:
                errors.append({"file_id": fid, "detail": str(exc)})

    if files:
        for upload in files:
            if not (upload and upload.filename):
                continue
            try:
                fid, _, _ = core.upload_CAD_file_persistent(upload)
                resolved.append((fid, upload.filename))
            except Exception as exc:
                errors.append({"filename": upload.filename or "unknown", "detail": str(exc)})

    if zip_file and zip_file.filename:
        try:
            zip_data = zip_file.file.read()
            _process_zip(zip_data, resolved, errors)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to process ZIP archive: {exc}"
            ) from exc

    if not resolved and not errors:
        raise HTTPException(
            status_code=422,
            detail="No valid input provided. Supply file_ids, files, or zip_file.",
        )

    all_file_ids = [fid for fid, _ in resolved]

    try:
        result = core.add_to_index(name, all_file_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to add to index: {exc}") from exc

    # merge router-level resolve errors into core-level errors
    result["errors"] = errors + result.get("errors", [])
    return IndexAddResponse(**result)


# ---------------------------------------------------------------------------
# POST /similarity/index/{name}/search
# ---------------------------------------------------------------------------


@router.post("/index/{name}/search", response_model=IndexSearchResponse)
def search_named_index(
    name: str,
    file: Optional[UploadFile] = File(None),
    file_id: Optional[str] = Query(None, description="file_id from a previous upload."),
    top_k: int = Query(10, ge=1, description="Number of similar shapes to return."),
):
    """Search a named index for the most similar parts to a query shape.

    Supply **either** a file upload *or* a ``file_id``.  Returns an empty ``hits``
    list when the index contains zero entries (no error).
    """
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise

    try:
        if file_id:
            resolved_id = file_id
        elif file:
            resolved_id, _, _ = core.upload_CAD_file_persistent(file)
        else:
            raise HTTPException(status_code=422, detail="Either 'file' or 'file_id' is required.")

        result = core.search_index(name, resolved_id, top_k)
        return IndexSearchResponse(
            hits=[IndexSearchHit(**h) for h in result["hits"]],
            count=result["count"],
        )
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc


# ---------------------------------------------------------------------------
# DELETE /similarity/index/{name}/parts
# ---------------------------------------------------------------------------


@router.delete("/index/{name}/parts", response_model=IndexRemoveResponse)
def remove_parts_from_index(
    name: str,
    part_ids: str = Query(..., description="Comma-separated part IDs (file_ids) to remove."),
):
    """Remove registered parts from a named index by their part IDs (file_ids).

    Returns **403** when targeting the read-only ``default`` index.
    """
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise

    ids = [p.strip() for p in part_ids.split(",") if p.strip()]
    if not ids:
        raise HTTPException(status_code=422, detail="No part_ids supplied.")

    try:
        result = core.remove_from_index(name, ids)
        return IndexRemoveResponse(**result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Remove failed: {exc}") from exc


# ---------------------------------------------------------------------------
# DELETE /similarity/index/{name}
# ---------------------------------------------------------------------------


@router.delete("/index/{name}", response_model=IndexDeleteResponse)
def delete_index(
    name: str,
    confirm: bool = Query(
        False,
        description="Must be ``true`` to confirm destructive deletion.",
    ),
):
    """Delete a named index and all its on-disk data.

    * Requires ``?confirm=true``; without it returns **409** with an instruction.
    * Returns **403** for the read-only ``default`` index.
    * Returns **404** when the index does not exist.
    """
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise

    if not confirm:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Deletion of index '{name}' is destructive and irreversible. "
                "Re-send this request with ?confirm=true to proceed."
            ),
        )

    try:
        result = core.delete_index(name)
        return IndexDeleteResponse(**result)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Internal helper – validate index name at router boundary
# ---------------------------------------------------------------------------


def _validate_name_or_raise(name: str) -> None:
    """Convert core validation errors to HTTPExceptions at the router boundary."""
    try:
        core._validate_index_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


# ===========================================================================
# ZIP extraction helper (shared by /compare and /index/add)
# ===========================================================================


def _process_zip(
    zip_data: bytes,
    resolved: list[tuple[str, str]],
    errors: list[dict],
) -> None:
    """Extract a ZIP archive and upload each valid CAD file persistently.

    Raises ``HTTPException(400)`` on Zip Slip and ``HTTPException(413)`` when
    the uncompressed size or file count exceeds the configured limits.
    """
    import tempfile

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
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Zip Slip detected: '{member.filename}' "
                            "resolves outside the extraction directory."
                        ),
                    )

                suffix = pathlib.Path(member.filename).suffix.lower()
                if suffix not in core.CAD_ALLOWED_EXTENSIONS:
                    continue  # skip non-CAD entries silently

                total_size += member.file_size
                if total_size > _ZIP_MAX_TOTAL_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"ZIP archive total uncompressed size exceeds the "
                            f"{_ZIP_MAX_TOTAL_BYTES // (1024 * 1024)} MB limit."
                        ),
                    )

                file_count += 1
                if file_count > _ZIP_MAX_FILES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"ZIP archive contains more than {_ZIP_MAX_FILES} CAD files.",
                    )

                member_dest.parent.mkdir(parents=True, exist_ok=True)
                member_dest.write_bytes(zf.read(member.filename))

                display_name = pathlib.Path(member.filename).name
                try:
                    fake = _BytesUploadFile(member_dest.read_bytes(), display_name)
                    fid, _, _ = core.upload_CAD_file_persistent(fake)
                    resolved.append((fid, display_name))
                except Exception as exc:
                    errors.append({"filename": display_name, "detail": str(exc)})


