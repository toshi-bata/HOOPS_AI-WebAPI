from typing import Optional

import core
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

router = APIRouter(prefix="/BRep", tags=["BRep"])


@router.post("/adjacency-graph")
def brep_adjacency_graph(
    request: Request,
    file: Optional[UploadFile] = File(None),
    file_id: Optional[str] = Query(None, description="file_id returned by POST /files/upload"),
):
    """Build a face adjacency graph from the B-rep model of a CAD file.

    Supply **either** a file upload *or* a ``file_id`` from a previous upload.
    """
    try:
        if file_id:
            cad_file_path = core.find_persistent_CAD_file(file_id)
        elif file:
            _, cad_file_path, _ = core.upload_CAD_file_persistent(file)
        else:
            raise HTTPException(status_code=422, detail="Either 'file' or 'file_id' is required.")
        result = core.build_brep_adjacency_graph(cad_file_path)
        if result.get("image_url"):
            result["image_url"] = str(request.base_url).rstrip("/") + result["image_url"]
        return result
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"BRep encoding failed: {exc}") from exc


@router.post("/attributes")
def brep_attributes(
    file: Optional[UploadFile] = File(None),
    file_id: Optional[str] = Query(None, description="file_id returned by POST /files/upload"),
):
    """Extract face and edge attributes from the B-rep model of a CAD file.

    Supply **either** a file upload *or* a ``file_id`` from a previous upload.
    """
    try:
        if file_id:
            cad_file_path = core.find_persistent_CAD_file(file_id)
        elif file:
            _, cad_file_path, _ = core.upload_CAD_file_persistent(file)
        else:
            raise HTTPException(status_code=422, detail="Either 'file' or 'file_id' is required.")
        return core.get_brep_attributes(cad_file_path)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"BRep attribute extraction failed: {exc}") from exc


@router.post("/type-counts")
def brep_type_counts(
    file: Optional[UploadFile] = File(None),
    file_id: Optional[str] = Query(None, description="file_id returned by POST /files/upload"),
):
    """Return face and edge counts grouped by type, aggregated server-side.

    Use this endpoint instead of ``/BRep/attributes`` whenever only counts per type
    (e.g. "how many cylindrical faces", "how many faces and edges in total") are
    needed — it returns small, pre-aggregated totals rather than a few hundred raw
    per-face/per-edge entries.

    Supply **either** a file upload *or* a ``file_id`` from a previous upload.
    """
    try:
        if file_id:
            cad_file_path = core.find_persistent_CAD_file(file_id)
        elif file:
            _, cad_file_path, _ = core.upload_CAD_file_persistent(file)
        else:
            raise HTTPException(status_code=422, detail="Either 'file' or 'file_id' is required.")
        return core.get_brep_type_counts(cad_file_path)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"BRep type-count aggregation failed: {exc}") from exc

