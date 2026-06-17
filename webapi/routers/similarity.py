import core
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

router = APIRouter(prefix="/similarity", tags=["CAD Similarity Search"])


@router.post("/search")
def similarity_search(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(10, ge=1, description="Number of similar shapes to return."),
):
    cad_file_path = None
    try:
        cad_file_path = core.save_uploaded_CAD_file(file)
        result = core.search_by_shape(cad_file_path, top_k=top_k)
        image_filename = result["image_url"].lstrip("/out/")
        result["image_url"] = str(request.url_for("out", path=image_filename))
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {exc}") from exc
    finally:
        if cad_file_path is not None:
            cad_file_path.unlink(missing_ok=True)
