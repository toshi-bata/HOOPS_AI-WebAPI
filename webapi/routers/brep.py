import core
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/BRep", tags=["BRep"])


@router.post("/adjacency-graph")
def brep_adjacency_graph(file: UploadFile = File(...)):
    cad_file_path = None
    try:
        cad_file_path = core.save_uploaded_CAD_file(file)
        return core.build_brep_adjacency_graph(cad_file_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"BRep encoding failed: {exc}") from exc
    finally:
        if cad_file_path is not None:
            cad_file_path.unlink(missing_ok=True)


@router.post("/attributes")
def brep_attributes(file: UploadFile = File(...)):
    cad_file_path = None
    try:
        cad_file_path = core.save_uploaded_CAD_file(file)
        return core.get_brep_attributes(cad_file_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"BRep attribute extraction failed: {exc}") from exc
    finally:
        if cad_file_path is not None:
            cad_file_path.unlink(missing_ok=True)
