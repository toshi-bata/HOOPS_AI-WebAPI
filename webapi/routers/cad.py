import core
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/CAD", tags=["CAD Viewer"])


@router.get("/viewer", response_class=HTMLResponse)
def CAD_viewer_upload_page():
    CAD_shared_dir = core.get_CAD_shared_dir()
    return f"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>CAD Viewer</title>
    <style>
        body {{ font-family: Segoe UI, sans-serif; margin: 2rem; max-width: 760px; }}
        form {{ border: 1px solid #ddd; padding: 1rem; margin: 1rem 0; }}
        label, input, button {{ display: block; margin: 0.5rem 0; }}
        input[type="text"] {{ width: 100%; box-sizing: border-box; }}
    </style>
</head>
<body>
    <h1>CAD Viewer</h1>
    <p>Shared folder: {CAD_shared_dir}</p>
    <form action="/CAD/viewer" method="post" enctype="multipart/form-data">
        <label for="file">Upload CAD file</label>
        <input id="file" name="file" type="file" required>
        <button type="submit">Open viewer</button>
    </form>
    <form action="/CAD/viewer/from-path" method="post">
        <label for="cad_file_path">CAD file path in shared folder</label>
        <input id="cad_file_path" name="cad_file_path" type="text" required>
        <button type="submit">Open viewer from path</button>
    </form>
</body>
</html>
"""


@router.post("/viewer")
def CAD_viewer(file: UploadFile = File(...)):
    try:
        cad_file_path = core.save_uploaded_CAD_file(file)
        return {"viewer_url": core.create_CAD_viewer(cad_file_path)}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CADViewer failed: {exc}") from exc


@router.delete("/viewer")
def CAD_viewer_terminate(all: bool = False):
    try:
        return core.terminate_CAD_viewer(terminate_all=all)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Terminate failed: {exc}") from exc


@router.post("/viewer/from-path")
def CAD_viewer_from_path(cad_file_path: str = Form(...)):
    try:
        return {"viewer_url": core.create_CAD_viewer(core.get_shared_CAD_file(cad_file_path))}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CADViewer failed: {exc}") from exc
