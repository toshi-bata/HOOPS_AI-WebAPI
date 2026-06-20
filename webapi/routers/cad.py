import core
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from typing import Optional

router = APIRouter(prefix="/CAD", tags=["CAD Viewer"])


def _get_session_id(request: Request) -> Optional[str]:
    return request.headers.get("X-Session-ID") or None


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


@router.get("/viewer/colors")
def CAD_viewer_colors(scs: str = Query(..., description="SCS filename")):
    """Return face color data for the given SCS file (if colorize has been called)."""
    colors = core.CAD_face_colors.get(scs)
    return {"colors": colors}


@router.get("/viewer/show", response_class=HTMLResponse)
def CAD_viewer_show(scs: str = Query(..., description="SCS filename in out/ directory")):
    """Serve the HOOPS web viewer HTML page for the given SCS file."""
    return HTMLResponse(content=f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HOOPS CAD Viewer</title>
  <style>
    html, body {{ margin: 0; padding: 0; overflow: hidden; }}
    #viewerContainer {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <div id="viewerContainer"></div>
  <div id="errorMsg" style="display:none;position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
    background:#1e1e2e;color:#f8f8f2;padding:2em 2.5em;border-radius:10px;border:1px solid #ff5555;
    font-family:sans-serif;max-width:480px;text-align:center;z-index:999;">
    <h2 style="color:#ff5555;margin:0 0 .5em">WebGL Error</h2>
    <p id="errorDetail" style="margin:0 0 1em;font-size:.95em"></p>
    <details style="text-align:left;font-size:.85em;color:#aaa">
      <summary style="cursor:pointer;color:#ccc">Checklist for Firefox users</summary>
      <ul style="margin:.5em 0 0 1em;line-height:1.8">
        <li>Type <code>about:config</code> in the address bar</li>
        <li><code>webgl.disabled</code> → must be <strong>false</strong></li>
        <li><code>privacy.resistFingerprinting</code> → must be <strong>false</strong></li>
        <li>Settings → General → Performance → "Use hardware acceleration" must be enabled</li>
      </ul>
    </details>
  </div>
  <script type="module">
    // WebGL availability check
    (function checkWebGL() {{
      const testCanvas = document.createElement('canvas');
      const gl = testCanvas.getContext('webgl2') || testCanvas.getContext('webgl') || testCanvas.getContext('experimental-webgl');
      if (!gl) {{
        const msg = document.getElementById('errorMsg');
        document.getElementById('errorDetail').textContent =
          'WebGL is not available in this browser. WebGL may be disabled in Firefox settings, or hardware acceleration may be turned off.';
        msg.style.display = 'block';
        return false;
      }}
      return true;
    }})();

    import {{ WebViewer, Color }} from '/static/hoops-web-viewer-monolith.mjs';

    const scsFile = {repr(scs)};
    const container = document.getElementById('viewerContainer');
    const hwv = new WebViewer({{
      container,
      endpointUri: '/out/' + scsFile,
    }});

    hwv.setCallbacks({{
      sceneReady: function () {{
        hwv.focusInput(true);
        container.addEventListener('mouseenter', () => hwv.focusInput(true));
        window.addEventListener('resize', () => hwv.resizeCanvas());

        if (hwv.view.getAxisTriad) hwv.view.getAxisTriad().enable();
        if (hwv.view.getNavCube) hwv.view.getNavCube().enable();
      }},
      modelStructureReady: async function () {{
        const res = await fetch('/CAD/viewer/colors?scs=' + scsFile);
        const data = await res.json();
        if (!data.colors || data.colors.length === 0) return;

        const model = hwv.model;
        const rootNode = model.getAbsoluteRootNode();
        const children = model.getNodeChildren(rootNode);
        if (children.length === 0) return;
        const modelNodeId = children[0];

        data.colors.forEach((rgb, faceId) => {{
          if (rgb) {{
            model.setNodeFaceColor(modelNodeId, faceId, new Color(rgb[0], rgb[1], rgb[2]));
          }}
        }});
      }},
    }});

    hwv.start().catch(err => {{
      const msg = document.getElementById('errorMsg');
      document.getElementById('errorDetail').textContent = err.message || String(err);
      msg.style.display = 'block';
    }});
  </script>
</body>
</html>""")


def _resolve_urls(result: dict, base_url: str) -> dict:
    for key in ("viewer_url", "image_url"):
        if result.get(key) and result[key].startswith("/"):
            result[key] = base_url.rstrip("/") + result[key]
    result.pop("_scs_filename", None)
    return result


@router.post("/viewer")
def CAD_viewer(
    request: Request,
    file: Optional[UploadFile] = File(None),
    file_id: Optional[str] = Query(None, description="file_id returned by POST /files/upload"),
):
    """Open a CAD file in the viewer.

    Supply **either** a file upload *or* a ``file_id`` from a previous upload.
    """
    try:
        if file_id:
            cad_file_path = core.find_persistent_CAD_file(file_id)
        elif file:
            _, cad_file_path, _ = core.upload_CAD_file_persistent(file)
        else:
            raise HTTPException(status_code=422, detail="Either 'file' or 'file_id' is required.")
        return _resolve_urls(core.create_CAD_viewer(cad_file_path, _get_session_id(request)), str(request.base_url))
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CADViewer failed: {exc}") from exc


@router.delete("/viewer")
def CAD_viewer_terminate(request: Request, all: bool = False):
    try:
        return core.terminate_CAD_viewer(session_id=_get_session_id(request), terminate_all=all)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Terminate failed: {exc}") from exc


@router.post("/viewer/from-path")
def CAD_viewer_from_path(request: Request, cad_file_path: str = Form(...)):
    try:
        return _resolve_urls(core.create_CAD_viewer(core.get_shared_CAD_file(cad_file_path), _get_session_id(request)), str(request.base_url))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CADViewer failed: {exc}") from exc

