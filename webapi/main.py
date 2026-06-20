from contextlib import asynccontextmanager
import mimetypes
import pathlib

mimetypes.init()
mimetypes.add_type("application/javascript", ".mjs")

import core
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers import brep, cad, files, mfr, similarity


@asynccontextmanager
async def lifespan(app: FastAPI):
    import shutil
    for folder in (core.CAD_UPLOAD_DIR, core.CAD_VIEWER_OUTPUT_DIR):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    core.init_hoops_license()
    yield
    if core.MFR_dataset_explorer is not None and hasattr(core.MFR_dataset_explorer, "close"):
        core.MFR_dataset_explorer.close()
    core.CAD_viewers.clear()


app = FastAPI(
    title="HOOPS AI File Search API",
    lifespan=lifespan,
)

app.include_router(files.router)
app.include_router(mfr.router)
app.include_router(cad.router)
app.include_router(brep.router)
app.include_router(similarity.router)

core.CAD_VIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/out", StaticFiles(directory=str(core.CAD_VIEWER_OUTPUT_DIR)), name="out")

_static_dir = pathlib.Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


if __name__ == "__main__":
    import argparse
    import socket
    import sys
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the HOOPS AI WebAPI server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        if _s.connect_ex(("127.0.0.1", args.port)) == 0:
            print(
                f"Error: port {args.port} is already in use. "
                f"Use --port <number> to specify a different port.",
                file=sys.stderr,
            )
            sys.exit(1)

    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)

