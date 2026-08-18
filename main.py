from contextlib import asynccontextmanager
import logging
import mimetypes
import os
import pathlib

mimetypes.init()
mimetypes.add_type("application/javascript", ".mjs")

import core
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from routers import brep, cad, context_layer, files, mfr, part_classification, similarity


def _configure_logging() -> None:
    """Give the application's own loggers a handler and a level.

    uvicorn only configures its own ``uvicorn.*`` loggers, so without this the
    root logger keeps its WARNING default and every ``logger.info()`` in core
    and the routers is silently dropped -- including the diagnostics that report
    how long expensive one-off work took. ``force=True`` so we win even when an
    embedded runner (or a library import) has already installed a root handler.
    """
    core.load_env_file()
    level = os.environ.get("HOOPS_AI_LOG_LEVEL", "INFO").strip().upper()
    if not isinstance(logging.getLevelName(level), int):
        level = "INFO"
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )


_configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Process-local caches only. The on-disk directories are NOT wiped here:
    # uploads/ is the CAD store backing every registered index, and deleting
    # out/ or embeddings_cache/ would also clobber a second instance running
    # on another port. Transient directories are TTL-swept instead.
    core._embedding_memory_cache.clear()
    core.CAD_viewers.clear()
    core.CAD_face_colors.clear()
    core.run_startup_maintenance()
    core.init_hoops_license()
    yield
    if core.MFR_dataset_explorer is not None and hasattr(core.MFR_dataset_explorer, "close"):
        core.MFR_dataset_explorer.close()
    if core.PART_CLASS_dataset_explorer is not None and hasattr(core.PART_CLASS_dataset_explorer, "close"):
        core.PART_CLASS_dataset_explorer.close()
    core.CAD_viewers.clear()


app = FastAPI(
    title="HOOPS AI File Search API",
    lifespan=lifespan,
)


@app.exception_handler(core.EnvConfigError)
async def env_config_error_handler(request: Request, exc: core.EnvConfigError):
    return JSONResponse(
        status_code=503,
        content={"detail": "Service unavailable: server configuration error. Check the server logs for details."},
    )


@app.exception_handler(core.PathConfigError)
async def path_config_error_handler(request: Request, exc: core.PathConfigError):
    return JSONResponse(
        status_code=503,
        content={"detail": "Service unavailable: required server resource not found. Check the server logs for details."},
    )

app.include_router(files.router)
app.include_router(mfr.router)
app.include_router(cad.router)
app.include_router(brep.router)
app.include_router(similarity.router)
app.include_router(part_classification.router)
app.include_router(context_layer.router)

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

