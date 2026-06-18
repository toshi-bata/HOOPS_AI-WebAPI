from contextlib import asynccontextmanager

import core
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers import brep, cad, files, mfr, similarity


@asynccontextmanager
async def lifespan(app: FastAPI):
    core.init_hoops_license()
    yield
    if core.MFR_dataset_explorer is not None and hasattr(core.MFR_dataset_explorer, "close"):
        core.MFR_dataset_explorer.close()


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

