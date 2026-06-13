# C:\Users\toshi\miniconda3\envs\hoops_ai_cpu\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
# C:\Users\toshi\miniconda3\envs\hoops_ai_cpu\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload

from contextlib import asynccontextmanager

import core
from fastapi import FastAPI
from routers import cad, mfr


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

app.include_router(mfr.router)
app.include_router(cad.router)


