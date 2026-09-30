from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.delivery.dependencies import build_services
from src.delivery.routes.inspections import router as inspections_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = build_services()
    yield
    
app = FastAPI(
    title="UAV PV API",
    version="1.0.0",
    lifespan = lifespan
)

prefix = "/api/v1"
app.include_router(inspections_router,prefix=prefix)