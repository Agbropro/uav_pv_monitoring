from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from src.delivery.dependencies import build_services
from src.delivery.routes.inspections import router as inspections_router
from src.delivery.routes.sites import router as sites_router
from src.infrastructure.loguru_logger import LoguruLogger, configure_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    startup_logger = LoguruLogger("STARTUP")
    startup_logger.info("Backend starting")
    try:
        app.state.services = build_services()
    except Exception:
        startup_logger.exception("Backend startup failed")
        raise
    startup_logger.info("Backend ready")
    try:
        yield
    finally:
        startup_logger.info("Backend stopped")
    
app = FastAPI(
    title="UAV PV API",
    version="1.0.0",
    lifespan = lifespan
)

@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    app.state.services.api_logger.log(
        "ERROR",
        f"Unexpected request failure | method={request.method} | exception_type={type(exc).__name__}",
        process_id=getattr(request.state, "process_id", None),
        exception=exc,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

prefix = "/api/v1"
app.include_router(inspections_router,prefix=prefix)
app.include_router(sites_router,prefix=prefix)
app.mount(
    "/dashboard",
    StaticFiles(directory=Path(__file__).resolve().parents[1] / "frontend", html=True),
    name="dashboard",
)
