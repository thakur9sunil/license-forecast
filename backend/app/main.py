import logging
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.loader import init_loader, get_loader
from app.models.schemas import HealthResponse
from app.routers import forecast, products, metrics

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading models...")
    loader = init_loader(
        tracking_uri=settings.mlflow_tracking_uri,
        cache_dir=settings.model_cache_dir,
    )
    loader.reload_all()
    loaded = loader.health_check()
    logger.info(f"Models loaded: {loaded}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="License Usage Forecasting API",
    description="Forecast software license needs before contract renewals.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router)
app.include_router(products.router)
app.include_router(metrics.router)


def _check_mlflow() -> bool:
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        from mlflow import MlflowClient
        MlflowClient().search_experiments()
        return True
    except Exception:
        return False


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    try:
        loader = get_loader()
        models_loaded = loader.health_check()
    except RuntimeError:
        models_loaded = {}

    all_loaded = bool(models_loaded) and all(models_loaded.values())
    mlflow_ok = _check_mlflow()

    if all_loaded and mlflow_ok:
        status = "healthy"
    elif all_loaded or mlflow_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        mlflow_connected=mlflow_ok,
        version="1.0.0",
    )
