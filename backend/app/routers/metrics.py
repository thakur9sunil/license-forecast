from fastapi import APIRouter, HTTPException

import mlflow
from mlflow import MlflowClient

from app.config import settings
from app.models.schemas import ModelMetrics

router = APIRouter(prefix="/model-metrics", tags=["metrics"])


def _fetch_metrics_from_mlflow() -> list[ModelMetrics]:
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    results = []

    for product in ["Jira", "Slack", "Zoom"]:
        model_name = f"license_forecast_{product.lower()}"
        try:
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                versions = client.get_latest_versions(model_name)
            if not versions:
                continue
            v = versions[0]
            run = client.get_run(v.run_id)
            m = run.data.metrics
            results.append(ModelMetrics(
                product=product,
                model_type="Prophet",
                mae=round(m.get("mae", 0.0), 4),
                rmse=round(m.get("rmse", 0.0), 4),
                mape=round(m.get("mape", 0.0), 4),
                r2=round(m.get("r2", 0.0), 4),
                training_date=str(run.info.start_time),
                mlflow_run_id=v.run_id,
                model_version=str(v.version),
            ))
        except Exception:
            continue

    return results


@router.get("/", response_model=list[ModelMetrics])
async def get_all_metrics() -> list[ModelMetrics]:
    try:
        return _fetch_metrics_from_mlflow()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {e}")


@router.get("/{product_name}", response_model=ModelMetrics)
async def get_product_metrics(product_name: str) -> ModelMetrics:
    if product_name not in ("Jira", "Slack", "Zoom"):
        raise HTTPException(status_code=404, detail=f"Unknown product '{product_name}'")
    try:
        all_metrics = _fetch_metrics_from_mlflow()
        for m in all_metrics:
            if m.product == product_name:
                return m
        raise HTTPException(status_code=404, detail=f"No metrics found for '{product_name}'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {e}")
