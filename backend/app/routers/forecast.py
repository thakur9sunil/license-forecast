from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from app.models.loader import ModelLoader, get_loader
from app.models.schemas import ForecastRequest, ForecastResponse
from app.services.forecaster import generate_forecast
from app.services.recommender import compute_recommendation, PRODUCT_LICENSE_COUNTS

router = APIRouter(prefix="/forecast", tags=["forecast"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def _load_historical(product: str) -> pd.DataFrame:
    csv_path = DATA_DIR / "license_usage.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=503, detail="Training data not found. Run data_generator first.")
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    product_df = df[df["product"] == product][["ds", "y"]].copy()
    if product_df.empty:
        raise HTTPException(status_code=404, detail=f"No data for product '{product}'")
    return product_df


def _build_response(
    product: str,
    horizon_months: int,
    forecast_result: dict,
    loader: ModelLoader,
) -> ForecastResponse:
    forecast_points = forecast_result["forecast"]
    renewal_yhat_upper = forecast_points[-1]["yhat_upper"] if forecast_points else 0.0
    recommendation, detail = compute_recommendation(
        product=product,
        predicted_at_renewal=forecast_result["predicted_at_renewal"],
        yhat_upper=renewal_yhat_upper,
        current_licensed=PRODUCT_LICENSE_COUNTS.get(product),
    )

    return ForecastResponse(
        product=product,
        horizon_months=horizon_months,
        historical=forecast_result["historical"],
        forecast=forecast_points,
        trend_direction=forecast_result["trend_direction"],
        recommendation=recommendation,
        recommendation_detail=detail,
        current_usage=forecast_result["current_usage"],
        predicted_at_renewal=forecast_result["predicted_at_renewal"],
        percent_change=forecast_result["percent_change"],
        renewal_date=forecast_result["renewal_date"],
        model_version=loader.get_model_version(product),
        generated_at=datetime.utcnow().isoformat() + "Z",
    )


@router.post("/", response_model=ForecastResponse)
async def create_forecast(
    request: ForecastRequest,
    loader: ModelLoader = Depends(get_loader),
) -> ForecastResponse:
    try:
        model = loader.load_production_model(request.product)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    hist_df = _load_historical(request.product)
    result = generate_forecast(
        model=model,
        historical_df=hist_df,
        horizon_months=request.horizon_months,
        renewal_date=request.renewal_date,
    )
    return _build_response(request.product, request.horizon_months, result, loader)


@router.get("/{product}", response_model=ForecastResponse)
async def get_forecast(
    product: str,
    horizon_months: int = Query(default=6, description="3, 6, or 12"),
    loader: ModelLoader = Depends(get_loader),
) -> ForecastResponse:
    if product not in ("Jira", "Slack", "Zoom"):
        raise HTTPException(status_code=404, detail=f"Unknown product '{product}'")
    if horizon_months not in (3, 6, 12):
        raise HTTPException(status_code=422, detail="horizon_months must be 3, 6, or 12")

    try:
        model = loader.load_production_model(product)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    hist_df = _load_historical(product)
    result = generate_forecast(
        model=model,
        historical_df=hist_df,
        horizon_months=horizon_months,
    )
    return _build_response(product, horizon_months, result, loader)
