from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    ds: date
    yhat: float = Field(..., description="Predicted license count")
    yhat_lower: float
    yhat_upper: float


class HistoricalPoint(BaseModel):
    ds: date
    y: float


class ForecastRequest(BaseModel):
    product: Literal["Jira", "Slack", "Zoom"]
    horizon_months: Literal[3, 6, 12] = 6
    renewal_date: date | None = None


class ForecastResponse(BaseModel):
    product: str
    horizon_months: int
    historical: list[HistoricalPoint]
    forecast: list[ForecastPoint]
    trend_direction: Literal["increasing", "decreasing", "stable"]
    recommendation: Literal["buy_more", "reduce", "hold"]
    recommendation_detail: str
    current_usage: float
    predicted_at_renewal: float
    percent_change: float
    renewal_date: date
    model_version: str
    generated_at: str


class ProductInfo(BaseModel):
    name: str
    current_licenses: int
    current_usage: int
    last_updated: date
    contract_renewal: date


class ModelMetrics(BaseModel):
    product: str
    model_type: str
    mae: float
    rmse: float
    mape: float
    r2: float
    training_date: str
    mlflow_run_id: str
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    models_loaded: dict[str, bool]
    mlflow_connected: bool
    version: str
