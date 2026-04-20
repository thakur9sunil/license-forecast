"""
Core forecasting logic: takes a loaded Prophet model and produces
structured output suitable for the ForecastResponse schema.
"""
from datetime import date, timedelta
from typing import Any

import pandas as pd


def generate_forecast(
    model: Any,
    historical_df: pd.DataFrame,
    horizon_months: int,
    renewal_date: date | None = None,
) -> dict:
    future = model.make_future_dataframe(periods=horizon_months, freq="MS")
    forecast_df = model.predict(future)

    historical = extract_historical(forecast_df, historical_df)
    forecast_points = extract_forecast_points(forecast_df, horizon_months)

    current_usage = float(historical_df["y"].iloc[-1]) if len(historical_df) > 0 else 0.0
    predicted_at_renewal = float(forecast_points[-1]["yhat"]) if forecast_points else current_usage

    if renewal_date is None:
        last_ds = pd.Timestamp(historical_df["ds"].max())
        renewal_date = (last_ds + pd.DateOffset(months=horizon_months)).date()

    trend_direction = compute_trend_direction(forecast_points)
    percent_change = compute_percent_change(current_usage, predicted_at_renewal)

    return {
        "historical": historical,
        "forecast": forecast_points,
        "current_usage": current_usage,
        "predicted_at_renewal": predicted_at_renewal,
        "renewal_date": renewal_date,
        "trend_direction": trend_direction,
        "percent_change": percent_change,
    }


def extract_historical(
    forecast_df: pd.DataFrame,
    original_df: pd.DataFrame,
) -> list[dict]:
    merged = original_df[["ds", "y"]].copy()
    merged["ds"] = pd.to_datetime(merged["ds"])
    return [
        {"ds": row["ds"].date(), "y": float(row["y"])}
        for _, row in merged.iterrows()
    ]


def extract_forecast_points(
    forecast_df: pd.DataFrame,
    horizon_months: int,
) -> list[dict]:
    future_rows = forecast_df.tail(horizon_months)
    points = []
    for _, row in future_rows.iterrows():
        points.append({
            "ds": pd.Timestamp(row["ds"]).date(),
            "yhat": max(0.0, float(row["yhat"])),
            "yhat_lower": max(0.0, float(row["yhat_lower"])),
            "yhat_upper": max(0.0, float(row["yhat_upper"])),
        })
    return points


def compute_trend_direction(
    forecast_points: list[dict],
    threshold_pct: float = 0.03,
) -> str:
    if len(forecast_points) < 2:
        return "stable"
    first = forecast_points[0]["yhat"]
    last = forecast_points[-1]["yhat"]
    if first == 0:
        return "stable"
    change = (last - first) / first
    if change > threshold_pct:
        return "increasing"
    if change < -threshold_pct:
        return "decreasing"
    return "stable"


def compute_percent_change(current: float, predicted: float) -> float:
    if current == 0:
        return 0.0
    return round(((predicted - current) / current) * 100, 2)
