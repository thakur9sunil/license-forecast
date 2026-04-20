import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.forecaster import (
    compute_percent_change,
    compute_trend_direction,
    extract_forecast_points,
    extract_historical,
)
from app.services.recommender import compute_recommendation


# ── forecaster.py ─────────────────────────────────────────────────────────────

def _make_forecast_df(n_hist: int = 12, n_future: int = 6) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n_hist + n_future, freq="MS")
    yhat = np.linspace(500, 560, n_hist + n_future)
    return pd.DataFrame({
        "ds": dates,
        "yhat": yhat,
        "yhat_lower": yhat - 20,
        "yhat_upper": yhat + 20,
    })


def test_extract_historical_length():
    original = pd.DataFrame({
        "ds": pd.date_range("2023-01-01", periods=12, freq="MS"),
        "y": np.random.randint(480, 560, 12),
    })
    forecast_df = _make_forecast_df()
    hist = extract_historical(forecast_df, original)
    assert len(hist) == 12


def test_extract_historical_fields():
    original = pd.DataFrame({
        "ds": pd.date_range("2023-01-01", periods=3, freq="MS"),
        "y": [500, 510, 520],
    })
    hist = extract_historical(_make_forecast_df(), original)
    assert all("ds" in h and "y" in h for h in hist)


def test_extract_forecast_points_count():
    forecast_df = _make_forecast_df(n_hist=12, n_future=6)
    points = extract_forecast_points(forecast_df, horizon_months=6)
    assert len(points) == 6


def test_extract_forecast_points_no_negatives():
    forecast_df = _make_forecast_df()
    forecast_df["yhat_lower"] = -10  # force negative
    points = extract_forecast_points(forecast_df, horizon_months=6)
    assert all(p["yhat_lower"] >= 0 for p in points)


def test_compute_trend_direction_increasing():
    points = [{"yhat": 500 + i * 10} for i in range(6)]
    assert compute_trend_direction(points) == "increasing"


def test_compute_trend_direction_decreasing():
    points = [{"yhat": 500 - i * 10} for i in range(6)]
    assert compute_trend_direction(points) == "decreasing"


def test_compute_trend_direction_stable():
    points = [{"yhat": 500 + i * 0.5} for i in range(6)]  # <3% change
    assert compute_trend_direction(points) == "stable"


def test_compute_percent_change():
    assert compute_percent_change(500, 550) == 10.0
    assert compute_percent_change(500, 450) == -10.0
    assert compute_percent_change(0, 100) == 0.0


# ── recommender.py ────────────────────────────────────────────────────────────

def test_recommendation_buy_more():
    rec, detail = compute_recommendation(
        product="Jira",
        predicted_at_renewal=600,
        yhat_upper=620,
        current_licensed=550,
    )
    assert rec == "buy_more"
    assert "additional" in detail.lower()


def test_recommendation_reduce():
    rec, detail = compute_recommendation(
        product="Jira",
        predicted_at_renewal=400,
        yhat_upper=420,
        current_licensed=550,
    )
    assert rec == "reduce"
    assert "drop" in detail.lower() or "reduc" in detail.lower()


def test_recommendation_hold():
    rec, detail = compute_recommendation(
        product="Jira",
        predicted_at_renewal=520,
        yhat_upper=540,
        current_licensed=550,
    )
    assert rec == "hold"


def test_recommendation_detail_nonempty():
    rec, detail = compute_recommendation("Slack", 650, 680, 700)
    assert len(detail) > 10
