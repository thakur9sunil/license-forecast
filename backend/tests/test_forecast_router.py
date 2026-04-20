"""
Integration tests for the FastAPI forecast router.
Uses a mock ModelLoader so Prophet models are not required at test time.
"""
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mock_loader():
    """Returns a ModelLoader mock that yields a tiny stub Prophet model."""
    from unittest.mock import MagicMock

    loader = MagicMock()
    loader.get_model_version.return_value = "test-v1"
    loader.health_check.return_value = {"Jira": True, "Slack": True, "Zoom": True}

    # Stub model whose predict() returns a plausible forecast dataframe
    def _fake_predict(future_df: pd.DataFrame) -> pd.DataFrame:
        n = len(future_df)
        yhat = np.linspace(500, 540, n)
        return pd.DataFrame({
            "ds": future_df["ds"],
            "yhat": yhat,
            "yhat_lower": yhat - 20,
            "yhat_upper": yhat + 20,
        })

    stub_model = MagicMock()
    stub_model.make_future_dataframe.return_value = pd.DataFrame({
        "ds": pd.date_range("2023-01-01", periods=18, freq="MS"),
    })
    stub_model.predict.side_effect = _fake_predict
    loader.load_production_model.return_value = stub_model
    return loader


@pytest.fixture
def client(mock_loader, tmp_path):
    """FastAPI test client with mocked loader and temp CSV data."""
    from fastapi.testclient import TestClient
    from app.main import app
    import app.models.loader as loader_module
    import app.routers.forecast as forecast_module

    # Write synthetic CSV to tmp_path so the router can load it
    from ml.data_generator import generate_all_products
    csv_path = tmp_path / "raw" / "license_usage.csv"
    csv_path.parent.mkdir(parents=True)
    generate_all_products(output_path=str(csv_path))

    # Patch DATA_DIR in the router to point to tmp_path
    with patch.object(forecast_module, "DATA_DIR", tmp_path / "raw"):
        with patch.object(loader_module, "_loader_instance", mock_loader):
            with TestClient(app) as c:
                yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("healthy", "degraded", "unhealthy")
    assert "models_loaded" in body


def test_products_list(client):
    resp = client.get("/products/")
    assert resp.status_code == 200
    products = resp.json()
    assert len(products) == 3
    names = {p["name"] for p in products}
    assert names == {"Jira", "Slack", "Zoom"}


def test_products_single(client):
    resp = client.get("/products/Jira")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Jira"
    assert "current_licenses" in body


def test_products_unknown_returns_404(client):
    resp = client.get("/products/UnknownSoftware")
    assert resp.status_code == 404


def test_forecast_post_valid(client):
    payload = {"product": "Jira", "horizon_months": 6}
    resp = client.post("/forecast/", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["product"] == "Jira"
    assert body["horizon_months"] == 6
    assert len(body["forecast"]) == 6
    assert body["trend_direction"] in ("increasing", "decreasing", "stable")
    assert body["recommendation"] in ("buy_more", "reduce", "hold")
    assert "recommendation_detail" in body
    assert body["percent_change"] is not None


def test_forecast_post_all_horizons(client):
    for horizon in (3, 6, 12):
        resp = client.post("/forecast/", json={"product": "Slack", "horizon_months": horizon})
        assert resp.status_code == 200
        assert resp.json()["horizon_months"] == horizon


def test_forecast_get_valid(client):
    resp = client.get("/forecast/Zoom?horizon_months=3")
    assert resp.status_code == 200
    assert resp.json()["product"] == "Zoom"


def test_forecast_invalid_product_422(client):
    resp = client.post("/forecast/", json={"product": "UnknownApp", "horizon_months": 6})
    assert resp.status_code == 422


def test_forecast_confidence_interval_positive(client):
    resp = client.post("/forecast/", json={"product": "Jira", "horizon_months": 3})
    body = resp.json()
    for point in body["forecast"]:
        assert point["yhat_lower"] >= 0
        assert point["yhat_upper"] >= point["yhat_lower"]
