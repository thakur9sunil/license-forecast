"""
SARIMA baseline model for each product.
Logs comparison metrics to MLflow as tags alongside Prophet runs.

Usage:
    python -m ml.train_arima
"""
import json
import os
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ml.evaluate import compute_metrics

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "license_usage.csv"
METRICS_DIR = Path(__file__).resolve().parent / "metrics"


def fit_sarima(
    series: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12),
) -> tuple:
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted


def evaluate_arima_for_product(
    df: pd.DataFrame,
    product_name: str,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12),
) -> dict:
    series = df.set_index("ds")["y"].asfreq("MS")

    # Leave-one-out: train on first 10, test on last 2
    train = series.iloc[:10]
    test = series.iloc[10:]

    try:
        fitted = fit_sarima(train, order=order, seasonal_order=seasonal_order)
        forecast = fitted.forecast(steps=len(test))
        metrics = compute_metrics(test.values, forecast.values)
    except Exception as e:
        print(f"  ARIMA failed for {product_name}: {e}")
        metrics = {"mae": 9999, "rmse": 9999, "mape": 9999, "r2": -9999}

    return metrics


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "license_forecasting")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(DATA_PATH, parse_dates=["ds"])
    all_metrics: dict[str, dict] = {}

    for product_name in ["Jira", "Slack", "Zoom"]:
        print(f"\n[{product_name}] Fitting SARIMA(1,1,1)(1,1,1,12)...")
        df = df_all[df_all["product"] == product_name][["ds", "y"]].copy()
        metrics = evaluate_arima_for_product(df, product_name)
        all_metrics[product_name] = metrics

        with mlflow.start_run(run_name=f"arima_{product_name.lower()}"):
            mlflow.log_params({
                "product": product_name,
                "model_type": "SARIMA",
                "order": "(1,1,1)",
                "seasonal_order": "(1,1,1,12)",
            })
            mlflow.log_metrics({
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "r2": metrics["r2"],
            })

        print(f"  MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
              f"MAPE={metrics['mape']:.4f}  R²={metrics['r2']:.4f}")

    metrics_path = METRICS_DIR / "arima_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nARIMA metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
