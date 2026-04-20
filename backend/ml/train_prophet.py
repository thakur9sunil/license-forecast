"""
Trains one Prophet model per product with hyperparameter search.
Logs all runs to MLflow, registers best model per product, promotes to Production.

Usage:
    python -m ml.train_prophet
"""
import json
import logging
import os
import sys
from itertools import product as itertools_product
from pathlib import Path

import mlflow
import mlflow.prophet
import pandas as pd
from mlflow import MlflowClient
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

PARAM_GRID = {
    "seasonality_mode": ["multiplicative", "additive"],
    "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.3],
    "seasonality_prior_scale": [1.0, 10.0],
}

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "license_usage.csv"
METRICS_DIR = Path(__file__).resolve().parent / "metrics"


def _all_param_combos(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools_product(*values)]


def train_prophet_for_product(
    df: pd.DataFrame,
    product_name: str,
    seasonality_mode: str,
    changepoint_prior_scale: float,
    seasonality_prior_scale: float,
    mlflow_run_name: str,
) -> tuple[Prophet, dict]:
    with mlflow.start_run(run_name=mlflow_run_name, nested=True) as run:
        mlflow.log_params({
            "product": product_name,
            "seasonality_mode": seasonality_mode,
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            "n_training_points": len(df),
        })

        model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        model.add_seasonality(name="quarterly", period=91.25, fourier_order=3)
        model.fit(df)

        # Cross-validation with the data we have (12 points, use 8 for initial)
        try:
            cv_df = cross_validation(
                model,
                initial="240 days",
                period="30 days",
                horizon="90 days",
                disable_tqdm=True,
            )
            perf = performance_metrics(cv_df, rolling_window=1)
            metrics = {
                "mae": float(perf["mae"].mean()),
                "rmse": float(perf["rmse"].mean()),
                "mape": float(perf["mape"].mean()),
            }
        except Exception:
            # Fallback: in-sample metrics when CV window is too small
            future = model.make_future_dataframe(periods=0, freq="MS")
            forecast = model.predict(future)
            import numpy as np
            actuals = df["y"].values
            preds = forecast["yhat"].values[-len(actuals):]
            mask = actuals != 0
            metrics = {
                "mae": float(np.mean(np.abs(actuals - preds))),
                "rmse": float(np.sqrt(np.mean((actuals - preds) ** 2))),
                "mape": float(np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask]))),
            }

        mlflow.log_metrics(metrics)
        mlflow.set_tag("run_id", run.info.run_id)

        mlflow.prophet.log_model(
            model,
            artifact_path="prophet_model",
            registered_model_name=f"license_forecast_{product_name.lower()}",
        )

        return model, {**metrics, "run_id": run.info.run_id}


def run_hyperparameter_search(
    df: pd.DataFrame,
    product_name: str,
    experiment_name: str,
) -> dict:
    combos = _all_param_combos(PARAM_GRID)
    results = []

    print(f"\n[{product_name}] Running {len(combos)} hyperparameter combinations...")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{product_name}_search"):
        for i, params in enumerate(combos):
            run_name = (
                f"{product_name}_{params['seasonality_mode']}"
                f"_cp{params['changepoint_prior_scale']}"
                f"_sp{params['seasonality_prior_scale']}"
            )
            _, metrics = train_prophet_for_product(
                df=df,
                product_name=product_name,
                mlflow_run_name=run_name,
                **params,
            )
            results.append({**params, **metrics})
            print(f"  [{i+1}/{len(combos)}] MAPE={metrics['mape']:.4f}  {run_name}")

    best = min(results, key=lambda x: x["mape"])
    print(f"  -> Best MAPE={best['mape']:.4f}  run_id={best['run_id']}")
    return best


def promote_best_model(product_name: str) -> None:
    client = MlflowClient()
    model_name = f"license_forecast_{product_name.lower()}"

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        print(f"  Could not query registry for {model_name}: {e}")
        return

    if not versions:
        print(f"  No versions found for {model_name}")
        return

    # Find the version with best (lowest) MAPE metric
    best_version = None
    best_mape = float("inf")
    for v in versions:
        run = client.get_run(v.run_id)
        # Nested runs may not have direct metrics — walk up to parent if needed
        mape = run.data.metrics.get("mape", float("inf"))
        if mape < best_mape:
            best_mape = mape
            best_version = v

    if best_version is None:
        return

    # Archive any current Production versions
    for v in versions:
        if v.current_stage == "Production":
            client.transition_model_version_stage(
                name=model_name, version=v.version, stage="Archived"
            )

    # Promote best to Production
    client.transition_model_version_stage(
        name=model_name,
        version=best_version.version,
        stage="Production",
    )
    print(f"  Promoted {model_name} v{best_version.version} -> Production (MAPE={best_mape:.4f})")


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "license_forecasting")

    mlflow.set_tracking_uri(tracking_uri)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(DATA_PATH, parse_dates=["ds"])
    all_metrics: dict[str, dict] = {}

    for product_name in ["Jira", "Slack", "Zoom"]:
        df = df_all[df_all["product"] == product_name][["ds", "y"]].copy()
        best = run_hyperparameter_search(df, product_name, experiment_name)
        all_metrics[product_name] = best
        promote_best_model(product_name)

    metrics_path = METRICS_DIR / "prophet_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
