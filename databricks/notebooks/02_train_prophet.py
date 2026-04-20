# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 2 — Train Prophet Models
# MAGIC
# MAGIC Trains one Prophet model per product (Jira, Slack, Zoom) using hyperparameter search.
# MAGIC - Logs all 16 runs per product to the built-in Databricks MLflow
# MAGIC - Registers best model per product in Unity Catalog Model Registry
# MAGIC - Promotes best version to "Champion" alias
# MAGIC
# MAGIC **Prerequisites:** Run Notebook 01 first to generate the data.
# MAGIC
# MAGIC **Cluster requirements:** Use a cluster with ML Runtime (e.g. 14.x ML).
# MAGIC Prophet and cmdstanpy are pre-installed on Databricks ML Runtime clusters.

# COMMAND ----------

# MAGIC %pip install prophet mlflow

# COMMAND ----------

import json
import logging
import warnings
from itertools import product as itertools_product

import mlflow
import mlflow.prophet
import pandas as pd
from mlflow import MlflowClient
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# On Databricks, MLflow tracking is automatic — no URI needed.
# mlflow.set_tracking_uri() is NOT required here.

# Use Unity Catalog for model registry (recommended on Databricks).
# Format: <catalog>.<schema>.<model_name>
# If you are on the free Community Edition, use the classic registry instead:
#   model_name = "license_forecast_jira"  (no catalog prefix)
USE_UNITY_CATALOG = False  # Set to True if your workspace has Unity Catalog enabled
CATALOG = "main"
SCHEMA = "license_forecast"

DBFS_DATA_PATH = "/dbfs/FileStore/license_forecast/data/license_usage.csv"
EXPERIMENT_NAME = "/Users/license_forecasting"  # Databricks experiment paths start with /Users/

PARAM_GRID = {
    "seasonality_mode": ["multiplicative", "additive"],
    "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.3],
    "seasonality_prior_scale": [1.0, 10.0],
}


def get_model_registry_name(product_name: str) -> str:
    if USE_UNITY_CATALOG:
        return f"{CATALOG}.{SCHEMA}.license_forecast_{product_name.lower()}"
    return f"license_forecast_{product_name.lower()}"


def all_param_combos(grid):
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools_product(*values)]


def train_prophet_for_product(df, product_name, seasonality_mode,
                               changepoint_prior_scale, seasonality_prior_scale,
                               run_name):
    with mlflow.start_run(run_name=run_name, nested=True) as run:
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

        try:
            cv_df = cross_validation(model, initial="240 days", period="30 days",
                                     horizon="90 days", disable_tqdm=True)
            perf = performance_metrics(cv_df, rolling_window=1)
            metrics = {
                "mae": float(perf["mae"].mean()),
                "rmse": float(perf["rmse"].mean()),
                "mape": float(perf["mape"].mean()),
            }
        except Exception:
            import numpy as np
            future = model.make_future_dataframe(periods=0, freq="MS")
            forecast = model.predict(future)
            actuals = df["y"].values
            preds = forecast["yhat"].values[-len(actuals):]
            mask = actuals != 0
            metrics = {
                "mae": float(np.mean(np.abs(actuals - preds))),
                "rmse": float(np.sqrt(np.mean((actuals - preds) ** 2))),
                "mape": float(np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask]))),
            }

        mlflow.log_metrics(metrics)

        mlflow.prophet.log_model(
            model,
            artifact_path="prophet_model",
            registered_model_name=get_model_registry_name(product_name),
        )

        return model, {**metrics, "run_id": run.info.run_id}


def run_hyperparameter_search(df, product_name):
    combos = all_param_combos(PARAM_GRID)
    results = []

    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n[{product_name}] Running {len(combos)} hyperparameter combinations...")

    with mlflow.start_run(run_name=f"{product_name}_search"):
        for i, params in enumerate(combos):
            run_name = (
                f"{product_name}_{params['seasonality_mode']}"
                f"_cp{params['changepoint_prior_scale']}"
                f"_sp{params['seasonality_prior_scale']}"
            )
            _, metrics = train_prophet_for_product(
                df=df, product_name=product_name, run_name=run_name, **params
            )
            results.append({**params, **metrics})
            print(f"  [{i+1}/{len(combos)}] MAPE={metrics['mape']:.4f}  {run_name}")

    best = min(results, key=lambda x: x["mape"])
    print(f"  -> Best MAPE={best['mape']:.4f}  run_id={best['run_id']}")
    return best


def promote_best_model(product_name):
    client = MlflowClient()
    model_name = get_model_registry_name(product_name)

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"  No versions found for {model_name}")
        return

    best_version = min(
        versions,
        key=lambda v: client.get_run(v.run_id).data.metrics.get("mape", float("inf"))
    )

    # On Databricks with Unity Catalog use aliases instead of stages
    if USE_UNITY_CATALOG:
        client.set_registered_model_alias(model_name, "champion", best_version.version)
        print(f"  Set alias 'champion' -> {model_name} v{best_version.version}")
    else:
        for v in versions:
            if v.current_stage == "Production":
                client.transition_model_version_stage(model_name, v.version, "Archived")
        client.transition_model_version_stage(model_name, best_version.version, "Production")
        print(f"  Promoted {model_name} v{best_version.version} -> Production")


# COMMAND ----------

# Load data from DBFS
df_all = pd.read_csv(DBFS_DATA_PATH, parse_dates=["ds"])
print(f"Loaded {len(df_all)} rows from {DBFS_DATA_PATH}")
display(df_all.head())

# COMMAND ----------

# Train and register models for all products
all_metrics = {}

for product_name in ["Jira", "Slack", "Zoom"]:
    df = df_all[df_all["product"] == product_name][["ds", "y"]].copy()
    best = run_hyperparameter_search(df, product_name)
    all_metrics[product_name] = best
    promote_best_model(product_name)

print("\n=== Training Complete ===")
print(json.dumps({k: {"mape": round(v["mape"], 4)} for k, v in all_metrics.items()}, indent=2))

# COMMAND ----------

# Save metrics summary to DBFS
import os
metrics_path = "/dbfs/FileStore/license_forecast/metrics/prophet_metrics.json"
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"Metrics saved to {metrics_path}")
