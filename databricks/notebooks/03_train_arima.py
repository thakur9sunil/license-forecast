# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 3 — Train ARIMA Benchmark Models
# MAGIC
# MAGIC Fits a SARIMA(1,1,1)(1,1,1,12) model for each product as a benchmark.
# MAGIC Logs metrics to Databricks MLflow for comparison against Prophet.
# MAGIC
# MAGIC **Prerequisites:** Run Notebook 01 first.

# COMMAND ----------

# MAGIC %pip install statsmodels mlflow scikit-learn

# COMMAND ----------

import json
import os
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DBFS_DATA_PATH = "/dbfs/FileStore/license_forecast/data/license_usage.csv"
EXPERIMENT_NAME = "/Users/license_forecasting"


def compute_metrics(actuals, predictions):
    mae = float(mean_absolute_error(actuals, predictions))
    rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
    mask = actuals != 0
    mape = float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])))
    r2 = float(r2_score(actuals, predictions))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def fit_and_evaluate_sarima(df, product_name,
                             order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    series = df.set_index("ds")["y"].asfreq("MS")
    train = series.iloc[:10]
    test = series.iloc[10:]

    try:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=len(test))
        metrics = compute_metrics(test.values, forecast.values)
    except Exception as e:
        print(f"  ARIMA failed for {product_name}: {e}")
        metrics = {"mae": 9999, "rmse": 9999, "mape": 9999, "r2": -9999}

    return metrics


# COMMAND ----------

df_all = pd.read_csv(DBFS_DATA_PATH, parse_dates=["ds"])

mlflow.set_experiment(EXPERIMENT_NAME)
all_metrics = {}

for product_name in ["Jira", "Slack", "Zoom"]:
    print(f"\n[{product_name}] Fitting SARIMA(1,1,1)(1,1,1,12)...")
    df = df_all[df_all["product"] == product_name][["ds", "y"]].copy()
    metrics = fit_and_evaluate_sarima(df, product_name)
    all_metrics[product_name] = metrics

    with mlflow.start_run(run_name=f"arima_{product_name.lower()}"):
        mlflow.log_params({
            "product": product_name,
            "model_type": "SARIMA",
            "order": "(1,1,1)",
            "seasonal_order": "(1,1,1,12)",
        })
        mlflow.log_metrics({k: v for k, v in metrics.items()})

    print(f"  MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
          f"MAPE={metrics['mape']:.4f}  R²={metrics['r2']:.4f}")

# COMMAND ----------

metrics_path = "/dbfs/FileStore/license_forecast/metrics/arima_metrics.json"
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\nARIMA metrics saved to {metrics_path}")

print("\n=== ARIMA Results ===")
for product, m in all_metrics.items():
    print(f"  {product}: MAPE={m['mape']:.4f}  RMSE={m['rmse']:.2f}")
