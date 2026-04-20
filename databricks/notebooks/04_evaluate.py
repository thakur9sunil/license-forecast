# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 4 — Compare Prophet vs ARIMA
# MAGIC
# MAGIC Loads saved metrics for both models and produces a comparison table.
# MAGIC Run this after both training notebooks have completed.

# COMMAND ----------

import json
import pandas as pd

prophet_path = "/dbfs/FileStore/license_forecast/metrics/prophet_metrics.json"
arima_path   = "/dbfs/FileStore/license_forecast/metrics/arima_metrics.json"

with open(prophet_path) as f:
    prophet_metrics = json.load(f)

with open(arima_path) as f:
    arima_metrics = json.load(f)

# COMMAND ----------

rows = []
for product in ["Jira", "Slack", "Zoom"]:
    p = prophet_metrics[product]
    a = arima_metrics[product]
    rows.append({
        "Product": product,
        "Prophet MAPE": f"{p['mape']*100:.2f}%",
        "ARIMA MAPE":   f"{a['mape']*100:.2f}%",
        "Prophet RMSE": round(p["rmse"], 2),
        "ARIMA RMSE":   round(a["rmse"], 2),
        "Winner": "Prophet" if p["mape"] < a["mape"] else "ARIMA",
    })

comparison_df = pd.DataFrame(rows)
print("=== Model Comparison ===")
display(comparison_df)
