# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 1 — Generate Synthetic License Usage Data
# MAGIC
# MAGIC Generates 12 months of synthetic license usage data for Jira, Slack, and Zoom.
# MAGIC Saves the result to DBFS (Databricks File System) as a CSV and as a Delta table.
# MAGIC
# MAGIC **Run this notebook first before any training notebooks.**

# COMMAND ----------

# MAGIC %pip install numpy pandas

# COMMAND ----------

import numpy as np
import pandas as pd

# On Databricks, DBFS is the shared file system.
# /dbfs/... is the local path. dbfs:/... is the Spark path.
DBFS_OUTPUT_PATH = "/dbfs/FileStore/license_forecast/data/license_usage.csv"
DELTA_TABLE_NAME = "license_forecast.usage_data"

PRODUCT_CONFIGS = {
    "Jira": {
        "base_licenses": 480,
        "trend_per_month": 4.5,
        "seasonal_pattern": [1.12, 0.95, 0.97, 1.00, 0.98, 0.95,
                              0.92, 0.94, 1.02, 1.05, 1.08, 1.10],
        "noise_std": 12.0,
        "outlier_indices": [3],
        "outlier_magnitude": 1.18,
    },
    "Slack": {
        "base_licenses": 620,
        "trend_per_month": 1.8,
        "seasonal_pattern": [1.05, 0.98, 0.99, 1.00, 1.00, 0.97,
                              0.93, 0.95, 1.02, 1.04, 1.07, 1.14],
        "noise_std": 8.0,
        "outlier_indices": [11],
        "outlier_magnitude": 1.22,
    },
    "Zoom": {
        "base_licenses": 390,
        "trend_per_month": -2.1,
        "seasonal_pattern": [1.08, 1.02, 0.99, 0.97, 0.95, 0.88,
                              0.84, 0.86, 0.98, 1.02, 1.04, 1.06],
        "noise_std": 18.0,
        "outlier_indices": [1, 7],
        "outlier_magnitude": 0.78,
    },
}


def generate_product_data(product_name, start_date, periods, base_licenses,
                          trend_per_month, seasonal_pattern, noise_std,
                          outlier_indices, outlier_magnitude, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="MS")
    values = []
    for i in range(periods):
        trend = base_licenses + trend_per_month * i
        seasonal = seasonal_pattern[i % 12]
        noise = rng.normal(0, noise_std)
        value = trend * seasonal + noise
        if i in outlier_indices:
            value *= outlier_magnitude
        values.append(max(0, int(round(value))))
    return pd.DataFrame({"ds": dates, "y": values, "product": product_name})


frames = [
    generate_product_data(name, "2023-01-01", 12, **cfg)
    for name, cfg in PRODUCT_CONFIGS.items()
]
df = pd.concat(frames, ignore_index=True)
print(f"Generated {len(df)} rows")
display(df)

# COMMAND ----------

# Save to DBFS as CSV (readable by all notebooks)
import os
os.makedirs(os.path.dirname(DBFS_OUTPUT_PATH), exist_ok=True)
df.to_csv(DBFS_OUTPUT_PATH, index=False)
print(f"Saved CSV to {DBFS_OUTPUT_PATH}")

# COMMAND ----------

# Also save as a Delta table for SQL access and versioning
spark_df = spark.createDataFrame(df)

spark.sql("CREATE DATABASE IF NOT EXISTS license_forecast")
spark_df.write.format("delta").mode("overwrite").saveAsTable(DELTA_TABLE_NAME)
print(f"Saved Delta table: {DELTA_TABLE_NAME}")

# COMMAND ----------

# Verify the data
display(spark.table(DELTA_TABLE_NAME))
