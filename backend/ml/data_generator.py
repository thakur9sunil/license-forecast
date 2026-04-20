"""
Generates 12 months of synthetic software license usage data for three products.
Each product has a distinct trend, seasonality pattern, noise level, and outliers.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PRODUCT_CONFIGS: dict = {
    "Jira": {
        "base_licenses": 480,
        "trend_per_month": 4.5,
        # Jan spike (Q1 planning), summer dip, Dec ramp-up
        "seasonal_pattern": [1.12, 0.95, 0.97, 1.00, 0.98, 0.95,
                              0.92, 0.94, 1.02, 1.05, 1.08, 1.10],
        "noise_std": 12.0,
        "outlier_indices": [3],        # April: budget project spike
        "outlier_magnitude": 1.18,
    },
    "Slack": {
        "base_licenses": 620,
        "trend_per_month": 1.8,
        # Stable through year, July dip, Dec onboarding spike
        "seasonal_pattern": [1.05, 0.98, 0.99, 1.00, 1.00, 0.97,
                              0.93, 0.95, 1.02, 1.04, 1.07, 1.14],
        "noise_std": 8.0,
        "outlier_indices": [11],       # December: bulk onboarding
        "outlier_magnitude": 1.22,
    },
    "Zoom": {
        "base_licenses": 390,
        "trend_per_month": -2.1,       # return-to-office trend
        # Deep summer dip, slight Q4 recovery
        "seasonal_pattern": [1.08, 1.02, 0.99, 0.97, 0.95, 0.88,
                              0.84, 0.86, 0.98, 1.02, 1.04, 1.06],
        "noise_std": 18.0,
        "outlier_indices": [1, 7],     # Feb/Aug: anomalous dips
        "outlier_magnitude": 0.78,
    },
}


def generate_product_data(
    product_name: str,
    start_date: str,
    periods: int,
    base_licenses: int,
    trend_per_month: float,
    seasonal_pattern: list[float],
    noise_std: float,
    outlier_indices: list[int],
    outlier_magnitude: float,
    seed: int = 42,
) -> pd.DataFrame:
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


def generate_all_products(output_path: str | None = None) -> pd.DataFrame:
    start_date = "2023-01-01"
    frames = []

    for product_name, cfg in PRODUCT_CONFIGS.items():
        df = generate_product_data(
            product_name=product_name,
            start_date=start_date,
            periods=12,
            **cfg,
        )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"Saved {len(combined)} rows to {output_path}")

    return combined


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    out = str(base / "data" / "raw" / "license_usage.csv")
    df = generate_all_products(output_path=out)
    print(df.to_string())
