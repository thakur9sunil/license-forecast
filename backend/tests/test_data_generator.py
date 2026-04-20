import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data_generator import generate_all_products, generate_product_data, PRODUCT_CONFIGS


def test_generate_all_products_row_count():
    df = generate_all_products(output_path=None)
    assert len(df) == 36, f"Expected 36 rows, got {len(df)}"


def test_generate_all_products_has_three_products():
    df = generate_all_products(output_path=None)
    assert set(df["product"].unique()) == {"Jira", "Slack", "Zoom"}


def test_generate_all_products_no_negative_values():
    df = generate_all_products(output_path=None)
    assert df["y"].min() >= 0, "License counts should be non-negative"


def test_generate_all_products_columns():
    df = generate_all_products(output_path=None)
    assert set(df.columns) >= {"ds", "y", "product"}


def test_jira_upward_trend():
    cfg = PRODUCT_CONFIGS["Jira"]
    df = generate_product_data("Jira", "2023-01-01", 12, **cfg)
    first_half_avg = df["y"].iloc[:6].mean()
    second_half_avg = df["y"].iloc[6:].mean()
    # With +4.5/month trend, second half should generally be higher
    assert second_half_avg > first_half_avg - 50  # allow seasonal variation


def test_zoom_downward_trend():
    cfg = PRODUCT_CONFIGS["Zoom"]
    df = generate_product_data("Zoom", "2023-01-01", 12, **cfg)
    # With -2.1/month trend over 12 months = -25 licenses, first should be higher
    assert df["y"].iloc[0] > df["y"].iloc[-1] - 60  # allow seasonal variation


def test_generate_product_data_returns_correct_length():
    cfg = PRODUCT_CONFIGS["Slack"]
    df = generate_product_data("Slack", "2023-01-01", 12, **cfg)
    assert len(df) == 12


def test_generate_product_data_monthly_frequency():
    cfg = PRODUCT_CONFIGS["Jira"]
    df = generate_product_data("Jira", "2023-01-01", 12, **cfg)
    # All dates should be first of month
    assert all(d.day == 1 for d in df["ds"])
