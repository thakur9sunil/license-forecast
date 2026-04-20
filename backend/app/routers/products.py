from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.models.schemas import ProductInfo

router = APIRouter(prefix="/products", tags=["products"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

_PRODUCT_META = {
    "Jira": {"current_licenses": 550, "contract_renewal": date(2025, 1, 1)},
    "Slack": {"current_licenses": 700, "contract_renewal": date(2024, 12, 1)},
    "Zoom": {"current_licenses": 420, "contract_renewal": date(2025, 3, 1)},
}


def _read_latest_usage() -> dict[str, int]:
    csv_path = DATA_DIR / "license_usage.csv"
    if not csv_path.exists():
        return {p: 0 for p in _PRODUCT_META}
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    result = {}
    for product in _PRODUCT_META:
        rows = df[df["product"] == product].sort_values("ds")
        result[product] = int(rows["y"].iloc[-1]) if not rows.empty else 0
    return result


def _read_last_updated() -> dict[str, date]:
    csv_path = DATA_DIR / "license_usage.csv"
    if not csv_path.exists():
        return {p: date.today() for p in _PRODUCT_META}
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    result = {}
    for product in _PRODUCT_META:
        rows = df[df["product"] == product].sort_values("ds")
        result[product] = rows["ds"].iloc[-1].date() if not rows.empty else date.today()
    return result


@router.get("/", response_model=list[ProductInfo])
async def list_products() -> list[ProductInfo]:
    usage = _read_latest_usage()
    last_updated = _read_last_updated()
    return [
        ProductInfo(
            name=product,
            current_licenses=meta["current_licenses"],
            current_usage=usage.get(product, 0),
            last_updated=last_updated.get(product, date.today()),
            contract_renewal=meta["contract_renewal"],
        )
        for product, meta in _PRODUCT_META.items()
    ]


@router.get("/{product_name}", response_model=ProductInfo)
async def get_product(product_name: str) -> ProductInfo:
    if product_name not in _PRODUCT_META:
        raise HTTPException(status_code=404, detail=f"Product '{product_name}' not found")
    meta = _PRODUCT_META[product_name]
    usage = _read_latest_usage()
    last_updated = _read_last_updated()
    return ProductInfo(
        name=product_name,
        current_licenses=meta["current_licenses"],
        current_usage=usage.get(product_name, 0),
        last_updated=last_updated.get(product_name, date.today()),
        contract_renewal=meta["contract_renewal"],
    )
