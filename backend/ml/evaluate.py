import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict:
    mae = float(mean_absolute_error(actuals, predictions))
    rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
    # avoid division by zero in MAPE
    mask = actuals != 0
    mape = float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])))
    r2 = float(r2_score(actuals, predictions))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def format_metrics_for_display(metrics: dict) -> dict:
    return {
        "mae": round(metrics["mae"], 4),
        "rmse": round(metrics["rmse"], 4),
        "mape": round(metrics["mape"], 4),
        "mape_pct": f"{metrics['mape'] * 100:.2f}%",
        "r2": round(metrics["r2"], 4),
    }
