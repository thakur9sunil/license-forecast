"""
Business logic for license renewal recommendations.
Uses upper confidence interval + 10% buffer to bias toward avoiding under-licensing.
"""

PRODUCT_LICENSE_COUNTS = {
    "Jira": 550,
    "Slack": 700,
    "Zoom": 420,
}

BUFFER_PCT = 0.10


def compute_recommendation(
    product: str,
    predicted_at_renewal: float,
    yhat_upper: float,
    current_licensed: int | None = None,
    buffer_pct: float = BUFFER_PCT,
) -> tuple[str, str]:
    if current_licensed is None:
        current_licensed = PRODUCT_LICENSE_COUNTS.get(product, int(predicted_at_renewal))

    if yhat_upper > current_licensed:
        predicted_need = yhat_upper * (1 + buffer_pct)
        shortfall = int(predicted_need - current_licensed)
        return (
            "buy_more",
            f"Usage is projected to reach ~{int(predicted_at_renewal)} licenses "
            f"(upper bound ~{int(yhat_upper)}). Consider purchasing {shortfall} "
            f"additional licenses to avoid gaps at renewal.",
        )
    elif predicted_at_renewal < current_licensed * 0.80:
        surplus = int(current_licensed - predicted_at_renewal)
        return (
            "reduce",
            f"Usage is projected to drop to ~{int(predicted_at_renewal)} licenses, "
            f"~{surplus} below your current count of {current_licensed}. "
            f"Reducing licenses at renewal could lower costs.",
        )
    else:
        return (
            "hold",
            f"Usage is projected at ~{int(predicted_at_renewal)} licenses, "
            f"within the comfortable range of your current {current_licensed} licenses. "
            f"No change recommended at renewal.",
        )
