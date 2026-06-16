"""Fraud review recommendations (ported from woodwide-starters SDK heuristics)."""

from __future__ import annotations

import pandas as pd

from woodwide.core import as_number


def _number(value):
    parsed = as_number(value)
    return parsed


def _boolean(value):
    if value is True:
        return True
    text = str(value).strip().lower()
    return text in {"true", "1", "yes"}


def _choose_signal(record):
    device_mismatch = _boolean(record.get("device_mismatch") or record.get("address_mismatch"))
    velocity = _number(record.get("velocity_score") or record.get("high_risk_ip"))
    attempts = _number(record.get("payment_attempts") or record.get("previous_orders"))
    geo_distance = _number(record.get("geo_distance_miles") or record.get("shipping_distance_km"))
    chargebacks = _number(record.get("chargeback_history") or record.get("review_score"))
    amount = _number(record.get("amount") or record.get("order_value_eur") or record.get("avg_order_value_eur"))

    if device_mismatch and (velocity or 0) > 0:
        return "Velocity + device mismatch"
    if (chargebacks or 0) > 1:
        return "Chargeback history"
    if (geo_distance or 0) > 1500:
        return "Impossible travel pattern"
    if (attempts or 0) >= 4:
        return "High-value retry pattern"
    if (amount or 0) > 1500:
        return "High-value transaction"
    return "Unusual transaction profile"


def _explanation_for_signal(signal: str) -> str:
    explanations = {
        "Velocity + device mismatch": "Recent activity is unusually fast and coming from a device profile that does not match the account baseline.",
        "Chargeback history": "The account has prior chargeback activity, increasing review priority for this transaction.",
        "Impossible travel pattern": "Location movement is unusually large for the observed account timeline.",
        "High-value retry pattern": "Multiple payment attempts around a high-value event can indicate abuse or card testing.",
        "High-value transaction": "The transaction value is large enough to merit additional verification.",
        "Unusual transaction profile": "The record differs from normal transaction patterns across several moderate signals.",
    }
    return explanations.get(signal, explanations["Unusual transaction profile"])


def _severity_from_score(score: float) -> str:
    if score >= 0.82:
        return "high"
    if score >= 0.62:
        return "medium"
    return "low"


def _recommendation_for_signal(signal: str, severity: str) -> tuple[str, str]:
    if severity == "high":
        return (
            "Manual fraud review",
            "Hold fulfillment, verify identity, and review device, payment, and location evidence before approval.",
        )
    if "device" in signal.lower() or "travel" in signal.lower():
        return (
            "Step-up verification",
            "Request additional verification before letting the transaction continue.",
        )
    return (
        "Monitor transaction",
        "Allow the transaction with enhanced monitoring and add the account to the review queue.",
    )


def analyze_fraud_row(row: pd.Series, api_score=None) -> dict:
    record = row.to_dict()
    amount = _number(record.get("order_value_eur") or record.get("amount"))
    velocity = _number(record.get("high_risk_ip"))
    attempts = _number(record.get("previous_orders"))
    geo_distance = _number(record.get("shipping_distance_km"))
    chargebacks = _number(record.get("review_score"))
    device_mismatch = _boolean(record.get("address_mismatch"))

    heuristic_score = 0.18
    if amount is not None and amount > 500:
        heuristic_score += min(0.18, amount / 6000)
    if velocity is not None:
        heuristic_score += min(0.28, velocity / 360)
    if attempts is not None and attempts > 1:
        heuristic_score += min(0.16, attempts * 0.04)
    if geo_distance is not None and geo_distance > 500:
        heuristic_score += min(0.16, geo_distance / 9000)
    if chargebacks is not None:
        heuristic_score += min(0.14, chargebacks * 0.07)
    if device_mismatch:
        heuristic_score += 0.16

    score = min(0.98, max(0.02, api_score if api_score is not None else heuristic_score))
    signal = _choose_signal(record)
    severity = _severity_from_score(score)
    category, action = _recommendation_for_signal(signal, severity)
    return {
        "fraud_score": round(score, 3),
        "likely_signal": signal,
        "signal_explanation": _explanation_for_signal(signal),
        "review_severity": severity,
        "review_category": category,
        "recommended_action": action,
    }


def build_fraud_review_plan(flagged: pd.DataFrame, score_column=None) -> pd.DataFrame:
    if flagged.empty:
        return flagged

    from woodwide.core import anomaly_score_column, probability_sort_column

    score_column = score_column or probability_sort_column(flagged) or anomaly_score_column(flagged)
    rows = []
    for _, row in flagged.iterrows():
        api_score = None
        if score_column and score_column in row.index:
            api_score = _number(row[score_column])
            if api_score is not None and api_score > 1:
                api_score = api_score / 100
        rows.append(analyze_fraud_row(row, api_score=api_score))
    return pd.concat([flagged.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
