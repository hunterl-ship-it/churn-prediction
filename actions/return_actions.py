"""Return-risk action recommendations (ported from woodwide-starters SDK heuristics)."""

from __future__ import annotations

import pandas as pd

from woodwide.core import as_number, probability_sort_column


def _number(value):
    return as_number(value)


def _boolean(value):
    if value is True:
        return True
    text = str(value).strip().lower()
    return text in {"true", "1", "yes"}


def _choose_signal(record):
    late_delivery = _boolean(record.get("late_delivery_risk") or record.get("late_delivery"))
    return_rate = _number(record.get("customer_return_rate"))
    support_contacts = _number(record.get("customer_support_contacts") or record.get("support_contacts"))
    size_exchange_history = _number(record.get("size_exchange_history"))
    order_value = _number(record.get("order_value_eur") or record.get("order_value"))
    final_sale = _boolean(record.get("final_sale"))

    if late_delivery and (support_contacts or 0) > 1:
        return "Delivery friction + support contact"
    if (return_rate or 0) > 0.55:
        return "High customer return history"
    if (size_exchange_history or 0) >= 3:
        return "Sizing uncertainty"
    if final_sale:
        return "Policy mismatch risk"
    if (order_value or 0) > 650:
        return "High-value return exposure"
    return "Unusual return profile"


def _explanation_for_signal(signal: str) -> str:
    explanations = {
        "Delivery friction + support contact": "The order arrived late and the customer contacted support, increasing return or refund pressure.",
        "High customer return history": "The customer has a higher-than-normal return pattern across recent orders.",
        "Sizing uncertainty": "Prior size exchanges suggest this order may need fit guidance before the return window closes.",
        "Policy mismatch risk": "The order has policy constraints that may create dissatisfaction if expectations are not clarified.",
        "High-value return exposure": "The order value is high enough to prioritize proactive retention or exchange support.",
        "Unusual return profile": "The order differs from normal return patterns across several moderate signals.",
    }
    return explanations.get(signal, explanations["Unusual return profile"])


def _severity_from_score(score: float) -> str:
    if score >= 0.82:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _recommendation_for_signal(signal: str, severity: str) -> tuple[str, str]:
    if "Delivery" in signal:
        return (
            "Service recovery",
            "Send a proactive apology, shipping credit, and exchange-first support option.",
        )
    if "Sizing" in signal:
        return (
            "Fit guidance",
            "Trigger size guidance and exchange recommendations before the customer starts a return.",
        )
    if severity == "high":
        return (
            "High-touch save",
            "Route to a specialist with exchange incentives and policy clarification.",
        )
    return (
        "Proactive return prevention",
        "Send targeted care instructions, exchange options, and support contact shortcuts.",
    )


def analyze_return_row(row: pd.Series, api_score=None) -> dict:
    record = row.to_dict()
    order_value = _number(record.get("order_value_eur") or record.get("order_value"))
    return_rate = _number(record.get("customer_return_rate"))
    days_since_purchase = _number(record.get("days_since_purchase"))
    support_contacts = _number(record.get("customer_support_contacts") or record.get("support_contacts"))
    size_exchange_history = _number(record.get("size_exchange_history"))
    late_delivery = _boolean(record.get("late_delivery_risk") or record.get("late_delivery"))
    final_sale = _boolean(record.get("final_sale"))

    heuristic_score = 0.16
    if order_value is not None and order_value > 180:
        heuristic_score += min(0.16, order_value / 2500)
    if return_rate is not None:
        heuristic_score += min(0.26, return_rate * 0.32)
    if days_since_purchase is not None and days_since_purchase > 18:
        heuristic_score += min(0.12, days_since_purchase / 220)
    if support_contacts is not None:
        heuristic_score += min(0.18, support_contacts * 0.06)
    if size_exchange_history is not None:
        heuristic_score += min(0.15, size_exchange_history * 0.05)
    if late_delivery:
        heuristic_score += 0.13
    if final_sale:
        heuristic_score += 0.08

    score = min(0.98, max(0.02, api_score if api_score is not None else heuristic_score))
    signal = _choose_signal(record)
    severity = _severity_from_score(score)
    category, action = _recommendation_for_signal(signal, severity)
    return {
        "return_score": round(score, 3),
        "likely_signal": signal,
        "signal_explanation": _explanation_for_signal(signal),
        "action_severity": severity,
        "action_category": category,
        "recommended_action": action,
    }


def build_return_action_plan(flagged: pd.DataFrame) -> pd.DataFrame:
    if flagged.empty:
        return flagged

    score_column = probability_sort_column(flagged)
    rows = []
    for _, row in flagged.iterrows():
        api_score = None
        if score_column and score_column in row.index:
            api_score = _number(row[score_column])
            if api_score is not None and api_score > 1:
                api_score = api_score / 100
        rows.append(analyze_return_row(row, api_score=api_score))
    return pd.concat([flagged.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
