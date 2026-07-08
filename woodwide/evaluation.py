"""Model evaluation metrics for binary outcome demos."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


def _as_binary_labels(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().mean() > 0.9:
        return values.fillna(0).astype(int).to_numpy()
    normalized = series.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    positive = {"1", "true", "yes", "y", "churn", "churned", "noshow", "missed"}
    return normalized.isin(positive).astype(int).to_numpy()


def _as_scores(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy()


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = positives.sum()
    n_neg = negatives.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg
    return float(np.trapezoid(tpr, fpr))


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_true == 1
    n_pos = positives.sum()
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted)
    precision = tp_cum / (np.arange(len(y_sorted)) + 1)
    return float(precision[y_sorted == 1].sum() / n_pos)


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, int]:
    predicted = (y_score >= threshold).astype(int)
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    tn = int(((predicted == 0) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    counts = confusion_at_threshold(y_true, y_score, threshold)
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0.0
    return {"precision": float(precision), "recall": float(recall)}


def lift_at_top_fraction(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.10) -> float:
    if len(y_true) == 0:
        return float("nan")
    baseline = y_true.mean()
    if baseline == 0:
        return float("nan")
    top_n = max(1, int(len(y_true) * fraction))
    order = np.argsort(-y_score)
    top_rate = y_true[order[:top_n]].mean()
    return float(top_rate / baseline)


def capture_rate_at_top_fraction(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.10) -> float:
    positives = y_true.sum()
    if positives == 0:
        return float("nan")
    top_n = max(1, int(len(y_true) * fraction))
    order = np.argsort(-y_score)
    captured = y_true[order[:top_n]].sum()
    return float(captured / positives)


def compute_binary_metrics(
    labels: pd.Series,
    scores: pd.Series,
    threshold: float = 0.5,
    top_fraction: float = 0.10,
) -> dict[str, Any]:
    y_true = _as_binary_labels(labels)
    y_score = _as_scores(scores)
    y_score, _ = orient_scores_for_labels(y_true, y_score)
    pr = precision_recall_at_threshold(y_true, y_score, threshold)
    counts = confusion_at_threshold(y_true, y_score, threshold)
    return {
        "auc_roc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "precision": pr["precision"],
        "recall": pr["recall"],
        "lift_top_decile": lift_at_top_fraction(y_true, y_score, top_fraction),
        "capture_top_decile": capture_rate_at_top_fraction(y_true, y_score, top_fraction),
        "threshold": threshold,
        "top_fraction": top_fraction,
        "positive_rate": float(y_true.mean()),
        "eval_rows": int(len(y_true)),
        "confusion": counts,
    }


def orient_scores_for_labels(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Return scores oriented so higher values mean positive class, flipping if needed."""
    scores = np.asarray(y_score, dtype=float)
    if len(scores) == 0:
        return scores, False
    auc = roc_auc_score(y_true, scores)
    if np.isnan(auc) or auc >= 0.5:
        return scores, False
    return 1.0 - scores, True


def oriented_probability_series(
    labels: pd.Series,
    scores: pd.Series,
) -> tuple[pd.Series, bool]:
    y_true = _as_binary_labels(labels)
    y_score = _as_scores(scores)
    oriented, flipped = orient_scores_for_labels(y_true, y_score)
    return pd.Series(oriented, index=scores.index), flipped


def metrics_to_json(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, indent=2)


def load_metrics_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def render_performance_panel(
    metrics: dict[str, Any],
    title: str = "Model Performance",
    animate: bool = False,
    skip_animation: bool = False,
):
    with st.expander(title, expanded=True):
        st.markdown(
            '<div class="section-note">Holdout evaluation on a labeled sample from the demo dataset.</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        columns = [col1, col2, col3, col4, col5]
        specs = [
            ("AUC-ROC", float(metrics.get("auc_roc", 0)), lambda value: f"{value:.3f}"),
            ("PR-AUC", float(metrics.get("pr_auc", 0)), lambda value: f"{value:.3f}"),
            ("Precision @ threshold", float(metrics.get("precision", 0)), lambda value: f"{value:.1%}"),
            ("Lift @ top 10%", float(metrics.get("lift_top_decile", 0)), lambda value: f"{value:.1f}x"),
            ("Capture @ top 10%", float(metrics.get("capture_top_decile", 0)), lambda value: f"{value:.1%}"),
        ]

        if animate and not skip_animation:
            from shared.instant_playback import animate_metrics_row

            animate_metrics_row(columns, specs, duration=1.0, skip=False)
        else:
            for column, (label, target, formatter) in zip(columns, specs):
                column.metric(label, formatter(target))

        confusion = metrics.get("confusion") or {}
        if confusion:
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.caption(
                    f"Threshold {metrics.get('threshold', 0.5):.0%} · "
                    f"Recall {metrics.get('recall', 0):.1%} · "
                    f"{metrics.get('eval_rows', 0):,} eval rows"
                )
            with detail_col2:
                st.caption(
                    f"TP {confusion.get('tp', 0):,} · FP {confusion.get('fp', 0):,} · "
                    f"FN {confusion.get('fn', 0):,} · TN {confusion.get('tn', 0):,}"
                )
