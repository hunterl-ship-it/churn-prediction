#!/usr/bin/env python3
"""Compute PCA patterns and KMeans segments from real at_risk demo artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from woodwide.core import analysis_dataframe_for_modeling, clustering_dataframe_for_modeling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMOS = ["churn", "noshow"]
N_PATTERNS = 6
N_CLUSTERS = 4
RANDOM_STATE = 42


def compute_patterns(modeling_df: pd.DataFrame, input_columns: list[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(modeling_df[input_columns].fillna(0))
    n_components = min(N_PATTERNS, len(input_columns), scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(scaled)

    rows = []
    for i, (var, loadings) in enumerate(zip(pca.explained_variance_ratio_, pca.components_)):
        top_feature_idx = np.argsort(np.abs(loadings))[::-1][:3]
        top_features = ", ".join(input_columns[j] for j in top_feature_idx)
        rows.append({
            "pattern": f"Pattern {chr(65 + i)}",
            "captured_variance": round(float(var), 4),
            "top_drivers": top_features,
        })
    return pd.DataFrame(rows)


def compute_cluster_labels(cluster_df: pd.DataFrame, input_columns: list[str], n: int) -> np.ndarray:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df[input_columns].fillna(0))
    n_clusters = min(n, len(cluster_df))
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    raw_labels = km.fit_predict(scaled)
    # Relabel by cluster size descending so Segment 1 is largest
    counts = np.bincount(raw_labels)
    rank = np.argsort(-counts)
    remap = {old: new for new, old in enumerate(rank)}
    return np.array([remap[l] for l in raw_labels])


def process(page_id: str) -> None:
    artifact_dir = PROJECT_ROOT / "demo_artifacts" / page_id
    at_risk = pd.read_csv(artifact_dir / "at_risk.csv")
    print(f"{page_id}: {len(at_risk)} at-risk rows, {len(at_risk.columns)} columns")

    modeling_df, _ = analysis_dataframe_for_modeling(at_risk)
    cluster_df, cluster_columns = clustering_dataframe_for_modeling(modeling_df)

    # If preferred columns don't yield numeric features, fall back to all numeric columns
    if not cluster_columns:
        cluster_df, cluster_columns = clustering_dataframe_for_modeling(at_risk)

    print(f"  numeric columns ({len(cluster_columns)}): {cluster_columns[:6]}...")

    patterns = compute_patterns(cluster_df, cluster_columns)
    patterns_path = artifact_dir / "patterns.csv"
    patterns.to_csv(patterns_path, index=False)
    print(f"  saved {patterns_path} ({len(patterns)} patterns)")

    if not cluster_df.empty and cluster_columns:
        labels = compute_cluster_labels(cluster_df, cluster_columns, N_CLUSTERS)
        at_risk = at_risk.copy()
        at_risk["cluster_label"] = None
        at_risk.loc[cluster_df.index, "cluster_label"] = [f"Segment {l + 1}" for l in labels]
        at_risk.to_csv(artifact_dir / "at_risk.csv", index=False)
        print(f"  cluster distribution: {dict(zip(*np.unique(at_risk['cluster_label'], return_counts=True)))}")
    else:
        print("  skipping clusters: insufficient numeric columns")


if __name__ == "__main__":
    for demo in DEMOS:
        artifact_dir = PROJECT_ROOT / "demo_artifacts" / demo
        if not artifact_dir.exists():
            print(f"skipping {demo}: no artifacts dir")
            continue
        process(demo)
