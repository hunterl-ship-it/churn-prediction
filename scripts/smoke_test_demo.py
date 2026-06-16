#!/usr/bin/env python3
"""Smoke-test instant demo artifacts and eval splits."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from workflows.instant_demo import artifacts_available, load_metadata, load_metrics


def main():
    failures = []
    for page_id in ("churn", "noshow"):
        if not artifacts_available(page_id):
            failures.append(f"Missing artifacts for {page_id}")
            continue
        metrics = load_metrics(page_id)
        if not metrics or metrics.get("auc_roc", 0) < 0.6:
            failures.append(f"Weak or missing metrics for {page_id}: {metrics}")

        metadata = load_metadata(page_id)
        if metadata and metadata.get("source") != "woodwide_api":
            failures.append(f"Artifacts for {page_id} are not from Wood Wide API capture")
        elif not metadata:
            failures.append(f"Missing metadata.json for {page_id} (run capture_demo_artifacts.py)")

        eval_path = PROJECT_ROOT / "datasets" / ("churn" if page_id == "churn" else "healthcare") / "eval.csv"
        if not eval_path.exists():
            failures.append(f"Missing eval split: {eval_path}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        sys.exit(1)

    print("OK: instant demo artifacts and eval splits look healthy.")


if __name__ == "__main__":
    main()
