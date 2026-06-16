# Risk Operations Demo

**Live app:** [woodwide-churn.streamlit.app](https://woodwide-churn.streamlit.app/) — deploy with `Home.py`

A multipage Streamlit demo for **Churn Intervention** and **Patient No-Show** hero workflows, plus fraud, returns, and segmentation examples. Built on the [Wood Wide](https://woodwide.ai) API with a **hybrid instant + live** demo experience.

---

## Hero demos (start here)

| Page | Instant demo | Live API |
|------|--------------|----------|
| Churn Intervention | Pre-computed metrics, at-risk list, drivers, action plan | Train + score ~270k rows via Wood Wide |
| Patient No-Show | Same workflow for appointment no-shows | Same pipeline on healthcare dataset |

**Instant mode (default):** loads `demo_artifacts/` captured from a real Wood Wide train + infer run — no API key required.

**Live mode:** trains and scores via Wood Wide; requires `WOODWIDE_API_KEY` pointed at production (`https://api.woodwide.ai`).

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run Home.py
```

Environment variables (`.env` or Streamlit secrets):

| Variable | Purpose |
|----------|---------|
| `WOODWIDE_API_KEY` | Required for **Live API** mode |
| `WOODWIDE_BASE_URL` | Defaults to `https://api.woodwide.ai` |
| `PILOT_CTA_URL` | Book-a-pilot link (defaults to `https://woodwide.ai`) |

Copy [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) for Streamlit Cloud.

---

## Demo datasets

| Path | Used by |
|------|---------|
| `datasets/churn/train.csv`, `test.csv`, `eval.csv` | Churn hero demo |
| `datasets/healthcare/train.csv`, `test.csv`, `eval.csv` | Patient no-show hero demo |
| `demo_artifacts/churn/`, `demo_artifacts/noshow/` | Instant demo bundles |

Regenerate eval holdouts:

```bash
python scripts/prepare_eval_splits.py
```

Capture instant artifacts from a real Wood Wide API run (requires `WOODWIDE_API_KEY`):

```bash
PYTHONPATH=. python scripts/capture_demo_artifacts.py
```

Options:

```bash
# Reuse an already-trained model (skip training)
PYTHONPATH=. python scripts/capture_demo_artifacts.py --reuse-model-id churn:YOUR_MODEL_ID

# Faster dev capture on a training sample
PYTHONPATH=. python scripts/capture_demo_artifacts.py --sample-train 20000 --demos noshow
```

Each bundle includes `metadata.json` (model id, job ids, capture timestamp) plus `metrics.json`, `at_risk.csv`, `driver_chart.csv`, `intervention_plan.csv`, and `explanations.json`.

Offline sklearn baseline (not used for shipped demos):

```bash
PYTHONPATH=. python scripts/generate_demo_artifacts.py
```

Smoke test:

```bash
python scripts/smoke_test_demo.py
```

---

## Architecture

```
Home.py                          # Prospect landing + hero CTAs
pages/1_Churn_Intervention.py    # Thin wrapper
pages/5_Patient_No_Show.py       # Thin wrapper
workflows/supervised_outcome.py  # Shared hero workflow (instant + live)
workflows/instant_demo.py        # Artifact loader
woodwide/evaluation.py           # AUC, PR-AUC, lift, confusion matrix
demo_artifacts/{churn,noshow}/   # metrics.json, at_risk.csv, etc.
```

Hero workflow: **Performance → high-risk list → drivers → patterns → segments → action plan → pilot CTA**

---

## Secondary pages

| Page | Model type | Notes |
|------|------------|-------|
| Fraud Detection | prediction / anomaly | E-commerce demo |
| Return Risk | prediction | Return prevention actions |
| Customer Segments | clustering | Segment export |

---

Built as a Wood Wide API workflow demo.
