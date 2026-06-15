# Churn Intervention Studio

**Live app:** [woodwide-churn.streamlit.app](https://woodwide-churn.streamlit.app/)

A Streamlit app that turns historical churn data into a ranked list of at-risk customers, explains *why* they look risky, and suggests what to do about it. The heavy lifting lives on the [Wood Wide](https://woodwide.ai) API; this repo is the workflow and presentation layer on top.

Upload your CSVs in the browser — no setup required. Sample train and test files are available for download inside the app if you want to try it without client data.

---

## The problem we're solving

Most retention teams already know churn is expensive. What they often lack is a repeatable way to answer three questions at once:

1. **Who** should we worry about right now?
2. **Why** does the model think they'll leave?
3. **What** should we actually do about each account?

Spreadsheets and generic "win-back" playbooks don't connect those dots. A model score alone isn't enough — CSMs and sales need context they can act on in a conversation, not a probability floating in a dashboard.

This app is built around that gap. You upload historical customers with a churn label, score your active base, review risk drivers and patterns, then get a draft intervention plan per customer with urgency levels and suggested actions.

---

## How the Wood Wide API is used

The app talks to Wood Wide over HTTPS with a bearer token. The flow looks like this:

```
Upload CSV → create/reuse dataset → train models → run inference → optional row explanations → build plan
```

**Datasets.** Training and scoring files are uploaded once and cached by content hash. If the same file comes back on a later run, the app reuses the existing dataset instead of uploading again.

**Churn prediction model.** A classification model is trained on historical data (`model_type: prediction`). The app auto-detects the churn label column and excludes IDs and label-like fields from inputs. Inference runs asynchronously (`/models/{id}/infer-async`), polls until the job completes, and returns per-row predictions plus probabilities.

**Explanations.** For the highest-risk rows, the app can call `/jobs/{inference_job_id}/explain` to get natural-language, customer-specific explanations. These feed both the UI and the intervention matcher.

**Pattern models (at-risk subset only).** After isolating predicted churners, the app trains two more models on that slice:

- **Factor analysis** — finds latent patterns in *why* at-risk customers look similar (variance / themes).
- **Clustering** — groups at-risk customers into segments for context.

**Feature weights.** When the API returns class-level `prediction_descriptions` (with `feature_contributions` and weights), the app surfaces them as model risk drivers. If that metadata is missing, it falls back to other sources (refreshed job results, cached training descriptions, or a feature-contrast summary).

**Caching.** Job IDs, inference results, and trained model IDs are cached so reruns don't re-upload data or retrain models unnecessarily. Long-running jobs can be resumed after a page refresh.

---

## How the intervention layer works

The intervention system is **not** a second ML model. It's a rules-plus-matching layer that maps rich customer context to a catalog of playbooks.

**Intervention catalogs.** The app ships three template libraries — generic B2B, SaaS-specific, and streaming/subscription — plus an auto-detect mode that picks one based on column names in the training data. You can also edit the catalog in the sidebar.

Each playbook entry has:

- A **name** (e.g. "Engagement recovery", "Renewal save motion")
- A **description** used for semantic matching
- **Keywords** matched against actual column names and values
- Three **actions** at low / medium / high urgency

**Building context per customer.** For each at-risk row, the app assembles a text bundle from:

- Primary at-risk **factor** (from factor analysis)
- **Cluster** description
- Model **feature weights** / risk drivers
- Row-level **prediction explanation** (if generated)
- Class-level **prediction description**

**Matching.** Each catalog entry is scored against that context using:

- Cosine similarity on tokenized text (explanations and descriptions weigh more than generic text)
- Keyword hits on the customer's feature values
- Boosts when the primary factor theme aligns with certain playbook names (billing → billing recovery, usage → adoption plan, etc.)

**Urgency.** High / medium / low is derived from the customer's churn probability relative to other at-risk customers (top ~20% → high, bottom ~30% → low, middle → medium). The matched playbook's action text for that urgency level becomes the recommendation.

**Diversity.** As the plan is built row by row, the matcher slightly penalizes playbooks already assigned many times globally and within the same cluster — so the output isn't "send the same email to everyone."

The result is a table you can download: customer ID, probability, segment context, intervention category, specific action, urgency, and what signal drove the match.

---

## Limitations (honest ones)

This is a demo, not a production retention platform.

- **Depends on API metadata.** Feature-weight charts need `prediction_descriptions` from Wood Wide. Some inference runs return that; others don't. Fallbacks exist but aren't as precise as true model attributions.
- **Interventions are templated, not learned.** Matching is heuristic (similarity + keywords), not optimized from historical save rates or campaign outcomes.
- **Single-user Streamlit app.** No auth, multi-tenancy, audit logs, or role-based access.
- **Synchronous UX with polling.** Training and inference can take minutes on large files. The app waits in-process (with timeouts) rather than using a proper job queue and notifications.
- **No closed loop.** There's no tracking of whether an intervention was sent, accepted, or worked. No A/B testing, no feedback into the model.
- **Data handling.** Uploaded files are processed in-session; fine for a pilot, not a full enterprise data-governance story.
- **Column assumptions.** Auto-detection works well on telco and SaaS-style schemas; messy client data may need manual column mapping (not exposed in the UI today).

---

## What we'd improve for production

If this were going beyond a sales engineering demo, the roadmap would look roughly like this:

1. **Reliable attributions** — Treat feature weights as a first-class API contract; fail loudly or queue a refresh when missing, rather than silently falling back.
2. **Intervention outcomes** — Store recommended actions in a CRM or CS platform, track execution, and measure save rate by playbook and segment.
3. **Proper orchestration** — Move training/inference to background workers (or Wood Wide webhooks), with email/Slack when jobs finish instead of blocking Streamlit.
4. **Governance** — Tenant isolation, encrypted secrets, PII handling, and configurable retention for cached artifacts.
5. **Human-in-the-loop review** — Let CSMs approve, edit, or reject suggestions before anything customer-facing goes out.
6. **Configurable business rules** — ARR thresholds, account tier overrides, and "never suggest discount to enterprise" style constraints on top of the matcher.
7. **Monitoring** — Drift detection on input features, model freshness, and intervention distribution (catch if one playbook dominates because of a bug).
8. **Tests** — Contract tests against Wood Wide API response shapes so parsing doesn't break silently on API changes.

---

## Project structure

| Path | Purpose |
|------|---------|
| `streamlit.py` | Full app: API integration, analysis workflow, intervention engine |
| `datasets/` | Default train/test CSVs bundled for demos |
| `requirements.txt` | Python dependencies |

---

Built as a Wood Wide API workflow demo. Questions about the API itself belong with Wood Wide; questions about this app's matching logic and UI are fair game in the repo.
