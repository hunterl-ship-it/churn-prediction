"""Shared Streamlit UI helpers for the multipage demo."""

from __future__ import annotations

import base64
import html
import os
from dataclasses import dataclass
from functools import lru_cache

import streamlit as st

from woodwide.core import (
    MODEL_TRAIN_TIMEOUT_SECONDS,
    PREVIEW_ROW_COUNT,
    api_key,
    base_url,
    cache_load_pending_jobs,
    cache_load_ready_models,
    init_local_cache,
)

try:
    from woodwide.core import PILOT_CTA_URL
except ImportError:
    PILOT_CTA_URL = os.environ.get("PILOT_CTA_URL", "https://woodwide.ai")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGO_PATH = os.path.join(REPO_ROOT, "wood-wide-ai-logo.svg")
FAVICON_PATH = os.path.join(REPO_ROOT, "woodwide-icon.png")

PAGE_CHURN = "pages/1_Churn_Intervention.py"
PAGE_FRAUD = "pages/2_Fraud_Detection.py"
PAGE_RETURNS = "pages/3_Return_Risk.py"
PAGE_SEGMENTS = "pages/4_Customer_Segments.py"
PAGE_NOSHOW = "pages/5_Patient_No_Show.py"

APP_CSS = """
<style>
:root {
    --ww-green: #0B3D2E;
    --ww-green-soft: #123f2b;
    --ww-cream: #F5F3E7;
    --ww-border: #d8e2dc;
    --ww-muted: #668575;
    --ww-text: #11231a;
    --ww-card: #ffffff;
}
.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fbfaf6 0%, #f5f3e7 100%);
    border-right: 1px solid var(--ww-border);
}
div[data-testid="stMetric"] {
    border: 1px solid var(--ww-border);
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    background: var(--ww-card);
}
div[data-testid="stMetricLabel"] {
    color: var(--ww-muted);
}
div[data-testid="stMetricValue"] {
    color: var(--ww-green);
}
.section-note {
    color: var(--ww-muted);
    font-size: 0.95rem;
    margin-top: -0.35rem;
    margin-bottom: 0.75rem;
    line-height: 1.5;
}
.pilot-cta {
    border: 1px solid var(--ww-border);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    background: var(--ww-cream);
    margin-top: 1rem;
    color: var(--ww-text);
}
.pilot-cta a {
    color: var(--ww-green);
    font-weight: 650;
    text-decoration: none;
}
.pilot-cta a:hover {
    text-decoration: underline;
}
.ww-brand-header {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--ww-border);
}
.ww-brand-header-logo-link {
    flex: 0 0 auto;
    line-height: 0;
}
.ww-brand-header-logo {
    height: 42px;
    width: auto;
    display: block;
}
.ww-brand-header-copy {
    min-width: 0;
}
.ww-brand-header-eyebrow {
    margin: 0 0 0.35rem;
    color: var(--ww-muted);
    font-size: 0.72rem;
    font-weight: 750;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.ww-brand-header-title {
    margin: 0;
    color: var(--ww-green);
    font-size: clamp(1.35rem, 2vw, 1.85rem);
    line-height: 1.2;
    font-weight: 700;
}
.ww-brand-header-subtitle {
    margin: 0.45rem 0 0;
    color: var(--ww-muted);
    font-size: 1rem;
    line-height: 1.5;
    max-width: 52rem;
}
.ww-library-intro {
    border: 1px solid var(--ww-border);
    border-radius: 14px;
    background: linear-gradient(135deg, #ffffff 0%, #f8fbf9 100%);
    padding: 1.1rem 1.25rem;
    margin-bottom: 1.5rem;
}
.ww-library-intro p {
    margin: 0;
    color: var(--ww-text);
    line-height: 1.55;
}
.ww-library-intro strong {
    color: var(--ww-green);
}
.ww-demo-section-title {
    margin: 0 0 0.35rem;
    color: var(--ww-green);
    font-size: 1.05rem;
    font-weight: 700;
}
.ww-demo-section-note {
    margin: 0 0 1rem;
    color: var(--ww-muted);
    font-size: 0.92rem;
}
.ww-demo-grid {
    display: grid;
    gap: 1rem;
    margin-bottom: 1.5rem;
    width: 100%;
    align-items: stretch;
}
.ww-demo-grid_hero {
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));
}
.ww-demo-grid_secondary {
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));
}
.ww-demo-card {
    display: flex;
    flex-direction: column;
    height: 100%;
    border: 1px solid var(--ww-border);
    border-radius: 14px;
    background: var(--ww-card);
    padding: 1.15rem 1.2rem 1.2rem;
    box-shadow: 0 1px 2px rgba(11, 61, 46, 0.04);
    box-sizing: border-box;
}
.ww-demo-card_featured {
    background: linear-gradient(180deg, #ffffff 0%, #fbfaf6 100%);
    border-color: #c9d8d0;
}
.ww-demo-card__badge {
    display: inline-block;
    align-self: flex-start;
    margin-bottom: 0.75rem;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    background: var(--ww-cream);
    color: var(--ww-green);
    border: 1px solid var(--ww-border);
    font-size: 0.68rem;
    font-weight: 750;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.ww-demo-card__badge_secondary {
    background: #f8fbf9;
    color: var(--ww-muted);
}
.ww-demo-card__title {
    margin: 0 0 0.55rem;
    color: var(--ww-green);
    font-size: 1.15rem;
    line-height: 1.25;
    font-weight: 700;
}
.ww-demo-card__body {
    margin: 0;
    color: var(--ww-muted);
    font-size: 0.94rem;
    line-height: 1.55;
    flex: 1;
}
.ww-demo-card__meta {
    margin: 0.85rem 0 1rem;
    color: var(--ww-green-soft);
    font-size: 0.86rem;
    line-height: 1.45;
}
.ww-demo-card__cta {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    min-height: 2.65rem;
    margin-top: auto;
    border-radius: 10px;
    font-size: 0.92rem;
    font-weight: 650;
    text-decoration: none;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}
.ww-demo-card__cta_primary {
    background: var(--ww-green);
    color: #ffffff !important;
    border: 1px solid var(--ww-green);
}
.ww-demo-card__cta_primary:hover {
    background: var(--ww-green-soft);
    border-color: var(--ww-green-soft);
}
.ww-demo-card__cta_secondary {
    background: #ffffff;
    color: var(--ww-green) !important;
    border: 1px solid var(--ww-border);
}
.ww-demo-card__cta_secondary:hover {
    background: var(--ww-cream);
    border-color: #c9d8d0;
}
.ww-tech-panel {
    color: var(--ww-text);
    line-height: 1.55;
}
.ww-tech-panel h4 {
    margin: 1rem 0 0.35rem;
    color: var(--ww-green);
    font-size: 0.95rem;
}
.ww-tech-panel ul {
    margin: 0.2rem 0 0.75rem;
    padding-left: 1.2rem;
    color: var(--ww-muted);
}
.ww-tech-panel code {
    display: block;
    margin-top: 0.75rem;
    padding: 0.85rem 1rem;
    border-radius: 10px;
    background: #f8fbf9;
    border: 1px solid var(--ww-border);
    color: var(--ww-green-soft);
    font-size: 0.82rem;
    white-space: pre-wrap;
}
.ww-sidebar-note {
    color: var(--ww-muted);
    font-size: 0.88rem;
    line-height: 1.5;
}
.ww-sidebar-note strong {
    color: var(--ww-green);
}
</style>
"""


@dataclass(frozen=True)
class DemoCard:
    title: str
    description: str
    meta: str
    page: str
    cta: str
    featured: bool = False
    query: str = ""


@lru_cache(maxsize=1)
def _logo_data_uri() -> str:
    with open(LOGO_PATH, "rb") as logo_file:
        encoded = base64.b64encode(logo_file.read()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def logo_page_icon() -> str:
    return FAVICON_PATH if os.path.exists(FAVICON_PATH) else LOGO_PATH


def page_href(page_script: str, query: str = "") -> str:
    filename = os.path.basename(page_script).removesuffix(".py")
    if filename[:1].isdigit() and "_" in filename:
        filename = filename.split("_", 1)[1]
    href = f"/{filename}"
    if query:
        href = f"{href}?{query}"
    return href


def configure_demo_app(page_title: str):
    st.set_page_config(
        page_title=page_title,
        page_icon=logo_page_icon(),
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_brand_header(title: str | None = None, subtitle: str | None = None, *, home: bool = False):
    logo = _logo_data_uri()
    if home:
        title_text = "Interactive demo library"
        subtitle_text = (
            "Self-guided workflows for evaluating Wood Wide on realistic business data. "
            "Open any demo, explore end to end, and share the link with your team."
        )
        eyebrow = "Wood Wide AI"
    else:
        title_text = title or ""
        subtitle_text = subtitle or ""
        eyebrow = ""

    eyebrow_html = (
        f'<p class="ww-brand-header-eyebrow">{html.escape(eyebrow)}</p>' if eyebrow else ""
    )
    subtitle_html = (
        f'<p class="ww-brand-header-subtitle">{html.escape(subtitle_text)}</p>'
        if subtitle_text
        else ""
    )
    st.html(
        f"""
        <div class="ww-brand-header">
            <a href="/" class="ww-brand-header-logo-link">
                <img src="{logo}" alt="Wood Wide AI" class="ww-brand-header-logo" />
            </a>
            <div class="ww-brand-header-copy">
                {eyebrow_html}
                <h1 class="ww-brand-header-title">{html.escape(title_text)}</h1>
                {subtitle_html}
            </div>
        </div>
        """
    )


def render_demo_library_intro():
    st.markdown(
        """
        <div class="ww-library-intro">
            <p>
                <strong>Built for customer exploration.</strong>
                Each page is a standalone walkthrough of a real Wood Wide workflow—predictions,
                explanations, and recommended actions—so prospects can navigate on their own and
                see where numeric reasoning fits in their stack.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_demo_card(card: DemoCard) -> str:
    badge_class = "ww-demo-card__badge"
    if not card.featured:
        badge_class += " ww-demo-card__badge_secondary"
    badge = "Featured demo" if card.featured else "Interactive demo"
    cta_class = "ww-demo-card__cta_primary" if card.featured else "ww-demo-card__cta_secondary"
    card_class = "ww-demo-card"
    if card.featured:
        card_class += " ww-demo-card_featured"
    href = html.escape(page_href(card.page, card.query))
    return f"""
    <article class="{card_class}">
        <span class="{badge_class}">{badge}</span>
        <h3 class="ww-demo-card__title">{html.escape(card.title)}</h3>
        <p class="ww-demo-card__body">{html.escape(card.description)}</p>
        <p class="ww-demo-card__meta">{html.escape(card.meta)}</p>
        <a class="ww-demo-card__cta {cta_class}" href="{href}">{html.escape(card.cta)}</a>
    </article>
    """


def render_demo_card_grid(cards: list[DemoCard], *, featured: bool = False):
    grid_class = "ww-demo-grid_hero" if featured else "ww-demo-grid_secondary"
    cards_html = "".join(_render_demo_card(card) for card in cards)
    st.html(f'<div class="ww-demo-grid {grid_class}">{cards_html}</div>')


def render_home_sidebar():
    st.markdown(
        """
        <div class="ww-sidebar-note">
            <strong>How to explore</strong><br />
            Start with a featured demo for the full instant experience, or browse any workflow
            from the cards. Use the page menu to jump between demos at any time.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    render_pilot_cta()


class DemoCsvUpload:
    """Minimal UploadedFile stand-in backed by a local demo CSV."""

    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self._bytes: bytes | None = None

    def getvalue(self) -> bytes:
        if self._bytes is None:
            with open(self.path, "rb") as file:
                self._bytes = file.read()
        return self._bytes

    def seek(self, position: int):
        return None


def demo_csv(path: str | None):
    if path and os.path.exists(path):
        return DemoCsvUpload(path)
    return None


def demo_paths_available(*demo_paths: str) -> bool:
    paths = [path for path in demo_paths if path]
    return bool(paths) and all(os.path.exists(path) for path in paths)


def demo_mode_session_key(page_id: str) -> str:
    return f"{page_id}_demo_mode"


def get_demo_mode(page_id: str) -> str:
    return st.session_state.get(demo_mode_session_key(page_id), "instant")


def apply_demo_query_params(*session_keys: str):
    query = st.query_params
    demo_flag = query.get("demo")
    if demo_flag in ("churn", "noshow", "1", "true", "yes"):
        for key in session_keys:
            st.session_state[key] = True
    if demo_flag == "churn":
        st.session_state["churn_use_demo"] = True
    if demo_flag == "noshow":
        st.session_state["noshow_use_demo"] = True
    if query.get("mode") == "live":
        target = demo_flag if demo_flag in ("churn", "noshow") else session_keys[0].split("_")[0]
        st.session_state[demo_mode_session_key(target)] = "live"


def render_demo_mode_toggle(page_id: str, artifacts_available: bool) -> str:
    st.header("Demo mode")
    options = ["instant", "live"]
    labels = {
        "instant": "Instant (recommended)",
        "live": "Live Wood Wide API",
    }
    if not artifacts_available:
        st.caption("Instant demo artifacts were not found; using live API.")
        st.session_state[demo_mode_session_key(page_id)] = "live"
        return "live"

    choice = st.radio(
        "Experience",
        options,
        format_func=lambda value: labels[value],
        index=0 if get_demo_mode(page_id) == "instant" else 1,
        key=f"{page_id}_demo_mode_radio",
        help="Instant mode loads pre-computed results in seconds. Live mode trains and scores via the Wood Wide API.",
    )
    st.session_state[demo_mode_session_key(page_id)] = choice
    return choice


def render_demo_run_button(session_key: str, *demo_paths: str, label: str = "Run with demo datasets"):
    if not demo_paths_available(*demo_paths):
        st.caption("Demo dataset files were not found locally.")
        return

    names = ", ".join(f"`{os.path.basename(path)}`" for path in demo_paths if path)
    st.caption(f"Available demo files: {names}")
    if st.session_state.get(session_key, False):
        st.success("Demo datasets loaded.")
    elif st.button(label, use_container_width=True, key=f"{session_key}:button"):
        st.session_state[session_key] = True
        st.rerun()


def render_custom_csv_uploads(page_id: str, entity_name_plural: str):
    """Render CSV upload controls for rerunning a hero workflow with user data."""
    with st.expander("Run with your own CSVs", expanded=False):
        st.caption(
            "Upload a labeled training CSV and a scoring CSV with matching feature columns. "
            "An optional labeled eval CSV enables holdout performance metrics for your data."
        )
        training_data = st.file_uploader(
            "Training CSV",
            type=["csv"],
            key=f"{page_id}_custom_train_csv",
            help="Include the outcome label column so Wood Wide can train a prediction model.",
        )
        scoring_data = st.file_uploader(
            f"Scoring CSV ({entity_name_plural})",
            type=["csv"],
            key=f"{page_id}_custom_test_csv",
            help="Rows to score with the trained model.",
        )
        eval_data = st.file_uploader(
            "Optional eval CSV",
            type=["csv"],
            key=f"{page_id}_custom_eval_csv",
            help="Optional labeled holdout with the same schema as the training CSV.",
        )

    has_custom_uploads = any(source is not None for source in (training_data, scoring_data, eval_data))
    if has_custom_uploads:
        if training_data and scoring_data:
            st.success("Custom CSVs loaded. The workflow will rerun in Live API mode.")
        else:
            st.warning("Upload both a training CSV and a scoring CSV to rerun the workflow.")

    return training_data, scoring_data, eval_data, has_custom_uploads


def resolve_csv_sources(
    session_key: str,
    train_demo_path: str | None,
    test_demo_path: str | None,
):
    use_demo = st.session_state.get(session_key, False)

    train_source = demo_csv(train_demo_path) if use_demo else None
    test_source = demo_csv(test_demo_path) if use_demo else None

    return train_source, test_source


def resolve_single_csv_source(session_key: str, demo_path: str | None):
    use_demo = st.session_state.get(session_key, False)
    if use_demo:
        return demo_csv(demo_path)
    return None


def dataset_source_label(source) -> str:
    if source is None:
        return "Not available"
    if is_demo_csv(source):
        return f"Demo: {source.name}"
    return source.name


def is_demo_csv(source) -> bool:
    return isinstance(source, DemoCsvUpload)


def workflow_not_ready_message():
    st.info("Click **Run with demo datasets** in the sidebar to start the analysis.")


def apply_page_style():
    st.markdown(APP_CSS, unsafe_allow_html=True)


def init_shared_session():
    init_local_cache()

    if "model_ids" not in st.session_state:
        st.session_state.model_ids = cache_load_ready_models()
    else:
        st.session_state.model_ids = {
            **cache_load_ready_models(),
            **st.session_state.model_ids,
        }

    if "pending_model_jobs" not in st.session_state:
        st.session_state.pending_model_jobs = cache_load_pending_jobs()
    else:
        st.session_state.pending_model_jobs = {
            **cache_load_pending_jobs(),
            **st.session_state.pending_model_jobs,
        }

    if "model_wait_timeout_seconds" not in st.session_state:
        st.session_state.model_wait_timeout_seconds = MODEL_TRAIN_TIMEOUT_SECONDS


def render_api_sidebar(show_advanced: bool = True, show_endpoint: bool = True):
    if api_key:
        st.success("Wood Wide API key loaded")
    else:
        st.caption("Live API mode requires WOODWIDE_API_KEY in secrets or .env.")

    if show_endpoint:
        st.caption(f"API: {base_url}")

    if not show_advanced:
        return

    with st.expander("Advanced API settings"):
        model_wait_timeout_minutes = st.slider(
            "Model wait timeout",
            min_value=10,
            max_value=180,
            value=min(180, max(10, st.session_state.model_wait_timeout_seconds // 60)),
            step=10,
            help="How long this Streamlit run should wait for Woodwide jobs before pausing.",
            key="shared_model_wait_timeout_minutes",
        )
        st.session_state.model_wait_timeout_seconds = model_wait_timeout_minutes * 60


def render_pilot_cta(headline_metric: str | None = None):
    metric_line = f"**{headline_metric}** — " if headline_metric else ""
    st.markdown(
        f"""
        <div class="pilot-cta">
            Want this workflow on your data?
            <a href="{PILOT_CTA_URL}" target="_blank">Book a pilot conversation →</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress_stepper(current_step: int):
    steps = [
        "Performance",
        "High-risk list",
        "Drivers",
        "Patterns",
        "Segments",
        "Action plan",
    ]
    labels = " → ".join(
        f"**{label}**" if index + 1 == current_step else label
        for index, label in enumerate(steps)
    )
    st.caption(labels)


def download_button_for_path(label, path, download_name):
    if os.path.exists(path):
        with open(path, "rb") as file:
            st.download_button(label, file, download_name, "text/csv", use_container_width=True)
    else:
        st.caption(f"{download_name} was not found.")


def section_note(text: str):
    st.markdown(f'<div class="section-note">{text}</div>', unsafe_allow_html=True)
