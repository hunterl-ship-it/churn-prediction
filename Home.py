"""Wood Wide AI Use Cases — landing page."""

import streamlit as st

from shared import (
    DemoCard,
    PAGE_CHURN,
    PAGE_FRAUD,
    PAGE_NOSHOW,
    PAGE_RETURNS,
    PAGE_SEGMENTS,
    apply_demo_query_params,
    apply_page_style,
    configure_demo_app,
    init_shared_session,
    render_brand_header,
    render_demo_card_grid,
    render_demo_library_intro,
    render_home_sidebar,
    render_pilot_cta,
)
from woodwide.core import api_key

configure_demo_app("Wood Wide AI Use Cases")

init_shared_session()
apply_page_style()
apply_demo_query_params("churn_use_demo", "noshow_use_demo")

with st.sidebar:
    render_home_sidebar()

render_brand_header(home=True)
render_demo_library_intro()

st.markdown(
    '<p class="ww-demo-section-title">Featured demos</p>'
    '<p class="ww-demo-section-note">'
    "Full end-to-end walkthroughs with instant results—no setup required."
    "</p>",
    unsafe_allow_html=True,
)
render_demo_card_grid(
    [
        DemoCard(
            title="Churn Intervention",
            description=(
                "Predict which subscribers will churn, review model performance on a labeled holdout, "
                "and generate a prioritized retention plan."
            ),
            meta="Instant demo: scores, drivers, and actions in seconds.",
            page=PAGE_CHURN,
            query="demo=churn",
            cta="Try churn demo",
            featured=True,
        ),
        DemoCard(
            title="Patient No-Show",
            description=(
                "Predict appointment no-shows, explain risk drivers, and produce outreach actions—"
                "the same workflow as churn, applied to healthcare scheduling."
            ),
            meta="Instant demo: 15k holdout eval with pre-computed lift metrics.",
            page=PAGE_NOSHOW,
            query="demo=noshow",
            cta="Try no-show demo",
            featured=True,
        ),
    ],
    featured=True,
)

st.markdown(
    '<p class="ww-demo-section-title">More workflows</p>'
    '<p class="ww-demo-section-note">'
    "Additional examples showing how Wood Wide fits fraud, commerce, and segmentation use cases."
    "</p>",
    unsafe_allow_html=True,
)
render_demo_card_grid(
    [
        DemoCard(
            title="Fraud Detection",
            description="Flag risky transactions, prioritize review queues, and explain suspicious signals.",
            meta="Prediction and anomaly scoring on e-commerce transactions.",
            page=PAGE_FRAUD,
            cta="Explore fraud demo",
        ),
        DemoCard(
            title="Return Risk",
            description="Score orders for return likelihood and generate proactive prevention actions.",
            meta="Return prediction workflow for e-commerce operations teams.",
            page=PAGE_RETURNS,
            cta="Explore return risk demo",
        ),
        DemoCard(
            title="Customer Segments",
            description="Discover customer groups from behavioral features and export segment assignments.",
            meta="Unsupervised clustering for marketing and lifecycle planning.",
            page=PAGE_SEGMENTS,
            cta="Explore segments demo",
        ),
    ],
)

st.divider()
st.metric("Wood Wide API", "Connected" if api_key else "Instant demo mode")

with st.expander("How Wood Wide fits your workflow"):
    st.markdown(
        """
        <div class="ww-tech-panel">
            <p>
                Wood Wide gives product and data teams a single API for numeric reasoning over
                structured business data—predictions, explanations, segmentation, and anomaly
                detection—without building bespoke ML pipelines for every use case.
            </p>
            <h4>What these demos show</h4>
            <ul>
                <li>Upload or connect tabular data and train a reusable dataset representation once.</li>
                <li>Score holdout or production records asynchronously at scale.</li>
                <li>Explain individual predictions so operators can act with confidence.</li>
                <li>Turn model output into prioritized business actions—not just scores.</li>
            </ul>
            <h4>Core API endpoints</h4>
            <ul>
                <li><strong>POST /datasets</strong> — register training or scoring tables with schema context.</li>
                <li><strong>POST /models/train</strong> — learn a reusable representation for prediction, clustering, or anomaly tasks.</li>
                <li><strong>POST /models/&#123;id&#125;/infer-async</strong> — score new rows without blocking your application.</li>
                <li><strong>GET /jobs/&#123;id&#125;/results</strong> — retrieve predictions, clusters, or anomaly scores when jobs complete.</li>
                <li><strong>POST /jobs/&#123;inference_job_id&#125;/explain</strong> — generate row-level drivers for review workflows.</li>
            </ul>
            <p>
                Each demo page walks through one of these patterns on realistic sample data.
                Share this library with prospects so they can explore independently, then
                <a href="https://woodwide.ai" target="_blank">talk to Wood Wide</a>
                about running the same workflows on their datasets.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_pilot_cta()
