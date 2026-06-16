"""Customer segmentation workflow page."""

import streamlit as st

from shared.ui import (
    apply_page_style,
    configure_demo_app,
    dataset_source_label,
    download_button_for_path,
    init_shared_session,
    render_api_sidebar,
    render_brand_header,
    render_demo_run_button,
    resolve_single_csv_source,
    workflow_not_ready_message,
)
from woodwide.core import (
    DEFAULT_CHURN_TRAIN_DATASET_PATH,
    PREVIEW_ROW_COUNT,
    api_key,
    clustering_dataframe_for_modeling,
    dataframe_to_csv_bytes,
    dataset_name_for_upload,
    get_or_create_dataset,
    get_or_start_model_training,
    inference_cache_key,
    parse_cluster_inference_payload,
    read_uploaded_csv,
    run_cached_model_inference,
    train_model,
)
import hashlib

configure_demo_app("Customer Segments")
init_shared_session()
apply_page_style()

render_brand_header(
    title="Customer Segments",
    subtitle="Train a clustering model on customer features and export segment assignments.",
)

with st.sidebar:
    render_api_sidebar()
    st.divider()
    st.header("Inputs")
    uploaded_customer = st.file_uploader(
        "Customer dataset",
        type=["csv"],
        help="Upload your own CSV, or use the demo dataset button below.",
    )
    holdout_file = st.file_uploader(
        "Optional holdout file to assign clusters",
        type=["csv"],
        help="Defaults to the customer dataset above.",
    )

    st.divider()
    st.header("Demo Datasets")
    render_demo_run_button("segments_use_demo", DEFAULT_CHURN_TRAIN_DATASET_PATH)
    customer_file = resolve_single_csv_source(
        "segments_use_demo",
        uploaded_customer,
        DEFAULT_CHURN_TRAIN_DATASET_PATH,
    )

    with st.expander("Download demo CSV"):
        download_button_for_path("Download sample train.csv", DEFAULT_CHURN_TRAIN_DATASET_PATH, "train.csv")

    st.divider()
    st.header("Demo Settings")
    preview_row_count = st.slider("Preview rows", 100, 5000, PREVIEW_ROW_COUNT, 100)
    show_raw_model_outputs = st.toggle("Show raw model outputs", value=False)

if not customer_file:
    workflow_not_ready_message()
    st.stop()

metric_col_1, metric_col_2 = st.columns(2)
metric_col_1.metric("Customer file", dataset_source_label(customer_file))
metric_col_2.metric("API connection", "Ready" if api_key else "Missing key")

segmented = None

if customer_file:
    st.subheader("1. Segmentation Model")
    section_note("Unsupervised clustering (model_type: clustering, no label column).")

    customers = read_uploaded_csv(customer_file)
    cluster_data, cluster_columns = clustering_dataframe_for_modeling(customers)
    if cluster_data.empty or not cluster_columns:
        st.error("No numeric feature columns were available for clustering.")
        st.stop()

    st.caption(f"Using {len(cluster_columns)} numeric columns for clustering.")

    dataset_name = dataset_name_for_upload("segment_customers", customer_file)
    cluster_csv = dataframe_to_csv_bytes(cluster_data)

    with st.spinner("Uploading dataset...", show_time=True):
        from woodwide.core import get_or_create_dataset_from_bytes
        dataset_id = get_or_create_dataset_from_bytes(cluster_csv, "segment_customers.csv", dataset_name)

    with st.spinner("Training clustering model...", show_time=True):
        model_id = get_or_start_model_training(
            st.session_state.model_ids,
            dataset_name,
            f"segments:{dataset_name}",
            "customer segmentation",
            "Segmentation model is ready.",
            lambda: train_model(
                dataset_name.replace("segment_customers", "customer_segments", 1),
                "clustering",
                dataset_id,
                input_columns=cluster_columns,
            ),
        )

    scoring_file = holdout_file or customer_file
    scoring_df = read_uploaded_csv(scoring_file)
    score_data, score_columns = clustering_dataframe_for_modeling(scoring_df)
    if score_data.empty:
        score_data = cluster_data
        scoring_df = customers

    score_csv = dataframe_to_csv_bytes(score_data)

    st.subheader("2. Segment Assignments")
    with st.spinner("Assigning clusters...", show_time=True):
        cache_key = inference_cache_key(
            "segment_inference",
            model_id,
            hashlib.sha256(score_csv).hexdigest(),
            "json",
        )
        clusters, cluster_payload, _ = run_cached_model_inference(
            model_id,
            score_csv,
            "segment_customers.csv",
            "json",
            cache_key,
            "segment inference",
        )
        labels, descriptions = parse_cluster_inference_payload(
            cluster_payload,
            len(scoring_df),
            clusters,
        )

    segmented = scoring_df.copy()
    segmented["cluster_label"] = labels
    segmented["cluster_description"] = [
        descriptions.get(label) or descriptions.get(str(label), "")
        if isinstance(descriptions, dict)
        else ""
        for label in labels
    ]

    if show_raw_model_outputs:
        st.dataframe(clusters, use_container_width=True, height=360)

    summary = (
        segmented.groupby("cluster_label", dropna=False)
        .size()
        .reset_index(name="customers")
        .sort_values("customers", ascending=False)
    )

    st.subheader("3. Segment Summary")
    summary_col, preview_col = st.columns([1, 2])
    with summary_col:
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.bar_chart(summary, x="cluster_label", y="customers")
    with preview_col:
        st.dataframe(segmented.head(preview_row_count), use_container_width=True, height=420)

    st.download_button(
        "Download customer segments CSV",
        segmented.to_csv(index=False).encode("utf-8"),
        "customer_segments.csv",
        "text/csv",
        use_container_width=True,
    )
