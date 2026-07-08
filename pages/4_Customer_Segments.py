"""Customer segmentation workflow page."""

import hashlib

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from shared.dashboard import init_dashboard_state, render_widget_controls, render_dashboard_header
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
    section_note,
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
    get_or_create_dataset_from_bytes,
    get_or_start_model_training,
    inference_cache_key,
    parse_cluster_inference_payload,
    read_uploaded_csv,
    run_cached_model_inference,
    segment_description_for_label,
    segment_name_for_label,
    train_model,
)

PAGE_ID = "segments"

ALL_SEGMENT_WIDGETS = {
    "raw_output": "Raw Cluster Inference Output",
    "segment_metrics": "Segment Summary Metrics",
    "segment_bar": "Segment Distribution Bar Chart",
    "segment_pie": "Segment Distribution Pie Chart",
    "segment_table": "Segment Assignments Table",
}

configure_demo_app("Customer Segments")
init_shared_session()
apply_page_style()
init_dashboard_state(PAGE_ID, list(ALL_SEGMENT_WIDGETS.keys()))

render_brand_header(
    title="Customer Segments",
    subtitle="Train a clustering model on customer features and export segment assignments.",
)

with st.sidebar:
    render_api_sidebar()
    st.divider()
    st.header("Demo Datasets")
    render_demo_run_button("segments_use_demo", DEFAULT_CHURN_TRAIN_DATASET_PATH)
    customer_file = resolve_single_csv_source(
        "segments_use_demo",
        DEFAULT_CHURN_TRAIN_DATASET_PATH,
    )

    with st.expander("Download demo CSV"):
        download_button_for_path("Download sample train.csv", DEFAULT_CHURN_TRAIN_DATASET_PATH, "train.csv")

    st.divider()
    st.header("Demo Settings")
    show_raw_model_outputs = st.toggle("Show raw model outputs", value=False)

if not customer_file:
    workflow_not_ready_message()
    st.stop()

metric_col_1, metric_col_2 = st.columns(2)
metric_col_1.metric("Customer file", dataset_source_label(customer_file))
metric_col_2.metric("API connection", "Ready" if api_key else "Missing key")

segmented = None
clusters = None

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

    scoring_df = read_uploaded_csv(customer_file)
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
    segmented["cluster_label"] = [
        segment_name_for_label(label, segmented)
        for label in labels
    ]
    segmented["cluster_description"] = [
        descriptions.get(label) or descriptions.get(str(label), "")
        if isinstance(descriptions, dict)
        else ""
        for label in labels
    ]
    segmented["cluster_description"] = [
        description or segment_description_for_label(label, segmented)
        for description, label in zip(segmented["cluster_description"], labels)
    ]

# ── Dashboard widget rendering ──────────────────────────────────────────────────
if segmented is not None:
    summary = (
        segmented.groupby("cluster_label", dropna=False)
        .size()
        .reset_index(name="customers")
        .sort_values("customers", ascending=False)
    )

    segment_pie_data = {
        str(row["cluster_label"]): int(row["customers"])
        for _, row in summary.iterrows()
    }

    data_dict = {
        "metrics": {
            "Total customers": f"{len(segmented):,}",
            "Segments found": f"{segmented['cluster_label'].nunique()}",
            "Numeric features used": f"{len(cluster_columns)}",
        },
        "segment_pie_data": segment_pie_data,
        "segment_table_df": segmented,
        "raw_output_df": clusters if clusters is not None else pd.DataFrame(),
    }

    render_dashboard_header(PAGE_ID, ALL_SEGMENT_WIDGETS, data_dict=data_dict)

    active_widgets = st.session_state[f"{PAGE_ID}_active_widgets"]

    def _pie(data: dict, title: str):
        if not data:
            st.info("No data available.")
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#0B3D2E", "#175cd3", "#f04438", "#f79009", "#668575", "#d8e2dc",
                  "#7c3aed", "#059669", "#dc2626", "#ea580c"]
        ax.pie(
            list(data.values()),
            labels=list(data.keys()),
            autopct="%1.1f%%",
            colors=colors[: len(data)],
            startangle=140,
        )
        ax.axis("equal")
        ax.set_title(title, fontsize=12, weight="bold", color="#0B3D2E")
        fig.patch.set_facecolor("#ffffff")
        st.pyplot(fig)
        plt.close()

    for idx, widget_id in enumerate(active_widgets):
        if widget_id not in ALL_SEGMENT_WIDGETS:
            continue
        widget_name = ALL_SEGMENT_WIDGETS[widget_id]
        collapsed = render_widget_controls(PAGE_ID, idx, widget_id, widget_name)
        if collapsed:
            continue

        if widget_id == "raw_output":
            if not show_raw_model_outputs:
                st.info("Enable **Show raw model outputs** in the sidebar to display this widget.")
            elif clusters is not None:
                row_sel, _ = st.columns([1, 3])
                with row_sel:
                    n = st.selectbox(
                        "Rows to display",
                        [10, 25, 50, 100, 250, 500, 1000],
                        index=3,
                        key=f"{PAGE_ID}_{widget_id}_rows",
                    )
                st.dataframe(clusters.head(n), use_container_width=True, height=360)
            else:
                st.info("No raw cluster output available yet.")

        elif widget_id == "segment_metrics":
            m1, m2, m3 = st.columns(3)
            m1.metric("Total customers", f"{len(segmented):,}")
            m2.metric("Segments found", f"{segmented['cluster_label'].nunique()}")
            m3.metric("Numeric features used", f"{len(cluster_columns)}")

        elif widget_id == "segment_bar":
            st.bar_chart(summary, x="cluster_label", y="customers")

        elif widget_id == "segment_pie":
            _pie(segment_pie_data, "Customer Segment Distribution")

        elif widget_id == "segment_table":
            summary_col, preview_col = st.columns([1, 2])
            with summary_col:
                st.markdown("**Segment Summary**")
                st.dataframe(summary, use_container_width=True, hide_index=True)
            with preview_col:
                row_sel, _ = st.columns([1, 2])
                with row_sel:
                    n = st.selectbox(
                        "Rows to display",
                        [10, 25, 50, 100, 250, 500, 1000],
                        index=3,
                        key=f"{PAGE_ID}_{widget_id}_rows",
                    )
                st.dataframe(segmented.head(n), use_container_width=True, height=420)
            st.download_button(
                "Download customer segments CSV",
                segmented.to_csv(index=False).encode("utf-8"),
                "customer_segments.csv",
                "text/csv",
                use_container_width=True,
                key=f"{PAGE_ID}_{widget_id}_dl",
            )

        st.divider()
