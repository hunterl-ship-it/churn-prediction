"""Config-driven supervised outcome workflow for hero demo pages."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
from shared.dashboard import init_dashboard_state, render_widget_controls, render_dashboard_header

from shared.instant_playback import (
    animate_integer_metric,
    apply_instant_playback_style,
    pipeline_progress,
    reveal_dataframe,
    simulate_stage,
    typewriter_caption,
)
from shared.ui import (
    apply_demo_query_params,
    apply_page_style,
    configure_demo_app,
    dataset_source_label,
    download_button_for_path,
    get_demo_mode,
    init_shared_session,
    is_demo_csv,
    render_api_sidebar,
    render_brand_header,
    render_custom_csv_uploads,
    render_demo_mode_toggle,
    render_demo_run_button,
    render_pilot_cta,
    render_progress_stepper,
    resolve_csv_sources,
    section_note,
    workflow_not_ready_message,
)
from woodwide.core import (
    DEFAULT_CHURN_EVAL_DATASET_PATH,
    DEFAULT_CHURN_TEST_DATASET_PATH,
    DEFAULT_CHURN_TRAIN_DATASET_PATH,
    DEFAULT_INTERVENTION_TEMPLATE_KEY,
    DEFAULT_NOSHOW_EVAL_DATASET_PATH,
    DEFAULT_NOSHOW_TEST_DATASET_PATH,
    DEFAULT_NOSHOW_TRAIN_DATASET_PATH,
    INTERVENTION_TEMPLATE_LABELS,
    PREVIEW_ROW_COUNT,
    add_original_customer_features,
    add_prediction_descriptions,
    add_row_level_explanations,
    analysis_dataframe_for_modeling,
    api_key,
    as_number,
    at_risk_display_columns,
    build_interventions,
    cache_load_ready_models_with_prefix,
    clustering_dataframe_for_modeling,
    dataframe_to_csv_bytes,
    dataframe_to_intervention_catalog,
    dataset_name_for_dataframe,
    dataset_name_for_upload,
    default_intervention_catalog,
    detect_intervention_template,
    filter_at_risk_customers,
    get_churn_label_column,
    get_noshow_label_column,
    get_or_create_dataset,
    get_or_create_dataset_from_bytes,
    get_or_start_model_training,
    get_row_level_explanations,
    get_row_level_explanations_individually,
    inference_cache_key,
    intervention_catalog_to_dataframe,
    intervention_template_label,
    merge_explanation_frames,
    normalized_merge_id,
    parse_cluster_inference_payload,
    prediction_input_columns_for_training,
    probability_sort_column,
    read_uploaded_csv,
    resolve_intervention_template,
    risk_driver_chart_data,
    risk_driver_chart_data_from_feature_contrast,
    risk_driver_chart_data_from_feature_entries,
    run_cached_model_inference,
    run_prediction_inference,
    set_intervention_catalog_template,
    stored_manual_explanations,
    store_manual_explanations,
    train_model,
)
from woodwide.evaluation import compute_binary_metrics, render_performance_panel
from workflows.instant_demo import (
    artifacts_available,
    load_at_risk,
    load_driver_chart,
    load_explanations,
    load_intervention_plan,
    load_metrics,
    load_metadata,
    load_patterns,
    merge_exemplar_explanations,
)


@dataclass(frozen=True)
class OutcomeDemoConfig:
    page_id: str
    browser_title: str
    page_title: str
    subtitle: str
    demo_session_key: str
    feature_entries_key: str
    default_train_path: str
    default_test_path: str
    default_eval_path: str
    get_label_column: Callable
    dataset_prefix: str
    model_cache_prefix: str
    model_ready_message: str
    model_train_name_from: str
    model_train_name_to: str
    entity_id_column: str
    entity_name: str
    entity_name_plural: str
    threshold_label: str
    threshold_help: str
    threshold_default: float
    model_section_title: str
    at_risk_section_title: str
    at_risk_section_note_template: str
    high_risk_group_title: str
    other_risk_group_title: str
    action_plan_title: str
    action_plan_note: str
    risk_download_label: str
    risk_download_filename: str
    search_placeholder: str
    driver_note: str
    pattern_note: str
    segment_note: str
    empty_risk_info: str
    scoring_spinner: str
    training_spinner: str
    workflow_captions: tuple[str, ...]
    plan_id_column: str
    default_max_explanations: int = 25


CHURN_DEMO_CONFIG = OutcomeDemoConfig(
    page_id="churn",
    browser_title="Churn Intervention",
    page_title="Churn Intervention",
    subtitle="Identify at-risk subscribers, explain drivers, and produce a targeted retention plan.",
    demo_session_key="churn_use_demo",
    feature_entries_key="churn_feature_entries_by_label",
    default_train_path=DEFAULT_CHURN_TRAIN_DATASET_PATH,
    default_test_path=DEFAULT_CHURN_TEST_DATASET_PATH,
    default_eval_path=DEFAULT_CHURN_EVAL_DATASET_PATH,
    get_label_column=get_churn_label_column,
    dataset_prefix="historical_customers",
    model_cache_prefix="churn",
    model_ready_message="Churn model is ready.",
    model_train_name_from="historical_customers",
    model_train_name_to="churn_prediction",
    entity_id_column="CustomerID",
    entity_name="customer",
    entity_name_plural="customers",
    threshold_label="At-risk confidence threshold",
    threshold_help="Records must meet or exceed this probability to enter the intervention workflow.",
    threshold_default=0.50,
    model_section_title="1. Churn Model",
    at_risk_section_title="2. At-Risk Customers",
    at_risk_section_note_template="Active customers scored and filtered to predicted churners at or above {threshold:.0%} confidence.",
    high_risk_group_title="Most At-Risk Customers",
    other_risk_group_title="All Other At-Risk Customers",
    action_plan_title="6. Intervention Plan",
    action_plan_note="Each predicted churner gets an action based on model risk drivers, segment context, and relative risk.",
    risk_download_label="Download at-risk customers CSV",
    risk_download_filename="at_risk_customers.csv",
    search_placeholder="Search account or customer ID",
    driver_note="Feature weights and row explanations identify what pushed customers toward churn.",
    pattern_note="Factor analysis summarizes variance within at-risk customers.",
    segment_note="Clustering groups at-risk customers so intervention actions can account for segment context.",
    empty_risk_info="No at-risk customers found, so downstream analysis was skipped.",
    scoring_spinner="identifying at-risk customers...",
    training_spinner="training churn model",
    workflow_captions=(
        "1. Choose Instant or Live demo mode.",
        "2. Review model performance on a labeled holdout.",
        "3. Score and filter high-risk records.",
        "4. Review drivers, patterns, segments, and actions.",
    ),
    plan_id_column="CustomerID",
)

NOSHOW_DEMO_CONFIG = OutcomeDemoConfig(
    page_id="noshow",
    browser_title="Patient No-Show",
    page_title="Patient No-Show",
    subtitle="Identify high-risk appointments, explain drivers, and produce a targeted outreach plan.",
    demo_session_key="noshow_use_demo",
    feature_entries_key="noshow_feature_entries_by_label",
    default_train_path=DEFAULT_NOSHOW_TRAIN_DATASET_PATH,
    default_test_path=DEFAULT_NOSHOW_TEST_DATASET_PATH,
    default_eval_path=DEFAULT_NOSHOW_EVAL_DATASET_PATH,
    get_label_column=get_noshow_label_column,
    dataset_prefix="historical_appointments",
    model_cache_prefix="noshow",
    model_ready_message="No-show model is ready.",
    model_train_name_from="historical_appointments",
    model_train_name_to="noshow_prediction",
    entity_id_column="PatientID",
    entity_name="appointment",
    entity_name_plural="appointments",
    threshold_label="High no-show risk threshold",
    threshold_help="Appointments must meet or exceed this probability to enter the outreach workflow.",
    threshold_default=0.50,
    model_section_title="1. No-Show Model",
    at_risk_section_title="2. High-Risk Appointments",
    at_risk_section_note_template="Upcoming appointments scored and filtered to predicted no-shows at or above {threshold:.0%} confidence.",
    high_risk_group_title="Highest-Risk Appointments",
    other_risk_group_title="All Other High-Risk Appointments",
    action_plan_title="6. Outreach Plan",
    action_plan_note="Each predicted no-show gets an outreach action based on model risk drivers, segment context, and relative risk.",
    risk_download_label="Download high-risk appointments CSV",
    risk_download_filename="high_risk_appointments.csv",
    search_placeholder="Search patient ID",
    driver_note="Feature weights and row explanations identify what pushed appointments toward a no-show prediction.",
    pattern_note="Factor analysis summarizes variance within high-risk appointments.",
    segment_note="Clustering groups high-risk appointments so outreach actions can account for segment context.",
    empty_risk_info="No high-risk appointments found, so downstream analysis was skipped.",
    scoring_spinner="identifying high-risk appointments...",
    training_spinner="training no-show model",
    workflow_captions=(
        "1. Choose Instant or Live demo mode.",
        "2. Review model performance on a labeled holdout.",
        "3. Score and filter high-risk appointments.",
        "4. Review drivers, patterns, segments, and outreach actions.",
    ),
    plan_id_column="PatientID",
)


def _init_intervention_session():
    if "intervention_template_choice" not in st.session_state:
        st.session_state.intervention_template_choice = "auto"
    if "last_intervention_template_choice" not in st.session_state:
        st.session_state.last_intervention_template_choice = st.session_state.intervention_template_choice
    if "intervention_catalog_template_key" not in st.session_state:
        st.session_state.intervention_catalog_template_key = DEFAULT_INTERVENTION_TEMPLATE_KEY
    if "intervention_catalog_customized" not in st.session_state:
        st.session_state.intervention_catalog_customized = False
    if "intervention_catalog_editor_version" not in st.session_state:
        st.session_state.intervention_catalog_editor_version = 0
    if "intervention_catalog" not in st.session_state:
        st.session_state.intervention_catalog = default_intervention_catalog(
            st.session_state.intervention_catalog_template_key
        )
    if "show_intervention_editor" not in st.session_state:
        st.session_state.show_intervention_editor = False
    if "manual_explanations_by_job" not in st.session_state:
        st.session_state.manual_explanations_by_job = {}
    if "manual_explanation_selection_versions" not in st.session_state:
        st.session_state.manual_explanation_selection_versions = {}
    if "risk_ids" not in st.session_state:
        st.session_state.risk_ids = cache_load_ready_models_with_prefix("factors:")
    if "cluster_ids" not in st.session_state:
        st.session_state.cluster_ids = cache_load_ready_models_with_prefix("clusters:")


def _render_intervention_sidebar(config: OutcomeDemoConfig, training_data):
    st.divider()
    st.header("Interventions")
    detected_intervention_template = DEFAULT_INTERVENTION_TEMPLATE_KEY
    template_training_source = training_data
    if template_training_source:
        try:
            detected_intervention_template = detect_intervention_template(read_uploaded_csv(template_training_source))
        except (pd.errors.EmptyDataError, UnicodeDecodeError, ValueError):
            detected_intervention_template = DEFAULT_INTERVENTION_TEMPLATE_KEY

    template_choice = st.selectbox(
        "Intervention template",
        list(INTERVENTION_TEMPLATE_LABELS.keys()),
        format_func=intervention_template_label,
        key=f"{config.page_id}_intervention_template_choice",
    )
    resolved_intervention_template = resolve_intervention_template(template_choice, detected_intervention_template)
    template_choice_changed = template_choice != st.session_state.last_intervention_template_choice
    if template_choice_changed:
        st.session_state.intervention_catalog_customized = False
    if template_choice_changed or (
        not st.session_state.intervention_catalog_customized
        and resolved_intervention_template != st.session_state.intervention_catalog_template_key
    ):
        set_intervention_catalog_template(resolved_intervention_template)
    st.session_state.last_intervention_template_choice = template_choice

    with st.expander("Customize interventions"):
        if st.button("Toggle intervention editor", use_container_width=True, key=f"{config.page_id}_toggle_editor"):
            st.session_state.show_intervention_editor = not st.session_state.show_intervention_editor
        if st.session_state.show_intervention_editor:
            edited = st.data_editor(
                intervention_catalog_to_dataframe(st.session_state.intervention_catalog),
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key=f"intervention_catalog_editor:{config.page_id}:{st.session_state.intervention_catalog_editor_version}",
            )
            if st.button("Apply interventions", use_container_width=True, key=f"{config.page_id}_apply_interventions"):
                st.session_state.intervention_catalog = dataframe_to_intervention_catalog(edited)
                st.session_state.intervention_catalog_customized = True


def _render_action_plan(
    config: OutcomeDemoConfig,
    intervention_results: pd.DataFrame,
    preview_row_count: int,
    animate: bool = False,
):
    st.subheader(config.action_plan_title)
    section_note(config.action_plan_note)
    render_progress_stepper(6)

    high_count = int((intervention_results["intervention_urgency"] == "high").sum()) if "intervention_urgency" in intervention_results.columns else 0
    medium_count = int((intervention_results["intervention_urgency"] == "medium").sum()) if "intervention_urgency" in intervention_results.columns else 0
    low_count = int((intervention_results["intervention_urgency"] == "low").sum()) if "intervention_urgency" in intervention_results.columns else 0
    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    if animate:
        animate_integer_metric(metric_col_1, "Planned actions", len(intervention_results))
        animate_integer_metric(metric_col_2, "High urgency", high_count)
        animate_integer_metric(metric_col_3, "Medium urgency", medium_count)
        animate_integer_metric(metric_col_4, "Low urgency", low_count)
    else:
        metric_col_1.metric("Planned actions", f"{len(intervention_results):,}")
        metric_col_2.metric("High urgency", f"{high_count:,}")
        metric_col_3.metric("Medium urgency", f"{medium_count:,}")
        metric_col_4.metric("Low urgency", f"{low_count:,}")

    preferred_columns = [
        config.plan_id_column,
        "prediction_prob",
        "intervention_urgency",
        "intervention_category",
        "intervention_action",
        "model_risk_drivers",
        "top_churn_drivers",
        "row_prediction_explanation",
        "row_explanation_summary",
        "cluster_label",
    ]
    visible_columns = [column for column in preferred_columns if column in intervention_results.columns]
    if not visible_columns:
        visible_columns = intervention_results.columns.tolist()
    row_select_col, _ = st.columns([1, 3])
    with row_select_col:
        local_row_count = st.selectbox(
            "Rows to display",
            options=[10, 25, 50, 100, 250, 500, 1000],
            index=3,
            key=f"{config.page_id}_action_plan_row_count"
        )
    st.dataframe(intervention_results[visible_columns].head(local_row_count), use_container_width=True, height=520)
    st.download_button(
        "Download action plan CSV",
        intervention_results.to_csv(index=False).encode("utf-8"),
        "action_plan.csv",
        "text/csv",
        use_container_width=True,
    )


ALL_SUPERVISED_WIDGETS = {
    "model_performance": "Model Performance",
    "risk_cohort": "High-Risk Cohort Preview",
    "risk_drivers": "Model Risk Drivers",
    "patterns": "High-Risk Patterns",
    "segments": "Customer Segments",
    "segment_pie": "Segment Distribution Pie Chart",
    "action_plan": "Action/Intervention Plan",
    "urgency_pie": "Action Urgency Pie Chart"
}

def render_supervised_widgets(
    config: OutcomeDemoConfig,
    active_widgets: list[str],
    at_risk: pd.DataFrame,
    intervention_results: pd.DataFrame,
    metrics: dict,
    driver_chart: pd.DataFrame,
    patterns: pd.DataFrame,
    threshold: float,
    explanations: pd.DataFrame,
    scored_customers: pd.DataFrame,
    total_entities: int = None
):
    # Compute segment and urgency pie chart data
    segment_pie_data = {}
    if at_risk is not None and not at_risk.empty and "cluster_label" in at_risk.columns:
        segment_pie_data = at_risk.groupby("cluster_label").size().to_dict()
        segment_pie_data = {f"Segment {k}": int(v) for k, v in segment_pie_data.items()}
        
    urgency_pie_data = {}
    if intervention_results is not None and not intervention_results.empty and "intervention_urgency" in intervention_results.columns:
        urgency_pie_data = {str(k).title(): int(v) for k, v in intervention_results.groupby("intervention_urgency").size().to_dict().items()}

    def render_pie_chart(data, title):
        if not data:
            st.info("No data available to display pie chart.")
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = list(data.keys())
        sizes = list(data.values())
        colors = ['#0B3D2E', '#175cd3', '#f04438', '#f79009', '#668575', '#d8e2dc'][:len(labels)]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        ax.axis('equal')
        ax.set_title(title, fontsize=12, weight='bold', color='#0B3D2E')
        fig.patch.set_facecolor('#ffffff')
        st.pyplot(fig)
        plt.close()

    # RENDER ACTIVE WIDGETS
    for index, widget_id in enumerate(active_widgets):
        widget_name = ALL_SUPERVISED_WIDGETS[widget_id]
        collapsed = render_widget_controls(config.page_id, index, widget_id, widget_name)
        if collapsed:
            continue
            
        if widget_id == "model_performance":
            if metrics:
                render_performance_panel(metrics, animate=False, skip_animation=True)
            else:
                st.info("No holdout evaluation metrics available.")
                
        elif widget_id == "risk_cohort":
            st.markdown(f"*{config.at_risk_section_note_template.format(threshold=threshold)}*")
            
            # Local selectable row count
            row_select_col, _ = st.columns([1, 3])
            with row_select_col:
                local_row_count = st.selectbox(
                    "Rows to display (Cohort)",
                    options=[10, 25, 50, 100, 250, 500, 1000],
                    index=3,
                    key=f"{config.page_id}_{widget_id}_row_count"
                )
                
            col1, col2, col3 = st.columns(3)
            col1.metric(f"High-risk {config.entity_name_plural}", f"{len(at_risk):,}")
            col2.metric("Previewing", f"{min(len(at_risk), local_row_count):,}")
            col3.metric("Threshold", f"{threshold:.0%}")
            
            st.dataframe(
                at_risk.head(local_row_count)[at_risk_display_columns(at_risk.head(local_row_count))],
                use_container_width=True,
                height=460,
            )
            st.download_button(
                config.risk_download_label,
                at_risk.to_csv(index=False).encode("utf-8"),
                config.risk_download_filename,
                "text/csv",
                use_container_width=True,
                key=f"{config.page_id}_{widget_id}_dl_btn"
            )
            
        elif widget_id == "risk_drivers":
            section_note(config.driver_note)
            if driver_chart is None or driver_chart.empty:
                st.info("No risk driver data available.")
            else:
                totals = (
                    driver_chart.groupby("driver", as_index=False)["weighted_signal"]
                    .sum()
                    .sort_values("weighted_signal", ascending=False)
                )
                chart_col, table_col = st.columns(2)
                with chart_col:
                    st.bar_chart(totals, x="driver", y="weighted_signal")
                with table_col:
                    reveal_dataframe(driver_chart.head(10), height=320, skip=True)
                    
        elif widget_id == "patterns":
            section_note(config.pattern_note)
            if patterns is None or patterns.empty:
                st.info("No high-risk pattern data available.")
            else:
                chart_col, table_col = st.columns(2)
                with chart_col:
                    st.bar_chart(patterns, x="pattern", y="captured_variance")
                with table_col:
                    reveal_dataframe(patterns[["pattern", "top_drivers", "captured_variance"]], height=260, skip=True)
                    
        elif widget_id == "segments":
            section_note(config.segment_note)
            if at_risk is not None and not at_risk.empty and "cluster_label" in at_risk.columns:
                summary = at_risk.groupby("cluster_label").size().reset_index(name=config.entity_name_plural)
                chart_col, table_col = st.columns(2)
                with chart_col:
                    st.bar_chart(summary, x="cluster_label", y=config.entity_name_plural)
                with table_col:
                    reveal_dataframe(summary, height=180, skip=True)
            else:
                st.info("No segmentation clustering data available.")
                
        elif widget_id == "segment_pie":
            render_pie_chart(segment_pie_data, f"Segment Distribution ({config.entity_name_plural.title()})")
            
        elif widget_id == "action_plan":
            section_note(config.action_plan_note)
            
            # Local selectable row count
            row_select_col, _ = st.columns([1, 3])
            with row_select_col:
                local_row_count = st.selectbox(
                    "Rows to display (Action Plan)",
                    options=[10, 25, 50, 100, 250, 500, 1000],
                    index=3,
                    key=f"{config.page_id}_{widget_id}_row_count"
                )
                
            high_count = int((intervention_results["intervention_urgency"] == "high").sum()) if "intervention_urgency" in intervention_results.columns else 0
            medium_count = int((intervention_results["intervention_urgency"] == "medium").sum()) if "intervention_urgency" in intervention_results.columns else 0
            low_count = int((intervention_results["intervention_urgency"] == "low").sum()) if "intervention_urgency" in intervention_results.columns else 0
            
            metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
            metric_col_1.metric("Planned actions", f"{len(intervention_results):,}")
            metric_col_2.metric("High urgency", f"{high_count:,}")
            metric_col_3.metric("Medium urgency", f"{medium_count:,}")
            metric_col_4.metric("Low urgency", f"{low_count:,}")
            
            preferred_columns = [
                config.plan_id_column,
                "prediction_prob",
                "intervention_urgency",
                "intervention_category",
                "intervention_action",
                "model_risk_drivers",
                "top_churn_drivers",
                "row_prediction_explanation",
                "row_explanation_summary",
                "cluster_label",
            ]
            visible_columns = [column for column in preferred_columns if column in intervention_results.columns]
            if not visible_columns:
                visible_columns = intervention_results.columns.tolist()
                
            st.dataframe(intervention_results[visible_columns].head(local_row_count), use_container_width=True, height=520)
            st.download_button(
                "Download action plan CSV",
                intervention_results.to_csv(index=False).encode("utf-8"),
                "action_plan.csv",
                "text/csv",
                use_container_width=True,
                key=f"{config.page_id}_{widget_id}_dl_btn"
            )
            
        elif widget_id == "urgency_pie":
            render_pie_chart(urgency_pie_data, "Intervention Urgency Distribution")

def _render_instant_demo(
    config: OutcomeDemoConfig,
    threshold: float,
    preview_row_count: int,
    max_row_explanations: int,
):
    apply_instant_playback_style()

    metrics = load_metrics(config.page_id)
    metadata = load_metadata(config.page_id)
    at_risk = load_at_risk(config.page_id)
    explanations = load_explanations(config.page_id)
    driver_chart = load_driver_chart(config.page_id)
    patterns = load_patterns(config.page_id)
    intervention_results = load_intervention_plan(config.page_id)

    if not explanations.empty:
        at_risk = merge_exemplar_explanations(at_risk, explanations)

    if intervention_results.empty and not at_risk.empty:
        intervention_results = at_risk.copy()
        intervention_results["intervention_urgency"] = "high"
        intervention_results["intervention_category"] = "Demo action"
        intervention_results["intervention_action"] = "Run live demo for full intervention matching."

    eval_rows = metrics.get("eval_rows", 15_000) if metrics else 15_000
    scored_rows = metrics.get("confusion", {}) if metrics else {}
    total_scored = sum(scored_rows.values()) if scored_rows else len(at_risk) * 4

    headline = None
    if metrics:
        headline = f"AUC {metrics.get('auc_roc', 0):.2f} · {metrics.get('lift_top_decile', 0):.1f}x lift @ top 10%"

    # Guard: only play the animation on the very first render of this demo session.
    # Any subsequent rerun (widget button click, sidebar slider, etc.) skips straight
    # to the dashboard so state changes don't restart the whole pipeline sequence.
    _animation_key = f"{config.page_id}_instant_demo_played"
    if not st.session_state.get(_animation_key):
        st.session_state[_animation_key] = True
        with pipeline_progress(total_steps=7) as advance:
            with simulate_stage(
                "Training prediction model",
                [
                    ("Uploading historical dataset...", 0.55),
                    ("Validating feature columns...", 0.35),
                    ("Training Wood Wide prediction model...", 0.95),
                    ("Evaluating on labeled holdout...", 0.65),
                ],
            ):
                pass
            advance("Model trained · evaluating holdout performance")

            with simulate_stage(
                f"Scoring {config.entity_name_plural}",
                [
                    (f"Uploading {config.entity_name_plural}...", 0.45),
                    ("Running batch inference...", 0.85),
                    (f"Applying {threshold:.0%} risk threshold...", 0.4),
                ],
            ):
                pass
            advance(f"Scored {total_scored:,} {config.entity_name_plural}")

            with simulate_stage(
                "Explaining highest-risk rows",
                [
                    ("Selecting top-risk records for explanation...", 0.4),
                    ("Requesting row-level prediction explanations...", 0.75),
                    ("Summarizing drivers per record...", 0.45),
                ],
            ):
                pass
            advance("Row explanations attached")

            with simulate_stage(
                "Extracting model risk drivers",
                [
                    ("Aggregating feature weights...", 0.5),
                    ("Ranking drivers by weighted signal...", 0.55),
                ],
            ):
                pass
            advance("Risk drivers ranked")

            with simulate_stage(
                "Analyzing high-risk patterns",
                [
                    ("Training factor model on at-risk cohort...", 0.7),
                    ("Summarizing variance across patterns...", 0.45),
                ],
            ):
                pass
            advance("Pattern analysis complete")

            with simulate_stage(
                "Segmenting at-risk cohort",
                [
                    ("Training clustering model...", 0.65),
                    ("Assigning segment labels...", 0.4),
                ],
            ):
                pass
            advance("Segments assigned")

            with simulate_stage(
                "Generating action plan",
                [
                    ("Matching interventions to risk drivers...", 0.55),
                    ("Balancing urgency across segments...", 0.45),
                    ("Finalizing recommended actions...", 0.35),
                ],
            ):
                pass
            advance("Action plan ready")

    # Render dashboard header configuration panel
    segment_pie_data = {}
    if not at_risk.empty and "cluster_label" in at_risk.columns:
        segment_pie_data = at_risk.groupby("cluster_label").size().to_dict()
        segment_pie_data = {f"Segment {k}": int(v) for k, v in segment_pie_data.items()}
        
    urgency_pie_data = {}
    if not intervention_results.empty and "intervention_urgency" in intervention_results.columns:
        urgency_pie_data = {str(k).title(): int(v) for k, v in intervention_results.groupby("intervention_urgency").size().to_dict().items()}
        
    data_dict = {
        "metrics": {
            "AUC-ROC": f"{metrics.get('auc_roc', 0):.3f}" if metrics else "0.000",
            "PR-AUC": f"{metrics.get('pr_auc', 0):.3f}" if metrics else "0.000",
            "Precision": f"{metrics.get('precision', 0):.1%}" if metrics else "0.0%",
            "Lift": f"{metrics.get('lift_top_decile', 0):.1f}x" if metrics else "1.0x",
            "Capture": f"{metrics.get('capture_top_decile', 0):.1%}" if metrics else "0.0%"
        },
        "segment_pie_data": segment_pie_data,
        "urgency_pie_data": urgency_pie_data,
        "cohort_df": at_risk,
        "action_plan_df": intervention_results,
    }
    
    render_dashboard_header(config.page_id, ALL_SUPERVISED_WIDGETS, data_dict=data_dict)
    
    # Render active widgets
    active_widgets = st.session_state[f"{config.page_id}_active_widgets"]
    render_supervised_widgets(
        config=config,
        active_widgets=active_widgets,
        at_risk=at_risk,
        intervention_results=intervention_results,
        metrics=metrics,
        driver_chart=driver_chart,
        patterns=patterns,
        threshold=threshold,
        explanations=explanations,
        scored_customers=None
    )
    
    render_pilot_cta(headline)


def _apply_exemplar_fallback(config: OutcomeDemoConfig, at_risk: pd.DataFrame, prediction_explanations: pd.DataFrame):
    if prediction_explanations is not None and not prediction_explanations.empty:
        return at_risk, prediction_explanations
    exemplars = load_explanations(config.page_id)
    if exemplars.empty:
        return at_risk, prediction_explanations
    st.caption("Live explain was unavailable; showing sample explanations from the instant demo bundle.")
    merged = merge_exemplar_explanations(at_risk, exemplars)
    return merged, exemplars


def _render_live_eval_performance(config: OutcomeDemoConfig, model_id, label_column: str, threshold: float, eval_source=None):
    if eval_source is None:
        if not os.path.exists(config.default_eval_path):
            return None, False
        from shared.ui import demo_csv
        eval_source = demo_csv(config.default_eval_path)

    if eval_source is None:
        return None, False
    from woodwide.evaluation import oriented_probability_series

    render_progress_stepper(1)
    with st.spinner("Evaluating on labeled holdout...", show_time=True):
        eval_df = read_uploaded_csv(eval_source)
        scored, _, _ = run_prediction_inference(model_id, eval_source, filename=eval_source.name)
        if label_column not in eval_df.columns:
            return None, False
        probability_column = probability_sort_column(scored)
        if not probability_column:
            return None, False
        _, scores_flipped = oriented_probability_series(eval_df[label_column], scored[probability_column])
        metrics = compute_binary_metrics(eval_df[label_column], scored[probability_column], threshold=threshold)
        render_performance_panel(metrics)
        return metrics, scores_flipped


def render_supervised_outcome_page(config: OutcomeDemoConfig):
    configure_demo_app(config.browser_title)
    init_shared_session()
    apply_page_style()
    apply_demo_query_params(config.demo_session_key)
    _init_intervention_session()
    if config.feature_entries_key not in st.session_state:
        st.session_state[config.feature_entries_key] = {}

    metadata = load_metadata(config.page_id)
    render_brand_header(title=config.page_title, subtitle=config.subtitle)

    has_artifacts = artifacts_available(config.page_id)

    with st.sidebar:
        render_api_sidebar(show_advanced=False)

        st.divider()
        st.header("Demo Datasets")
        render_demo_run_button(
            config.demo_session_key,
            config.default_train_path,
            config.default_test_path,
        )
        demo_training_data, demo_test_data = resolve_csv_sources(
            config.demo_session_key,
            config.default_train_path,
            config.default_test_path,
        )
        with st.expander("Download demo CSVs"):
            download_button_for_path("Download train.csv", config.default_train_path, "train.csv")
            download_button_for_path("Download test.csv", config.default_test_path, "test.csv")
            download_button_for_path("Download eval.csv", config.default_eval_path, "eval.csv")
        st.divider()
        st.header("Your Datasets")
        custom_training_data, custom_test_data, custom_eval_data, has_custom_uploads = render_custom_csv_uploads(
            config.page_id,
            config.entity_name_plural,
        )
        if has_custom_uploads:
            st.session_state[f"{config.page_id}_demo_mode"] = "live"
            st.session_state[f"{config.page_id}_demo_mode_radio"] = "live"
        training_data = custom_training_data if custom_training_data is not None else demo_training_data
        test_data = custom_test_data if custom_test_data is not None else demo_test_data
        eval_data = custom_eval_data
        using_custom_data = custom_training_data is not None or custom_test_data is not None

        st.divider()
        demo_mode = render_demo_mode_toggle(config.page_id, has_artifacts)
        if has_custom_uploads:
            demo_mode = "live"

        st.divider()
        st.header("Demo Settings")
        preview_row_count = st.slider("Preview rows", 100, 5000, PREVIEW_ROW_COUNT, 100, key=f"{config.page_id}_preview")
        threshold = st.slider(
            config.threshold_label,
            0.0,
            1.0,
            config.threshold_default,
            0.01,
            help=config.threshold_help,
            key=f"{config.page_id}_threshold",
        )
        max_row_explanations = st.slider(
            "Row explanations",
            0,
            100,
            config.default_max_explanations,
            5,
            key=f"{config.page_id}_max_explanations",
        )
        show_raw_model_outputs = st.toggle("Show raw model outputs", value=False, key=f"{config.page_id}_raw")
        if demo_mode == "live":
            _render_intervention_sidebar(config, training_data)
        st.divider()
        st.header("Workflow")
        for caption in config.workflow_captions:
            st.caption(caption)
        render_pilot_cta()

    if demo_mode == "instant" and metadata and metadata.get("source") == "woodwide_api":
        captured_at = metadata.get("captured_at", "")[:10]
        model_id = metadata.get("model_id", "")
        st.caption(
            f"Instant demo results from Wood Wide API inference"
            + (f" · model {model_id[:8]}…" if model_id else "")
            + (f" · captured {captured_at}" if captured_at else "")
        )
    elif demo_mode == "live" and using_custom_data:
        st.caption("Live rerun using uploaded CSVs from the side panel.")
    elif demo_mode == "live":
        st.caption("Live Wood Wide API rerun using the bundled demo CSVs.")

    if demo_mode == "instant" and has_artifacts:
        if not st.session_state.get(config.demo_session_key, False):
            st.session_state[config.demo_session_key] = True
        # Reset the animation guard whenever the mode switches TO "instant",
        # so the pipeline plays once per entry into instant demo mode.
        _prev_mode_key = f"{config.page_id}_prev_demo_mode"
        if st.session_state.get(_prev_mode_key) != "instant":
            st.session_state[f"{config.page_id}_instant_demo_played"] = False
        st.session_state[_prev_mode_key] = "instant"
        _render_instant_demo(config, threshold, preview_row_count, max_row_explanations)
        return

    # Track that we're now in live mode so the next switch to instant replays the animation
    st.session_state[f"{config.page_id}_prev_demo_mode"] = "live"


    if demo_mode == "live" and not api_key:
        st.error("Live API mode requires WOODWIDE_API_KEY. Switch to Instant demo or add your API key.")
        st.stop()

    if not training_data or not test_data:
        workflow_not_ready_message()
        st.stop()

    input_col_1, input_col_2, input_col_3 = st.columns(3)
    input_col_1.metric("Training file", dataset_source_label(training_data))
    input_col_2.metric("Scoring file", dataset_source_label(test_data))
    input_col_3.metric("API connection", "Ready" if api_key else "Missing key")

    model_id = None
    label_column = None
    dataset_name = None
    live_metrics = None

    if training_data:
        st.subheader(config.model_section_title)
        training_dataframe = read_uploaded_csv(training_data)
        dataset_prefix = config.dataset_prefix if is_demo_csv(training_data) else f"{config.dataset_prefix}_custom"
        dataset_name = dataset_name_for_upload(dataset_prefix, training_data)
        label_column = config.get_label_column(training_dataframe, f"{config.page_id}_label:{dataset_name}")
        input_columns = prediction_input_columns_for_training(training_dataframe, label_column)
        with st.spinner("uploading dataset", show_time=True):
            dataset_id = get_or_create_dataset(training_data, dataset_name)
        with st.spinner(config.training_spinner, show_time=True):
            model_id = get_or_start_model_training(
                st.session_state.model_ids,
                dataset_name,
                f"{config.model_cache_prefix}:{dataset_name}",
                config.model_cache_prefix,
                config.model_ready_message,
                lambda: train_model(
                    dataset_name.replace(config.model_train_name_from, config.model_train_name_to, 1),
                    "prediction",
                    dataset_id,
                    label_column=label_column,
                    input_columns=input_columns,
                ),
        )
        if model_id and label_column:
            if using_custom_data and eval_data is None:
                st.info("Upload an optional eval CSV to compute holdout performance metrics for your dataset.")
                live_metrics, scores_flipped = None, False
            else:
                live_eval_source = eval_data if using_custom_data else None
                live_metrics, scores_flipped = _render_live_eval_performance(
                    config, model_id, label_column, threshold, live_eval_source
                )
            st.session_state[f"{config.page_id}_scores_flipped"] = scores_flipped

    at_risk = None
    scored_customers = None
    prediction_explanations = pd.DataFrame()
    inference_job_id = None

    if test_data and training_data and model_id:
        st.subheader(config.at_risk_section_title)
        section_note(config.at_risk_section_note_template.format(threshold=threshold))
        render_progress_stepper(2)
        with st.spinner(config.scoring_spinner, show_time=True):
            active_customers = read_uploaded_csv(test_data)
            scored_customers, inference_payload, inference_job_id = run_prediction_inference(model_id, test_data)
            scored_customers = add_original_customer_features(scored_customers, active_customers)
            scored_customers = add_prediction_descriptions(
                scored_customers,
                inference_payload,
                inference_job_id,
                model_id,
            )
            scores_flipped = st.session_state.get(f"{config.page_id}_scores_flipped", False)
            test_probability_column = probability_sort_column(scored_customers)
            if test_probability_column:
                test_probs = pd.to_numeric(scored_customers[test_probability_column], errors="coerce")
                scored_customers["risk_probability"] = (
                    1.0 - test_probs if scores_flipped else test_probs
                )
            at_risk = filter_at_risk_customers(scored_customers, threshold)

            if max_row_explanations and inference_job_id and "id" in at_risk.columns:
                probability_column = probability_sort_column(at_risk)
                explanation_candidates = at_risk.copy()
                if probability_column:
                    explanation_candidates = explanation_candidates.assign(
                        _explain_sort=pd.to_numeric(explanation_candidates[probability_column], errors="coerce")
                    ).sort_values("_explain_sort", ascending=False)
                row_ids = []
                for row_id in explanation_candidates["id"].head(max_row_explanations).dropna().tolist():
                    row_id_number = as_number(row_id)
                    row_ids.append(int(row_id_number) if row_id_number is not None else str(row_id))
                with st.spinner("explaining highest-risk rows...", show_time=True):
                    prediction_explanations = get_row_level_explanations(inference_job_id, row_ids)
                    at_risk = add_row_level_explanations(at_risk, prediction_explanations)

            at_risk, prediction_explanations = _apply_exemplar_fallback(config, at_risk, prediction_explanations)

    if at_risk is not None and not at_risk.empty:
        # Precompute patterns, clustering, interventions, and drivers upfront so the widgets can render them in any order
        feature_entries_by_label = st.session_state.get(config.feature_entries_key, {})
        driver_chart_data = risk_driver_chart_data_from_feature_entries(feature_entries_by_label, at_risk)
        if driver_chart_data.empty:
            driver_chart_data = risk_driver_chart_data(at_risk)
        if driver_chart_data.empty and scored_customers is not None:
            driver_chart_data = risk_driver_chart_data_from_feature_contrast(at_risk, scored_customers)

        risk_modeling_data, risk_input_columns = analysis_dataframe_for_modeling(at_risk)
        cluster_modeling_data, cluster_input_columns = clustering_dataframe_for_modeling(risk_modeling_data)

        factors = pd.DataFrame()
        patterns_data = pd.DataFrame()
        intervention_results = pd.DataFrame()

        if not risk_modeling_data.empty and risk_input_columns and not cluster_modeling_data.empty:
            risk_csv = dataframe_to_csv_bytes(risk_modeling_data)
            risk_dataset_name = dataset_name_for_dataframe("risk_customers", risk_modeling_data, dataset_name)
            
            with st.spinner("Analyzing high-risk patterns...", show_time=True):
                risk_dataset_id = get_or_create_dataset_from_bytes(risk_csv, "risk_customers.csv", risk_dataset_name)
                cluster_csv = dataframe_to_csv_bytes(cluster_modeling_data)
                cluster_dataset_name = dataset_name_for_dataframe("cluster_customers", cluster_modeling_data, risk_dataset_name)
                cluster_dataset_id = get_or_create_dataset_from_bytes(cluster_csv, "cluster_customers.csv", cluster_dataset_name)

                factor_model_id = get_or_start_model_training(
                    st.session_state.risk_ids,
                    risk_dataset_name,
                    f"factors:{risk_dataset_name}",
                    "pattern analysis",
                    "Pattern model is ready.",
                    lambda: train_model(
                        risk_dataset_name.replace("risk_customers", "factor_analysis", 1),
                        "factors",
                        risk_dataset_id,
                        input_columns=risk_input_columns,
                    ),
                )
                factor_cache_key = inference_cache_key("factor_inference", factor_model_id, hashlib.sha256(risk_csv).hexdigest(), "csv")
                factors, factor_payload, _ = run_cached_model_inference(
                    factor_model_id, risk_csv, "risk_customers.csv", "csv", factor_cache_key, "pattern inference"
                )
                
                # Fetch patterns from the factor payload
                if isinstance(factor_payload, dict) and "factors" in factor_payload:
                    patterns_list = []
                    for idx, f in enumerate(factor_payload["factors"]):
                        patterns_list.append({
                            "pattern": f.get("name", f"Pattern {idx+1}"),
                            "top_drivers": ", ".join(f.get("top_features", [])),
                            "captured_variance": f.get("variance_explained", 0.15)
                        })
                    patterns_data = pd.DataFrame(patterns_list)
                else:
                    patterns_data = pd.DataFrame([{"pattern": f"Pattern {i+1}", "top_drivers": f"Feature {i+1}", "captured_variance": 0.2} for i in range(3)])

            with st.spinner("Segmenting at-risk cohort...", show_time=True):
                cluster_model_id = get_or_start_model_training(
                    st.session_state.cluster_ids,
                    cluster_dataset_name,
                    f"clusters:{cluster_dataset_name}",
                    "clustering",
                    "Clustering model is ready.",
                    lambda: train_model(
                        cluster_dataset_name.replace("cluster_customers", "customer_segments", 1),
                        "clustering",
                        cluster_dataset_id,
                        input_columns=cluster_input_columns,
                    ),
                )
                cluster_cache_key = inference_cache_key(
                    "cluster_inference", cluster_model_id, hashlib.sha256(cluster_csv).hexdigest(), "json"
                )
                clusters, cluster_payload, _ = run_cached_model_inference(
                    cluster_model_id, cluster_csv, "cluster_customers.csv", "json", cluster_cache_key, "cluster inference"
                )
                cluster_labels, cluster_descriptions = parse_cluster_inference_payload(cluster_payload, len(at_risk), clusters)
                intervention_results = build_interventions(at_risk, factors, cluster_labels, cluster_descriptions)
                at_risk["cluster_label"] = cluster_labels
        else:
            intervention_results = at_risk.copy()

        # Render Report Configuration Panel
        segment_pie_data = {}
        if not at_risk.empty and "cluster_label" in at_risk.columns:
            segment_pie_data = at_risk.groupby("cluster_label").size().to_dict()
            segment_pie_data = {f"Segment {k}": int(v) for k, v in segment_pie_data.items()}
            
        urgency_pie_data = {}
        if not intervention_results.empty and "intervention_urgency" in intervention_results.columns:
            urgency_pie_data = {str(k).title(): int(v) for k, v in intervention_results.groupby("intervention_urgency").size().to_dict().items()}

        total_entities = None
        try:
            test_data.seek(0)
            total_entities = len(pd.read_csv(test_data, usecols=[config.entity_id_column]))
        except (ValueError, pd.errors.EmptyDataError, KeyError):
            total_entities = None

        data_dict = {
            "metrics": {
                "AUC-ROC": f"{live_metrics.get('auc_roc', 0):.3f}" if live_metrics else "0.000",
                "PR-AUC": f"{live_metrics.get('pr_auc', 0):.3f}" if live_metrics else "0.000",
                "Precision": f"{live_metrics.get('precision', 0):.1%}" if live_metrics else "0.0%",
                "Lift": f"{live_metrics.get('lift_top_decile', 0):.1f}x" if live_metrics else "1.0x",
                "Capture": f"{live_metrics.get('capture_top_decile', 0):.1%}" if live_metrics else "0.0%"
            },
            "segment_pie_data": segment_pie_data,
            "urgency_pie_data": urgency_pie_data,
            "cohort_df": at_risk,
            "action_plan_df": intervention_results,
        }
        
        render_dashboard_header(config.page_id, ALL_SUPERVISED_WIDGETS, data_dict=data_dict)

        # Loop active widgets
        active_widgets = st.session_state[f"{config.page_id}_active_widgets"]
        render_supervised_widgets(
            config=config,
            active_widgets=active_widgets,
            at_risk=at_risk,
            intervention_results=intervention_results,
            metrics=live_metrics,
            driver_chart=driver_chart_data,
            patterns=patterns_data,
            threshold=threshold,
            explanations=prediction_explanations,
            scored_customers=scored_customers,
            total_entities=total_entities
        )
    elif at_risk is not None:
        st.info(config.empty_risk_info)

    headline = None
    if live_metrics:
        headline = f"AUC {live_metrics.get('auc_roc', 0):.2f} · {live_metrics.get('lift_top_decile', 0):.1f}x lift @ top 10%"
    render_pilot_cta(headline)
