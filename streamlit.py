import streamlit as st
import pandas as pd
import os
import requests
import time
import hashlib
import re
import math
from collections import Counter
from io import StringIO
from dotenv import load_dotenv
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Churn Intervention Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# authentication

api_key = os.getenv("WOODWIDE_API_KEY")
base_url = "https://api.woodwide.ai"
headers = {"Authorization": f"Bearer {api_key}"}
MODEL_POLL_INTERVAL_SECONDS = 5
MODEL_TRAIN_TIMEOUT_SECONDS = 30 * 60
PREVIEW_ROW_COUNT = 1000
SUCCESS_STATUS_CODES = (200, 201, 202)

INTERVENTION_CATALOG = [
    {
        "name": "High-value retention save",
        "description": "long account age high total charges loyal subscriber premium plan high lifetime value severe churn risk",
        "keywords": ["AccountAge", "TotalCharges", "SubscriptionType", "Premium", "CustomerID"],
        "low": "Send a loyalty thank-you message with a personalized watch recommendation.",
        "medium": "Offer a loyalty perk such as a one-month upgrade or exclusive content bundle.",
        "high": "Send a high-touch retention offer with a limited-time discount or premium upgrade.",
    },
    {
        "name": "Price sensitivity offer",
        "description": "high monthly charges expensive plan subscription price sensitivity downgrade cancellation cost concern",
        "keywords": ["MonthlyCharges", "SubscriptionType", "Basic", "Standard", "Premium", "TotalCharges"],
        "low": "Recommend a better-fit plan based on recent viewing habits.",
        "medium": "Offer a temporary discount or annual-plan savings option.",
        "high": "Offer an immediate save discount and a lower-cost plan alternative.",
    },
    {
        "name": "Payment friction recovery",
        "description": "payment method paperless billing billing friction payment update invoice card automatic payment",
        "keywords": ["PaymentMethod", "PaperlessBilling", "MonthlyCharges", "TotalCharges"],
        "low": "Send a reminder to enable paperless billing and automatic payments.",
        "medium": "Send a payment-method update prompt with billing support options.",
        "high": "Offer billing support plus a short grace-period or save discount.",
    },
    {
        "name": "Viewing engagement boost",
        "description": "low viewing hours short viewing duration inactive disengaged low watch time content consumption drop",
        "keywords": ["ViewingHoursPerWeek", "AverageViewingDuration", "WatchlistSize", "ContentDownloadsPerMonth"],
        "low": "Send a weekly personalized watchlist based on their preferred genre.",
        "medium": "Send a re-engagement campaign with trending titles and continue-watching prompts.",
        "high": "Offer a free premium-content preview and a personalized comeback playlist.",
    },
    {
        "name": "Content personalization",
        "description": "genre preference content type low content fit weak recommendations movie series sports documentary preference",
        "keywords": ["ContentType", "GenrePreference", "UserRating", "WatchlistSize"],
        "low": "Send personalized recommendations in their preferred genre.",
        "medium": "Send a curated content collection and ask for preference feedback.",
        "high": "Offer a premium-title preview in their preferred genre plus personalized recommendations.",
    },
    {
        "name": "Device access enablement",
        "description": "multi device access registered device device setup streaming device mobile tv tablet access issue",
        "keywords": ["MultiDeviceAccess", "DeviceRegistered", "ContentDownloadsPerMonth"],
        "low": "Send device setup tips for their registered device.",
        "medium": "Promote multi-device viewing and offline download instructions.",
        "high": "Offer assisted device setup and highlight multi-device plan benefits.",
    },
    {
        "name": "Support recovery",
        "description": "support tickets customer issue service problem unresolved frustration help center technical support",
        "keywords": ["SupportTicketsPerMonth"],
        "low": "Send a help-center follow-up for common streaming issues.",
        "medium": "Escalate recent support issues and confirm resolution.",
        "high": "Create an urgent support recovery case with a named support owner.",
    },
    {
        "name": "Experience quality recovery",
        "description": "low user rating poor satisfaction bad experience quality complaint unhappy rating churn risk",
        "keywords": ["UserRating", "SupportTicketsPerMonth", "AverageViewingDuration"],
        "low": "Ask for quick feedback and send recommendations to improve their experience.",
        "medium": "Send a satisfaction recovery survey with a targeted content or plan offer.",
        "high": "Send a save offer and route the customer to support for experience recovery.",
    },
    {
        "name": "Family feature activation",
        "description": "parental control subtitles family household accessibility children captions language preferences",
        "keywords": ["ParentalControl", "SubtitlesEnabled", "MultiDeviceAccess", "WatchlistSize"],
        "low": "Send tips for subtitles, profiles, and parental controls.",
        "medium": "Promote family-friendly watchlists and profile setup guidance.",
        "high": "Offer assisted setup for family controls, subtitles, and multi-device viewing.",
    },
    {
        "name": "Offline viewing activation",
        "description": "content downloads offline viewing mobile travel commute download feature low downloads",
        "keywords": ["ContentDownloadsPerMonth", "DeviceRegistered", "MultiDeviceAccess"],
        "low": "Send a reminder about offline downloads for mobile viewing.",
        "medium": "Recommend downloadable titles from their favorite genre.",
        "high": "Offer a premium mobile/offline viewing bundle or download-focused content pack.",
    },
]


def default_intervention_catalog():
    return [
        {
            **intervention,
            "keywords": list(intervention["keywords"]),
        }
        for intervention in INTERVENTION_CATALOG
    ]


def current_intervention_catalog():
    return st.session_state.get("intervention_catalog") or INTERVENTION_CATALOG


def intervention_catalog_to_dataframe(catalog):
    return pd.DataFrame(
        [
            {
                "name": intervention["name"],
                "description": intervention["description"],
                "keywords": ", ".join(intervention["keywords"]),
                "low": intervention["low"],
                "medium": intervention["medium"],
                "high": intervention["high"],
            }
            for intervention in catalog
        ]
    )


def dataframe_to_intervention_catalog(dataframe):
    catalog = []
    for _, row in dataframe.dropna(subset=["name"]).iterrows():
        keywords = [
            keyword.strip()
            for keyword in str(row.get("keywords", "")).split(",")
            if keyword.strip()
        ]
        catalog.append(
            {
                "name": str(row.get("name", "")).strip(),
                "description": str(row.get("description", "")).strip(),
                "keywords": keywords,
                "low": str(row.get("low", "")).strip(),
                "medium": str(row.get("medium", "")).strip(),
                "high": str(row.get("high", "")).strip(),
            }
        )
    return catalog


def find_intervention_by_name(name):
    for intervention in current_intervention_catalog():
        if intervention["name"] == name:
            return intervention
    return None


def uploaded_file_hash(uploaded_file):
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def safe_name_part(value):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def dataset_name_for_upload(prefix, uploaded_file):
    file_stem = os.path.splitext(uploaded_file.name)[0]
    safe_stem = safe_name_part(file_stem)
    file_hash = uploaded_file_hash(uploaded_file)[:12]
    return f"{prefix}_{safe_stem}_{file_hash}"


def dataframe_to_csv_bytes(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


def dataset_name_for_dataframe(prefix, dataframe, source_name):
    safe_source = safe_name_part(source_name)
    dataframe_hash = hashlib.sha256(dataframe_to_csv_bytes(dataframe)).hexdigest()[:12]
    return f"{prefix}_{safe_source}_{dataframe_hash}"


def find_dataset_by_name(dataset_name):
    for params in ({"name": dataset_name}, {"dataset_name": dataset_name}, None):
        response = requests.get(
            f"{base_url}/datasets",
            headers=headers,
            params=params,
        )
        response.raise_for_status()

        datasets_response = response.json()
        if isinstance(datasets_response, list):
            datasets = datasets_response
        else:
            datasets = []
            for key in ("datasets", "data", "items", "results"):
                value = datasets_response.get(key)
                if isinstance(value, list):
                    datasets = value
                    break

        for dataset in datasets:
            possible_names = (
                dataset.get("name"),
                dataset.get("dataset_name"),
                dataset.get("display_name"),
            )
            if dataset_name in possible_names:
                return dataset

    return None


def dataset_id_from_dataset(dataset):
    return dataset.get("id") or dataset.get("dataset_id")


def get_or_create_dataset(uploaded_file, dataset_name):
    uploaded_file.seek(0)
    return get_or_create_dataset_from_bytes(
        uploaded_file.getvalue(),
        uploaded_file.name,
        dataset_name,
    )


def get_or_create_dataset_from_bytes(file_bytes, filename, dataset_name):
    existing_dataset = find_dataset_by_name(dataset_name)
    if existing_dataset:
        st.write(f"Reusing existing dataset: {dataset_name}")
        return dataset_id_from_dataset(existing_dataset)

    response = requests.post(
        f"{base_url}/datasets",
        headers=headers,
        files={"file": (filename, file_bytes, "text/csv")},
        data={"dataset_name": dataset_name},
    )

    if response.status_code not in SUCCESS_STATUS_CODES:
        if response.status_code == 400 and "already exists" in response.text:
            existing_dataset = find_dataset_by_name(dataset_name)
            if existing_dataset:
                st.write(f"Reusing existing dataset: {dataset_name}")
                return dataset_id_from_dataset(existing_dataset)

        st.write(response.status_code)
        st.write(response.text)
        response.raise_for_status()

    return response.json()["dataset"]["id"]


def wait_for_model_ready(model_id):
    started_at = time.monotonic()
    status_placeholder = st.empty()

    while True:
        status_response = requests.get(
            f"{base_url}/models/{model_id}",
            headers=headers,
        )

        if status_response.status_code != 200:
            st.write(status_response.status_code)
            st.write(status_response.text)
            status_response.raise_for_status()

        model_response = status_response.json()
        status = model_response.get("status")
        elapsed_seconds = int(time.monotonic() - started_at)
        status_placeholder.write(
            f"Model status: {status or 'unknown'} ({elapsed_seconds // 60}m {elapsed_seconds % 60}s)"
        )

        if status in ("ready", "completed", "succeeded", "success"):
            return True

        if status in ("failed", "error"):
            st.write(model_response)
            return False

        if elapsed_seconds >= MODEL_TRAIN_TIMEOUT_SECONDS:
            st.write("Timed out waiting for model training.")
            st.write(model_response)
            return False

        time.sleep(MODEL_POLL_INTERVAL_SECONDS)


def read_inference_csv(response):
    content_type = response.headers.get("content-type", "")

    if "application/json" in content_type:
        payload = response.json()
        data = payload.get("data")

        if isinstance(data, str):
            return pd.read_csv(StringIO(data))

        if isinstance(data, list):
            return pd.DataFrame(data)

        st.write(payload)
        raise ValueError("Inference response JSON did not include CSV or row data.")

    return pd.read_csv(StringIO(response.text))


def token_vector(text):
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return Counter(tokens)


def cosine_similarity(left_text, right_text):
    left = token_vector(left_text)
    right = token_vector(right_text)
    if not left or not right:
        return 0

    shared_score = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return shared_score / (left_norm * right_norm)


def match_intervention(context_text):
    matches = [
        (
            cosine_similarity(context_text, intervention["description"]),
            intervention,
        )
        for intervention in current_intervention_catalog()
    ]
    return max(matches, key=lambda match: match[0])


def row_signal_score(row, keywords):
    score = 0
    for column, value in row.items():
        text = f"{column} {value}".lower()
        for keyword in keywords:
            if keyword in text:
                numeric_value = as_number(value)
                if numeric_value is None:
                    score += 1
                elif numeric_value > 0:
                    score += min(abs(numeric_value), 10)
    return score


def choose_intervention(row, context_text):
    primary_factor_description = str(row.get("primary_factor_description", "")).lower()
    if "billing" in primary_factor_description or "payment" in primary_factor_description:
        intervention = find_intervention_by_name("Payment friction recovery")
        if intervention:
            return 1, intervention, "primary factor"
    if "usage" in primary_factor_description or "adoption" in primary_factor_description:
        intervention = find_intervention_by_name("Viewing engagement boost")
        if intervention:
            return 1, intervention, "primary factor"
    if "support" in primary_factor_description or "experience" in primary_factor_description:
        intervention = find_intervention_by_name("Support recovery")
        if intervention:
            return 1, intervention, "primary factor"
    if "value" in primary_factor_description or "qualitative" in primary_factor_description or "insight" in primary_factor_description:
        intervention = find_intervention_by_name("Content personalization")
        if intervention:
            return 1, intervention, "primary factor"
    if "confidence" in primary_factor_description or "risk score" in primary_factor_description:
        intervention = find_intervention_by_name("High-value retention save")
        if intervention:
            return 1, intervention, "primary factor"

    semantic_matches = [
        (
            cosine_similarity(context_text, intervention["description"]),
            intervention,
        )
        for intervention in current_intervention_catalog()
    ]
    best_semantic_score, best_semantic_intervention = max(
        semantic_matches,
        key=lambda match: match[0],
    )

    signal_matches = [
        (
            row_signal_score(row, intervention["keywords"]),
            intervention,
        )
        for intervention in current_intervention_catalog()
    ]
    best_signal_score, best_signal_intervention = max(
        signal_matches,
        key=lambda match: match[0],
    )

    if best_signal_score > 0:
        return best_signal_score, best_signal_intervention, "customer columns"

    return best_semantic_score, best_semantic_intervention, "factor/cluster text"


def risk_probability(row):
    probability_columns = [
        "prediction_prob",
        "prediction_probability",
        "probability",
        "confidence",
        "prediction_confidence",
        "score",
    ]
    for column in probability_columns:
        if column in row.index and pd.notna(row[column]):
            try:
                value = float(row[column])
            except (TypeError, ValueError):
                continue
            return value if value <= 1 else value / 100

    prediction_columns = [column for column in row.index if "prediction" in column.lower()]
    for column in prediction_columns:
        value = as_number(row[column])
        if value is not None and value not in (0, 1):
            return value if value <= 1 else value / 100

    return None


def as_number(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def urgency_from_probability(probability):
    if probability is None:
        return "medium"
    if probability >= 0.9:
        return "high"
    if probability >= 0.75:
        return "medium"
    return "low"


def urgency_from_relative_risk(probability, high_cutoff, medium_cutoff):
    if probability is None:
        return None
    if probability >= 0.9:
        return "high"
    if probability >= high_cutoff:
        return "high"
    if probability >= medium_cutoff:
        return "medium"
    return "low"


def probability_sort_column(dataframe):
    candidate_columns = [
        "prediction_prob",
        "prediction_probability",
        "probability",
        "confidence",
        "prediction_confidence",
        "score",
    ]
    for column in candidate_columns:
        if column in dataframe.columns:
            return column

    prediction_columns = [column for column in dataframe.columns if "prediction" in column.lower()]
    for column in prediction_columns:
        if column != "prediction":
            return column

    return None


def fallback_urgency_from_rank(row_index, total_rows):
    if total_rows <= 1:
        return "medium"
    percentile = row_index / (total_rows - 1)
    if percentile < 0.2:
        return "high"
    if percentile < 0.7:
        return "medium"
    return "low"


def probability_series(dataframe):
    probability_column = probability_sort_column(dataframe)
    if not probability_column:
        return None

    probabilities = pd.to_numeric(dataframe[probability_column], errors="coerce")
    if probabilities.notna().any():
        return probabilities.where(probabilities <= 1, probabilities / 100)

    return None


def filter_at_risk_customers(scored_customers, churn_confidence_threshold):
    at_risk_customers = scored_customers.copy()

    if "prediction" in at_risk_customers.columns:
        at_risk_customers = at_risk_customers[at_risk_customers["prediction"] == 1]

    probabilities = probability_series(at_risk_customers)
    if probabilities is None:
        st.warning("No probability column was found, so at-risk filtering used only the prediction label.")
        return at_risk_customers

    return at_risk_customers[probabilities >= churn_confidence_threshold]


def parse_cluster_inference(response, expected_rows):
    payload = response.json()
    data = payload.get("data", payload)
    cluster_labels = data.get("cluster_label") or data.get("cluster_labels") or data.get("clusters")
    cluster_descriptions = data.get("cluster_descriptions") or data.get("cluster_description") or {}

    if isinstance(cluster_labels, dict):
        sorted_indices = sorted(
            cluster_labels,
            key=lambda key: (0, int(key)) if str(key).isdigit() else (1, str(key)),
        )
        labels = [cluster_labels[key] for key in sorted_indices]
    elif isinstance(cluster_labels, list):
        labels = cluster_labels
    else:
        labels = [None] * expected_rows

    if len(labels) < expected_rows:
        labels = labels + [None] * (expected_rows - len(labels))

    return labels[:expected_rows], cluster_descriptions


def build_factor_metadata(factors):
    metadata = {}
    factor_prefixes = [
        column.removesuffix("score")
        for column in factors.columns
        if column.endswith("score")
    ]

    for factor_prefix in factor_prefixes:
        description_column = f"{factor_prefix}description"
        variance_column = f"{factor_prefix}captured_variance"
        metadata[factor_prefix] = {
            "description": factors[description_column].iloc[0]
            if description_column in factors.columns
            else factor_prefix,
            "captured_variance": factors[variance_column].iloc[0]
            if variance_column in factors.columns
            else 0,
        }

    return metadata


def primary_factor_from_row(row, factor_metadata):
    scored_factors = []
    for factor_prefix, metadata in factor_metadata.items():
        score_column = f"{factor_prefix}score"
        score = as_number(row[score_column]) if score_column in row.index else None
        if score is not None:
            scored_factors.append((score, factor_prefix, metadata["description"]))

    if scored_factors:
        score, factor_prefix, description = max(scored_factors, key=lambda item: abs(item[0]))
        return factor_prefix, description, score

    if factor_metadata:
        factor_prefix, metadata = max(
            factor_metadata.items(),
            key=lambda item: as_number(item[1]["captured_variance"], 0),
        )
        return factor_prefix, metadata["description"], None

    return None, "General churn risk", None


def cluster_description_for_label(cluster_label, cluster_descriptions):
    if cluster_label is None:
        return ""

    if isinstance(cluster_descriptions, dict):
        description = cluster_descriptions.get(cluster_label) or cluster_descriptions.get(str(cluster_label))
        if description:
            return description
        if str(cluster_label).isdigit():
            return cluster_descriptions.get(int(cluster_label), "")
        return ""

    if isinstance(cluster_descriptions, list):
        cluster_index = as_number(cluster_label)
        if cluster_index is not None and int(cluster_index) < len(cluster_descriptions):
            return cluster_descriptions[int(cluster_index)]

    return ""


def build_interventions(at_risk, factors, cluster_labels, cluster_descriptions):
    result = at_risk.reset_index(drop=True).copy()
    result["_original_row_index"] = result.index
    factor_metadata = build_factor_metadata(factors)

    if len(factors) == len(result):
        factor_score_columns = [
            column
            for column in factors.columns
            if column.endswith("score") and column not in result.columns
        ]
        result = pd.concat(
            [result, factors[factor_score_columns].reset_index(drop=True)],
            axis=1,
        )

    result["cluster_label"] = cluster_labels
    result["cluster_description"] = [
        cluster_description_for_label(label, cluster_descriptions)
        for label in cluster_labels
    ]

    sort_column = probability_sort_column(result)
    high_cutoff = None
    medium_cutoff = None
    if sort_column:
        sort_values = pd.to_numeric(result[sort_column], errors="coerce")
        if sort_values.notna().any():
            normalized_sort_values = sort_values.where(sort_values <= 1, sort_values / 100)
            high_cutoff = normalized_sort_values.quantile(0.8)
            medium_cutoff = normalized_sort_values.quantile(0.3)
            result = result.assign(_risk_sort=sort_values).sort_values("_risk_sort", ascending=False)

    interventions = []
    for row_rank, (_, row) in enumerate(result.iterrows()):
        factor_prefix, factor_description, factor_score = primary_factor_from_row(row, factor_metadata)
        probability = risk_probability(row)
        if high_cutoff is not None and medium_cutoff is not None:
            urgency = urgency_from_relative_risk(probability, high_cutoff, medium_cutoff)
        else:
            urgency = urgency_from_probability(probability)

        if urgency is None:
            urgency = fallback_urgency_from_rank(row_rank, len(result))

        context = f"{factor_description} {row.get('cluster_description', '')}"
        row_context = row.to_dict()
        row_context["primary_factor_description"] = factor_description
        match_score, intervention, match_source = choose_intervention(row_context, context)
        interventions.append(
            {
                "primary_factor": factor_prefix,
                "primary_factor_description": factor_description,
                "primary_factor_score": factor_score,
                "intervention_category": intervention["name"],
                "intervention_action": intervention[urgency],
                "intervention_urgency": urgency,
                "intervention_match_source": match_source,
                "intervention_match_score": round(match_score, 3),
            }
        )

    return (
        pd.concat([result.reset_index(drop=True), pd.DataFrame(interventions)], axis=1)
        .drop(columns=["_risk_sort", "_original_row_index"], errors="ignore")
    )


if "model_ids" not in st.session_state:
    st.session_state.model_ids = {}

if "intervention_catalog" not in st.session_state:
    st.session_state.intervention_catalog = default_intervention_catalog()

if "show_intervention_editor" not in st.session_state:
    st.session_state.show_intervention_editor = False

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e6e8ef;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        color: #667085;
    }
    .section-note {
        color: #667085;
        font-size: 0.95rem;
        margin-top: -0.35rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Churn Intervention Studio")
st.caption("Train a churn model, identify at-risk subscribers, explain the risk drivers, and produce a targeted intervention plan.")

'''
workflow:
user uploads historical customer dataset
(later) use pandas to do some feature engineering, data cleanup (maybe LLM to do feature engineering/selection?)
train wood wide prediction model on historical customer dataset, input label is churn

user uploads current customer dataset (no churn label)
run inference using prediction model to discover at-risk customers
display a list to the user (with probabilities)
run unsupervised factor analysis or clustering to discover factors/group customers that are at risk
based on groups, recommend interventions (LLM?)
display interventions

Next steps:
Allow user to decide what criteria is needed for intervention (not just the ones that are predicted to churn, but also the ones that are low confidence of not churning)
What threshold of confidence does the user want? Maybe 55% confidence churn shouldn't be given an intervention
Use LLM to draft email intervention, discount
Allow user-established rules (if _% confidence to churn, do this. if this category and _% confidence churn, do whatever.)
Connect program to database (supabase/databricks) and run churn prediction every time interval -> sales/CEO dashboard 
-> model can retrain itself based on new data (e.g customer just churned)
'''

with st.sidebar:
    st.header("Inputs")
    if api_key:
        st.success("Wood Wide API key loaded")
    else:
        st.error("Missing WOODWIDE_API_KEY")

    training_data = st.file_uploader(
        "Historical customers",
        type=["csv"],
        help="Training data with a churn label column.",
    )
    test_data = st.file_uploader(
        "Active customers",
        type=["csv"],
        help="Current customers to score for churn risk.",
    )

    st.divider()
    st.header("Demo Datasets")
    st.caption("Use these defaults if you do not have client data ready.")
    if os.path.exists("train.csv"):
        with open("train.csv", "rb") as file:
            st.download_button(
                "Download default train.csv",
                file,
                "train.csv",
                "text/csv",
                use_container_width=True,
            )
    else:
        st.caption("Default train.csv was not found.")

    if os.path.exists("test.csv"):
        with open("test.csv", "rb") as file:
            st.download_button(
                "Download default test.csv",
                file,
                "test.csv",
                "text/csv",
                use_container_width=True,
            )
    else:
        st.caption("Default test.csv was not found.")

    st.divider()
    st.header("Demo Settings")
    preview_row_count = st.slider(
        "Preview rows",
        min_value=100,
        max_value=5000,
        value=PREVIEW_ROW_COUNT,
        step=100,
    )
    churn_confidence_threshold = st.slider(
        "At-risk confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        help="Customers must meet or exceed this churn probability to be included in the at-risk workflow.",
    )
    show_raw_model_outputs = st.toggle("Show raw model outputs", value=False)

    st.divider()
    st.header("Interventions")
    if st.button("Customize interventions", use_container_width=True):
        st.session_state.show_intervention_editor = not st.session_state.show_intervention_editor

    if st.session_state.show_intervention_editor:
        edited_interventions = st.data_editor(
            intervention_catalog_to_dataframe(st.session_state.intervention_catalog),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="intervention_catalog_editor",
            column_config={
                "description": st.column_config.TextColumn(width="medium"),
                "keywords": st.column_config.TextColumn(
                    width="medium",
                    help="Comma-separated words or dataset column names used for matching.",
                ),
                "low": st.column_config.TextColumn(width="medium"),
                "medium": st.column_config.TextColumn(width="medium"),
                "high": st.column_config.TextColumn(width="medium"),
            },
        )
        apply_col, reset_col = st.columns(2)
        with apply_col:
            if st.button("Apply", use_container_width=True):
                st.session_state.intervention_catalog = dataframe_to_intervention_catalog(edited_interventions)
                st.success("Interventions updated.")
        with reset_col:
            if st.button("Reset", use_container_width=True):
                st.session_state.intervention_catalog = default_intervention_catalog()
                st.success("Defaults restored.")

    st.divider()
    st.header("Workflow")
    st.caption("1. Upload training and active-customer CSVs.")
    st.caption("2. Train or reuse the churn model.")
    st.caption("3. Score active customers and isolate predicted churners.")
    st.caption("4. Run factor analysis and clustering.")
    st.caption("5. Generate intervention actions.")

if not training_data or not test_data:
    st.info("Upload both CSV files in the sidebar to run the full demo.")

input_col_1, input_col_2, input_col_3 = st.columns(3)
input_col_1.metric("Training file", training_data.name if training_data else "Not uploaded")
input_col_2.metric("Active-customer file", test_data.name if test_data else "Not uploaded")
input_col_3.metric("API connection", "Ready" if api_key else "Missing key")

model_id = None

# on file upload
if training_data:
    st.subheader("1. Churn Model")
    st.markdown('<div class="section-note">Historical data is uploaded once, then reused by content hash on later runs.</div>', unsafe_allow_html=True)
    with st.spinner("uploading dataset", show_time=True):
        dataset_name = dataset_name_for_upload("historical_customers", training_data)
        dataset_id = get_or_create_dataset(training_data, dataset_name)

    if dataset_name in st.session_state.model_ids:
        model_id = st.session_state.model_ids[dataset_name]
        st.success("Reusing trained churn model for this upload.")
    else:
        with st.spinner("training churn model", show_time=True):
            model_name = dataset_name.replace("historical_customers", "churn_prediction", 1)
            response = requests.post(
                f"{base_url}/models/train",
                headers = headers,
                json = {
                    "model_name": model_name,
                    "model_type": "prediction",
                    "dataset_id": dataset_id,
                    "label_column": "Churn"
                }
            )

            if response.status_code not in SUCCESS_STATUS_CODES:
                st.write(response.status_code)
                st.write(response.text)
                response.raise_for_status()

            model_id = response.json()["model"]["id"]

            if wait_for_model_ready(model_id):
                st.success("Churn model is ready.")
                st.session_state.model_ids[dataset_name] = model_id
            else:
                st.error("Churn model training failed or timed out.")
                st.stop()

at_risk = None

if test_data:
    if not training_data:
        st.warning("Upload training data first.")
    elif not model_id:
        st.warning("Wait for model training to finish before running inference.")
    else:
        st.subheader("2. At-Risk Customers")
        st.markdown(
            f'<div class="section-note">Active customers are scored and filtered to predicted churners at or above {churn_confidence_threshold:.0%} confidence.</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("identifying at risk customers...", show_time=True):
            test_data.seek(0)
            response = requests.post(
                f"{base_url}/models/{model_id}/infer",
                headers=headers,
                files={"file": ("test.csv", test_data, "text/csv")},
                data={"output_type": "csv"}
            )

            if response.status_code not in SUCCESS_STATUS_CODES:
                st.write(response.status_code)
                st.write(response.text)
                response.raise_for_status()

            at_risk = read_inference_csv(response)
            scored_customers = at_risk
            at_risk = filter_at_risk_customers(scored_customers, churn_confidence_threshold)

if at_risk is not None:
    total_customers = None
    try:
        test_data.seek(0)
        total_customers = len(pd.read_csv(test_data, usecols=["CustomerID"]))
    except (ValueError, pd.errors.EmptyDataError):
        total_customers = None

    risk_metric_1, risk_metric_2, risk_metric_3, risk_metric_4 = st.columns(4)
    risk_metric_1.metric("At-risk customers", f"{len(at_risk):,}")
    risk_metric_2.metric("Previewing", f"{min(len(at_risk), preview_row_count):,}")
    risk_metric_3.metric("Active customers", f"{total_customers:,}" if total_customers is not None else "Uploaded")
    risk_metric_4.metric("Threshold", f"{churn_confidence_threshold:.0%}")

    at_risk_tab, at_risk_download_tab = st.tabs(["Preview", "Download"])
    with at_risk_tab:
        st.dataframe(
            at_risk.head(preview_row_count),
            use_container_width=True,
            height=460,
        )
    with at_risk_download_tab:
        st.download_button(
            "Download at-risk customers CSV",
            at_risk.to_csv(index=False).encode("utf-8"),
            "at_risk_customers.csv",
            "text/csv",
            use_container_width=True,
        )

if "risk_ids" not in st.session_state:
    st.session_state.risk_ids = {}

if "cluster_ids" not in st.session_state:
    st.session_state.cluster_ids = {}

if at_risk is not None:
    if at_risk.empty:
        st.info("No at-risk customers found, so factor analysis was skipped.")
    else:
        st.subheader("3. Risk Drivers")
        st.markdown('<div class="section-note">Factor analysis summarizes why customers are at risk.</div>', unsafe_allow_html=True)
        risk_csv = dataframe_to_csv_bytes(at_risk)
        risk_dataset_name = dataset_name_for_dataframe(
            "risk_customers",
            at_risk,
            dataset_name,
        )
        risk_dataset_id = get_or_create_dataset_from_bytes(
            risk_csv,
            "risk_customers.csv",
            risk_dataset_name,
        )

        if risk_dataset_name in st.session_state.risk_ids:
            factor_model_id = st.session_state.risk_ids[risk_dataset_name]
            st.success("Reusing trained factor analysis model for these at-risk customers.")
        else:
            with st.spinner("training factor analysis model", show_time=True):
                factor_model_name = risk_dataset_name.replace("risk_customers", "factor_analysis", 1)
                response = requests.post(
                    f"{base_url}/models/train",
                    headers=headers,
                    json={
                        "model_name": factor_model_name,
                        "model_type": "factors",
                        "dataset_id": risk_dataset_id,
                    },
                )

                if response.status_code not in SUCCESS_STATUS_CODES:
                    st.write(response.status_code)
                    st.write(response.text)
                    response.raise_for_status()

                factor_model_id = response.json()["model"]["id"]

                if wait_for_model_ready(factor_model_id):
                    st.success("Factor analysis model is ready.")
                    st.session_state.risk_ids[risk_dataset_name] = factor_model_id
                else:
                    st.error("Factor analysis model training failed or timed out.")
                    st.stop()

        with st.spinner("factoring...", show_time=True):
            response = requests.post(
                f"{base_url}/models/{factor_model_id}/infer",
                headers=headers,
                files={"file": ("risk_customers.csv", risk_csv, "text/csv")},
                data={"output_type": "csv"},
            )

            if response.status_code not in SUCCESS_STATUS_CODES:
                st.write(response.status_code)
                st.write(response.text)
                response.raise_for_status()

            factors = read_inference_csv(response)

        if show_raw_model_outputs:
            st.dataframe(factors, use_container_width=True, height=360)

        factor_prefixes = [
            column.removesuffix("score")
            for column in factors.columns
            if column.endswith("score")
        ]
        chart_rows = []
        for factor_prefix in factor_prefixes:
            description_column = f"{factor_prefix}description"
            variance_column = f"{factor_prefix}captured_variance"
            if description_column not in factors.columns or variance_column not in factors.columns:
                continue

            captured_variance = as_number(factors[variance_column].iloc[0], 0)
            if captured_variance <= 0:
                continue

            chart_rows.append(
                {
                    "label": factors[description_column].iloc[0],
                    "size": captured_variance,
                }
            )

        if chart_rows:
            chart_data = pd.DataFrame(chart_rows)
            chart_col, detail_col = st.columns([1, 1])
            fig, ax = plt.subplots()
            ax.pie(
                chart_data["size"],
                labels=chart_data["label"],
                autopct="%1.1f%%",
            )
            ax.set_title("Captured variance by churn-risk factor")
            ax.axis("equal")
            with chart_col:
                st.pyplot(fig)
            with detail_col:
                st.dataframe(chart_data, use_container_width=True, hide_index=True)
        else:
            st.info("No factor variance columns were available for the pie chart.")

        st.subheader("4. Customer Segments")
        st.markdown('<div class="section-note">Clustering groups at-risk customers so intervention actions can account for segment context.</div>', unsafe_allow_html=True)
        if risk_dataset_name in st.session_state.cluster_ids:
            cluster_model_id = st.session_state.cluster_ids[risk_dataset_name]
            st.success("Reusing trained clustering model for these at-risk customers.")
        else:
            with st.spinner("training clustering model", show_time=True):
                cluster_model_name = risk_dataset_name.replace("risk_customers", "customer_segments", 1)
                response = requests.post(
                    f"{base_url}/models/train",
                    headers=headers,
                    json={
                        "model_name": cluster_model_name,
                        "model_type": "clustering",
                        "dataset_id": risk_dataset_id,
                    },
                )

                if response.status_code not in SUCCESS_STATUS_CODES:
                    st.write(response.status_code)
                    st.write(response.text)
                    response.raise_for_status()

                cluster_model_id = response.json()["model"]["id"]

                if wait_for_model_ready(cluster_model_id):
                    st.success("Clustering model is ready.")
                    st.session_state.cluster_ids[risk_dataset_name] = cluster_model_id
                else:
                    st.error("Clustering model training failed or timed out.")
                    st.stop()

        with st.spinner("segmenting at-risk customers...", show_time=True):
            response = requests.post(
                f"{base_url}/models/{cluster_model_id}/infer",
                headers=headers,
                files={"file": ("risk_customers.csv", risk_csv, "text/csv")},
                data={"output_type": "json"},
            )

            if response.status_code not in SUCCESS_STATUS_CODES:
                st.write(response.status_code)
                st.write(response.text)
                response.raise_for_status()

            cluster_labels, cluster_descriptions = parse_cluster_inference(response, len(at_risk))

        intervention_results = build_interventions(
            at_risk,
            factors,
            cluster_labels,
            cluster_descriptions,
        )

        st.subheader("5. Intervention Plan")
        st.markdown('<div class="section-note">Each predicted churner gets an action based on factor scores, segment context, and relative risk.</div>', unsafe_allow_html=True)

        high_count = (intervention_results["intervention_urgency"] == "high").sum()
        medium_count = (intervention_results["intervention_urgency"] == "medium").sum()
        low_count = (intervention_results["intervention_urgency"] == "low").sum()
        metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
        metric_col_1.metric("Planned actions", f"{len(intervention_results):,}")
        metric_col_2.metric("High urgency", f"{high_count:,}")
        metric_col_3.metric("Medium urgency", f"{medium_count:,}")
        metric_col_4.metric("Low urgency", f"{low_count:,}")

        plan_tab, summary_tab, download_tab = st.tabs(["Plan", "Summary", "Download"])
        with plan_tab:
            preferred_columns = [
                "CustomerID",
                "prediction_prob",
                "intervention_urgency",
                "intervention_category",
                "intervention_action",
                "primary_factor_description",
                "cluster_label",
                "intervention_match_source",
            ]
            visible_columns = [
                column for column in preferred_columns if column in intervention_results.columns
            ]
            if not visible_columns:
                visible_columns = intervention_results.columns.to_list()
            st.dataframe(
                intervention_results[visible_columns].head(preview_row_count),
                use_container_width=True,
                height=520,
            )

        with summary_tab:
            summary_col_1, summary_col_2 = st.columns(2)
            category_counts = (
                intervention_results["intervention_category"]
                .value_counts()
                .rename_axis("category")
                .reset_index(name="customers")
            )
            urgency_counts = (
                intervention_results["intervention_urgency"]
                .value_counts()
                .rename_axis("urgency")
                .reset_index(name="customers")
            )
            with summary_col_1:
                st.dataframe(category_counts, use_container_width=True, hide_index=True)
                st.bar_chart(category_counts, x="category", y="customers")
            with summary_col_2:
                st.dataframe(urgency_counts, use_container_width=True, hide_index=True)
                st.bar_chart(urgency_counts, x="urgency", y="customers")

        with download_tab:
            st.download_button(
                "Download intervention plan CSV",
                intervention_results.to_csv(index=False).encode("utf-8"),
                "intervention_plan.csv",
                "text/csv",
                use_container_width=True,
            )
