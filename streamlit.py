import streamlit as st
import pandas as pd
import os
import requests
import time
import hashlib
import re
import math
import json
import uuid
import sqlite3
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
base_url = os.getenv("WOODWIDE_BASE_URL", "https://api.woodwide.ai").rstrip("/")
headers = {"Authorization": f"Bearer {api_key}"}
MODEL_POLL_INTERVAL_SECONDS = 5
MODEL_TRAIN_TIMEOUT_SECONDS = 30 * 60
PREVIEW_ROW_COUNT = 1000
SUCCESS_STATUS_CODES = (200, 201, 202)
READY_STATUSES = ("ready", "completed", "succeeded", "success")
FAILED_STATUSES = ("failed", "error")
PENDING_STATUSES = ("pending", "queued", "running", "processing", "in_progress", "started")
NO_READY_DATASET_VERSION_TEXT = "No ready dataset version found"
DEFAULT_TRAIN_DATASET_PATH = os.path.join("datasets", "train.csv")
DEFAULT_TEST_DATASET_PATH = os.path.join("datasets", "test.csv")
LOCAL_CACHE_DIR = ".streamlit_cache"
LOCAL_CACHE_DB_PATH = os.path.join(LOCAL_CACHE_DIR, "woodwide_jobs.sqlite3")

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


def local_cache_connection():
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
    connection = sqlite3.connect(LOCAL_CACHE_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_local_cache():
    with local_cache_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_name TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ready_models (
                ready_key TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_jobs (
                pending_key TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                job_id TEXT,
                updated_at REAL NOT NULL
            )
            """
        )


def cache_get_dataset_id(dataset_name):
    with local_cache_connection() as connection:
        row = connection.execute(
            "SELECT dataset_id FROM datasets WHERE dataset_name = ?",
            (dataset_name,),
        ).fetchone()
    return row["dataset_id"] if row else None


def cache_save_dataset(dataset_name, dataset_id):
    if not dataset_id:
        return

    with local_cache_connection() as connection:
        connection.execute(
            """
            INSERT INTO datasets (dataset_name, dataset_id, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(dataset_name) DO UPDATE SET
                dataset_id = excluded.dataset_id,
                updated_at = excluded.updated_at
            """,
            (dataset_name, dataset_id, time.time()),
        )


def cache_load_ready_models():
    with local_cache_connection() as connection:
        rows = connection.execute("SELECT ready_key, model_id FROM ready_models").fetchall()
    return {row["ready_key"]: row["model_id"] for row in rows}


def cache_load_ready_models_with_prefix(prefix):
    with local_cache_connection() as connection:
        rows = connection.execute(
            "SELECT ready_key, model_id FROM ready_models WHERE ready_key LIKE ?",
            (f"{prefix}%",),
        ).fetchall()
    return {row["ready_key"]: row["model_id"] for row in rows}


def cache_save_ready_model(ready_key, model_id):
    if not model_id:
        return

    with local_cache_connection() as connection:
        connection.execute(
            """
            INSERT INTO ready_models (ready_key, model_id, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(ready_key) DO UPDATE SET
                model_id = excluded.model_id,
                updated_at = excluded.updated_at
            """,
            (ready_key, model_id, time.time()),
        )


def cache_delete_ready_model(ready_key):
    with local_cache_connection() as connection:
        connection.execute("DELETE FROM ready_models WHERE ready_key = ?", (ready_key,))


def cache_load_pending_jobs():
    with local_cache_connection() as connection:
        rows = connection.execute("SELECT pending_key, model_id, job_id FROM pending_jobs").fetchall()
    return {
        row["pending_key"]: {
            "model_id": row["model_id"],
            "job_id": row["job_id"],
        }
        for row in rows
    }


def cache_save_pending_job(pending_key, model_id, job_id):
    if not model_id:
        return

    with local_cache_connection() as connection:
        connection.execute(
            """
            INSERT INTO pending_jobs (pending_key, model_id, job_id, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(pending_key) DO UPDATE SET
                model_id = excluded.model_id,
                job_id = excluded.job_id,
                updated_at = excluded.updated_at
            """,
            (pending_key, model_id, job_id, time.time()),
        )


def cache_delete_pending_job(pending_key):
    with local_cache_connection() as connection:
        connection.execute("DELETE FROM pending_jobs WHERE pending_key = ?", (pending_key,))


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
    if not isinstance(dataset, dict):
        return None

    return dataset.get("id") or dataset.get("dataset_id")


def datasets_from_response_payload(payload):
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    for key in ("datasets", "data", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    return []


def dataset_status_value(record):
    if not isinstance(record, dict):
        return None

    for key in ("status", "state", "processing_status", "ingestion_status"):
        value = record.get(key)
        if isinstance(value, str):
            return value.lower()

    return None


def dataset_version_records(dataset):
    if not isinstance(dataset, dict):
        return []

    versions = []
    for key in ("versions", "dataset_versions"):
        value = dataset.get(key)
        if isinstance(value, list):
            versions.extend(value)

    for key in ("version", "latest_version", "ready_version", "current_version"):
        value = dataset.get(key)
        if isinstance(value, dict):
            versions.append(value)

    return versions


def dataset_readiness(dataset):
    if not isinstance(dataset, dict):
        return "unknown"

    records = [dataset, *dataset_version_records(dataset)]
    statuses = [status for status in (dataset_status_value(record) for record in records) if status]

    if any(status in READY_STATUSES for status in statuses):
        return "ready"

    if any(status in FAILED_STATUSES for status in statuses):
        return "failed"

    if any(status in PENDING_STATUSES for status in statuses):
        return "pending"

    if dataset.get("ready_version_id") or dataset.get("current_version_id"):
        return "ready"

    return "unknown"


def fetch_dataset(dataset_id, dataset_name=None):
    response = requests.get(
        f"{base_url}/datasets/{dataset_id}",
        headers=headers,
    )
    if response.status_code == 200:
        payload = response.json()
        if isinstance(payload, dict) and isinstance(payload.get("dataset"), dict):
            return payload["dataset"]
        return payload

    if response.status_code not in (404, 405):
        st.write(response.status_code)
        st.write(response.text)
        response.raise_for_status()

    if dataset_name:
        return find_dataset_by_name(dataset_name)

    response = requests.get(
        f"{base_url}/datasets",
        headers=headers,
    )
    response.raise_for_status()
    for dataset in datasets_from_response_payload(response.json()):
        if dataset_id_from_dataset(dataset) == dataset_id:
            return dataset

    return None


def wait_for_dataset_ready(
    dataset_id,
    dataset_name=None,
    job_id=None,
    timeout_seconds=None,
    allow_unknown_ready=True,
):
    if job_id:
        job_status = wait_for_job_succeeded(job_id, "dataset processing", timeout_seconds)
        if job_status is not True:
            return job_status

    started_at = time.monotonic()
    status_placeholder = st.empty()
    timeout_seconds = timeout_seconds or current_model_wait_timeout_seconds()

    while True:
        dataset = fetch_dataset(dataset_id, dataset_name)
        readiness = dataset_readiness(dataset)
        elapsed_seconds = int(time.monotonic() - started_at)

        if readiness == "ready":
            status_placeholder.write(
                f"Dataset status: ready ({elapsed_seconds // 60}m {elapsed_seconds % 60}s)"
            )
            return True

        if readiness == "failed":
            st.write(dataset)
            return False

        if readiness == "unknown" and job_id:
            status_placeholder.write(
                f"Dataset status: processed ({elapsed_seconds // 60}m {elapsed_seconds % 60}s)"
            )
            return True

        if readiness == "unknown" and not dataset:
            st.write("Dataset could not be found after upload.")
            return False

        status_placeholder.write(
            f"Dataset status: {readiness} ({elapsed_seconds // 60}m {elapsed_seconds % 60}s)"
        )

        if readiness == "unknown" and allow_unknown_ready:
            return True

        if elapsed_seconds >= timeout_seconds:
            st.write("Stopped waiting locally while dataset processing is still pending.")
            st.write(dataset)
            return None

        time.sleep(MODEL_POLL_INTERVAL_SECONDS)


def get_or_create_dataset(uploaded_file, dataset_name):
    uploaded_file.seek(0)
    return get_or_create_dataset_from_bytes(
        uploaded_file.getvalue(),
        uploaded_file.name,
        dataset_name,
    )


def get_or_create_dataset_from_bytes(file_bytes, filename, dataset_name):
    cached_dataset_id = cache_get_dataset_id(dataset_name)
    if cached_dataset_id:
        st.write(f"Reusing cached dataset: {dataset_name}")
        wait_status = wait_for_dataset_ready(cached_dataset_id, dataset_name)
        if wait_status is True:
            return cached_dataset_id
        if wait_status is False:
            st.error(f"Dataset processing failed for {dataset_name}.")
            st.stop()
        st.warning("Dataset processing is still running. Refresh or rerun the app later to resume.")
        st.stop()

    existing_dataset = find_dataset_by_name(dataset_name)
    if existing_dataset:
        st.write(f"Reusing existing dataset: {dataset_name}")
        dataset_id = dataset_id_from_dataset(existing_dataset)
        wait_status = wait_for_dataset_ready(dataset_id, dataset_name)
        if wait_status is True:
            cache_save_dataset(dataset_name, dataset_id)
            return dataset_id
        if wait_status is False:
            st.error(f"Dataset processing failed for {dataset_name}.")
            st.stop()
        st.warning("Dataset processing is still running. Refresh or rerun the app later to resume.")
        st.stop()

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
                dataset_id = dataset_id_from_dataset(existing_dataset)
                wait_status = wait_for_dataset_ready(dataset_id, dataset_name)
                if wait_status is True:
                    cache_save_dataset(dataset_name, dataset_id)
                    return dataset_id
                if wait_status is False:
                    st.error(f"Dataset processing failed for {dataset_name}.")
                    st.stop()
                st.warning("Dataset processing is still running. Refresh or rerun the app later to resume.")
                st.stop()

        st.write(response.status_code)
        st.write(response.text)
        response.raise_for_status()

    payload = response.json()
    dataset_id = dataset_id_from_dataset(payload.get("dataset", {})) or dataset_id_from_dataset(payload)
    if not dataset_id:
        st.write(payload)
        raise ValueError("Dataset creation response did not include a dataset id.")

    wait_status = wait_for_dataset_ready(dataset_id, dataset_name, payload.get("job_id"))
    if wait_status is True:
        cache_save_dataset(dataset_name, dataset_id)
        return dataset_id
    if wait_status is False:
        st.error(f"Dataset processing failed for {dataset_name}.")
        st.stop()
    st.warning("Dataset processing is still running. Refresh or rerun the app later to resume.")
    st.stop()


def current_model_wait_timeout_seconds():
    return st.session_state.get("model_wait_timeout_seconds", MODEL_TRAIN_TIMEOUT_SECONDS)


def wait_for_model_ready(model_id, timeout_seconds=None):
    started_at = time.monotonic()
    status_placeholder = st.empty()
    timeout_seconds = timeout_seconds or current_model_wait_timeout_seconds()

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

        if elapsed_seconds >= timeout_seconds:
            st.write("Stopped waiting locally while model training is still pending.")
            st.write(model_response)
            return None

        time.sleep(MODEL_POLL_INTERVAL_SECONDS)


def model_is_ready(model_id):
    response = requests.get(
        f"{base_url}/models/{model_id}",
        headers=headers,
    )

    if response.status_code != 200:
        return False

    status = response.json().get("status")
    return status in READY_STATUSES


def model_status(model_id):
    response = requests.get(
        f"{base_url}/models/{model_id}",
        headers=headers,
    )

    if response.status_code != 200:
        return None

    return response.json().get("status")


def wait_for_training_complete(model_id, job_id):
    if job_id:
        job_status = wait_for_job_succeeded(job_id, "model training")
        if job_status is not True:
            return job_status

    return wait_for_model_ready(model_id)


def train_model(model_name, model_type, dataset_id, label_column=None, input_columns=None):
    payload = {
        "model_name": model_name,
        "model_type": model_type,
        "dataset_id": dataset_id,
    }
    if label_column:
        payload["label_column"] = label_column
    if input_columns:
        payload["input_columns"] = input_columns

    response = requests.post(
        f"{base_url}/models/train",
        headers=headers,
        json=payload,
    )

    if response.status_code == 409 and NO_READY_DATASET_VERSION_TEXT not in response.text:
        unique_model_name = f"{model_name}_{uuid.uuid4().hex[:8]}"
        st.warning(f"Model name already exists. Training as {unique_model_name}.")
        payload["model_name"] = unique_model_name
        response = requests.post(
            f"{base_url}/models/train",
            headers=headers,
            json=payload,
        )

    if response.status_code == 409 and NO_READY_DATASET_VERSION_TEXT in response.text:
        st.info("Dataset upload is still being prepared for training. Waiting for a ready dataset version.")
        wait_status = wait_for_dataset_ready(dataset_id, allow_unknown_ready=False)
        if wait_status is True:
            response = requests.post(
                f"{base_url}/models/train",
                headers=headers,
                json=payload,
            )
        elif wait_status is False:
            st.error("Dataset processing failed before model training could start.")
            st.stop()
        else:
            st.warning("Dataset processing is still running. Refresh or rerun the app later to resume.")
            st.stop()

    if response.status_code not in SUCCESS_STATUS_CODES:
        st.write(response.status_code)
        st.write(response.text)
        response.raise_for_status()

    payload = response.json()
    return payload["model"]["id"], payload.get("job_id")


def get_or_start_model_training(
    ready_models,
    ready_key,
    pending_key,
    training_label,
    ready_message,
    start_training,
):
    if ready_key in ready_models:
        model_id = ready_models[ready_key]
        if model_is_ready(model_id):
            st.success(f"Reusing trained {training_label} model.")
            cache_save_ready_model(ready_key, model_id)
            return model_id

        ready_models.pop(ready_key, None)
        cache_delete_ready_model(ready_key)
        st.warning(f"Cached {training_label} model is not ready anymore. Starting or resuming training.")

    pending_training = st.session_state.pending_model_jobs.get(pending_key)
    if pending_training:
        model_id = pending_training["model_id"]
        training_job_id = pending_training.get("job_id")
        status = model_status(model_id)
        if status in READY_STATUSES:
            st.success(f"Reusing trained {training_label} model.")
            ready_models[ready_key] = model_id
            st.session_state.pending_model_jobs.pop(pending_key, None)
            cache_save_ready_model(ready_key, model_id)
            cache_delete_pending_job(pending_key)
            return model_id

        if status in FAILED_STATUSES:
            st.session_state.pending_model_jobs.pop(pending_key, None)
            cache_delete_pending_job(pending_key)
            st.warning(f"Cached {training_label} model failed. Starting a fresh training job.")
            model_id, training_job_id = start_training()
            st.session_state.pending_model_jobs[pending_key] = {
                "model_id": model_id,
                "job_id": training_job_id,
            }
            cache_save_pending_job(pending_key, model_id, training_job_id)

        st.info(f"Resuming wait for existing {training_label} training job.")
        training_status = wait_for_training_complete(model_id, training_job_id)
        if training_status is True:
            st.success(ready_message)
            ready_models[ready_key] = model_id
            st.session_state.pending_model_jobs.pop(pending_key, None)
            cache_save_ready_model(ready_key, model_id)
            cache_delete_pending_job(pending_key)
            return model_id

        if training_status is False:
            st.session_state.pending_model_jobs.pop(pending_key, None)
            cache_delete_pending_job(pending_key)
            st.warning(f"Previous {training_label} training job failed. Starting a fresh training job.")
            model_id, training_job_id = start_training()
            st.session_state.pending_model_jobs[pending_key] = {
                "model_id": model_id,
                "job_id": training_job_id,
            }
            cache_save_pending_job(pending_key, model_id, training_job_id)
        else:
            st.warning(
                f"{training_label.title()} model training is still queued or running. "
                "This is not a failure; refresh or rerun the app later to resume polling the same job."
            )
            st.stop()
    else:
        model_id, training_job_id = start_training()
        st.session_state.pending_model_jobs[pending_key] = {
            "model_id": model_id,
            "job_id": training_job_id,
        }
        cache_save_pending_job(pending_key, model_id, training_job_id)

    training_status = wait_for_training_complete(model_id, training_job_id)
    if training_status is True:
        st.success(ready_message)
        ready_models[ready_key] = model_id
        st.session_state.pending_model_jobs.pop(pending_key, None)
        cache_save_ready_model(ready_key, model_id)
        cache_delete_pending_job(pending_key)
        return model_id

    if training_status is False:
        st.session_state.pending_model_jobs.pop(pending_key, None)
        cache_delete_pending_job(pending_key)
        st.error(f"{training_label.title()} model training failed.")
        st.stop()

    st.warning(
        f"{training_label.title()} model training is still queued or running. "
        "This is not a failure; refresh or rerun the app later to resume polling the same job."
    )
    st.stop()


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


def dataframe_from_column_oriented_data(data):
    if isinstance(data, dict):
        return pd.DataFrame(data)
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, str):
        return pd.read_csv(StringIO(data))
    return pd.DataFrame()


def read_inference_json(response):
    payload = response.json()
    return dataframe_from_column_oriented_data(payload.get("data", payload)), payload


def read_uploaded_csv(uploaded_file):
    uploaded_file.seek(0)
    dataframe = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)
    return dataframe


def normalized_column_name(column):
    return re.sub(r"[^a-z0-9]+", "", str(column).lower())


def is_binary_target_series(series):
    values = series.dropna()
    if values.empty:
        return False

    normalized_values = {
        re.sub(r"[^a-z0-9]+", "", str(value).lower())
        for value in values.unique()
    }
    if len(normalized_values) > 2:
        return False

    binary_values = {
        "0",
        "1",
        "false",
        "true",
        "f",
        "t",
        "no",
        "yes",
        "n",
        "y",
        "notchurn",
        "churn",
        "notchurned",
        "churned",
        "retained",
        "lost",
    }
    return normalized_values.issubset(binary_values)


def detect_churn_label_column(dataframe):
    exact_matches = {
        "churn",
        "churned",
        "ischurn",
        "ischurned",
        "haschurn",
        "haschurned",
        "customerchurn",
        "customerchurned",
        "churnlabel",
        "churntarget",
    }

    fallback_matches = []
    for column in dataframe.columns:
        normalized_column = normalized_column_name(column)
        if "churn" not in normalized_column:
            continue

        if normalized_column in exact_matches and is_binary_target_series(dataframe[column]):
            return column

        if is_binary_target_series(dataframe[column]):
            fallback_matches.append(column)

    if len(fallback_matches) == 1:
        return fallback_matches[0]

    return None


def churn_label_column_options(dataframe):
    likely_columns = []
    other_columns = []

    for column in dataframe.columns:
        normalized_column = normalized_column_name(column)
        if "churn" in normalized_column or is_binary_target_series(dataframe[column]):
            likely_columns.append(column)
        else:
            other_columns.append(column)

    return likely_columns + other_columns


def get_churn_label_column(dataframe, key):
    detected_column = detect_churn_label_column(dataframe)
    if detected_column:
        return detected_column

    st.warning("I could not confidently identify the churn target column in the training data.")
    options = churn_label_column_options(dataframe)
    if not options:
        st.error("Training data does not contain any columns.")
        st.stop()

    placeholder = "Select a column"
    selected_column = st.selectbox(
        "Which column identifies churn?",
        [placeholder, *options],
        key=key,
        help="Choose the target column the churn model should learn to predict.",
    )
    if selected_column == placeholder:
        st.stop()

    if selected_column and not is_binary_target_series(dataframe[selected_column]):
        st.info(
            f'"{selected_column}" does not look like a binary churn column. '
            "Use it only if this is the target your model should predict."
        )

    return selected_column


def prediction_input_columns_for_training(dataframe, label_column):
    label_normalized = normalized_column_name(label_column)

    input_columns = []
    for column in dataframe.columns:
        normalized_column = normalized_column_name(column)
        if normalized_column == label_normalized or normalized_column in ("id", "customerid", "customer_id", "accountid", "account_id"):
            continue

        if "churn" in normalized_column and is_binary_target_series(dataframe[column]):
            continue

        if dataframe[column].dropna().empty:
            continue

        input_columns.append(column)

    if not input_columns:
        st.error("Training data must include at least one non-label feature column.")
        st.stop()

    return input_columns


def add_missing_columns(left, right):
    missing_columns = [column for column in right.columns if column not in left.columns]
    if not missing_columns:
        return left

    return pd.concat(
        [
            left.reset_index(drop=True),
            right[missing_columns].reset_index(drop=True),
        ],
        axis=1,
    )


def count_preferred_columns(dataframe):
    return sum(
        column in dataframe.columns and dataframe[column].notna().any()
        for column in PREFERRED_ANALYSIS_COLUMNS
    )


def add_original_customer_features(scored_customers, source_customers):
    scored_customers = scored_customers.copy()
    source_customers = source_customers.copy().reset_index(drop=True)

    if "CustomerID" in scored_customers.columns and "CustomerID" in source_customers.columns:
        merged = scored_customers.merge(
            source_customers,
            on="CustomerID",
            how="left",
            suffixes=("", "_source"),
        )
        if count_preferred_columns(merged) > count_preferred_columns(scored_customers):
            return merged

    if "id" in scored_customers.columns:
        source_by_row_id = source_customers.copy()
        source_by_row_id["_source_row_id"] = source_by_row_id.index.map(normalized_merge_id)
        scored_by_row_id = scored_customers.copy()
        scored_by_row_id["_source_row_id"] = scored_by_row_id["id"].map(normalized_merge_id)
        merged = scored_by_row_id.merge(
            source_by_row_id,
            on="_source_row_id",
            how="left",
            suffixes=("", "_source"),
        ).drop(columns=["_source_row_id"])
        if count_preferred_columns(merged) > count_preferred_columns(scored_customers):
            return merged

    if len(scored_customers) == len(source_customers):
        return add_missing_columns(scored_customers, source_customers)

    return scored_customers


def add_prediction_descriptions(scored_customers, inference_payload):
    prediction_descriptions = inference_payload.get("prediction_descriptions")
    if not prediction_descriptions and inference_payload.get("descriptions"):
        try:
            descriptions = inference_payload["descriptions"]
            if isinstance(descriptions, str):
                descriptions = json.loads(descriptions)
            prediction_descriptions = descriptions.get("prediction_descriptions")
        except (AttributeError, json.JSONDecodeError):
            prediction_descriptions = None

    if isinstance(prediction_descriptions, str):
        try:
            prediction_descriptions = json.loads(prediction_descriptions)
        except json.JSONDecodeError:
            prediction_descriptions = None

    if isinstance(prediction_descriptions, list):
        prediction_descriptions = {
            item.get("label", item.get("prediction")): item.get("description", item.get("explanation", ""))
            for item in prediction_descriptions
            if isinstance(item, dict)
        }

    if not isinstance(prediction_descriptions, dict):
        prediction_descriptions = None

    if not prediction_descriptions or "prediction" not in scored_customers.columns:
        return scored_customers

    descriptions_by_label = {
        str(label): description
        for label, description in prediction_descriptions.items()
    }
    scored_customers = scored_customers.copy()
    scored_customers["prediction_class_explanation"] = scored_customers["prediction"].map(
        lambda prediction: descriptions_by_label.get(str(prediction), "")
    )
    return scored_customers


def wait_for_job_succeeded(job_id, label, timeout_seconds=None):
    started_at = time.monotonic()
    status_placeholder = st.empty()
    timeout_seconds = timeout_seconds or current_model_wait_timeout_seconds()

    while True:
        job_response = requests.get(
            f"{base_url}/jobs/{job_id}",
            headers=headers,
        )

        if job_response.status_code != 200:
            st.write(job_response.status_code)
            st.write(job_response.text)
            job_response.raise_for_status()

        job_payload = job_response.json()
        status = job_payload.get("status")
        elapsed_seconds = int(time.monotonic() - started_at)
        status_placeholder.write(
            f"{label} status: {status or 'unknown'} ({elapsed_seconds // 60}m {elapsed_seconds % 60}s)"
        )

        if status in ("succeeded", "ready", "completed", "success"):
            return True

        if status in ("failed", "error"):
            st.write(job_payload)
            return False

        if elapsed_seconds >= timeout_seconds:
            st.write(f"Stopped waiting locally while {label} is still pending.")
            st.write(job_payload)
            return None

        time.sleep(MODEL_POLL_INTERVAL_SECONDS)


def fetch_job_results(job_id):
    response = requests.get(
        f"{base_url}/jobs/{job_id}/results",
        headers=headers,
    )

    if response.status_code != 200:
        st.write(response.status_code)
        st.write(response.text)
        response.raise_for_status()

    return response.json()


def download_result_artifact(results_payload):
    artifact_url = (
        results_payload.get("inference_results_uri")
        or results_payload.get("combined_results_uri")
        or results_payload.get("results_uri")
    )

    if not artifact_url:
        return None

    response = requests.get(artifact_url)
    response.raise_for_status()
    return response


def dataframe_from_result_artifact(results_payload):
    artifact_response = download_result_artifact(results_payload)
    if artifact_response is None:
        data = results_payload.get("data") or results_payload.get("rows") or results_payload.get("results")
        return dataframe_from_column_oriented_data(data)

    content_type = artifact_response.headers.get("content-type", "")
    if "application/json" in content_type or artifact_response.text.strip().startswith(("{", "[")):
        payload = artifact_response.json()
        return dataframe_from_column_oriented_data(payload.get("data", payload.get("rows", payload)))

    return pd.read_csv(StringIO(artifact_response.text))


def run_prediction_inference(model_id, uploaded_file):
    uploaded_file.seek(0)
    response = requests.post(
        f"{base_url}/models/{model_id}/infer-async",
        headers=headers,
        files={"file": ("test.csv", uploaded_file, "text/csv")},
        data={"output_type": "json"},
    )

    if response.status_code == 404:
        uploaded_file.seek(0)
        response = requests.post(
            f"{base_url}/models/{model_id}/infer",
            headers=headers,
            files={"file": ("test.csv", uploaded_file, "text/csv")},
            data={"output_type": "json"},
        )

        if response.status_code not in SUCCESS_STATUS_CODES:
            st.write(response.status_code)
            st.write(response.text)
            response.raise_for_status()

        scored_customers, inference_payload = read_inference_json(response)
        inference_job_id = (
            inference_payload.get("job_id")
            or inference_payload.get("inference_job_id")
            or inference_payload.get("id")
        )
        return scored_customers, inference_payload, inference_job_id

    if response.status_code not in SUCCESS_STATUS_CODES:
        st.write(response.status_code)
        st.write(response.text)
        response.raise_for_status()

    inference_job_id = response.json()["job_id"]
    if not wait_for_job_succeeded(inference_job_id, "prediction inference"):
        st.error("Prediction inference failed or timed out.")
        st.stop()

    inference_payload = fetch_job_results(inference_job_id)
    scored_customers = dataframe_from_result_artifact(inference_payload)
    return scored_customers, inference_payload, inference_job_id


def normalize_explanation_columns(explanations):
    rename_map = {}
    for column in explanations.columns:
        lower_column = column.lower()
        if lower_column in ("row_id", "row_ids", "input_id", "source_id", "index", "row_index"):
            rename_map[column] = "id"
        elif lower_column in ("explanation", "description", "prediction_explanation"):
            rename_map[column] = "row_prediction_explanation"
        elif lower_column in ("summary", "explanation_summary"):
            rename_map[column] = "row_explanation_summary"
    return explanations.rename(columns=rename_map)


def request_row_level_explanation_batch(inference_job_id, row_ids, output_type="json"):
    response = requests.post(
        f"{base_url}/jobs/{inference_job_id}/explain",
        headers=headers,
        json={
            "ids": row_ids,
            "output_type": output_type,
        },
    )

    if response.status_code not in SUCCESS_STATUS_CODES:
        if response.status_code == 404:
            st.warning(
                "Row-level explanations are not available for this inference job/API environment, so the app will continue without them."
            )
            return None

        st.write(response.status_code)
        st.write(response.text)
        response.raise_for_status()

    explanation_job_id = response.json()["job_id"]
    if not wait_for_job_succeeded(explanation_job_id, "row explanation"):
        st.warning("Row-level explanation job failed or timed out.")
        return pd.DataFrame()

    results_payload = fetch_job_results(explanation_job_id)
    explanations = dataframe_from_result_artifact(results_payload)
    if explanations.empty:
        return explanations

    return normalize_explanation_columns(explanations)


def get_row_level_explanations(inference_job_id, row_ids, output_type="json"):
    if not inference_job_id or not row_ids:
        return pd.DataFrame()

    explanations = request_row_level_explanation_batch(inference_job_id, row_ids, output_type)
    if explanations is None or explanations.empty or "id" not in explanations.columns:
        return pd.DataFrame() if explanations is None else explanations

    requested_ids = {normalized_merge_id(row_id) for row_id in row_ids}
    returned_ids = {normalized_merge_id(row_id) for row_id in explanations["id"].dropna().tolist()}
    missing_ids = [
        row_id
        for row_id in row_ids
        if normalized_merge_id(row_id) not in returned_ids
    ]

    if not missing_ids:
        return explanations

    st.info(
        f"Explanation batch returned {len(returned_ids)} of {len(requested_ids)} rows. "
        "Requesting missing rows individually."
    )
    explanation_frames = [explanations]
    for row_id in missing_ids:
        row_explanation = request_row_level_explanation_batch(
            inference_job_id,
            [row_id],
            output_type,
        )
        if row_explanation is not None and not row_explanation.empty:
            explanation_frames.append(row_explanation)

    combined_explanations = pd.concat(explanation_frames, ignore_index=True)
    if "id" not in combined_explanations.columns:
        return combined_explanations

    combined_explanations["_normalized_id"] = combined_explanations["id"].map(normalized_merge_id)
    return (
        combined_explanations
        .drop_duplicates("_normalized_id", keep="last")
        .drop(columns=["_normalized_id"])
    )


def normalized_merge_id(value):
    numeric_value = as_number(value)
    if numeric_value is not None and numeric_value.is_integer():
        return str(int(numeric_value))
    return str(value).strip()


def explanation_value_columns(explanations):
    return [
        column
        for column in explanations.columns
        if column != "id"
        and (
            "explanation" in column.lower()
            or "summary" in column.lower()
            or column == "prediction_class_explanation"
        )
    ]


def add_row_level_explanations(results, explanations):
    if explanations.empty:
        return results

    value_columns = explanation_value_columns(explanations)
    if not value_columns:
        return results

    if "id" not in results.columns or "id" not in explanations.columns:
        return results

    results_for_merge = results.copy()
    explanations_for_merge = explanations[["id", *value_columns]].copy()

    for column in value_columns:
        if column in results_for_merge.columns:
            results_for_merge = results_for_merge.drop(columns=[column])

    results_for_merge["_explanation_merge_id"] = results_for_merge["id"].map(normalized_merge_id)
    explanations_for_merge["_explanation_merge_id"] = explanations_for_merge["id"].map(normalized_merge_id)
    explanations_for_merge = explanations_for_merge.drop(columns=["id"])

    return (
        results_for_merge
        .merge(explanations_for_merge, on="_explanation_merge_id", how="left")
        .drop(columns=["_explanation_merge_id"])
    )


PREFERRED_ANALYSIS_COLUMNS = [
    "AccountAge",
    "MonthlyCharges",
    "TotalCharges",
    "SubscriptionType",
    "PaymentMethod",
    "PaperlessBilling",
    "ContentType",
    "MultiDeviceAccess",
    "DeviceRegistered",
    "ViewingHoursPerWeek",
    "AverageViewingDuration",
    "ContentDownloadsPerMonth",
    "GenrePreference",
    "UserRating",
    "SupportTicketsPerMonth",
    "Gender",
    "WatchlistSize",
    "ParentalControl",
    "SubtitlesEnabled",
]


EXCLUDED_ANALYSIS_COLUMNS = {
    "id",
    "customerid",
    "churn",
    "prediction",
    "prediction_prob",
    "prediction_probability",
    "probability",
    "confidence",
    "prediction_confidence",
    "score",
    "prediction_class_explanation",
    "row_prediction_explanation",
    "row_explanation_summary",
}


def is_useful_analysis_column(column, series):
    lowercase_column = str(column).lower()
    normalized_column = normalized_column_name(column)
    normalized_excluded_columns = {
        normalized_column_name(excluded_column)
        for excluded_column in EXCLUDED_ANALYSIS_COLUMNS
    }
    if lowercase_column in EXCLUDED_ANALYSIS_COLUMNS or normalized_column in normalized_excluded_columns:
        return False
    if "churn" in normalized_column and is_binary_target_series(series):
        return False
    if lowercase_column.startswith(("factor_", "cluster_", "intervention_")):
        return False
    if normalized_column.startswith(("factor", "cluster", "intervention")):
        return False

    non_null = series.dropna()
    if non_null.empty or non_null.nunique(dropna=True) <= 1:
        return False

    if pd.api.types.is_numeric_dtype(non_null) or pd.api.types.is_bool_dtype(non_null):
        return True

    text_values = non_null.astype(str)
    unique_count = text_values.nunique(dropna=True)
    unique_ratio = unique_count / max(len(text_values), 1)
    average_length = text_values.str.len().mean()
    return unique_count <= 50 and unique_ratio <= 0.8 and average_length <= 80


def analysis_dataframe_for_modeling(at_risk):
    preferred_columns = [
        column
        for column in PREFERRED_ANALYSIS_COLUMNS
        if column in at_risk.columns and is_useful_analysis_column(column, at_risk[column])
    ]
    if preferred_columns:
        return at_risk[preferred_columns].copy(), preferred_columns

    selected_columns = [
        column
        for column in at_risk.columns
        if is_useful_analysis_column(column, at_risk[column])
    ]
    return at_risk[selected_columns].copy(), selected_columns


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


def sort_by_probability_desc(dataframe):
    probability_column = probability_sort_column(dataframe)
    if not probability_column:
        return dataframe

    sort_values = pd.to_numeric(dataframe[probability_column], errors="coerce")
    if not sort_values.notna().any():
        return dataframe

    return (
        dataframe
        .assign(_risk_probability_sort=sort_values)
        .sort_values("_risk_probability_sort", ascending=False)
        .drop(columns=["_risk_probability_sort"])
    )


def filter_at_risk_customers(scored_customers, churn_confidence_threshold):
    at_risk_customers = scored_customers.copy()

    if "prediction" in at_risk_customers.columns:
        at_risk_customers = at_risk_customers[at_risk_customers["prediction"] == 1]

    probabilities = probability_series(at_risk_customers)
    if probabilities is None:
        st.warning("No probability column was found, so at-risk filtering used only the prediction label.")
        return at_risk_customers

    return sort_by_probability_desc(at_risk_customers[probabilities >= churn_confidence_threshold])


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

with st.sidebar:
    st.header("Inputs")
    if api_key:
        st.success("Wood Wide API key loaded")
    else:
        st.error("Missing WOODWIDE_API_KEY")
    st.caption(f"API: {base_url}")

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
    if os.path.exists(DEFAULT_TRAIN_DATASET_PATH):
        with open(DEFAULT_TRAIN_DATASET_PATH, "rb") as file:
            st.download_button(
                "Download default train.csv",
                file,
                "train.csv",
                "text/csv",
                use_container_width=True,
            )
    else:
        st.caption("Default train.csv was not found.")

    if os.path.exists(DEFAULT_TEST_DATASET_PATH):
        with open(DEFAULT_TEST_DATASET_PATH, "rb") as file:
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
    max_row_explanations = st.slider(
        "Row explanations",
        min_value=0,
        max_value=100,
        value=25,
        step=5,
        help="Generate customer-specific prediction explanations for the highest-risk rows. Set to 0 to skip.",
    )
    model_wait_timeout_minutes = st.slider(
        "Model wait timeout",
        min_value=10,
        max_value=180,
        value=min(180, max(10, st.session_state.model_wait_timeout_seconds // 60)),
        step=10,
        help="How long this Streamlit run should wait for Woodwide training jobs before pausing. Pending jobs are saved and resumed on rerun.",
    )
    st.session_state.model_wait_timeout_seconds = model_wait_timeout_minutes * 60
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
    training_dataframe = read_uploaded_csv(training_data)
    dataset_name = dataset_name_for_upload("historical_customers", training_data)
    churn_label_column = get_churn_label_column(
        training_dataframe,
        f"churn_label_column:{dataset_name}",
    )
    churn_input_columns = prediction_input_columns_for_training(training_dataframe, churn_label_column)
    with st.spinner("uploading dataset", show_time=True):
        dataset_id = get_or_create_dataset(training_data, dataset_name)

    with st.spinner("training churn model", show_time=True):
        model_id = get_or_start_model_training(
            st.session_state.model_ids,
            dataset_name,
            f"churn:{dataset_name}",
            "churn",
            "Churn model is ready.",
            lambda: train_model(
                dataset_name.replace("historical_customers", "churn_prediction", 1),
                "prediction",
                dataset_id,
                label_column=churn_label_column,
                input_columns=churn_input_columns,
            ),
        )

at_risk = None
prediction_explanations = pd.DataFrame()

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
            active_customers = read_uploaded_csv(test_data)
            scored_customers, inference_payload, inference_job_id = run_prediction_inference(model_id, test_data)
            scored_customers = add_original_customer_features(scored_customers, active_customers)
            scored_customers = add_prediction_descriptions(scored_customers, inference_payload)
            at_risk = scored_customers
            at_risk = filter_at_risk_customers(scored_customers, churn_confidence_threshold)

            if max_row_explanations and inference_job_id and "id" in at_risk.columns:
                probability_column = probability_sort_column(at_risk)
                explanation_candidates = at_risk.copy()
                if probability_column:
                    explanation_candidates = explanation_candidates.assign(
                        _explain_sort=pd.to_numeric(
                            explanation_candidates[probability_column],
                            errors="coerce",
                        )
                    ).sort_values("_explain_sort", ascending=False)

                row_ids = []
                for row_id in explanation_candidates["id"].head(max_row_explanations).dropna().tolist():
                    row_id_number = as_number(row_id)
                    row_ids.append(int(row_id_number) if row_id_number is not None else str(row_id))

                with st.spinner("explaining highest-risk rows...", show_time=True):
                    prediction_explanations = get_row_level_explanations(inference_job_id, row_ids)
                    at_risk = add_row_level_explanations(at_risk, prediction_explanations)
            elif max_row_explanations and not inference_job_id:
                st.warning("Inference response did not include a job ID, so row-level explanations were skipped.")

if at_risk is not None:
    total_customers = None
    try:
        test_data.seek(0)
        total_customers = len(pd.read_csv(test_data, usecols=["CustomerID"]))
    except (ValueError, pd.errors.EmptyDataError):
        total_customers = None

    risk_metric_1, risk_metric_2, risk_metric_3, risk_metric_4, risk_metric_5 = st.columns(5)
    risk_metric_1.metric("At-risk customers", f"{len(at_risk):,}")
    risk_metric_2.metric("Previewing", f"{min(len(at_risk), preview_row_count):,}")
    risk_metric_3.metric("Active customers", f"{total_customers:,}" if total_customers is not None else "Uploaded")
    risk_metric_4.metric("Threshold", f"{churn_confidence_threshold:.0%}")
    risk_metric_5.metric("Explained rows", f"{len(prediction_explanations):,}")

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
    st.session_state.risk_ids = cache_load_ready_models_with_prefix("factors:")
else:
    st.session_state.risk_ids = {
        **cache_load_ready_models_with_prefix("factors:"),
        **st.session_state.risk_ids,
    }

if "cluster_ids" not in st.session_state:
    st.session_state.cluster_ids = cache_load_ready_models_with_prefix("clusters:")
else:
    st.session_state.cluster_ids = {
        **cache_load_ready_models_with_prefix("clusters:"),
        **st.session_state.cluster_ids,
    }

if at_risk is not None:
    if at_risk.empty:
        st.info("No at-risk customers found, so factor analysis was skipped.")
    else:
        st.subheader("3. Risk Drivers")
        st.markdown('<div class="section-note">Factor analysis summarizes why customers are at risk.</div>', unsafe_allow_html=True)
        risk_modeling_data, risk_input_columns = analysis_dataframe_for_modeling(at_risk)
        if risk_modeling_data.empty or not risk_input_columns:
            st.error("No usable customer feature columns were available for factor analysis.")
            st.stop()

        st.caption(f"Using {len(risk_input_columns)} customer feature columns for factor analysis and clustering.")
        risk_csv = dataframe_to_csv_bytes(risk_modeling_data)
        risk_dataset_name = dataset_name_for_dataframe(
            "risk_customers",
            risk_modeling_data,
            dataset_name,
        )
        risk_dataset_id = get_or_create_dataset_from_bytes(
            risk_csv,
            "risk_customers.csv",
            risk_dataset_name,
        )

        with st.spinner("training factor analysis model", show_time=True):
            factor_model_id = get_or_start_model_training(
                st.session_state.risk_ids,
                risk_dataset_name,
                f"factors:{risk_dataset_name}",
                "factor analysis",
                "Factor analysis model is ready.",
                lambda: train_model(
                    risk_dataset_name.replace("risk_customers", "factor_analysis", 1),
                    "factors",
                    risk_dataset_id,
                    input_columns=risk_input_columns,
                ),
            )

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
        with st.spinner("training clustering model", show_time=True):
            cluster_model_id = get_or_start_model_training(
                st.session_state.cluster_ids,
                risk_dataset_name,
                f"clusters:{risk_dataset_name}",
                "clustering",
                "Clustering model is ready.",
                lambda: train_model(
                    risk_dataset_name.replace("risk_customers", "customer_segments", 1),
                    "clustering",
                    risk_dataset_id,
                    input_columns=risk_input_columns,
                ),
            )

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
                "row_prediction_explanation",
                "row_explanation_summary",
                "prediction_class_explanation",
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
