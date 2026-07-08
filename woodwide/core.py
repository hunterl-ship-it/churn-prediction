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
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")


def secret_or_env(name, default=None):
    value = os.getenv(name)
    try:
        value = st.secrets.get(name, value)
    except Exception:
        pass

    if isinstance(value, str):
        value = value.strip().strip('"').strip("'")

    return value if value else default


api_key = secret_or_env("WOODWIDE_API_KEY")
base_url = secret_or_env("WOODWIDE_BASE_URL", "https://api.woodwide.ai").rstrip("/")
headers = {"Authorization": f"Bearer {api_key}"}


def report_api_error(message: str, response=None):
    detail = ""
    if response is not None:
        response_text = getattr(response, "text", "") or ""
        if response_text:
            detail = f" ({response_text[:240]})"
    st.error(f"{message}{detail}")


MODEL_POLL_INTERVAL_SECONDS = 5
MODEL_TRAIN_TIMEOUT_SECONDS = 30 * 60
PREVIEW_ROW_COUNT = 1000
SUCCESS_STATUS_CODES = (200, 201, 202)
READY_STATUSES = ("ready", "completed", "succeeded", "success")
FAILED_STATUSES = ("failed", "error")
PENDING_STATUSES = ("pending", "queued", "running", "processing", "in_progress", "started")
NO_READY_DATASET_VERSION_TEXT = "No ready dataset version found"
DEFAULT_CHURN_TRAIN_DATASET_PATH = os.path.join("datasets", "churn", "train.csv")
DEFAULT_CHURN_TEST_DATASET_PATH = os.path.join("datasets", "churn", "test.csv")
DEFAULT_FRAUD_PREDICTION_TRAIN_DATASET_PATH = os.path.join("datasets", "ecommerce", "train.csv")
DEFAULT_FRAUD_SCORING_DATASET_PATH = os.path.join("datasets", "ecommerce", "fraud_test.csv")
DEFAULT_FRAUD_ANOMALY_TRAIN_DATASET_PATH = os.path.join("datasets", "ecommerce", "fraud_train.csv")
DEFAULT_FRAUD_TRAIN_DATASET_PATH = DEFAULT_FRAUD_PREDICTION_TRAIN_DATASET_PATH
DEFAULT_FRAUD_TEST_DATASET_PATH = DEFAULT_FRAUD_SCORING_DATASET_PATH
DEFAULT_RETURNS_TRAIN_DATASET_PATH = os.path.join("datasets", "ecommerce", "train.csv")
DEFAULT_RETURNS_TEST_DATASET_PATH = os.path.join("datasets", "ecommerce", "test.csv")
DEFAULT_NOSHOW_TRAIN_DATASET_PATH = os.path.join("datasets", "healthcare", "train.csv")
DEFAULT_NOSHOW_TEST_DATASET_PATH = os.path.join("datasets", "healthcare", "test.csv")
DEFAULT_CHURN_EVAL_DATASET_PATH = os.path.join("datasets", "churn", "eval.csv")
DEFAULT_NOSHOW_EVAL_DATASET_PATH = os.path.join("datasets", "healthcare", "eval.csv")
PILOT_CTA_URL = os.environ.get("PILOT_CTA_URL", "https://woodwide.ai")
DEFAULT_TRAIN_DATASET_PATH = DEFAULT_CHURN_TRAIN_DATASET_PATH
DEFAULT_TEST_DATASET_PATH = DEFAULT_CHURN_TEST_DATASET_PATH
LOCAL_CACHE_DIR = ".streamlit_cache"
LOCAL_CACHE_DB_PATH = os.path.join(LOCAL_CACHE_DIR, "woodwide_jobs.sqlite3")

GENERIC_INTERVENTION_CATALOG = [
    {
        "name": "High-value retention save",
        "description": "high value account long tenure important customer severe churn risk renewal retention save",
        "keywords": ["value", "revenue", "arr", "mrr", "charges", "tenure", "account_age", "CustomerID", "customer_id"],
        "low": "Send a personalized check-in that reinforces the value the customer has already received.",
        "medium": "Offer a retention review with a tailored next-best action or account adjustment.",
        "high": "Route to a high-touch save motion with an owner, deadline, and tailored retention offer.",
    },
    {
        "name": "Pricing and plan fit review",
        "description": "price sensitivity expensive plan monthly charge cost downgrade budget concern contract renewal",
        "keywords": ["price", "plan", "monthly", "charge", "cost", "contract", "subscription", "tier"],
        "low": "Send plan-fit guidance that highlights the best-value option for current usage.",
        "medium": "Offer a pricing or package review with a lower-friction plan alternative.",
        "high": "Escalate a save offer with a time-boxed discount, plan adjustment, or renewal incentive.",
    },
    {
        "name": "Billing friction recovery",
        "description": "billing payment invoice collection friction payment method failed charge renewal blocker",
        "keywords": ["billing", "payment", "invoice", "pay", "card", "method", "paperless"],
        "low": "Send a billing reminder with self-service payment update options.",
        "medium": "Offer billing support and confirm whether payment setup is blocking continued use.",
        "high": "Create an urgent billing recovery task with support follow-up and save-offer authority.",
    },
    {
        "name": "Engagement recovery",
        "description": "low usage inactive disengaged low activity low login frequency adoption drop utilization decline",
        "keywords": ["usage", "activity", "login", "engagement", "active", "utilization", "frequency"],
        "low": "Send a personalized re-engagement prompt based on the customer's recent activity pattern.",
        "medium": "Launch a guided activation campaign focused on the most relevant next action.",
        "high": "Assign an owner to run a rapid re-engagement plan with concrete adoption goals.",
    },
    {
        "name": "Onboarding completion",
        "description": "onboarding setup activation incomplete early lifecycle setup progress implementation incomplete",
        "keywords": ["onboarding", "setup", "activation", "implementation", "completed", "started"],
        "low": "Send a setup checklist with the next incomplete milestone highlighted.",
        "medium": "Offer guided onboarding support to complete the most important activation steps.",
        "high": "Schedule an onboarding recovery session and assign a completion owner.",
    },
    {
        "name": "Support recovery",
        "description": "support tickets unresolved issue service problem complaint technical support frustration help",
        "keywords": ["support", "ticket", "issue", "case", "complaint", "problem"],
        "low": "Send a support follow-up with the most likely self-service fix.",
        "medium": "Escalate open or repeated issues and confirm resolution with the customer.",
        "high": "Create an urgent support recovery case with a named owner and executive visibility.",
    },
    {
        "name": "Satisfaction recovery",
        "description": "low satisfaction bad experience unhappy negative nps csat rating sentiment complaint churn risk",
        "keywords": ["nps", "csat", "rating", "satisfaction", "sentiment", "score", "feedback"],
        "low": "Ask for targeted feedback and share one immediate improvement path.",
        "medium": "Run a satisfaction recovery touchpoint with a tailored offer or service action.",
        "high": "Escalate to a retention owner for a direct save conversation and recovery plan.",
    },
    {
        "name": "Feature adoption enablement",
        "description": "low feature adoption missing core capability underused features product value education enablement",
        "keywords": ["feature", "adoption", "module", "capability", "used", "usage_score"],
        "low": "Send a short enablement tip for the most relevant underused feature.",
        "medium": "Invite the customer to a guided feature walkthrough tied to their goals.",
        "high": "Create a focused adoption plan with success milestones and follow-up ownership.",
    },
]


B2B_SAAS_INTERVENTION_CATALOG = [
    {
        "name": "Executive sponsor outreach",
        "description": "enterprise account high arr executive sponsor missing relationship risk strategic account renewal churn",
        "keywords": ["arr", "mrr", "enterprise", "company_size", "executive", "sponsor", "account_age", "seats"],
        "low": "Send a relationship health check that reinforces business outcomes and next steps.",
        "medium": "Ask the account owner to confirm the executive sponsor and renewal priorities.",
        "high": "Schedule an executive sponsor touchpoint with a concrete retention plan and success owner.",
    },
    {
        "name": "Product adoption plan",
        "description": "low product usage low feature adoption low login frequency product usage score adoption risk",
        "keywords": ["product_usage_score", "features_adopted", "login_frequency", "days_since_last_login", "usage", "adoption"],
        "low": "Send a targeted adoption tip for the next feature most likely to create value.",
        "medium": "Offer a product success session focused on usage gaps and team activation.",
        "high": "Assign customer success to run a 30-day adoption recovery plan with measurable goals.",
    },
    {
        "name": "Onboarding recovery",
        "description": "onboarding incomplete implementation not completed setup activation early customer lifecycle risk",
        "keywords": ["onboarding_completed", "onboarding", "implementation", "activation", "setup"],
        "low": "Send the remaining onboarding checklist and highlight the highest-value next step.",
        "medium": "Offer a guided onboarding completion session with a customer success manager.",
        "high": "Escalate to onboarding recovery with a named owner and completion deadline.",
    },
    {
        "name": "Support escalation",
        "description": "many support tickets unresolved issues customer frustration technical problems service recovery",
        "keywords": ["support_tickets", "support", "ticket", "case", "issue", "problem"],
        "low": "Send a support follow-up summarizing recent fixes and self-service resources.",
        "medium": "Escalate recurring tickets and confirm resolution with the account contact.",
        "high": "Open an urgent success/support recovery case with account-owner visibility.",
    },
    {
        "name": "Renewal save motion",
        "description": "contract renewal annual contract month to month renewal risk low retention expansion contraction",
        "keywords": ["contract_type", "renewal", "annual", "monthly", "arr", "expansion_revenue_pct", "seats"],
        "low": "Send a renewal value recap that connects usage to business outcomes.",
        "medium": "Offer a renewal review with plan-fit and adoption recommendations.",
        "high": "Launch a save motion with commercial flexibility and executive/account-owner follow-up.",
    },
    {
        "name": "NPS recovery",
        "description": "low nps low csat poor customer sentiment detractor satisfaction recovery churn risk",
        "keywords": ["nps_score", "csat_avg", "nps", "csat", "satisfaction", "sentiment"],
        "low": "Ask for focused feedback and acknowledge the most likely experience gap.",
        "medium": "Schedule a customer success check-in to address satisfaction drivers.",
        "high": "Escalate detractor recovery to a retention owner with a corrective action plan.",
    },
    {
        "name": "Seat and team expansion review",
        "description": "seat count company size expansion revenue low seat penetration team growth adoption opportunity",
        "keywords": ["seats", "company_size", "expansion_revenue_pct", "arr", "team", "users"],
        "low": "Share team adoption guidance based on current seat usage.",
        "medium": "Offer a seat and workflow review to identify blocked teams or unused value.",
        "high": "Build a joint success plan for team adoption, renewal protection, and expansion potential.",
    },
]


STREAMING_INTERVENTION_CATALOG = [
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

HEALTHCARE_INTERVENTION_CATALOG = [
    {
        "name": "Reminder escalation",
        "description": "long lead time upcoming appointment no reminder missed prior visits scheduling risk",
        "keywords": ["LeadTimeDays", "SMSReminderSent", "CallReminderSent", "PreviousNoShows", "PatientID"],
        "low": "Send a standard appointment reminder with date, time, and location.",
        "medium": "Send a multi-channel reminder (SMS + call) two days before the visit.",
        "high": "Assign staff to call the patient directly and confirm attendance.",
    },
    {
        "name": "Transportation support",
        "description": "transportation barrier distance clinic access ride no car travel hardship missed appointment",
        "keywords": ["TransportationBarrier", "DistanceToClinicMiles", "InsuranceType", "Medicaid"],
        "low": "Include parking and transit directions in the appointment reminder.",
        "medium": "Offer ride-share voucher or clinic shuttle information.",
        "high": "Coordinate non-emergency medical transport and confirm pickup details.",
    },
    {
        "name": "Repeat no-show recovery",
        "description": "previous no shows missed appointments history attendance pattern chronic absenteeism",
        "keywords": ["PreviousNoShows", "PreviousAppointments", "LeadTimeDays"],
        "low": "Send a personalized attendance note highlighting the importance of follow-through.",
        "medium": "Require confirmation reply before the visit and offer flexible rescheduling.",
        "high": "Route to a care coordinator for attendance counseling and barrier review.",
    },
    {
        "name": "New patient onboarding",
        "description": "new patient first visit intake onboarding unfamiliar clinic paperwork anxiety",
        "keywords": ["AppointmentType", "New Patient", "LeadTimeDays", "Specialty"],
        "low": "Send new-patient welcome packet with forms and what to expect.",
        "medium": "Schedule a pre-visit call to answer questions and confirm insurance.",
        "high": "Assign a navigator for first-visit support and same-day check-in assistance.",
    },
    {
        "name": "Access and insurance barrier",
        "description": "medicaid self pay insurance coverage cost barrier underserved access difficulty",
        "keywords": ["InsuranceType", "Medicaid", "Self-Pay", "Specialty"],
        "low": "Confirm insurance eligibility and share financial assistance options.",
        "medium": "Offer a financial counseling call before the appointment.",
        "high": "Escalate to social work for coverage, copay, or access barrier resolution.",
    },
    {
        "name": "Wait time and scheduling friction",
        "description": "long wait time scheduling delay lead time appointment delay frustration access",
        "keywords": ["WaitTimeDays", "LeadTimeDays", "DayOfWeek", "AppointmentHour"],
        "low": "Offer earlier time slots or waitlist options in the reminder.",
        "medium": "Proactively reschedule to a more convenient day or time.",
        "high": "Priority-reschedule with a care coordinator and confirm attendance.",
    },
    {
        "name": "Behavioral health engagement",
        "description": "mental health specialty psychiatry therapy stigma engagement no show behavioral health",
        "keywords": ["Specialty", "Mental Health", "Age", "PreviousNoShows"],
        "low": "Send a supportive reminder emphasizing confidentiality and visit goals.",
        "medium": "Offer telehealth as an alternative to in-person attendance.",
        "high": "Assign a behavioral health navigator for outreach and barrier removal.",
    },
    {
        "name": "Chronic care continuity",
        "description": "chronic conditions follow up care continuity medication management comorbidity",
        "keywords": ["ChronicConditionCount", "AppointmentType", "Follow-up", "PreviousAppointments"],
        "low": "Highlight the clinical importance of keeping the follow-up visit.",
        "medium": "Send condition-specific prep instructions and offer telehealth check-in.",
        "high": "Escalate to the care team for urgent outreach and same-week rescheduling.",
    },
]

INTERVENTION_TEMPLATE_CATALOGS = {
    "generic": GENERIC_INTERVENTION_CATALOG,
    "b2b_saas": B2B_SAAS_INTERVENTION_CATALOG,
    "streaming": STREAMING_INTERVENTION_CATALOG,
    "healthcare": HEALTHCARE_INTERVENTION_CATALOG,
}

INTERVENTION_TEMPLATE_LABELS = {
    "auto": "Auto-detect",
    "generic": "Generic retention",
    "b2b_saas": "B2B SaaS",
    "streaming": "Streaming/media",
    "healthcare": "Healthcare scheduling",
}

DEFAULT_INTERVENTION_TEMPLATE_KEY = "generic"


def copied_intervention_catalog(catalog):
    return [
        {
            **intervention,
            "keywords": list(intervention["keywords"]),
        }
        for intervention in catalog
    ]


def default_intervention_catalog(template_key=DEFAULT_INTERVENTION_TEMPLATE_KEY):
    return copied_intervention_catalog(
        INTERVENTION_TEMPLATE_CATALOGS.get(
            template_key,
            INTERVENTION_TEMPLATE_CATALOGS[DEFAULT_INTERVENTION_TEMPLATE_KEY],
        )
    )


def intervention_template_label(template_key):
    return INTERVENTION_TEMPLATE_LABELS.get(template_key, template_key)


def intervention_template_score(dataframe, vocabulary):
    column_text = " ".join(str(column).lower() for column in dataframe.columns)
    normalized_column_text = " ".join(normalized_column_name(column) for column in dataframe.columns)
    score = 0
    for token in vocabulary:
        normalized_token = normalized_column_name(token)
        if token.lower() in column_text or normalized_token in normalized_column_text:
            score += 1
    return score


def detect_intervention_template(dataframe):
    b2b_saas_score = intervention_template_score(
        dataframe,
        [
            "arr",
            "mrr",
            "seats",
            "contract_type",
            "company_size",
            "industry",
            "nps",
            "csat",
            "support_tickets",
            "onboarding",
            "executive_sponsor",
            "product_usage",
            "features_adopted",
            "login_frequency",
            "expansion_revenue",
        ],
    )
    streaming_score = intervention_template_score(
        dataframe,
        [
            "viewing",
            "watch",
            "genre",
            "content",
            "subscription_type",
            "downloads",
            "device",
            "subtitles",
            "parental",
            "user_rating",
            "watchlist",
            "streaming",
        ],
    )
    healthcare_score = intervention_template_score(
        dataframe,
        [
            "patientid",
            "patient_id",
            "appointment",
            "specialty",
            "insurance",
            "leadtime",
            "waittime",
            "no_show",
            "noshow",
            "reminder",
            "clinic",
            "provider",
            "medicaid",
            "medicare",
        ],
    )

    scores = {
        "b2b_saas": b2b_saas_score,
        "streaming": streaming_score,
        "healthcare": healthcare_score,
    }
    best_template, best_score = max(scores.items(), key=lambda item: item[1])
    second_best_score = sorted(scores.values(), reverse=True)[1]
    if best_score >= 4 and best_score >= second_best_score + 2:
        return best_template

    if b2b_saas_score >= 4 and b2b_saas_score >= streaming_score + 2:
        return "b2b_saas"
    if streaming_score >= 4 and streaming_score >= b2b_saas_score + 2:
        return "streaming"
    return DEFAULT_INTERVENTION_TEMPLATE_KEY


def resolve_intervention_template(template_choice, detected_template):
    if template_choice == "auto":
        return detected_template or DEFAULT_INTERVENTION_TEMPLATE_KEY
    return template_choice or DEFAULT_INTERVENTION_TEMPLATE_KEY


def set_intervention_catalog_template(template_key):
    st.session_state.intervention_catalog = default_intervention_catalog(template_key)
    st.session_state.intervention_catalog_template_key = template_key
    st.session_state.intervention_catalog_editor_version = (
        st.session_state.get("intervention_catalog_editor_version", 0) + 1
    )


def current_intervention_catalog(template_key=None):
    if template_key:
        return default_intervention_catalog(template_key)
    return st.session_state.get("intervention_catalog") or default_intervention_catalog()


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


def find_intervention_by_name(name, template_key=None):
    for intervention in current_intervention_catalog(template_key):
        if intervention["name"] == name:
            return intervention
    return None


def find_first_intervention_by_names(names, template_key=None):
    for name in names:
        intervention = find_intervention_by_name(name, template_key)
        if intervention:
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
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_jobs (
                cache_key TEXT PRIMARY KEY,
                job_id TEXT,
                status TEXT NOT NULL,
                result_payload_json TEXT,
                result_csv TEXT,
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


MODEL_PREDICTION_DESCRIPTIONS_PREFIX = "model_prediction_descriptions:"


def cache_save_model_prediction_descriptions(model_id, prediction_descriptions, training_job_id=None):
    if not model_id or not prediction_descriptions:
        return

    cache_save_inference_job(
        f"{MODEL_PREDICTION_DESCRIPTIONS_PREFIX}{model_id}",
        training_job_id or model_id,
        "succeeded",
        {"prediction_descriptions": prediction_descriptions},
    )


def cache_get_model_prediction_descriptions(model_id):
    if not model_id:
        return None

    cached_job = cache_get_inference_job(f"{MODEL_PREDICTION_DESCRIPTIONS_PREFIX}{model_id}")
    if not cached_job:
        return None

    payload = payload_from_cached_json(cached_job.get("result_payload_json"))
    return normalize_prediction_descriptions_dict(payload.get("prediction_descriptions"))


def cache_get_training_job_id_for_model(model_id):
    if not model_id:
        return None

    cached_descriptions_job = cache_get_inference_job(f"{MODEL_PREDICTION_DESCRIPTIONS_PREFIX}{model_id}")
    if cached_descriptions_job and cached_descriptions_job.get("job_id"):
        return cached_descriptions_job["job_id"]

    with local_cache_connection() as connection:
        row = connection.execute(
            "SELECT job_id FROM pending_jobs WHERE model_id = ? AND job_id IS NOT NULL ORDER BY updated_at DESC LIMIT 1",
            (model_id,),
        ).fetchone()

    return row["job_id"] if row else None


def cache_training_prediction_descriptions(model_id, training_job_id):
    if not model_id or not training_job_id:
        return

    try:
        training_results = fetch_job_results(training_job_id)
    except requests.RequestException:
        return

    prediction_descriptions = normalize_prediction_descriptions_dict(
        extract_prediction_descriptions_value(training_results)
    )
    if prediction_descriptions:
        cache_save_model_prediction_descriptions(
            model_id,
            prediction_descriptions,
            training_job_id,
        )


def cache_get_inference_job(cache_key):
    with local_cache_connection() as connection:
        row = connection.execute(
            """
            SELECT cache_key, job_id, status, result_payload_json, result_csv
            FROM inference_jobs
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
    return dict(row) if row else None


def cache_save_inference_job(cache_key, job_id, status, result_payload=None, result_dataframe=None):
    result_payload_json = json.dumps(result_payload) if result_payload is not None else None
    result_csv = result_dataframe.to_csv(index=False) if result_dataframe is not None else None

    with local_cache_connection() as connection:
        connection.execute(
            """
            INSERT INTO inference_jobs (
                cache_key,
                job_id,
                status,
                result_payload_json,
                result_csv,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                job_id = excluded.job_id,
                status = excluded.status,
                result_payload_json = COALESCE(excluded.result_payload_json, inference_jobs.result_payload_json),
                result_csv = COALESCE(excluded.result_csv, inference_jobs.result_csv),
                updated_at = excluded.updated_at
            """,
            (cache_key, job_id, status, result_payload_json, result_csv, time.time()),
        )


def cache_delete_inference_job(cache_key):
    with local_cache_connection() as connection:
        connection.execute("DELETE FROM inference_jobs WHERE cache_key = ?", (cache_key,))


def dataframe_from_cached_csv(csv_text):
    if not csv_text:
        return pd.DataFrame()

    return pd.read_csv(StringIO(csv_text))


def payload_from_cached_json(payload_json):
    if not payload_json:
        return {}

    try:
        return json.loads(payload_json)
    except json.JSONDecodeError:
        return {}


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


def model_id_from_training_payload(payload):
    if not isinstance(payload, dict):
        return None

    for key in ("model", "data", "result", "job"):
        value = payload.get(key)
        if isinstance(value, dict):
            model_id = value.get("id") or value.get("model_id")
            if model_id:
                return model_id

    return payload.get("model_id") or payload.get("id")


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
        report_api_error("The Wood Wide API request failed.", response)
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

        report_api_error("The Wood Wide API request failed.", response)
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
        report_api_error("The Wood Wide API request failed.", response)
        response.raise_for_status()

    payload = response.json()
    model_id = model_id_from_training_payload(payload)
    if not model_id:
        st.write(payload)
        raise ValueError("Model training response did not include a model id.")
    return model_id, payload.get("job_id")


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
            cache_training_prediction_descriptions(model_id, training_job_id)
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
            cache_training_prediction_descriptions(model_id, training_job_id)
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
        cache_training_prediction_descriptions(model_id, training_job_id)
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
    if hasattr(uploaded_file, "getvalue"):
        content = uploaded_file.getvalue()
        buffer = StringIO(content.decode("utf-8") if isinstance(content, bytes) else content)
        dataframe = pd.read_csv(buffer)
    else:
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


def detect_noshow_label_column(dataframe):
    exact_matches = {
        "noshow",
        "no_show",
        "isnoshow",
        "hasnoshow",
        "missed",
        "missedappointment",
        "appointmentmissed",
        "showstatus",
        "appointmentstatus",
        "attendance",
        "attended",
    }

    fallback_matches = []
    for column in dataframe.columns:
        normalized_column = normalized_column_name(column)
        if not any(token in normalized_column for token in ("noshow", "missed", "attendance", "showstatus")):
            continue

        if normalized_column in exact_matches and is_binary_target_series(dataframe[column]):
            return column

        if is_binary_target_series(dataframe[column]):
            fallback_matches.append(column)

    if len(fallback_matches) == 1:
        return fallback_matches[0]

    return None


def noshow_label_column_options(dataframe):
    likely_columns = []
    other_columns = []

    for column in dataframe.columns:
        normalized_column = normalized_column_name(column)
        if any(token in normalized_column for token in ("noshow", "missed", "attendance", "show")) or is_binary_target_series(
            dataframe[column]
        ):
            likely_columns.append(column)
        else:
            other_columns.append(column)

    return likely_columns + other_columns


def get_noshow_label_column(dataframe, key):
    detected_column = detect_noshow_label_column(dataframe)
    if detected_column:
        return detected_column

    st.warning("I could not confidently identify the no-show target column in the training data.")
    options = noshow_label_column_options(dataframe)
    if not options:
        st.error("Training data does not contain any columns.")
        st.stop()

    placeholder = "Select a column"
    selected_column = st.selectbox(
        "Which column identifies no-shows?",
        [placeholder, *options],
        key=key,
        help="Choose the target column the model should learn to predict (1 = no-show, 0 = show).",
    )
    if selected_column == placeholder:
        st.stop()

    if selected_column and not is_binary_target_series(dataframe[selected_column]):
        st.info(
            f'"{selected_column}" does not look like a binary no-show column. '
            "Use it only if this is the target your model should predict."
        )

    return selected_column


def prediction_input_columns_for_training(dataframe, label_column):
    label_normalized = normalized_column_name(label_column)

    input_columns = []
    for column in dataframe.columns:
        normalized_column = normalized_column_name(column)
        if normalized_column == label_normalized or normalized_column in (
            "id",
            "customerid",
            "customer_id",
            "accountid",
            "account_id",
            "patientid",
            "patient_id",
        ):
            continue

        if "churn" in normalized_column and is_binary_target_series(dataframe[column]):
            continue

        if any(token in normalized_column for token in ("noshow", "missed", "attendance")) and is_binary_target_series(
            dataframe[column]
        ):
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


def parse_jsonish_value(value):
    if isinstance(value, str):
        stripped_value = value.strip()
        if stripped_value.startswith(("{", "[")):
            try:
                return json.loads(stripped_value)
            except json.JSONDecodeError:
                return value
    return value


def description_from_prediction_payload(payload):
    payload = parse_jsonish_value(payload)
    if isinstance(payload, dict):
        for key in ("description", "explanation", "summary", "text"):
            value = payload.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return json.dumps(payload)
    if payload is None:
        return ""
    return str(payload)


def numeric_weight(value):
    if isinstance(value, dict):
        for key in (
            "weight",
            "value",
            "score",
            "importance",
            "contribution",
            "normalized_weight",
        ):
            weight = as_number(value.get(key))
            if weight is not None:
                return weight
        return None
    return as_number(value)


def normalized_prediction_label(value):
    number = as_number(value)
    if number is not None and float(number).is_integer():
        return str(int(number))
    return str(value)


def is_positive_churn_label(value):
    number = as_number(value)
    if number is not None:
        return number == 1

    normalized_value = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    return normalized_value in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "churn",
        "churned",
        "noshow",
        "noshowed",
        "missed",
        "absent",
    }


def is_non_churn_label(value):
    number = as_number(value)
    if number is not None:
        return number == 0

    normalized_value = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    return normalized_value in {"0", "false", "f", "no", "n", "notchurn", "notchurned", "retained"}


def parse_descriptions_object(descriptions):
    descriptions = parse_jsonish_value(descriptions)
    if isinstance(descriptions, dict):
        return descriptions
    return None


def download_json_artifact(artifact_url):
    if not artifact_url:
        return None

    try:
        response = requests.get(artifact_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None

    if not response.text.strip().startswith(("{", "[")):
        return None

    try:
        return response.json()
    except json.JSONDecodeError:
        return None


def extract_prediction_descriptions_value(payload):
    if not isinstance(payload, dict):
        return None

    prediction_descriptions = payload.get("prediction_descriptions")
    descriptions = parse_descriptions_object(payload.get("descriptions"))
    if not prediction_descriptions and descriptions:
        prediction_descriptions = descriptions.get("prediction_descriptions")

    training_results = payload.get("training_results")
    if not prediction_descriptions and isinstance(training_results, dict):
        prediction_descriptions = extract_prediction_descriptions_value(training_results)

    return prediction_descriptions


def normalize_prediction_descriptions_dict(prediction_descriptions):
    prediction_descriptions = parse_jsonish_value(prediction_descriptions)
    if isinstance(prediction_descriptions, list):
        prediction_descriptions = {
            item.get("label", item.get("prediction")): item
            for item in prediction_descriptions
            if isinstance(item, dict)
        }

    if not isinstance(prediction_descriptions, dict):
        return None

    return {
        normalized_prediction_label(label): payload
        for label, payload in prediction_descriptions.items()
    }


def prediction_descriptions_from_artifact_urls(payload):
    for url_key in ("combined_results_uri", "inference_results_uri", "results_uri"):
        artifact_payload = download_json_artifact(payload.get(url_key))
        if not artifact_payload:
            continue

        prediction_descriptions = extract_prediction_descriptions_value(artifact_payload)
        normalized_descriptions = normalize_prediction_descriptions_dict(prediction_descriptions)
        if normalized_descriptions:
            return normalized_descriptions

    return None


def prediction_descriptions_from_payload_sources(payload):
    if not isinstance(payload, dict):
        return None

    normalized_descriptions = normalize_prediction_descriptions_dict(
        extract_prediction_descriptions_value(payload)
    )
    if normalized_descriptions:
        return normalized_descriptions

    return prediction_descriptions_from_artifact_urls(payload)


def refresh_inference_payload_descriptions(inference_payload, inference_job_id=None):
    if not isinstance(inference_payload, dict):
        return inference_payload

    if prediction_descriptions_from_payload_sources(inference_payload):
        return inference_payload

    if not inference_job_id:
        return inference_payload

    try:
        fresh_payload = fetch_job_results(inference_job_id)
    except requests.RequestException:
        return inference_payload

    if not isinstance(fresh_payload, dict):
        return inference_payload

    merged_payload = dict(inference_payload)
    for key in ("descriptions", "prediction_descriptions", "training_results"):
        if fresh_payload.get(key) is not None:
            merged_payload[key] = fresh_payload[key]

    for url_key in ("combined_results_uri", "inference_results_uri", "results_uri"):
        if fresh_payload.get(url_key) and not merged_payload.get(url_key):
            merged_payload[url_key] = fresh_payload[url_key]

    return merged_payload


def prediction_descriptions_for_model_run(inference_payload, inference_job_id=None, model_id=None):
    refreshed_payload = refresh_inference_payload_descriptions(inference_payload, inference_job_id)
    prediction_descriptions = prediction_descriptions_from_payload_sources(refreshed_payload)
    if prediction_descriptions:
        return prediction_descriptions

    if model_id:
        cached_descriptions = cache_get_model_prediction_descriptions(model_id)
        if cached_descriptions:
            return cached_descriptions

        training_job_id = cache_get_training_job_id_for_model(model_id)
        if training_job_id:
            cache_training_prediction_descriptions(model_id, training_job_id)
            cached_descriptions = cache_get_model_prediction_descriptions(model_id)
            if cached_descriptions:
                return cached_descriptions

    return None


def prediction_descriptions_from_inference_payload(inference_payload):
    return prediction_descriptions_for_model_run(inference_payload)


def feature_entries_by_label_from_descriptions(prediction_descriptions):
    if not prediction_descriptions:
        return {}

    return {
        label: feature_weight_entries_from_prediction_payload(payload)
        for label, payload in prediction_descriptions.items()
    }


def churn_class_label_from_dataframe(dataframe):
    if dataframe is None or dataframe.empty or "prediction" not in dataframe.columns:
        return "1"

    labels = [
        normalized_prediction_label(label)
        for label in dataframe["prediction"].dropna().unique()
    ]
    positive_labels = [
        label
        for label in labels
        if is_positive_churn_label(label)
    ]
    if len(positive_labels) == 1:
        return positive_labels[0]

    non_churn_labels = {
        label
        for label in labels
        if is_non_churn_label(label)
    }
    churn_labels = [label for label in labels if label not in non_churn_labels]
    if len(churn_labels) == 1:
        return churn_labels[0]

    return "1"


def feature_weight_entries_from_value(value):
    value = parse_jsonish_value(value)
    entries = []

    if isinstance(value, dict):
        feature_name = None
        for feature_key in ("feature", "feature_name", "name", "column", "field"):
            if feature_key in value:
                feature_name = str(value[feature_key])
                break

        weight = numeric_weight(value)
        if feature_name is not None and weight is not None:
            return [(feature_name, weight)]

        for key, weight_value in value.items():
            weight = numeric_weight(weight_value)
            if weight is not None:
                entries.append((str(key), weight))
        return entries

    if isinstance(value, list):
        for item in value:
            entries.extend(feature_weight_entries_from_value(item))

    return entries


def feature_weight_entries_from_prediction_payload(payload):
    payload = parse_jsonish_value(payload)
    if isinstance(payload, dict):
        for key in (
            "feature_weights",
            "featureWeights",
            "weights",
            "feature_importances",
            "featureImportances",
            "feature_contributions",
            "featureContributions",
            "features",
        ):
            if key in payload:
                return feature_weight_entries_from_value(payload[key])
        return feature_weight_entries_from_value(payload)

    return feature_weight_entries_from_value(payload)


def format_feature_weight(feature, weight):
    return f"{feature} ({weight:+.3g})"


def feature_weight_summary(entries, direction="positive", limit=3):
    if direction == "positive":
        selected_entries = [entry for entry in entries if entry[1] > 0]
        selected_entries = sorted(selected_entries, key=lambda entry: entry[1], reverse=True)
    elif direction == "negative":
        selected_entries = [entry for entry in entries if entry[1] < 0]
        selected_entries = sorted(selected_entries, key=lambda entry: entry[1])
    else:
        selected_entries = sorted(entries, key=lambda entry: abs(entry[1]), reverse=True)

    return ", ".join(
        format_feature_weight(feature, weight)
        for feature, weight in selected_entries[:limit]
    )


def feature_weight_context_from_entries(entries, limit=5):
    if not entries:
        return ""

    strongest_entries = sorted(entries, key=lambda entry: abs(entry[1]), reverse=True)[:limit]
    context_parts = []
    for feature, weight in strongest_entries:
        if weight > 0:
            context_parts.append(f"churn risk driver {feature} positive feature weight {weight:.3g}")
        elif weight < 0:
            context_parts.append(f"protective driver {feature} negative feature weight {weight:.3g}")
    return " ".join(context_parts)


def driver_group_for_feature(feature):
    normalized_feature = normalized_column_name(feature)
    if any(token in normalized_feature for token in ("usage", "login", "activity", "adoption", "feature", "utilization")):
        return "Usage/adoption"
    if any(token in normalized_feature for token in ("nps", "csat", "rating", "satisfaction", "sentiment", "feedback")):
        return "Satisfaction"
    if any(token in normalized_feature for token in ("support", "ticket", "case", "issue")):
        return "Support friction"
    if any(token in normalized_feature for token in ("contract", "arr", "mrr", "revenue", "seat", "price", "charge", "plan", "subscription")):
        return "Commercial/renewal"
    if any(token in normalized_feature for token in ("onboarding", "setup", "implementation", "activation")):
        return "Onboarding"
    if any(token in normalized_feature for token in ("billing", "payment", "invoice", "card")):
        return "Billing"
    if any(token in normalized_feature for token in ("view", "watch", "genre", "content", "download", "device")):
        return "Content engagement"
    if any(token in normalized_feature for token in ("lead", "wait", "schedule", "appointment", "appt", "dayof", "weekday", "hour")):
        return "Scheduling"
    if any(token in normalized_feature for token in ("noshow", "no_show", "previous", "history", "prior", "past")):
        return "Appointment history"
    if any(token in normalized_feature for token in ("chronic", "condition", "health", "diagnosis", "specialty", "procedure")):
        return "Health status"
    if any(token in normalized_feature for token in ("sms", "reminder", "call", "outreach", "notification")):
        return "Outreach"
    if any(token in normalized_feature for token in ("transport", "distance", "travel", "access", "barrier")):
        return "Access"
    if any(token in normalized_feature for token in ("age", "gender", "sex", "race", "ethnicity", "language")):
        return "Demographics"
    if any(token in normalized_feature for token in ("insurance", "coverage", "payer", "copay", "deductible")):
        return "Coverage"
    return "Other"


def row_value_for_feature(row, feature):
    if feature in row.index:
        value = row.get(feature)
        return "" if pd.isna(value) else value

    normalized_feature = normalized_column_name(feature)
    for column in row.index:
        if normalized_column_name(column) == normalized_feature:
            value = row.get(column)
            return "" if pd.isna(value) else value

    return ""


def model_risk_driver_summary_for_row(row, entries, limit=3):
    positive_entries = sorted(
        [entry for entry in entries if entry[1] > 0],
        key=lambda entry: entry[1],
        reverse=True,
    )
    if not positive_entries:
        return ""

    driver_parts = []
    for feature, weight in positive_entries[:limit]:
        group = driver_group_for_feature(feature)
        value = row_value_for_feature(row, feature)
        if value != "":
            driver_parts.append(f"{group}: {feature}={value} ({weight:+.3g})")
        else:
            driver_parts.append(f"{group}: {feature} ({weight:+.3g})")
    return "; ".join(driver_parts)


def parse_feature_weight_summary(summary):
    entries = []
    for part in str(summary or "").split(","):
        match = re.match(r"\s*(.*?)\s*\(([+-]?[0-9.eE-]+)\)\s*$", part)
        if not match:
            continue

        weight = as_number(match.group(2))
        if weight is not None:
            entries.append((match.group(1), weight))
    return entries


def risk_driver_chart_data(dataframe):
    if dataframe is None or dataframe.empty or "top_churn_drivers" not in dataframe.columns:
        return pd.DataFrame()

    driver_rows = []
    for summary in dataframe["top_churn_drivers"].dropna():
        summary_text = str(summary).strip()
        if not summary_text:
            continue
        for feature, weight in parse_feature_weight_summary(summary_text):
            if weight <= 0:
                continue
            driver_rows.append(
                {
                    "driver": driver_group_for_feature(feature),
                    "feature": feature,
                    "weighted_signal": abs(weight),
                    "customers": 1,
                }
            )

    if not driver_rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(driver_rows)
        .groupby(["driver", "feature"], as_index=False)
        .agg(weighted_signal=("weighted_signal", "sum"), customers=("customers", "sum"))
        .sort_values("weighted_signal", ascending=False)
    )


def risk_driver_chart_data_from_feature_entries(feature_entries_by_label, at_risk):
    if not feature_entries_by_label or at_risk is None or at_risk.empty:
        return pd.DataFrame()

    churn_label = churn_class_label_from_dataframe(at_risk)
    entries = feature_entries_by_label.get(churn_label, [])
    if not entries:
        for label, label_entries in feature_entries_by_label.items():
            if is_positive_churn_label(label):
                entries = label_entries
                churn_label = label
                break

    if not entries:
        return pd.DataFrame()

    customer_count = len(at_risk)
    driver_rows = []
    for feature, weight in entries:
        if weight <= 0:
            continue
        driver_rows.append(
            {
                "driver": driver_group_for_feature(feature),
                "feature": feature,
                "weighted_signal": abs(weight) * customer_count,
                "customers": customer_count,
            }
        )

    if not driver_rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(driver_rows)
        .groupby(["driver", "feature"], as_index=False)
        .agg(weighted_signal=("weighted_signal", "sum"), customers=("customers", "sum"))
        .sort_values("weighted_signal", ascending=False)
    )


def risk_driver_chart_data_from_feature_contrast(at_risk, scored_customers):
    if (
        at_risk is None
        or scored_customers is None
        or at_risk.empty
        or scored_customers.empty
    ):
        return pd.DataFrame()

    at_risk_modeling_data, feature_columns = analysis_dataframe_for_modeling(at_risk)
    if at_risk_modeling_data.empty or not feature_columns:
        return pd.DataFrame()

    if "prediction" in scored_customers.columns:
        retained_customers = scored_customers[
            ~scored_customers["prediction"].map(is_positive_churn_label)
        ]
    else:
        probability_column = probability_sort_column(scored_customers)
        if probability_column:
            probabilities = pd.to_numeric(scored_customers[probability_column], errors="coerce")
            retained_customers = scored_customers[probabilities < 0.5]
        else:
            retained_customers = scored_customers

    retained_modeling_data, _ = analysis_dataframe_for_modeling(retained_customers)
    driver_rows = []
    for feature in feature_columns:
        if feature not in retained_modeling_data.columns:
            continue

        at_risk_values = pd.to_numeric(at_risk_modeling_data[feature], errors="coerce").dropna()
        retained_values = pd.to_numeric(retained_modeling_data[feature], errors="coerce").dropna()
        if at_risk_values.empty or retained_values.empty:
            continue

        effect_size = abs(at_risk_values.mean() - retained_values.mean())
        pooled_std = pd.concat([at_risk_values, retained_values]).std()
        if pooled_std and pooled_std > 0:
            signal = effect_size / pooled_std
        else:
            signal = effect_size

        if signal <= 0:
            continue

        driver_rows.append(
            {
                "driver": driver_group_for_feature(feature),
                "feature": feature,
                "weighted_signal": signal * len(at_risk),
                "customers": len(at_risk),
            }
        )

    if not driver_rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(driver_rows)
        .groupby(["driver", "feature"], as_index=False)
        .agg(weighted_signal=("weighted_signal", "sum"), customers=("customers", "sum"))
        .sort_values("weighted_signal", ascending=False)
    )


def add_prediction_descriptions(
    scored_customers,
    inference_payload,
    inference_job_id=None,
    model_id=None,
):
    prediction_descriptions = prediction_descriptions_for_model_run(
        inference_payload,
        inference_job_id,
        model_id,
    )
    feature_entries_by_label = feature_entries_by_label_from_descriptions(prediction_descriptions)
    st.session_state.churn_feature_entries_by_label = feature_entries_by_label
    st.session_state.noshow_feature_entries_by_label = feature_entries_by_label

    if not prediction_descriptions or "prediction" not in scored_customers.columns:
        return scored_customers

    payloads_by_label = prediction_descriptions
    scored_customers = scored_customers.copy()
    scored_customers["prediction_class_explanation"] = scored_customers["prediction"].map(
        lambda prediction: description_from_prediction_payload(
            payloads_by_label.get(normalized_prediction_label(prediction), "")
        )
    )

    scored_customers["top_churn_drivers"] = scored_customers["prediction"].map(
        lambda prediction: feature_weight_summary(
            feature_entries_by_label.get(normalized_prediction_label(prediction), []),
            "positive",
        )
    )
    scored_customers["top_protective_drivers"] = scored_customers["prediction"].map(
        lambda prediction: feature_weight_summary(
            feature_entries_by_label.get(normalized_prediction_label(prediction), []),
            "negative",
        )
    )
    scored_customers["feature_weight_context"] = scored_customers["prediction"].map(
        lambda prediction: feature_weight_context_from_entries(
            feature_entries_by_label.get(normalized_prediction_label(prediction), [])
        )
    )
    scored_customers["model_risk_drivers"] = scored_customers.apply(
        lambda row: model_risk_driver_summary_for_row(
            row,
            feature_entries_by_label.get(normalized_prediction_label(row.get("prediction")), []),
        ),
        axis=1,
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


def fetch_job_results(job_id, raise_on_error=True):
    response = requests.get(
        f"{base_url}/jobs/{job_id}/results",
        headers=headers,
    )

    if response.status_code == 404:
        if not raise_on_error:
            return None
        report_api_error("The Wood Wide API request failed.", response)
        response.raise_for_status()

    if response.status_code != 200:
        if not raise_on_error:
            return None
        report_api_error("The Wood Wide API request failed.", response)
        response.raise_for_status()

    return response.json()


def load_job_result_dataframe(job_id, raise_on_error=False):
    results_payload = fetch_job_results(job_id, raise_on_error=raise_on_error)
    if results_payload is None:
        return None, None
    return dataframe_from_result_artifact(results_payload), results_payload


def job_status(job_id):
    response = requests.get(
        f"{base_url}/jobs/{job_id}",
        headers=headers,
    )

    if response.status_code != 200:
        return None

    return response.json().get("status")


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


def inference_cache_key(kind, *parts):
    raw_key = ":".join(str(part) for part in (kind, *parts))
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def cached_inference_result(cache_key, label):
    cached_job = cache_get_inference_job(cache_key)
    if not cached_job:
        return None

    if cached_job.get("status") in READY_STATUSES and cached_job.get("result_csv"):
        st.success(f"Reusing cached {label} result.")
        return (
            dataframe_from_cached_csv(cached_job.get("result_csv")),
            payload_from_cached_json(cached_job.get("result_payload_json")),
            cached_job.get("job_id"),
        )

    job_id = cached_job.get("job_id")
    if not job_id:
        cache_delete_inference_job(cache_key)
        return None

    status = job_status(job_id)
    if status in READY_STATUSES:
        result_dataframe, results_payload = load_job_result_dataframe(job_id, raise_on_error=False)
        if results_payload is None:
            cache_delete_inference_job(cache_key)
            return None
        cache_save_inference_job(
            cache_key,
            job_id,
            "succeeded",
            results_payload,
            result_dataframe,
        )
        st.success(f"Reusing completed {label} job.")
        return result_dataframe, results_payload, job_id

    if status in FAILED_STATUSES:
        cache_delete_inference_job(cache_key)
        return None

    st.info(f"Resuming cached {label} job.")
    job_completed = wait_for_job_succeeded(job_id, label)
    if job_completed is True:
        result_dataframe, results_payload = load_job_result_dataframe(job_id, raise_on_error=False)
        if results_payload is None:
            cache_delete_inference_job(cache_key)
            return None
        cache_save_inference_job(
            cache_key,
            job_id,
            "succeeded",
            results_payload,
            result_dataframe,
        )
        return result_dataframe, results_payload, job_id

    if job_completed is False:
        cache_delete_inference_job(cache_key)
        return None

    cache_save_inference_job(cache_key, job_id, status or "pending")
    st.warning(f"{label.title()} is still queued or running. Refresh or rerun the app later to resume.")
    st.stop()


def run_cached_model_inference(
    model_id,
    file_bytes,
    filename,
    output_type,
    cache_key,
    label,
    extra_form_data=None,
):
    cached_result = cached_inference_result(cache_key, label)
    if cached_result is not None:
        return cached_result

    form_data = {"output_type": output_type}
    if extra_form_data:
        form_data.update(extra_form_data)

    response = requests.post(
        f"{base_url}/models/{model_id}/infer-async",
        headers=headers,
        files={"file": (filename, file_bytes, "text/csv")},
        data=form_data,
    )

    if response.status_code == 404:
        response = requests.post(
            f"{base_url}/models/{model_id}/infer",
            headers=headers,
            files={"file": (filename, file_bytes, "text/csv")},
            data=form_data,
        )

        if response.status_code not in SUCCESS_STATUS_CODES:
            st.write(response.status_code)
            st.write(response.text)
            response.raise_for_status()

        if output_type == "json":
            result_dataframe, result_payload = read_inference_json(response)
        else:
            result_dataframe = read_inference_csv(response)
            result_payload = {"data": result_dataframe.to_dict(orient="records")}

        inference_job_id = (
            result_payload.get("job_id")
            or result_payload.get("inference_job_id")
            or result_payload.get("id")
        )
        cache_save_inference_job(
            cache_key,
            inference_job_id,
            "succeeded",
            result_payload,
            result_dataframe,
        )
        return result_dataframe, result_payload, inference_job_id

    if response.status_code not in SUCCESS_STATUS_CODES:
        report_api_error("The Wood Wide API request failed.", response)
        response.raise_for_status()

    inference_job_id = response.json()["job_id"]
    cache_save_inference_job(cache_key, inference_job_id, "pending")
    if not wait_for_job_succeeded(inference_job_id, label):
        cache_delete_inference_job(cache_key)
        st.error(f"{label.title()} failed or timed out.")
        st.stop()

    inference_payload = fetch_job_results(inference_job_id)
    result_dataframe = dataframe_from_result_artifact(inference_payload)
    cache_save_inference_job(
        cache_key,
        inference_job_id,
        "succeeded",
        inference_payload,
        result_dataframe,
    )
    return result_dataframe, inference_payload, inference_job_id


def run_prediction_inference(model_id, uploaded_file, filename="test.csv"):
    file_bytes = uploaded_file.getvalue()
    cache_key = inference_cache_key(
        "prediction",
        model_id,
        hashlib.sha256(file_bytes).hexdigest(),
        "json",
    )
    return run_cached_model_inference(
        model_id,
        file_bytes,
        filename,
        "json",
        cache_key,
        "prediction inference",
    )


def run_anomaly_inference(model_id, file_bytes, filename="fraud_test.csv", output_type="csv"):
    cache_key = inference_cache_key(
        "anomaly",
        model_id,
        hashlib.sha256(file_bytes).hexdigest(),
        output_type,
        "per_row",
    )
    return run_cached_model_inference(
        model_id,
        file_bytes,
        filename,
        output_type,
        cache_key,
        "anomaly inference",
        extra_form_data={"anomaly_format": "per_row"},
    )


def anomaly_score_column(dataframe):
    candidates = [
        "anomaly_score",
        "anomaly probability",
        "anomaly_probability",
        "score",
        "prediction_prob",
        "prediction_probability",
        "probability",
        "confidence",
        "is_anomaly",
    ]
    normalized = {normalized_column_name(column): column for column in dataframe.columns}
    for candidate in candidates:
        key = normalized_column_name(candidate)
        if key in normalized:
            return normalized[key]

    for column in dataframe.columns:
        lower = column.lower()
        if "anomaly" in lower or lower.endswith("_score"):
            return column
    return None


def filter_by_score_threshold(dataframe, threshold, score_column=None):
    result = dataframe.copy()
    score_column = score_column or anomaly_score_column(result) or probability_sort_column(result)
    if not score_column:
        st.warning("No score column found; returning all rows.")
        return sort_by_probability_desc(result)

    scores = pd.to_numeric(result[score_column], errors="coerce")
    if scores.notna().any() and scores.max() > 1:
        scores = scores / 100

    filtered = result[scores >= threshold].copy()
    return (
        filtered.assign(_score_sort=scores.loc[filtered.index])
        .sort_values("_score_sort", ascending=False)
        .drop(columns=["_score_sort"])
    )


def is_positive_label(value, positive_tokens=("1", "true", "yes", "returned", "return", "fraud", "anomaly")):
    if pd.isna(value):
        return False
    normalized = str(value).strip().lower()
    if normalized in positive_tokens:
        return True
    number = as_number(value)
    return number == 1


def filter_high_return_risk(scored_orders, threshold):
    filtered = scored_orders.copy()
    if "prediction" in filtered.columns:
        filtered = filtered[filtered["prediction"].map(lambda value: is_positive_label(value, ("1", "true", "yes", "returned", "return")))]
    return filter_by_score_threshold(filtered, threshold)


def filter_high_fraud_risk(scored_transactions, threshold):
    filtered = scored_transactions.copy()
    if "prediction" in filtered.columns:
        filtered = filtered[filtered["prediction"].map(lambda value: is_positive_label(value, ("1", "true", "yes", "fraud")))]
    return filter_by_score_threshold(filtered, threshold)


def filter_flagged_transactions(scored_transactions, threshold):
    return filter_by_score_threshold(scored_transactions, threshold)


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


EXPLAIN_UNAVAILABLE_STATUS_CODES = {400, 404, 405, 422, 501}


def explain_unavailable_session_key(inference_job_id):
    return f"explain_unavailable:{inference_job_id}"


def explanations_disabled_for_job(inference_job_id):
    return bool(st.session_state.get(explain_unavailable_session_key(inference_job_id)))


def mark_explanations_unavailable(inference_job_id):
    st.session_state[explain_unavailable_session_key(inference_job_id)] = True


def explain_request_failed(response):
    return response.status_code in EXPLAIN_UNAVAILABLE_STATUS_CODES


def warn_explanations_unavailable(response=None):
    detail = ""
    if response is not None and response.text:
        detail = f" API response: {response.text[:240]}"
    st.warning(
        "Row-level explanations are unavailable for this inference job or API environment, "
        f"so the app will continue without them.{detail}"
    )


def request_row_level_explanation_batch(inference_job_id, row_ids, output_type="json"):
    if explanations_disabled_for_job(inference_job_id):
        return None

    cache_key = row_explanation_cache_key(inference_job_id, row_ids, output_type)
    cached_result = cached_inference_result(cache_key, "row explanation")
    if cached_result is not None:
        explanations, results_payload, explanation_job_id = cached_result
        if explanations.empty:
            return explanations
        return normalize_explanation_columns(explanations)

    response = requests.post(
        f"{base_url}/jobs/{inference_job_id}/explain",
        headers=headers,
        json={
            "ids": row_ids,
            "output_type": output_type,
        },
    )

    if response.status_code not in SUCCESS_STATUS_CODES:
        if explain_request_failed(response):
            mark_explanations_unavailable(inference_job_id)
            warn_explanations_unavailable(response)
            return None

        report_api_error("The Wood Wide API request failed.", response)
        response.raise_for_status()

    explanation_job_id = response.json()["job_id"]
    cache_save_inference_job(cache_key, explanation_job_id, "pending")
    if not wait_for_job_succeeded(explanation_job_id, "row explanation"):
        cache_delete_inference_job(cache_key)
        st.warning("Row-level explanation job failed or timed out.")
        return pd.DataFrame()

    explanations, results_payload = load_job_result_dataframe(explanation_job_id, raise_on_error=False)
    if results_payload is None:
        cache_delete_inference_job(cache_key)
        st.warning(
            "Row-level explanation results were not available from the API for this job, so the app will continue without them."
        )
        return pd.DataFrame()

    cache_save_inference_job(
        cache_key,
        explanation_job_id,
        "succeeded",
        results_payload,
        explanations,
    )
    if explanations.empty:
        return explanations

    return normalize_explanation_columns(explanations)


def row_explanation_cache_key(inference_job_id, row_ids, output_type="json"):
    normalized_row_ids = sorted(str(row_id) for row_id in row_ids)
    return inference_cache_key(
        "row_explanations",
        inference_job_id,
        ",".join(normalized_row_ids),
        output_type,
    )


def cached_row_explanation_if_ready(cache_key):
    cached_job = cache_get_inference_job(cache_key)
    if not cached_job:
        return "missing", pd.DataFrame()

    if cached_job.get("status") in READY_STATUSES and cached_job.get("result_csv"):
        explanations = dataframe_from_cached_csv(cached_job.get("result_csv"))
        return "succeeded", normalize_explanation_columns(explanations)

    job_id = cached_job.get("job_id")
    if not job_id:
        cache_delete_inference_job(cache_key)
        return "missing", pd.DataFrame()

    status = job_status(job_id)
    if status in READY_STATUSES:
        explanations, results_payload = load_job_result_dataframe(job_id, raise_on_error=False)
        if results_payload is None:
            cache_delete_inference_job(cache_key)
            return "missing", pd.DataFrame()
        cache_save_inference_job(cache_key, job_id, "succeeded", results_payload, explanations)
        return "succeeded", normalize_explanation_columns(explanations)

    if status in FAILED_STATUSES:
        cache_delete_inference_job(cache_key)
        return "failed", pd.DataFrame()

    cache_save_inference_job(cache_key, job_id, status or "pending")
    return status or "pending", pd.DataFrame()


def start_row_level_explanation_job(inference_job_id, row_ids, output_type="json"):
    if explanations_disabled_for_job(inference_job_id):
        return row_explanation_cache_key(inference_job_id, row_ids, output_type), "unavailable", pd.DataFrame()

    cache_key = row_explanation_cache_key(inference_job_id, row_ids, output_type)
    status, explanations = cached_row_explanation_if_ready(cache_key)
    if status == "succeeded" or status in PENDING_STATUSES:
        return cache_key, status, explanations

    response = requests.post(
        f"{base_url}/jobs/{inference_job_id}/explain",
        headers=headers,
        json={
            "ids": row_ids,
            "output_type": output_type,
        },
    )

    if response.status_code not in SUCCESS_STATUS_CODES:
        if explain_request_failed(response):
            mark_explanations_unavailable(inference_job_id)
            warn_explanations_unavailable(response)
            return cache_key, "unavailable", pd.DataFrame()

        report_api_error("The Wood Wide API request failed.", response)
        response.raise_for_status()

    explanation_job_id = response.json()["job_id"]
    cache_save_inference_job(cache_key, explanation_job_id, "pending")
    return cache_key, "pending", pd.DataFrame()


def get_row_level_explanations(inference_job_id, row_ids, output_type="json"):
    if not inference_job_id or not row_ids:
        return pd.DataFrame()
    if explanations_disabled_for_job(inference_job_id):
        return pd.DataFrame()

    explanations = request_row_level_explanation_batch(inference_job_id, row_ids, output_type)
    if explanations is None:
        return pd.DataFrame()
    if explanations.empty or "id" not in explanations.columns:
        return explanations

    requested_ids = {normalized_merge_id(row_id) for row_id in row_ids}
    returned_ids = {normalized_merge_id(row_id) for row_id in explanations["id"].dropna().tolist()}
    missing_ids = [
        row_id
        for row_id in row_ids
        if normalized_merge_id(row_id) not in returned_ids
    ]

    if not missing_ids or explanations_disabled_for_job(inference_job_id):
        return explanations

    st.info(
        f"Explanation batch returned {len(returned_ids)} of {len(requested_ids)} rows. "
        "Requesting missing rows individually."
    )
    explanation_frames = [explanations]
    for row_id in missing_ids:
        if explanations_disabled_for_job(inference_job_id):
            break
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


def get_row_level_explanations_individually(inference_job_id, row_ids, output_type="json"):
    if explanations_disabled_for_job(inference_job_id):
        return pd.DataFrame()

    explanation_frames = []
    for row_id in row_ids:
        row_explanation = request_row_level_explanation_batch(
            inference_job_id,
            [row_id],
            output_type,
        )
        if row_explanation is not None and not row_explanation.empty:
            explanation_frames.append(row_explanation)

    if not explanation_frames:
        return pd.DataFrame()

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


def row_explanation_is_missing(row):
    for column in ("row_prediction_explanation", "row_explanation_summary"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return False
    return True


def row_prediction_explanation_context(row):
    context_parts = []
    for column in ("row_prediction_explanation", "row_explanation_summary"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            context_parts.append(str(value).strip())

    return " ".join(context_parts)


def prediction_class_explanation_context(row):
    value = row.get("prediction_class_explanation")
    if pd.notna(value) and str(value).strip():
        return str(value).strip()
    return ""


def feature_weight_context(row):
    context_parts = []
    for column in ("model_risk_drivers", "top_churn_drivers", "feature_weight_context"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            context_parts.append(str(value).strip())
    return " ".join(context_parts)


def merge_explanation_frames(*frames):
    non_empty_frames = [
        frame
        for frame in frames
        if frame is not None and not frame.empty
    ]
    if not non_empty_frames:
        return pd.DataFrame()

    combined = pd.concat(non_empty_frames, ignore_index=True)
    if "id" not in combined.columns:
        return combined

    combined["_normalized_id"] = combined["id"].map(normalized_merge_id)
    return (
        combined
        .drop_duplicates("_normalized_id", keep="last")
        .drop(columns=["_normalized_id"])
    )


def stored_manual_explanations(inference_job_id):
    return st.session_state.manual_explanations_by_job.get(inference_job_id, pd.DataFrame())


def store_manual_explanations(inference_job_id, explanations):
    if explanations is None or explanations.empty:
        return

    st.session_state.manual_explanations_by_job[inference_job_id] = merge_explanation_frames(
        stored_manual_explanations(inference_job_id),
        explanations,
    )


def identity_columns(dataframe):
    preferred_columns = [
        "CustomerID",
        "customer_id",
        "account_id",
        "AccountID",
        "accountid",
        "id",
    ]
    return [column for column in preferred_columns if column in dataframe.columns]


def display_id_for_row(row):
    for column in identity_columns(pd.DataFrame([row])):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value)
    return str(row.name)


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
    "arr_usd",
    "mrr_usd",
    "seats",
    "company_size",
    "industry",
    "contract_type",
    "account_age_months",
    "product_usage_score",
    "features_adopted_pct",
    "login_frequency_monthly",
    "days_since_last_login",
    "support_tickets_90d",
    "nps_score",
    "csat_avg",
    "expansion_revenue_pct",
    "has_executive_sponsor",
    "onboarding_completed",
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
    "model_risk_drivers",
    "top_churn_drivers",
    "top_protective_drivers",
    "feature_weight_context",
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


def numeric_feature_series(series):
    non_null = series.dropna()
    if non_null.empty:
        return None

    if pd.api.types.is_bool_dtype(non_null):
        return series.astype("boolean").astype("Int64")

    if pd.api.types.is_numeric_dtype(non_null):
        numeric_series = pd.to_numeric(series, errors="coerce")
        return numeric_series if numeric_series.dropna().nunique() > 1 else None

    normalized_values = non_null.astype(str).map(normalized_column_name)
    binary_map = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "t": 1,
        "1": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "f": 0,
        "0": 0,
    }
    if normalized_values.isin(binary_map.keys()).all():
        return series.astype(str).map(lambda value: binary_map.get(normalized_column_name(value)))

    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.dropna().nunique() > 1 and numeric_series.notna().sum() == non_null.shape[0]:
        return numeric_series

    return None


def clustering_dataframe_for_modeling(dataframe):
    cluster_columns = {}
    for column in dataframe.columns:
        numeric_series = numeric_feature_series(dataframe[column])
        if numeric_series is None or numeric_series.dropna().nunique() <= 1:
            continue

        cluster_columns[column] = numeric_series

    cluster_dataframe = pd.DataFrame(cluster_columns).dropna(axis=1, how="all")
    return cluster_dataframe.copy(), list(cluster_dataframe.columns)


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
            if str(keyword).lower() in text:
                numeric_value = as_number(value)
                if numeric_value is None:
                    score += 1
                elif numeric_value > 0:
                    score += min(abs(numeric_value), 10)
    return score


def primary_factor_intervention_names(primary_factor_description):
    primary_factor_description = str(primary_factor_description).lower()
    if "billing" in primary_factor_description or "payment" in primary_factor_description:
        return [
            "Billing friction recovery",
            "Payment friction recovery",
            "Pricing and plan fit review",
        ]
    if "usage" in primary_factor_description or "adoption" in primary_factor_description:
        return [
            "Product adoption plan",
            "Engagement recovery",
            "Viewing engagement boost",
            "Feature adoption enablement",
        ]
    if "support" in primary_factor_description or "experience" in primary_factor_description:
        return [
            "Support recovery",
            "Support escalation",
            "Satisfaction recovery",
            "Experience quality recovery",
        ]
    if "value" in primary_factor_description or "qualitative" in primary_factor_description or "insight" in primary_factor_description:
        return [
            "High-value retention save",
            "Executive sponsor outreach",
            "Content personalization",
            "Pricing and plan fit review",
        ]
    if "confidence" in primary_factor_description or "risk score" in primary_factor_description:
        return [
            "High-value retention save",
            "Renewal save motion",
            "Executive sponsor outreach",
        ]
    return []


def intervention_match_candidates(row, context_text, intervention_catalog=None):
    primary_factor_description = str(row.get("primary_factor_description", "")).lower()
    primary_names = set(primary_factor_intervention_names(primary_factor_description))
    row_explanation_text = row_prediction_explanation_context(row)
    class_explanation_text = prediction_class_explanation_context(row)
    feature_weight_text = feature_weight_context(row)
    candidates = []

    for intervention in intervention_catalog or current_intervention_catalog():
        semantic_score = cosine_similarity(context_text, intervention["description"])
        row_explanation_score = cosine_similarity(row_explanation_text, intervention["description"])
        class_explanation_score = cosine_similarity(class_explanation_text, intervention["description"])
        feature_weight_score = cosine_similarity(feature_weight_text, intervention["description"])
        signal_score = row_signal_score(row, intervention["keywords"])
        signal_weight = min(signal_score, 30) / 10
        primary_boost = 1.25 if intervention["name"] in primary_names else 0
        score = (
            semantic_score
            + (row_explanation_score * 1.6)
            + (feature_weight_score * 2.0)
            + (class_explanation_score * 0.5)
            + signal_weight
            + primary_boost
        )

        if feature_weight_score > 0:
            match_source = "feature weights"
        elif row_explanation_score > 0:
            match_source = "row explanation"
        elif primary_boost > 0:
            match_source = "primary factor"
        elif signal_score > 0:
            match_source = "customer columns"
        else:
            match_source = "factor/cluster text"

        candidates.append(
            {
                "score": score,
                "intervention": intervention,
                "match_source": match_source,
            }
        )

    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)


def choose_intervention(
    row,
    context_text,
    intervention_counts=None,
    cluster_intervention_counts=None,
    intervention_catalog=None,
):
    candidates = intervention_match_candidates(row, context_text, intervention_catalog)
    if not candidates:
        raise ValueError("No interventions are configured.")

    if not intervention_counts and not cluster_intervention_counts:
        best_candidate = candidates[0]
    else:
        cluster_label = str(row.get("cluster_label", ""))

        def diversified_score(candidate):
            intervention_name = candidate["intervention"]["name"]
            global_count = intervention_counts[intervention_name] if intervention_counts else 0
            cluster_count = (
                cluster_intervention_counts[(cluster_label, intervention_name)]
                if cluster_intervention_counts and cluster_label
                else 0
            )
            diversity_penalty = 1 + (global_count * 0.45) + (cluster_count * 0.35)
            return candidate["score"] / diversity_penalty

        best_candidate = max(
            candidates,
            key=diversified_score,
        )

    return (
        best_candidate["score"],
        best_candidate["intervention"],
        best_candidate["match_source"],
    )


def risk_probability(row):
    probability_columns = [
        "risk_probability",
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
    if probability >= high_cutoff:
        return "high"
    if probability >= medium_cutoff:
        return "medium"
    return "low"


def probability_sort_column(dataframe):
    candidate_columns = [
        "risk_probability",
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
    probabilities = probability_series(at_risk_customers)

    if probabilities is not None and "risk_probability" in at_risk_customers.columns:
        filtered = at_risk_customers[probabilities >= churn_confidence_threshold]
        if filtered.empty and probabilities.max() < churn_confidence_threshold:
            if "prediction" in at_risk_customers.columns:
                filtered = at_risk_customers[
                    at_risk_customers["prediction"].map(is_positive_churn_label)
                ]
            else:
                top_fraction = max(0.05, min(0.5, churn_confidence_threshold))
                cutoff = probabilities.quantile(1 - top_fraction)
                filtered = at_risk_customers[probabilities >= cutoff]
        return sort_by_probability_desc(filtered)

    if "prediction" in at_risk_customers.columns:
        at_risk_customers = at_risk_customers[
            at_risk_customers["prediction"].map(is_positive_churn_label)
        ]

    probabilities = probability_series(at_risk_customers)
    if probabilities is None:
        st.warning("No probability column was found, so at-risk filtering used only the prediction label.")
        return at_risk_customers

    return sort_by_probability_desc(at_risk_customers[probabilities >= churn_confidence_threshold])


def at_risk_display_columns(dataframe):
    probability_column = probability_sort_column(dataframe)
    preferred_columns = [
        probability_column,
        "model_risk_drivers",
        "top_churn_drivers",
        "row_prediction_explanation",
        "row_explanation_summary",
        "prediction_class_explanation",
        "CustomerID",
        "customer_id",
        "account_id",
        "AccountID",
        "PatientID",
        "patient_id",
        "Age",
        "Specialty",
        "InsuranceType",
        "LeadTimeDays",
        "PreviousNoShows",
        "industry",
        "arr_usd",
        "product_usage_score",
        "nps_score",
        "days_since_last_login",
        "support_tickets_90d",
    ]

    display_columns = []
    for column in preferred_columns:
        if column and column in dataframe.columns and column not in display_columns:
            display_columns.append(column)

    remaining_columns = [
        column
        for column in dataframe.columns
        if column not in display_columns
        and normalized_column_name(column) not in (
            "prediction",
            "churn",
            "churned",
            "ischurn",
            "ischurned",
            "noshow",
            "no_show",
            "missed",
        )
    ]
    return [*display_columns, *remaining_columns]


def filter_customers_for_search(dataframe, query):
    if dataframe.empty or not query:
        return dataframe

    normalized_query = str(query).strip().lower()
    if not normalized_query:
        return dataframe

    searchable_columns = identity_columns(dataframe) or dataframe.columns.tolist()
    search_text = (
        dataframe[searchable_columns]
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    return dataframe[search_text.str.contains(re.escape(normalized_query), na=False)]


def customer_selection_options(dataframe):
    options = {}
    if dataframe.empty or "id" not in dataframe.columns:
        return options

    selectable_dataframe = dataframe[dataframe.apply(row_explanation_is_missing, axis=1)]
    for _, row in selectable_dataframe.iterrows():
        label = display_id_for_row(row)
        probability = risk_probability(row)
        if probability is not None:
            label = f"{label} ({probability:.0%})"
        options[label] = row["id"]
    return options


def parse_cluster_inference_payload(payload, expected_rows, dataframe=None):
    data = payload.get("data", payload)
    cluster_labels = None
    if isinstance(data, dict):
        cluster_labels = data.get("cluster_label") or data.get("cluster_labels") or data.get("clusters")

    if cluster_labels is None and dataframe is not None:
        for column in ("cluster_label", "cluster_labels", "clusters"):
            if column in dataframe.columns:
                cluster_labels = dataframe[column].tolist()
                break

    cluster_descriptions = {}
    if isinstance(data, dict):
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


def parse_cluster_inference(response, expected_rows):
    return parse_cluster_inference_payload(response.json(), expected_rows)


SYNTHETIC_SEGMENT_PROFILES = {
    "streaming": [
        (
            "Premium Multi-Device Loyalists",
            "Long-tenured, higher-value subscribers with broad device access and premium-plan behavior.",
        ),
        (
            "Budget Plan Price Watchers",
            "Cost-sensitive subscribers whose risk is tied to plan fit, charges, and lower perceived value.",
        ),
        (
            "Low-Engagement Casual Viewers",
            "Subscribers with lighter viewing activity, smaller watchlists, or weaker content engagement.",
        ),
        (
            "Billing-Friction Accounts",
            "Accounts where payment method, paperless billing, or charge patterns suggest operational friction.",
        ),
        (
            "Family Feature Users",
            "Households using subtitles, parental controls, or multiple devices where retention depends on feature fit.",
        ),
        (
            "Support-Sensitive Streamers",
            "Subscribers with support contacts or lower satisfaction signals that warrant service recovery.",
        ),
    ],
    "healthcare": [
        (
            "Repeat No-Show Patients",
            "Patients with prior missed visits or attendance patterns that need stronger confirmation workflows.",
        ),
        (
            "Long-Lead Specialty Visits",
            "Appointments booked farther out or in specialty care where schedule drift raises no-show risk.",
        ),
        (
            "Access Barrier Patients",
            "Patients with distance, transportation, insurance, or access constraints that may block attendance.",
        ),
        (
            "Reminder-Sensitive Appointments",
            "Visits where SMS, call, timing, or outreach signals suggest reminder escalation can help.",
        ),
        (
            "New Patient Onboarding",
            "First-time or low-history patients who may need preparation, directions, or intake support.",
        ),
        (
            "Chronic Care Follow-Ups",
            "Patients with clinical complexity where care-continuity messaging can reduce missed appointments.",
        ),
    ],
    "generic": [
        (
            "High-Value Retention Accounts",
            "Important records with elevated risk and enough value to justify high-touch intervention.",
        ),
        (
            "Engagement Recovery Group",
            "Records whose risk is driven by lower activity, adoption, or recent interaction depth.",
        ),
        (
            "Operational Friction Group",
            "Records with billing, scheduling, support, or process signals that suggest fixable friction.",
        ),
        (
            "Price and Plan Fit Group",
            "Records where cost, package fit, or commercial terms appear connected to risk.",
        ),
        (
            "Low-Signal Monitoring Group",
            "Records with moderate risk and fewer strong drivers, best handled with lightweight monitoring.",
        ),
        (
            "Service Recovery Group",
            "Records with satisfaction, support, or experience signals that call for direct recovery action.",
        ),
    ],
}


def segment_template_for_dataframe(dataframe):
    if dataframe is None:
        return "generic"
    template = detect_intervention_template(dataframe)
    if template == "healthcare":
        return "healthcare"
    if template == "streaming":
        return "streaming"
    return "generic"


def segment_index_for_label(cluster_label):
    if cluster_label is None or (isinstance(cluster_label, float) and math.isnan(cluster_label)):
        return None

    text = str(cluster_label).strip()
    if not text:
        return None

    segment_match = re.search(r"\bsegment\s+(\d+)\b", text, flags=re.IGNORECASE)
    if segment_match:
        return max(0, int(segment_match.group(1)) - 1)

    number = as_number(text)
    if number is None:
        return None
    return int(number)


def segment_profile_for_label(cluster_label, dataframe=None, template_key=None):
    template = template_key or segment_template_for_dataframe(dataframe)
    profiles = SYNTHETIC_SEGMENT_PROFILES.get(template) or SYNTHETIC_SEGMENT_PROFILES["generic"]
    index = segment_index_for_label(cluster_label)
    if index is None:
        return str(cluster_label or "Unassigned Segment"), ""
    name, description = profiles[index % len(profiles)]
    return name, description


def segment_name_for_label(cluster_label, dataframe=None, template_key=None):
    text = str(cluster_label or "").strip()
    if text and segment_index_for_label(text) is None:
        return text
    name, _ = segment_profile_for_label(cluster_label, dataframe, template_key)
    return name


def segment_description_for_label(cluster_label, dataframe=None, template_key=None):
    _, description = segment_profile_for_label(cluster_label, dataframe, template_key)
    return description


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


def factor_chart_data_for_display(chart_data, max_slices=4, min_share=0.05):
    if chart_data.empty:
        return chart_data

    chart_data = chart_data.copy().sort_values("size", ascending=False).reset_index(drop=True)
    total_size = chart_data["size"].sum()
    if total_size <= 0:
        return chart_data

    chart_data["_share"] = chart_data["size"] / total_size
    keep_mask = (chart_data.index < max_slices) & (chart_data["_share"] >= min_share)
    kept_rows = chart_data[keep_mask].drop(columns=["_share"])
    other_rows = chart_data[~keep_mask]

    if other_rows.empty:
        return kept_rows

    other_size = other_rows["size"].sum()
    if other_size <= 0:
        return kept_rows

    return pd.concat(
        [
            kept_rows,
            pd.DataFrame([{"label": "Other", "size": other_size}]),
        ],
        ignore_index=True,
    )


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
    segment_template = segment_template_for_dataframe(result)

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

    result["cluster_label"] = [
        segment_name_for_label(label, result, segment_template)
        for label in cluster_labels
    ]
    result["cluster_description"] = [
        cluster_description_for_label(label, cluster_descriptions)
        or segment_description_for_label(label, result, segment_template)
        for label in cluster_labels
    ]

    sort_column = probability_sort_column(result)
    high_cutoff = None
    medium_cutoff = None
    use_relative_thresholds = False
    if sort_column:
        sort_values = pd.to_numeric(result[sort_column], errors="coerce")
        if sort_values.notna().any():
            normalized_sort_values = sort_values.where(sort_values <= 1, sort_values / 100)
            high_cutoff = normalized_sort_values.quantile(0.8)
            medium_cutoff = normalized_sort_values.quantile(0.3)
            use_relative_thresholds = (
                normalized_sort_values.nunique(dropna=True) > 2
                and high_cutoff is not None
                and medium_cutoff is not None
                and high_cutoff > medium_cutoff
            )
            result = result.assign(_risk_sort=sort_values).sort_values("_risk_sort", ascending=False)

    interventions = []
    intervention_counts = Counter()
    cluster_intervention_counts = Counter()
    intervention_catalog = current_intervention_catalog(segment_template)
    for row_rank, (_, row) in enumerate(result.iterrows()):
        factor_prefix, factor_description, factor_score = primary_factor_from_row(row, factor_metadata)
        probability = risk_probability(row)
        if use_relative_thresholds:
            urgency = urgency_from_relative_risk(probability, high_cutoff, medium_cutoff)
        else:
            urgency = None

        if urgency is None:
            urgency = fallback_urgency_from_rank(row_rank, len(result))

        explanation_context = row_prediction_explanation_context(row)
        class_explanation_context = prediction_class_explanation_context(row)
        model_driver_context = feature_weight_context(row)
        context = " ".join(
            part
            for part in (
                factor_description,
                row.get("cluster_description", ""),
                model_driver_context,
                explanation_context,
                class_explanation_context,
            )
            if pd.notna(part) and str(part).strip()
        )
        row_context = row.to_dict()
        row_context["primary_factor_description"] = factor_description
        match_score, intervention, match_source = choose_intervention(
            row_context,
            context,
            intervention_counts,
            cluster_intervention_counts,
            intervention_catalog,
        )
        intervention_counts[intervention["name"]] += 1
        cluster_intervention_counts[(str(row.get("cluster_label", "")), intervention["name"])] += 1
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
