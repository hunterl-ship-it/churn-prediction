"""Fraud detection workflow helpers."""

from __future__ import annotations

import pandas as pd


LABEL_COLUMNS = ("is_fraud", "risk_label", "is_returned")


FRAUD_LABEL_COLUMN = "is_fraud"


def prepare_fraud_prediction_datasets(raw: pd.DataFrame, train_fraction=0.8):
    """Split ecommerce data into labeled fraud training and label-stripped scoring files."""
    data = raw.copy()
    for column in ("order_id", "order_date", "customer_support_contacts"):
        if column in data.columns:
            data = data.drop(columns=[column])

    if FRAUD_LABEL_COLUMN not in data.columns:
        raise ValueError(f"Raw data must include `{FRAUD_LABEL_COLUMN}` for prediction mode.")

    split_index = int(train_fraction * len(data))
    fraud_train = data.iloc[:split_index].copy()
    fraud_test = data.iloc[split_index:].copy()
    for column in LABEL_COLUMNS:
        if column in fraud_test.columns:
            fraud_test = fraud_test.drop(columns=[column])
    return fraud_train.reset_index(drop=True), fraud_test.reset_index(drop=True)


def prepare_fraud_anomaly_datasets(raw: pd.DataFrame, train_fraction=0.8):
    """Split ecommerce data into normal-only training and label-stripped scoring files."""
    data = raw.copy()
    for column in ("order_id", "order_date", "customer_support_contacts"):
        if column in data.columns:
            data = data.drop(columns=[column])

    split_index = int(train_fraction * len(data))
    fraud_train = data.iloc[:split_index]
    if "is_fraud" in fraud_train.columns:
        fraud_train = fraud_train[fraud_train["is_fraud"] == 0]

    fraud_test = data.iloc[split_index:].copy()
    for frame in (fraud_train, fraud_test):
        for column in LABEL_COLUMNS:
            if column in frame.columns:
                frame.drop(columns=[column], inplace=True)

    return fraud_train.reset_index(drop=True), fraud_test.reset_index(drop=True)


prepare_fraud_datasets = prepare_fraud_anomaly_datasets


def prepare_returns_datasets(raw: pd.DataFrame, train_fraction=0.8):
    """Split ecommerce data into labeled training and unlabeled test files."""
    data = raw.copy()
    for column in ("order_id", "order_date", "customer_support_contacts"):
        if column in data.columns:
            data = data.drop(columns=[column])

    split_index = int(train_fraction * len(data))
    train_set = data.iloc[:split_index].copy()
    test_set = data.iloc[split_index:].copy()
    if "is_returned" in test_set.columns:
        test_set = test_set.drop(columns=["is_returned"])
    if "is_fraud" in test_set.columns:
        test_set = test_set.drop(columns=["is_fraud"])
    if "risk_label" in test_set.columns:
        test_set = test_set.drop(columns=["risk_label"])
    return train_set.reset_index(drop=True), test_set.reset_index(drop=True)
