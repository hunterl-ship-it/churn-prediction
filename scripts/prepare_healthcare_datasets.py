#!/usr/bin/env python3
"""Generate synthetic patient appointment no-show demo datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

INSURANCE_TYPES = ["Commercial", "Medicare", "Medicaid", "Self-Pay"]
SPECIALTIES = [
    "Primary Care",
    "Cardiology",
    "Orthopedics",
    "Dermatology",
    "Pediatrics",
    "Mental Health",
]
APPOINTMENT_TYPES = ["New Patient", "Follow-up", "Procedure"]
GENDERS = ["Female", "Male", "Non-binary"]
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
YES_NO = ["Yes", "No"]


def _random_patient_ids(count: int, rng: np.random.Generator) -> list[str]:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    ids = []
    for _ in range(count):
        ids.append("PT" + "".join(rng.choice(list(alphabet), size=10)))
    return ids


def generate_appointments(row_count: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 90, size=row_count)
    gender = rng.choice(GENDERS, size=row_count, p=[0.52, 0.46, 0.02])
    insurance = rng.choice(INSURANCE_TYPES, size=row_count, p=[0.42, 0.28, 0.18, 0.12])
    specialty = rng.choice(SPECIALTIES, size=row_count, p=[0.34, 0.14, 0.12, 0.10, 0.18, 0.12])
    appointment_type = rng.choice(APPOINTMENT_TYPES, size=row_count, p=[0.22, 0.58, 0.20])
    lead_time_days = rng.integers(1, 180, size=row_count)
    wait_time_days = np.clip(rng.integers(0, 90, size=row_count) + lead_time_days // 10, 0, 120)
    day_of_week = rng.choice(DAYS_OF_WEEK, size=row_count, p=[0.22, 0.20, 0.20, 0.18, 0.15, 0.05])
    appointment_hour = rng.integers(7, 18, size=row_count)
    distance_miles = np.round(rng.gamma(shape=2.2, scale=4.5, size=row_count), 1)
    previous_appointments = rng.integers(0, 40, size=row_count)
    previous_no_shows = np.minimum(
        previous_appointments,
        rng.poisson(lam=1.4, size=row_count) + (previous_appointments > 10).astype(int),
    )
    sms_reminder = rng.choice(YES_NO, size=row_count, p=[0.78, 0.22])
    call_reminder = rng.choice(YES_NO, size=row_count, p=[0.55, 0.45])
    transportation_barrier = rng.choice(YES_NO, size=row_count, p=[0.12, 0.88])
    chronic_condition_count = rng.integers(0, 6, size=row_count)

    logit = (
        -3.4
        + 0.014 * lead_time_days
        + 0.28 * (insurance == "Medicaid")
        + 0.22 * (insurance == "Self-Pay")
        + 0.18 * (specialty == "Mental Health")
        + 0.14 * (appointment_type == "New Patient")
        + 0.12 * (day_of_week == "Monday")
        + 0.10 * (day_of_week == "Saturday")
        + 0.008 * distance_miles
        + 0.35 * previous_no_shows
        - 0.012 * previous_appointments
        + 0.42 * (transportation_barrier == "Yes")
        - 0.38 * (sms_reminder == "Yes")
        - 0.18 * (call_reminder == "Yes")
        + 0.06 * (age < 30)
        + 0.03 * chronic_condition_count
        + rng.normal(0, 0.35, size=row_count)
    )
    no_show_probability = 1 / (1 + np.exp(-logit))
    no_show = (rng.random(row_count) < no_show_probability).astype(int)

    return pd.DataFrame(
        {
            "PatientID": _random_patient_ids(row_count, rng),
            "Age": age,
            "Gender": gender,
            "InsuranceType": insurance,
            "Specialty": specialty,
            "AppointmentType": appointment_type,
            "LeadTimeDays": lead_time_days,
            "WaitTimeDays": wait_time_days,
            "DayOfWeek": day_of_week,
            "AppointmentHour": appointment_hour,
            "DistanceToClinicMiles": distance_miles,
            "PreviousAppointments": previous_appointments,
            "PreviousNoShows": previous_no_shows,
            "SMSReminderSent": sms_reminder,
            "CallReminderSent": call_reminder,
            "TransportationBarrier": transportation_barrier,
            "ChronicConditionCount": chronic_condition_count,
            "no_show": no_show,
        }
    )


def split_train_eval_test(
    data: pd.DataFrame,
    label_column: str,
    train_fraction: float = 0.8,
    eval_rows: int = 15_000,
    seed: int = 42,
):
    split_index = int(train_fraction * len(data))
    train = data.iloc[:split_index].copy()
    remainder = data.iloc[split_index:].copy()

    eval_data = remainder
    if len(remainder) > eval_rows:
        fraction = min(1.0, eval_rows / len(remainder))
        eval_parts = []
        for _, group in remainder.groupby(label_column, sort=False):
            eval_parts.append(group.sample(frac=fraction, random_state=seed))
        eval_data = pd.concat(eval_parts, ignore_index=True).head(eval_rows)

    test = remainder.drop(columns=[label_column])
    return train, eval_data, test


def split_train_test(data: pd.DataFrame, train_fraction: float = 0.8, label_column: str = "no_show"):
    train, eval_data, test = split_train_eval_test(data, label_column, train_fraction)
    return train, test


def main():
    parser = argparse.ArgumentParser(description="Prepare healthcare no-show demo CSVs")
    parser.add_argument(
        "--total-rows",
        type=int,
        default=273_163,
        help="Total appointment rows to generate (default matches churn demo scale; use 990000 for full demo).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of rows reserved for training.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/healthcare",
        help="Directory for train.csv and test.csv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-rows", type=int, default=15_000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = generate_appointments(args.total_rows, seed=args.seed)
    train, eval_data, test = split_train_eval_test(
        data,
        "no_show",
        train_fraction=args.train_fraction,
        eval_rows=args.eval_rows,
        seed=args.seed,
    )

    train_path = output_dir / "train.csv"
    eval_path = output_dir / "eval.csv"
    test_path = output_dir / "test.csv"
    train.to_csv(train_path, index=False)
    eval_data.to_csv(eval_path, index=False)
    test.to_csv(test_path, index=False)

    no_show_rate = train["no_show"].mean()
    print(f"Wrote {len(train):,} training rows -> {train_path}")
    print(f"Wrote {len(eval_data):,} eval rows -> {eval_path}")
    print(f"Wrote {len(test):,} scoring rows -> {test_path}")
    print(f"Training no-show rate: {no_show_rate:.1%}")
    print("For a ~990k-row sales demo: --total-rows 990000")


if __name__ == "__main__":
    main()
