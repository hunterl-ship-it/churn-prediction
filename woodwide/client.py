"""Wood Wide API client — re-exports from core."""

from woodwide.core import (
    api_key,
    base_url,
    get_or_create_dataset,
    get_or_create_dataset_from_bytes,
    get_or_start_model_training,
    headers,
    train_model,
    wait_for_dataset_ready,
    wait_for_training_complete,
)

__all__ = [
    "api_key",
    "base_url",
    "get_or_create_dataset",
    "get_or_create_dataset_from_bytes",
    "get_or_start_model_training",
    "headers",
    "train_model",
    "wait_for_dataset_ready",
    "wait_for_training_complete",
]
