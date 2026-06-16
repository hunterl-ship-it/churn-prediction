"""SQLite cache — re-exports from core."""

from woodwide.core import (
    cache_delete_inference_job,
    cache_delete_pending_job,
    cache_delete_ready_model,
    cache_get_dataset_id,
    cache_get_inference_job,
    cache_load_pending_jobs,
    cache_load_ready_models,
    cache_load_ready_models_with_prefix,
    cache_save_dataset,
    cache_save_inference_job,
    cache_save_pending_job,
    cache_save_ready_model,
    init_local_cache,
)

__all__ = [
    "cache_delete_inference_job",
    "cache_delete_pending_job",
    "cache_delete_ready_model",
    "cache_get_dataset_id",
    "cache_get_inference_job",
    "cache_load_pending_jobs",
    "cache_load_ready_models",
    "cache_load_ready_models_with_prefix",
    "cache_save_dataset",
    "cache_save_inference_job",
    "cache_save_pending_job",
    "cache_save_ready_model",
    "init_local_cache",
]
