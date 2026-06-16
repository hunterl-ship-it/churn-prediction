"""Inference helpers — re-exports core inference with anomaly support."""

from woodwide.core import (
    cached_inference_result,
    inference_cache_key,
    run_anomaly_inference,
    run_cached_model_inference,
    run_prediction_inference,
)

__all__ = [
    "cached_inference_result",
    "inference_cache_key",
    "run_anomaly_inference",
    "run_cached_model_inference",
    "run_prediction_inference",
]
