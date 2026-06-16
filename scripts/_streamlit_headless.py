"""Minimal Streamlit stub for headless Wood Wide API scripts."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


class HeadlessStop(RuntimeError):
    """Raised when mocked st.stop() is called."""


class SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def install_streamlit_mock() -> None:
    if "streamlit" in sys.modules and getattr(sys.modules["streamlit"], "_headless_mock", False):
        return

    mock_st = MagicMock()
    mock_st._headless_mock = True
    mock_st.session_state = SessionState(
        model_wait_timeout_seconds=60 * 60,
        pending_model_jobs={},
        model_ids={},
        churn_feature_entries_by_label={},
        noshow_feature_entries_by_label={},
        manual_explanations_by_job={},
        manual_explanation_selection_versions={},
    )

    def stop():
        raise HeadlessStop("st.stop() called in headless mode")

    mock_st.stop = stop
    mock_st.empty.return_value = SimpleNamespace(write=lambda *args, **kwargs: None)
    mock_st.secrets = {}
    mock_st.write = print
    mock_st.error = lambda msg: print(f"[error] {msg}")
    mock_st.warning = lambda msg: print(f"[warning] {msg}")
    mock_st.info = lambda msg: print(f"[info] {msg}")
    mock_st.success = lambda msg: print(f"[success] {msg}")

    sys.modules["streamlit"] = mock_st
