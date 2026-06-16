"""Simulated live processing for instant demo mode."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable

import streamlit as st

INSTANT_PLAYBACK_CSS = """
<style>
@keyframes instantPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.55; }
}
.instant-pipeline-active {
    animation: instantPulse 1.4s ease-in-out infinite;
    color: #175cd3;
    font-weight: 600;
}
.instant-stage-complete {
    color: #027a48;
}
div[data-testid="stStatusWidget"] {
    border-radius: 8px;
}
</style>
"""


def apply_instant_playback_style():
    st.markdown(INSTANT_PLAYBACK_CSS, unsafe_allow_html=True)


@contextmanager
def pipeline_progress(total_steps: int):
    container = st.container()
    with container:
        bar = st.progress(0.0, text="Initializing demo pipeline...")
    state = {"step": 0, "total": total_steps, "bar": bar}

    def advance(label: str):
        state["step"] += 1
        fraction = min(1.0, state["step"] / state["total"])
        state["bar"].progress(fraction, text=label)

    yield advance
    state["bar"].progress(1.0, text="Analysis complete")
    time.sleep(0.25)


@contextmanager
def simulate_stage(title: str, steps: list[tuple[str, float]] | None = None, skip: bool = False):
    if skip:
        yield
        return

    with st.status(title, expanded=True) as status:
        if steps:
            for label, delay in steps:
                st.markdown(f'<span class="instant-pipeline-active">{label}</span>', unsafe_allow_html=True)
                time.sleep(delay)
        else:
            time.sleep(0.4)
        status.update(label=f"{title} — complete", state="complete")
    yield


def _ease_out_quad(t: float) -> float:
    return 1 - (1 - t) * (1 - t)


def animate_metric(
    column,
    label: str,
    target: float,
    formatter: Callable[[float], str],
    duration: float = 0.9,
    steps: int = 14,
    skip: bool = False,
):
    if skip:
        column.metric(label, formatter(target))
        return

    placeholder = column.empty()
    for step in range(steps + 1):
        progress = _ease_out_quad(step / steps)
        placeholder.metric(label, formatter(target * progress))
        if step < steps:
            time.sleep(duration / steps)


def animate_metrics_row(
    columns: list,
    specs: list[tuple[str, float, Callable[[float], str]]],
    duration: float = 0.9,
    skip: bool = False,
):
    if skip:
        for column, (label, target, formatter) in zip(columns, specs):
            column.metric(label, formatter(target))
        return

    placeholders = [column.empty() for column in columns]
    steps = 14
    for step in range(steps + 1):
        progress = _ease_out_quad(step / steps)
        for placeholder, (label, target, formatter) in zip(placeholders, specs):
            placeholder.metric(label, formatter(target * progress))
        if step < steps:
            time.sleep(duration / steps)


def animate_integer_metric(column, label: str, target: int, duration: float = 0.8, skip: bool = False):
    animate_metric(
        column,
        label,
        float(target),
        lambda value: f"{int(round(value)):,}",
        duration=duration,
        skip=skip,
    )


def typewriter_caption(text: str, delay: float = 0.02, skip: bool = False):
    if skip:
        st.caption(text)
        return

    placeholder = st.empty()
    shown = ""
    for character in text:
        shown += character
        placeholder.caption(shown + "▌")
        time.sleep(delay)
    placeholder.caption(text)


def reveal_dataframe(dataframe, height: int = 460, skip: bool = False):
    if skip or dataframe.empty:
        st.dataframe(dataframe, use_container_width=True, height=height)
        return

    placeholder = st.empty()
    with placeholder.container():
        with st.spinner("Rendering results..."):
            time.sleep(0.45)
    placeholder.dataframe(dataframe, use_container_width=True, height=height)
