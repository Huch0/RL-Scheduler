"""Plotly helpers for profit visualisations.
Includes deterministic curve preview and (legacy) sampling‑based preview.
"""
from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import plotly.graph_objs as go
import streamlit as st

__all__ = [
    "plot_profit_curves",
    "plot_profit_samples",
]

# -------------------------------------------------------------
# 📈 Deterministic curves (used by Contract Config tab)
# -------------------------------------------------------------

def plot_profit_curves(
    prices: Sequence[float],
    deadlines: Sequence[int],
    penalties: Sequence[float],
    *,
    key: Optional[str] = None,
    title: str = "Profit curves",
) -> None:
    """Render profit‑versus‑time lines for exact values."""
    if not prices:
        st.info("No instances to preview.")
        return

    traces = []
    for pr, dl, lp in zip(prices, deadlines, penalties):
        x = [0, dl]
        y = [pr, pr]
        if lp > 0:
            x.append(dl + pr / lp)
            y.append(0)
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"DL={dl}, P={pr:.0f}, LP={lp:.0f}",
            )
        )

    fig = go.Figure(
        traces,
        layout=dict(
            title=title,
            xaxis_title="Time",
            yaxis_title="Profit",
            template="plotly_white",
            showlegend=True,
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# -------------------------------------------------------------
# 📈 Sampling‑based curves (used by Sampling tab)
# -------------------------------------------------------------

def plot_profit_samples(
    mu_rep: float,
    sd_rep: float,
    mu_dl: float,
    sd_dl: float,
    mu_pr: float,
    sd_pr: float,
    mu_lp: float,
    sd_lp: float,
    *,
    key: Optional[str] = None,
    n_show: int = 5,
) -> None:
    """Sample instances from normal distributions and draw curves."""
    rep_count = max(1, int(np.random.normal(mu_rep, sd_rep, 1)[0]))
    dls = np.random.normal(mu_dl, sd_dl, rep_count).astype(int)
    prices = np.random.normal(mu_pr, sd_pr, rep_count)
    lps = np.random.normal(mu_lp, sd_lp, rep_count)

    dls = np.clip(dls, 1, None)
    prices = np.clip(prices, 0, None)
    lps = np.clip(lps, 0.1, None)

    traces = []
    for dl, pr, lp in zip(dls[:n_show], prices[:n_show], lps[:n_show]):
        x = [0, dl, dl + pr / lp]
        y = [pr, pr, 0]
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"DL={dl}, P={pr:.0f}, LP={lp:.0f}",
            )
        )

    fig = go.Figure(
        traces,
        layout=dict(
            title="Sampled profit curves",
            xaxis_title="Time",
            yaxis_title="Profit",
            template="plotly_white",
            showlegend=True,
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)