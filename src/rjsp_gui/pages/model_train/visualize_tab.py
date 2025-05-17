from __future__ import annotations
from pathlib import Path
from typing import Any, List, Dict

import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
from rjsp_gui.services.train_service import load_tb_scalars
import time

__all__ = ["render_visualize_tab"]

def _make_subplots(tags: List[str], pivot_df: pd.DataFrame):
    fig = make_subplots(rows=len(tags), cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for i, tag in enumerate(tags, start=1):
        if tag not in pivot_df.columns:
            continue
        fig.add_scatter(x=pivot_df["step"], y=pivot_df[tag], mode="lines", connectgaps=True, name=tag, row=i, col=1)
        fig.update_yaxes(title_text=tag, row=i, col=1)
    fig.update_xaxes(title_text="step", row=len(tags), col=1)
    fig.update_layout(height=200 * len(tags), showlegend=False, title="Training Scalars")
    return fig

def render_visualize_tab() -> None:
    # ── RL Training Dashboard & Scalars ──────────────────────────
    st.header("🚀 RL Training Dashboard")
    run_dir = st.session_state.get("run_dir")
    if not run_dir:
        st.info("학습을 먼저 실행하세요.")
        return
    # placeholders
    col1, col2 = st.columns([3, 1])
    progress_placeholder = col1.empty()
    reward_placeholder = col2.empty()
    progress_bar = progress_placeholder.progress(0.0)
    # load scalar logs
    tb_dir = Path(run_dir)
    df = load_tb_scalars(tb_dir)
    if df.empty:
        st.info("스칼라 로그가 아직 생성되지 않았습니다.")
        # auto-refresh
        if st.session_state.get("thread"):
            time.sleep(1)
            st.rerun()
        return
    # update metrics
    total_steps = st.session_state.get("total_steps", 1)
    max_step = df["step"].max() if "step" in df.columns else 0
    progress_bar.progress(min(max_step / total_steps, 1.0))
    if "rollout/ep_rew_mean" in df["tag"].values:
        latest_rew = df[df["tag"] == "rollout/ep_rew_mean"].iloc[-1]["value"]
        reward_placeholder.metric("Latest reward", f"{latest_rew:.2f}")
    # scalar plots
    avail_tags = sorted(df["tag"].unique())
    default_tags = [t for t in avail_tags if "reward" in t or "ep_len" in t][:3]
    sel_tags = st.multiselect("Select scalar tags", avail_tags, default=default_tags)
    smooth = st.slider("Smoothing", 0.0, 0.95, 0.6, 0.05)
    if sel_tags:
        sub_df = df[df["tag"].isin(sel_tags)].copy()
        sub_df["smoothed"] = sub_df.groupby("tag")["value"].transform(lambda s: s.ewm(alpha=1 - smooth).mean())
        pivot = sub_df.pivot(index="step", columns="tag", values="smoothed").reset_index().sort_values("step")
        # fill missing values to make continuous lines
        pivot = pivot.ffill()  # use ffill instead of deprecated fillna(method='ffill')
        fig = _make_subplots(sel_tags, pivot)
        st.plotly_chart(fig, use_container_width=True)
    # auto-refresh while training
    if st.session_state.get("thread"):
        time.sleep(1)
        st.rerun()