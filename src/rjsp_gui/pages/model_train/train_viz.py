from __future__ import annotations

import io, pickle, threading, time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, List

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from rjsp_gui.services.train_service import (
    make_run_dir,
    train_model,
    zip_artifacts,
    load_tb_scalars,
)

TB_SUBDIR = "tb"  # ensure train_model writes tensorboard_log=run_dir / TB_SUBDIR


def _init_state() -> None:
    for k in ("thread", "run_dir", "start_time", "log_buf"):
        st.session_state.setdefault(k, None)


def _clear_state() -> None:
    for k in ("thread", "run_dir", "start_time", "log_buf"):
        st.session_state.pop(k, None)


def _make_subplots(tags: List[str], pivot_df: pd.DataFrame):
    """Return a plotly figure with one subplot per tag."""
    fig = make_subplots(rows=len(tags), cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for i, tag in enumerate(tags, start=1):
        if tag not in pivot_df.columns:
            continue
        fig.add_scatter(
            x=pivot_df["step"],
            y=pivot_df[tag],
            mode="lines",
            name=tag,
            connectgaps=True,
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text=tag, row=i, col=1)
    fig.update_xaxes(title_text="step", row=len(tags), col=1)
    fig.update_layout(height=250 * len(tags), showlegend=False, title="Training Scalars")
    return fig


def render_train_viz_tab() -> None:
    """Streamlit tab to launch training and render live TensorBoard‑style plots."""

    _init_state()

    # ── Configuration UI ────────────────────────────────────────────────
    st.title("⚙️ Training Configuration")
    agent_name = st.text_input("Agent name", "AGENT_NAME")
    log_root = Path(st.text_input("Log directory", "./logs"))
    total_steps = st.number_input("Total timesteps", 10_000, 10_000_000, 100_000, 10_000)
    eval_freq = st.number_input("Eval freq (steps)", 1_000, 100_000, 10_000, 1_000)
    ckpt_freq = st.number_input("Checkpoint freq (steps)", 1_000, 100_000, 10_000, 1_000)
    env_file = st.file_uploader("Env.pkl (from Handler tab)", type="pkl")
    hp_cfg = st.session_state.get("hparam_cfg")
    if hp_cfg:
        st.success(f"Hyper‑params ✔ ({hp_cfg['algorithm']})")
    else:
        st.warning("먼저 Hyperparameter 탭에서 설정하세요.")

    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ Start", disabled=bool(st.session_state.get("thread")))
    stop_btn = col2.button("⏹ Stop", disabled=not bool(st.session_state.get("thread")))

    # ── Dashboard Skeleton ─────────────────────────────────────────────
    st.header("🚀 RL Training Dashboard")
    progress_placeholder = st.empty()
    reward_metric = st.empty()
    progress_bar = progress_placeholder.progress(0.0)

    tab_scalar, tab_logs = st.tabs(["📈 Scalars", "📝 Logs"])
    scalar_area = tab_scalar.container()
    log_container = tab_logs.container()

    # ── Start Training Thread ─────────────────────────────────────────
    if start_btn:
        if not env_file or not hp_cfg:
            st.error("Env.pkl 과 Hyperparameter JSON 이 필요합니다.")
            return

        run_dir = make_run_dir(log_root, agent_name)
        st.session_state.run_dir = run_dir
        st.session_state.start_time = time.time()

        raw_bytes = env_file.read(); env_file.seek(0)
        payload = pickle.load(env_file)
        sched_buf = io.BytesIO(payload.get("scheduler_pickle", raw_bytes))
        contract = ("sampling" in payload and {"sampling": payload.get("sampling")}) or None

        log_buf = io.StringIO(); st.session_state.log_buf = log_buf

        def _bg():
            try:
                with redirect_stdout(log_buf), redirect_stderr(log_buf):
                    run_path = train_model(
                        scheduler_buf=sched_buf,
                        contract_file=contract,
                        hp_cfg=hp_cfg,
                        agent_name=agent_name,
                        log_root=log_root,
                        total_steps=int(total_steps),
                        eval_freq=int(eval_freq),
                        ckpt_freq=int(ckpt_freq),
                        action_handler=payload.get("action_handler"),
                        observation_handler=payload.get("observation_handler"),
                        reward_handler=payload.get("reward_handler"),
                        info_handler=None,
                    )
                st.session_state.run_dir = run_path
            finally:
                st.session_state.thread = None

        th = threading.Thread(target=_bg, daemon=True)
        add_script_run_ctx(th)
        th.start(); st.session_state.thread = th
        st.success(f"Training started → {run_dir}")

    # ── Handle Stop Request ───────────────────────────────────────────
    if stop_btn and st.session_state.get("thread"):
        st.warning("종료 요청… 잠시 기다려주세요.")
        st.session_state.thread = None

    # ── Live Scalars & Progress ────────────────────────────────────────
    run_dir = st.session_state.get("run_dir")
    if run_dir:
        tb_dir = Path(run_dir)
        df = load_tb_scalars(tb_dir)
        if not df.empty:
            avail_tags = sorted(df["tag"].unique())
            default_tags = [t for t in avail_tags if "reward" in t or "ep_len" in t][:3]
            with scalar_area:
                sel_tags = st.multiselect("Scalar tags", avail_tags, default=default_tags)
                smooth = st.slider("Smoothing", 0.0, 0.95, 0.6, 0.05)

            if sel_tags:
                sub_df = df[df["tag"].isin(sel_tags)].copy()
                sub_df["smoothed"] = sub_df.groupby("tag")["value"].transform(
                    lambda s: s.ewm(alpha=1 - smooth).mean()
                )
                pivot = sub_df.pivot(index="step", columns="tag", values="smoothed").reset_index().sort_values("step")
                fig = _make_subplots(sel_tags, pivot)
                scalar_area.plotly_chart(fig, use_container_width=True)

            # progress & reward headline
            if "rollout/ep_rew_mean" in df["tag"].values:
                latest_rew = df[df["tag"] == "rollout/ep_rew_mean"].iloc[-1]["value"]
                reward_metric.metric("Latest reward", f"{latest_rew:.2f}")
            # fallback if tag missing: use max step
            max_step = df["step"].max()
            progress_bar.progress(min(max_step / total_steps, 1.0))
        else:
            scalar_area.info("TensorBoard scalar 로그가 아직 없습니다.")

    # ── Logs Pane ─────────────────────────────────────────────────────
    if st.session_state.get("log_buf"):
        log_container.text_area("Logs", st.session_state.log_buf.getvalue(), height=450, disabled=True)

    # ── Download artifacts when done ──────────────────────────────────
    if run_dir and not st.session_state.get("thread"):
        zbuf = zip_artifacts(Path(run_dir))
        st.download_button("Download artifacts.zip", zbuf, "artifacts.zip", "application/zip")
        st.success("Training finished ✅")
        _clear_state()

    # ── Auto-refresh ─────────────────────────────────────────────────
    if st.session_state.get("thread"):
        time.sleep(1)
        st.rerun()
