"""
Training & Visualization Tab
────────────────────────────
• Env.pkl + Hyper-Param(JSON) → background thread 학습
• ▶ Start / ⏹ Stop
• 진행 상황 Plotly 그래프
• 학습 끝나면 artifacts.zip 다운로드
"""
from __future__ import annotations

import io, pickle, threading, time, zipfile, sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from rjsp_gui.services.train_service import (
    make_run_dir,
    train_model,
    zip_artifacts,
)

def render_train_viz_tab():
    """Render the training & visualization UI for RL training."""
    # Initialize session state
    def _init_state():
        for key in ("thread", "run_dir", "start_time", "log_buf"): st.session_state.setdefault(key, None)
    def _clear_state():
        for key in ("thread", "run_dir", "start_time", "log_buf"): st.session_state.pop(key, None)
    _init_state()

    # Configuration inputs (no sidebar)
    st.title("⚙️ Training Configuration")
    agent_name = st.text_input("Agent name", "AGENT_NAME")
    log_root = Path(st.text_input("Log directory", "./logs"))
    total_steps = st.number_input("Total timesteps", 10_000, 10_000_000, 100_000, 10_000)
    eval_freq = st.number_input("Eval freq (steps)", 1_000, 100_000, 10_000, 1_000)
    ckpt_freq = st.number_input("Checkpoint freq (steps)", 1_000, 100_000, 10_000, 1_000)
    env_file = st.file_uploader("Env.pkl (from Handler tab)", type="pkl")
    hp_cfg = st.session_state.get("hparam_cfg")
    if hp_cfg:
        st.success(f"Hyper-params loaded ✔ ({hp_cfg['algorithm']})")
    else:
        st.warning("먼저 Hyper-Parameter 탭에서 설정하세요.")

    # Control buttons
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ Start", disabled=bool(st.session_state.thread))
    stop_btn = col2.button("⏹ Stop", disabled=not st.session_state.thread)

    # Dashboard header and placeholders
    st.header("🚀 RL Training Dashboard")
    progress_bar = st.progress(0.0)
    reward_metric = st.empty()
    tabs = st.tabs(["📈 Live Plot", "📝 Logs"])
    plot_placeholder = tabs[0].empty()
    log_container    = tabs[1].container()

    # Start training in background
    if start_btn:
        if not env_file or not hp_cfg:
            st.error("Env.pkl 과 Hyper-Parameter JSON 이 필요합니다.")
            return
        run_dir = make_run_dir(log_root, agent_name)
        st.session_state.run_dir = run_dir
        st.session_state.start_time = time.time()
        raw_bytes = env_file.read(); env_file.seek(0)
        payload = pickle.load(env_file)
        sched_buf = io.BytesIO(payload.get("scheduler_pickle", raw_bytes))
        contract = ({"sampling": payload.get("sampling")} if isinstance(payload, dict) and payload.get("sampling") else None)
        def _bg_learn():
            log_buf = io.StringIO()
            st.session_state.log_buf = log_buf  # expose to UI

            with redirect_stdout(log_buf), redirect_stderr(log_buf):
                run_path = train_model(
                    scheduler_buf   = sched_buf,
                    contract_file   = contract,
                    hp_cfg          = hp_cfg,
                    agent_name      = agent_name,
                    log_root        = log_root,
                    total_steps     = int(total_steps),
                    eval_freq       = int(eval_freq),
                    ckpt_freq       = int(ckpt_freq),
                    action_handler      = payload.get("action_handler"),
                    observation_handler = payload.get("observation_handler"),
                    reward_handler      = payload.get("reward_handler"),
                    info_handler        = None,
                )
            st.session_state.run_dir = run_path
            st.session_state.thread  = None
        threading.Thread(target=_bg_learn, daemon=True).start(); st.session_state.thread = True
        st.success(f"Training started → {run_dir}")

    # Stop logic
    if stop_btn and st.session_state.thread:
        st.warning("종료 요청… 잠시 기다려주세요.")
        st.session_state.thread = None

    # Live progress update
    if st.session_state.run_dir and isinstance(st.session_state.run_dir, (str, Path)):
        prog_csv = Path(st.session_state.run_dir) / "progress.csv"
        if prog_csv.exists():
            df = pd.read_csv(prog_csv)
            if not df.empty:
                latest = df.iloc[-1]
                reward_metric.metric("Latest reward", f"{latest['reward']:.2f}")
                progress_bar.progress(min(latest["timesteps"] / total_steps, 1.0))

                fig = px.line(
                    df,
                    x="timesteps",
                    y=["reward", "episode_length"],
                    labels={"value": "metric", "variable": "type"},
                    title="Training curves"
                )
                plot_placeholder.plotly_chart(fig, use_container_width=True)

    # ── ⑤‑b Live log stream ─────────────────────────────────────
    if st.session_state.get("log_buf"):
        log_container.code(st.session_state.log_buf.getvalue(), language="bash")

    # Download artifacts when done
    if st.session_state.run_dir and st.session_state.thread is None:
        zbuf = zip_artifacts(Path(st.session_state.run_dir))
        st.download_button("Download artifacts.zip", zbuf, file_name="artifacts.zip", mime="application/zip")
        st.success("Training finished ✅")
        _clear_state()