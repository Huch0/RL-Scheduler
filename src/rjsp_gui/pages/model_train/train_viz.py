from __future__ import annotations

import io, pickle, threading, time
from contextlib import redirect_stdout, redirect_stderr
from streamlit.runtime.scriptrunner import add_script_run_ctx, RerunException, RerunData
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from rjsp_gui.services.train_service import make_run_dir, train_model, zip_artifacts


def render_train_viz_tab():
    """Render the training & visualization UI for RL training."""
    # Initialize/clear session state
    def _init_state():
        for key in ("thread", "run_dir", "start_time", "log_buf"):  
            st.session_state.setdefault(key, None)
    def _clear_state():
        for key in ("thread", "run_dir", "start_time", "log_buf"):  
            st.session_state.pop(key, None)
    _init_state()

    # ── Configuration Inputs ─────────────────────────────────
    st.title("⚙️ Training Configuration")
    agent_name  = st.text_input("Agent name", "AGENT_NAME")
    log_root    = Path(st.text_input("Log directory", "./logs"))
    total_steps = st.number_input("Total timesteps", 10_000, 10_000_000, 100_000, 10_000)
    eval_freq   = st.number_input("Eval freq (steps)", 1_000, 100_000, 10_000, 1_000)
    ckpt_freq   = st.number_input("Checkpoint freq (steps)", 1_000, 100_000, 10_000, 1_000)
    env_file    = st.file_uploader("Env.pkl (from Handler tab)", type="pkl")
    hp_cfg      = st.session_state.get("hparam_cfg")
    if hp_cfg:
        st.success(f"Hyper-params loaded ✔ ({hp_cfg['algorithm']})")
    else:
        st.warning("먼저 Hyper-Parameter 탭에서 설정하세요.")

    # Control buttons
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ Start", disabled=bool(st.session_state.get("thread")))
    stop_btn  = col2.button("⏹ Stop",  disabled=not bool(st.session_state.get("thread")))

    # ── Dashboard Placeholders ───────────────────────────────
    st.header("🚀 RL Training Dashboard")
    progress_bar    = st.progress(0.0)
    reward_metric   = st.empty()
    tabs            = st.tabs(["📈 Live Plot", "📝 Logs"])
    plot_placeholder = tabs[0].empty()
    log_container    = tabs[1].container()

    # ── Start Training ───────────────────────────────────────
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
        contract = ("sampling" in payload and {"sampling": payload.get("sampling")}) or None

        # Shared log buffer
        log_buf = io.StringIO()
        st.session_state.log_buf = log_buf

        def _bg_learn():
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

        thread = threading.Thread(target=_bg_learn, daemon=True)
        add_script_run_ctx(thread)
        thread.start()
        st.session_state.thread = thread
        st.success(f"Training started → {run_dir}")

    # ── Stop Logic ────────────────────────────────────────────
    if stop_btn and st.session_state.get("thread"):
        st.warning("종료 요청… 잠시 기다려주세요.")
        st.session_state.thread = None

    # ── Live Progress & Metrics ──────────────────────────────
    run_dir = st.session_state.get("run_dir")
    if run_dir:
        prog_csv = Path(run_dir) / "progress.csv"
        if prog_csv.exists():
            try:
                df = pd.read_csv(prog_csv)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            # Adapt SB3 default progress column names
            if 'time/total_timesteps' in df.columns:
                df = df.rename(columns={
                    'time/total_timesteps': 'timesteps',
                    'rollout/ep_rew_mean':   'reward',
                    'rollout/ep_len_mean':   'episode_length'
                })
            if not df.empty:
                last = df.iloc[-1]
                reward_metric.metric("Latest reward", f"{last['reward']:.2f}")
                progress_bar.progress(min(last['timesteps'] / total_steps, 1.0))
                fig = px.line(
                    df,
                    x='timesteps',
                    y=['reward', 'episode_length'],
                    labels={'value': 'Metric', 'variable': 'Type'},
                    title='Training Curves'
                )
                plot_placeholder.plotly_chart(fig, use_container_width=True)

    # ── Live Logs ─────────────────────────────────────────────
    if st.session_state.log_buf:
        log_container.text_area(
            "Logs",
            value=st.session_state.log_buf.getvalue(),
            height=500,
            disabled=True
        )

    # ── Download Artifacts on Completion ──────────────────────
    if run_dir and not st.session_state.get("thread"):
        zbuf = zip_artifacts(Path(run_dir))
        st.download_button("Download artifacts.zip", zbuf, file_name="artifacts.zip", mime="application/zip")
        st.success("Training finished ✅")
        _clear_state()

    # ── Auto-refresh While Training ───────────────────────────
    if st.session_state.get("thread"):
        time.sleep(1)
        st.rerun()
