from __future__ import annotations
import io, pickle, threading, time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
typing_import = None
import streamlit as st
import html
import streamlit.components.v1 as components  # type: ignore
from streamlit.runtime.scriptrunner import add_script_run_ctx  # type: ignore

from rjsp_gui.services.train_service import make_run_dir, train_model, zip_artifacts

__all__ = ["render_train_tab"]

# initialize/clear session state for training
def _init_state():
    for k in ("thread", "run_dir", "log_buf"): st.session_state.setdefault(k, None)

def _clear_state():
    for k in ("thread", "run_dir", "log_buf"): st.session_state.pop(k, None)

def render_train_tab() -> None:
    """Run training in background and display progress, logs, and artifact download."""
    _init_state()
    # Hyperparameter 확인
    hp_cfg = st.session_state.get("hparam_cfg")
    if hp_cfg:
        st.success(f"Hyperparams ✔ ({hp_cfg['algorithm']})")
    else:
        st.warning("먼저 Hyperparameter 탭에서 설정하세요.")
    # Env.pkl 업로드
    env_file = st.file_uploader("Env.pkl (from Handler tab)", type=["pkl"], key="train_env")
    if env_file:
        st.success("Env.pkl 업로드 완료 ✅")
    # Agent name and training configuration
    agent_name = st.text_input("Agent name", hp_cfg.get("algorithm", "AGENT_NAME"))
    log_root = Path(st.text_input("Log directory", "./logs"))
    total_steps = st.number_input("Total timesteps", 10_000, 10_000_000, 100_000, 10_000)
    eval_freq = st.number_input("Eval freq (steps)", 1_000, 100_000, 10_000, 1_000)
    ckpt_freq = st.number_input("Checkpoint freq (steps)", 1_000, 100_000, 10_000, 1_000)

    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ Start Training", disabled=bool(st.session_state.get("thread")))
    stop_btn = col2.button("⏹ Stop Training", disabled=not bool(st.session_state.get("thread")))

    # Placeholders
    progress_bar = st.progress(0.0)
    log_area = st.empty()

    def _start():
        # Env.pkl 및 hyperparam 확인
        if not env_file or not hp_cfg:
            st.error("Env.pkl 과 Hyperparameter JSON 이 필요합니다.")
            return
        # Load payload
        raw_bytes = env_file.read(); env_file.seek(0)
        payload = pickle.load(env_file)
        sched_buf = io.BytesIO(payload.get("scheduler_pickle", raw_bytes))
        # Wrap sampling config under 'sampling' key for stochastic environment
        contract = ("sampling" in payload and {"sampling": payload.get("sampling")}) or None
        # 시작 디렉터리 및 로그 버퍼 설정
        run_dir = make_run_dir(log_root, agent_name)
        st.session_state.run_dir = run_dir
        # Store training parameters for visualization
        st.session_state.total_steps = total_steps
        st.session_state.eval_freq = eval_freq
        st.session_state.ckpt_freq = ckpt_freq
        st.session_state.log_root = str(log_root)
        log_buf = io.StringIO(); st.session_state.log_buf = log_buf
        def _bg():
            try:
                with redirect_stdout(st.session_state.log_buf), redirect_stderr(st.session_state.log_buf):
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
        st.session_state.thread = th
        th.start()

    if start_btn:
        _start()

    if stop_btn and st.session_state.get("thread"):
        st.session_state.thread = None
        st.warning("Training stop requested...")

    # Show logs
    if st.session_state.get("log_buf"):
        log_val = st.session_state.log_buf.getvalue()
        # escape HTML and preserve line breaks
        safe_log = html.escape(log_val).replace("\n", "<br>")
        # render scrollable div and auto-scroll to bottom
        components.html(
            f"""
            <div id='log' style='overflow-y: scroll; height:300px; white-space: pre-wrap; font-family: monospace;'>
                {safe_log}
            </div>
            <script>
            var ele = document.getElementById('log');
            ele.scrollTop = ele.scrollHeight;
            </script>
            """,
            height=500,
        )

    # Download artifacts when done
    run_dir = st.session_state.get("run_dir")
    if run_dir and not st.session_state.get("thread"):
        zbuf = zip_artifacts(Path(run_dir))
        st.download_button("Download artifacts.zip", zbuf, file_name="artifacts.zip", mime="application/zip")
        st.success("Training finished ✅")
        _clear_state()  # reset session state
