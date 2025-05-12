"""Handler Tab

• upload Scheduler pickle
• choose obs / act / reward handler IDs
• per‑JobTemplate sampling strategy (repetition & deadline ~ N(μ,σ))
• simple hist plot preview per template
• produces a config dict for downstream use
"""

from __future__ import annotations

import io, pickle, random
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from rl_scheduler.scheduler import Scheduler

__all__ = ["render_handler_tab"]

DEFAULT_OBS = ["DefaultObs"]
DEFAULT_ACT = ["DefaultAct"]
DEFAULT_RWD = ["DefaultReward"]


# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------

def _preview_hist(mu: float, sigma: float, title: str):
    """Render a mini‑histogram for the given normal params."""
    samples = np.random.normal(mu, sigma, size=200).astype(int)
    fig, ax = plt.subplots(figsize=(2.5, 1.6))
    ax.hist(samples, bins=15, edgecolor="black")
    ax.set_title(title, fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    st.pyplot(fig, clear_figure=True)


# ---------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------

def render_handler_tab() -> Dict[str, Any]:
    st.subheader("Environment Handler & Sampling Strategy")

    # 1️⃣ Scheduler pickle upload
    sched_file = st.file_uploader("Scheduler (.pkl)", type=["pkl"], key="hdl_sched")

    if sched_file is None:
        st.info("Upload a scheduler pickle first to configure sampling strategy.")
        return {}

    try:
        scheduler: Scheduler = pickle.load(sched_file)
    except Exception as e:
        st.error(f"Failed to load pickle: {e}")
        return {}

    # Populate job template IDs
    job_ids = [jt.job_template_id if hasattr(jt, "job_template_id") else jt["job_template_id"]  # type: ignore
               for jt in scheduler.job_templates]

    # 2️⃣ Handler selection
    obs = st.selectbox("Observation Handler", DEFAULT_OBS, key="hdl_obs")
    act = st.selectbox("Action Handler",      DEFAULT_ACT, key="hdl_act")
    rwd = st.selectbox("Reward Function",     DEFAULT_RWD, key="hdl_rew")

    # 3️⃣ Per‑template sampling parameters
    st.divider()
    st.markdown("### Sampling (Normal distribution, rounded to int)")

    sampling: Dict[str, Dict[str, Dict[str, float]]] = {}

    for jid in job_ids:
        with st.expander(f"Job Template {jid}"):
            col1, col2 = st.columns(2)
            with col1:
                rep_mu  = st.number_input("Repetition μ", 1, 100, 3, key=f"rep_mu_{jid}")
                rep_sd  = st.number_input("Repetition σ", 0.1, 50.0, 1.0, step=0.1, key=f"rep_sd_{jid}")
            with col2:
                dl_mu   = st.number_input("Deadline μ", 1, 500, 20, key=f"dl_mu_{jid}")
                dl_sd   = st.number_input("Deadline σ", 0.1, 200.0, 5.0, step=0.1, key=f"dl_sd_{jid}")

            # quick histogram preview
            st.caption("Preview (200 samples)")
            _preview_hist(rep_mu, rep_sd, "Repetition")
            _preview_hist(dl_mu,  dl_sd,  "Deadline")

            sampling[str(jid)] = {
                "repetition": {"mean": rep_mu, "std": rep_sd},
                "deadline":   {"mean": dl_mu,  "std": dl_sd},
            }

    # 4️⃣ Env.pkl builder (scheduler + sampling config)
    st.divider()
    if st.button("Build Env.pkl", key="build_envpkl"):
        env_dict = {
            "scheduler_pickle": sched_file.getvalue(),
            "observation": obs,
            "action": act,
            "reward": rwd,
            "sampling": sampling,
        }
        buf = io.BytesIO()
        pickle.dump(env_dict, buf)
        buf.seek(0)
        st.download_button("Download Env.pkl", data=buf, file_name="env.pkl", mime="application/octet-stream")

    return {
        "observation": obs,
        "action": act,
        "reward": rwd,
        "sampling": sampling,
        "scheduler_file_name": sched_file.name if sched_file else None,
    }
