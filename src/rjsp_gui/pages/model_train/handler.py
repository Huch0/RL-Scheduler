"""Handler Tab

• upload Scheduler pickle
• choose obs / act / reward handler IDs
• per‑JobTemplate sampling strategy (repetition, deadline, price, penalty ~ N(μ,σ))
• Plotly profit‑curve preview per template
• produces a config dict for downstream use
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import streamlit as st

from rl_scheduler.scheduler import Scheduler
from rl_scheduler.envs.registry import _ACTION_HANDLERS, _OBSERVATION_HANDLERS, _REWARD_HANDLERS
from rjsp_gui.pages.config.job import _timeline_html
from rjsp_gui.services.plotly_service import plot_profit_samples  # NEW: Plotly helper

__all__ = ["render_handler_tab"]

OBS_OPTIONS = list(_OBSERVATION_HANDLERS.keys())
ACT_OPTIONS = list(_ACTION_HANDLERS.keys())
RWD_OPTIONS = list(_REWARD_HANDLERS.keys())


# ---------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------

def render_handler_tab() -> Dict[str, Any]:
    """Render sampling/handler configuration tab and return the config dict."""

    st.subheader("Environment Handler & Sampling Strategy")

    # 1️⃣ Scheduler pickle upload ------------------------------------------------
    sched_file = st.file_uploader("Scheduler (.pkl)", type=["pkl"], key="hdl_sched")
    if sched_file is None:
        st.info("Upload a scheduler pickle first to configure sampling strategy.")
        return {}

    try:
        scheduler: Scheduler = pickle.load(sched_file)
    except Exception as exc:
        st.error(f"Failed to load pickle: {exc}")
        return {}

    # Populate job template IDs -------------------------------------------------
    job_ids = [
        jt.job_template_id if hasattr(jt, "job_template_id") else jt["job_template_id"]  # type: ignore
        for jt in scheduler.job_templates
    ]

    # 2️⃣ Handler selection ------------------------------------------------------
    obs = st.selectbox("Observation Handler", OBS_OPTIONS, index=0, key="hdl_obs")
    act = st.selectbox("Action Handler", ACT_OPTIONS, index=0, key="hdl_act")
    rwd = st.selectbox("Reward Function", RWD_OPTIONS, index=0, key="hdl_rew")

    # 3️⃣ Per‑template sampling parameters --------------------------------------
    st.divider()
    st.markdown("### Sampling (Normal distribution, rounded to int)")

    sampling: Dict[str, Dict[str, Dict[str, float]]] = {}

    for jid in job_ids:
        with st.expander(f"Job Template {jid}", expanded=False):
            # 3‑1) Timeline visualization ---------------------------------------
            jt = next(j for j in scheduler.job_templates if j.job_template_id == jid)
            ops = [
                next(o for o in scheduler.operation_templates if o.operation_template_id == oid)
                for oid in getattr(jt, "operation_template_sequence", [])
            ]
            ops_data = [{"duration": op.duration, "type_code": op.type_code} for op in ops]
            if ops_data:
                st.markdown(_timeline_html(ops_data, getattr(jt, "color", "#aaa")), unsafe_allow_html=True)
            else:
                st.warning(f"No operations defined for Job Template {jid}.")

            st.markdown("---")

            # 3‑2) Numeric inputs ----------------------------------------------
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                rep_mu = st.number_input("Repetition μ", 1, 100, 3, key=f"rep_mu_{jid}")
                rep_sd = st.number_input("Repetition σ", 0.0, 50.0, 1.0, 0.1, key=f"rep_sd_{jid}")
            with col2:
                dl_mu = st.number_input("Deadline μ", 1, 500, 20, key=f"dl_mu_{jid}")
                dl_sd = st.number_input("Deadline σ", 0.0, 200.0, 5.0, 0.1, key=f"dl_sd_{jid}")
            with col3:
                price_mu = st.number_input("Price μ", min_value=0.0, value=1000.0, key=f"price_mu_{jid}")
                price_sd = st.number_input("Price σ", min_value=0.0, value=0.0, key=f"price_sd_{jid}")
            with col4:
                pen_mu = st.number_input("Penalty μ", min_value=0.0, value=50.0, key=f"penalty_mu_{jid}")
                pen_sd = st.number_input("Penalty σ", min_value=0.0, value=0.0, key=f"penalty_sd_{jid}")

            # 3‑3) Plotly profit preview ----------------------------------------
            st.caption("Profit‑curve preview (Plotly)")
            plot_profit_samples(
                mu_rep=rep_mu, sd_rep=rep_sd,
                mu_dl=dl_mu,  sd_dl=dl_sd,
                mu_pr=price_mu, sd_pr=price_sd,
                mu_lp=pen_mu,   sd_lp=pen_sd,
            )

            # 3‑4) Save sampling dict -------------------------------------------
            sampling[str(jid)] = {
                "repetition":   {"mean": rep_mu,  "std": rep_sd},
                "deadline":     {"mean": dl_mu,   "std": dl_sd},
                "price":        {"mean": price_mu, "std": price_sd},
                "late_penalty": {"mean": pen_mu,   "std": pen_sd},
            }

    # 4️⃣ Export Sampling JSON ---------------------------------------------------
    if sampling:
        st.divider()
        sample_cfg = st.text_input("Sampling config name", key="sample_cfg")
        sample_ver = st.text_input("Version", "v1", key="sample_ver")
        sampling_json = json.dumps({"sampling": sampling}, indent=4)
        file_name = f"S-{sample_cfg or 'default'}-{sample_ver}.json"
        st.download_button("Download sampling JSON", sampling_json, file_name=file_name, mime="application/json")

    # 5️⃣ Env.pkl builder --------------------------------------------------------
    st.divider()
    if st.button("Build Env.pkl", key="build_envpkl"):
        env_dict = {
            "scheduler_pickle": sched_file.getvalue(),
            "observation_handler": obs,
            "action_handler": act,
            "reward_handler": rwd,
            "sampling": sampling,
        }
        buf = io.BytesIO()
        pickle.dump(env_dict, buf)
        buf.seek(0)
        st.download_button("Download Env.pkl", buf, file_name="env.pkl", mime="application/octet-stream")

    # -------------------------------------------------------------------------
    return {
        "observation_handler": obs,
        "action_handler": act,
        "reward_handler": rwd,
        "sampling": sampling,
        "scheduler_file_name": Path(sched_file.name).name if sched_file else None,
    }
