"""Hyper‑Parameter Tab

• 알고리즘 선택 후 **SB3 YAML** 업로드 또는 직접 입력
• 반복 수, Deadline 샘플링(정규분포) 파라미터 입력
• 전체 구성을 YAML 로 다운로드 & 상위 페이지로 반환
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import json

__all__ = ["render_hyperparam_tab"]

# -------------------------------------------------------------------
# Main Render Function
# -------------------------------------------------------------------

def render_hyperparam_tab() -> Dict[str, Any]:
    """Render the Hyperparameter tab and return config dict."""
    st.subheader("Model Algorithm & Hyperparameters")

    # ---------- Algorithm selector ----------
    algo = st.selectbox("Algorithm", ["PPO", "MaskablePPO", "DQN", "A2C", "DDPG"], key="mt_algo")

    # ---------- Upload or edit JSON ----------
    up_file = st.file_uploader("Upload SB3 hyper-param JSON (optional)", type=["json"], key="hp_json")

    default_json = {
        "hyperparameters": {
            "policy": "MultiInputPolicy",
            "n_steps": 2048,
            "batch_size": 256,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "n_epochs": 10,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "device": "auto",
            "seed": 42,
            "verbose": 1,
        }
    }

    if up_file:
        json_text = up_file.getvalue().decode()
        try:
            cfg_dict = json.loads(json_text)
            st.success("JSON loaded ✅")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            cfg_dict = default_json
    else:
        if "hp_text" not in st.session_state:
            st.session_state["hp_text"] = json.dumps(default_json, indent=4)
        json_text = st.text_area("Paste SB3 hyper-parameters JSON", height=260, key="hp_text")
        try:
            cfg_dict = json.loads(json_text)
        except Exception:
            st.warning("⚠️ Invalid JSON — using defaults.")
            cfg_dict = default_json

    # ---------- Save to session_state ----------
    final_cfg = {
        "algorithm": algo,
        "hyperparameters": cfg_dict.get("hyperparameters", {}),
    }
    # Store hyperparameter config for use in train_viz_tab
    st.session_state["hparam_cfg"] = final_cfg

    # Preview and download only when not loading from file
    if not up_file:
        st.markdown("### Generated Hyperparameter JSON")
        st.code(json.dumps(final_cfg, indent=4), language="json")
        st.download_button(
            "Download hyper-parameters JSON",
            data=json.dumps(final_cfg, indent=4),
            file_name=f"hp_{algo}.json",
            mime="application/json",
            key="hp_json_dl",
        )
    # Ensure hyperparams available for training tab
    st.session_state["hparam_cfg"] = final_cfg
    return final_cfg
