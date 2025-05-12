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
import yaml

__all__ = ["render_hyperparam_tab"]

# -------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------

def _load_yaml(text: str) -> Dict[str, Any]:
    """Safely parse YAML text → dict; fallback to empty dict on error."""
    try:
        return yaml.safe_load(text) or {}
    except Exception:
        st.warning("⚠️ Invalid YAML — ignored.")
        return {}


# -------------------------------------------------------------------
# Main Render Function
# -------------------------------------------------------------------

def render_hyperparam_tab() -> Dict[str, Any]:
    """Render the Hyper‑Parameter tab and return config dict."""
    st.subheader("Model Algorithm & Hyper‑Parameters")

    # SB3 algorithm documentation links
    DOCS = {
        "PPO": "https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters",
        "DQN": "https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html#parameters",
        "A2C": "https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html#parameters",
        "DDPG": "https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html#parameters",
    }
    # Algorithm selector with link to docs
    algo = st.selectbox("Algorithm", list(DOCS), key="mt_algo")
    st.markdown(
        f"ℹ️ Detailed hyper-parameter list: [SB3 {algo} docs]({DOCS[algo]})",
        unsafe_allow_html=True,
    )

    # --- SB3 YAML hyper‑parameters ----------------------------------
    up_file = st.file_uploader("Upload SB3 hyper‑param YAML (optional)", type=["yml", "yaml"], key="hp_yaml")

    if up_file:
        yaml_text = up_file.getvalue().decode()
        yaml_dict = _load_yaml(yaml_text)
        st.success("YAML loaded ✅")
    else:
        yaml_text = st.text_area("Paste / edit SB3 YAML", height=200, key="hp_text")
        yaml_dict = _load_yaml(yaml_text)

    # Sampling Strategy removed (handled by handler)
    cfg: Dict[str, Any] = {
        "algorithm": algo,
        "sb3_hyperparams": yaml_dict,
    }

    # YAML download ---------------------------------------------------
    yaml_text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    st.download_button(
        "Download train_config.yaml",
        data=yaml_text,
        file_name="train_config.yaml",
        mime="text/yaml",
    )

    # Persist to session_state for downstream tabs
    st.session_state["hparam_cfg"] = cfg
    return cfg
