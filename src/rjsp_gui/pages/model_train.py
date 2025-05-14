import streamlit as st
import sys
from pathlib import Path

# src 경로 PYTHONPATH 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 탭‑별 렌더러
from rjsp_gui.pages.model_train.hyperparam import render_hyperparam_tab
from rjsp_gui.pages.model_train.handler    import render_handler_tab
from rjsp_gui.pages.model_train.train_viz  import render_train_viz_tab

st.title("Model Train Page")

tab_hp, tab_hdl, tab_viz = st.tabs(
    ["Hyper Parameter", "Handler", "Training Visualize"]
)

with tab_hp:
    hparam_cfg = render_hyperparam_tab()

with tab_hdl:
    handler_cfg = render_handler_tab()

with tab_viz:
    render_train_viz_tab()
