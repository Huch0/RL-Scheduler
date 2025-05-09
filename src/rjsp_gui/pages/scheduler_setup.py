import streamlit as st
import sys
import os
from pathlib import Path

# 현재 파일의 디렉토리 경로를 구함
current_dir = Path(__file__).parent
# config 디렉토리 경로 추가
sys.path.append(str(current_dir))

from config.job import render_job_config
from config.machine import render_machine_config
from config.contract import render_contract_config
from config.training import render_training_config

st.title("Scheduler Setup Page")

# 서브페이지 탭 생성
tab_job, tab_machine, tab_contract, tab_train = st.tabs(["Job", "Machine", "Contract", "Training"])

# --- Job 탭 ---
with tab_job:
    job_saved = render_job_config()

# --- Machine 탭 ---
with tab_machine:
    machine_saved = render_machine_config()

# --- Contract 탭 ---
with tab_contract:
    contract_saved = render_contract_config()

# --- Training Hyperparameter 탭 ---
with tab_train:
    training_saved = render_training_config()

# --- Export All ---
st.subheader("Export All Configurations")
if st.button("Export All Configurations", key="export_all_configs"):
    st.success("All configurations have been exported successfully.")
