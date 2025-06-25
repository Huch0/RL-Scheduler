import streamlit as st
import sys
import os
import sys
from pathlib import Path

from rjsp_gui.pages.config.job import render_job_config
from rjsp_gui.pages.config.machine import render_machine_config
from rjsp_gui.pages.config.contract import render_contract_config
from rjsp_gui.services.scheduler_service import build_scheduler_pickle

st.title("Scheduler Setup Page")

# 서브페이지 탭 생성
tab_job, tab_machine, tab_contract = st.tabs(["Job", "Machine", "Contract"])

# --- Job 탭 ---
with tab_job:
    job_saved = render_job_config()

# --- Machine 탭 ---
with tab_machine:
    machine_saved = render_machine_config()

# --- Contract 탭 ---
with tab_contract:
    contract_saved = render_contract_config()

# --- Export Scheduler ---
st.subheader("Export Scheduler")
# 사용자 지정 파일명 입력
file_name = st.text_input("Scheduler file name", "scheduler.pkl", key="sched_file_name")
try:
    buf = build_scheduler_pickle()
    st.download_button(
        f"Download {file_name}",
        data=buf,
        file_name=file_name,
        mime="application/octet-stream",
        key="download_scheduler"
    )
except Exception as e:
    st.error(str(e))
