import streamlit as st

st.title("Interaction Page")

# --- Control Plane ---
st.subheader("Control Plane")
env_col, agent_col, manual_col, rand_col, log_col = st.columns([2, 2, 2, 1, 2])

# Environment panel
with env_col:
    st.markdown("**Environment**")
    machine = st.selectbox(
        "Machine", ["Machine 1", "Machine 2", "Machine 3"], key="env_machine"
    )
    job = st.selectbox("Job", ["Job 1", "Job 2", "Job 3"], key="env_job")
    operation = st.selectbox("Operation", ["Op A", "Op B"], key="env_operation")
    contract = st.selectbox("Contract", ["C1", "C2"], key="env_contract")
    st.button("Load", key="env_load")
    st.button("Reset", key="env_reset")

# Agent panel
with agent_col:
    st.markdown("**Agent**")
    path = st.selectbox("Path", ["Path A", "Path B"], key="agent_path")
    st.button("Load", key="agent_load")
    st.button("Do it!", key="agent_do")

# Manual Action panel
with manual_col:
    st.markdown("**Manual Action**")
    m_machine = st.selectbox("Machine", ["Machine 1", "Machine 2"], key="man_machine")
    m_job = st.selectbox("Job", ["Job 1", "Job 2"], key="man_job")
    repetition = st.number_input("Repetition", min_value=1, value=1, key="man_rept")
    st.button("Do it!", key="man_do")

# Random Action panel
with rand_col:
    st.markdown("**Random Action**")
    st.button("Do it!", key="rand_do")

# Log panel
with log_col:
    st.markdown("**Log**")
    st.button("Load Action Sequence", key="log_load")
    st.button("Save Current Action Sequence", key="log_save")

st.markdown("---")

# --- Machine Schedule ---
st.subheader("Machine Schedule")
st.markdown("Step **23**")  # dynamic step indicator

# Placeholder for Gantt chart
st.text("<< Gantt chart of machine schedule goes here >>")
st.button("Export", key="export_machine_schedule")

st.markdown("---")

# --- Job Queue ---
st.subheader("Job Queue")
# Placeholder for job queue sequences
st.text("<< Job 1: sequences of operations with expected profit graphs >>")
st.text("<< Job 2: sequences of operations with expected profit graphs >>")
st.button("Export", key="export_job_queue")
