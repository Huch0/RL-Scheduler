import streamlit as st
from streamlit_sortables import sort_items

st.title("Config Page")

# --- Job Template 1 ---
st.subheader("Job Template 1")
col1, col2 = st.columns(2)
with col1:
    job1_type = st.selectbox("Type", ["Type A", "Type B", "Type C"], key="job1_type")
    job1_duration = st.number_input(
        "Duration", min_value=0, value=5, step=1, key="job1_duration"
    )
with col2:
    job1_color = st.color_picker("Color", "#00FF00", key="job1_color")

# --- Job Template 2 ---
st.subheader("Job Template 2")
col3, col4 = st.columns(2)
with col3:
    job2_type = st.selectbox("Type", ["Type A", "Type B", "Type C"], key="job2_type")
    job2_duration = st.number_input(
        "Duration", min_value=0, value=10, step=1, key="job2_duration"
    )
with col4:
    job2_color = st.color_picker("Color", "#FF0000", key="job2_color")

# --- Operation palette & canvas ---
left_col, right_col = st.columns([1, 3])
if "operation_templates" not in st.session_state:
    st.session_state.operation_templates = []
if "job_sequence" not in st.session_state:
    st.session_state.job_sequence = []

with left_col:
    st.subheader("Operation Template")
    op_type = st.text_input("Type", key="op_type")
    op_duration = st.number_input(
        "Duration", min_value=0, value=1, step=1, key="op_duration"
    )
    op_color = st.color_picker("Color", "#FF0000", key="op_color")
    if st.button("Add Operation Template", key="add_operation"):
        st.session_state.operation_templates.append(
            {"type": op_type, "duration": op_duration, "color": op_color}
        )
    st.markdown("**Palette:**")
    for op in st.session_state.operation_templates:
        st.markdown(f"- {op['type']} ({op['duration']})")

with right_col:
    st.subheader("Build Job Sequence")
    if st.session_state.operation_templates:
        seq = sort_items(
            [
                f"{op['type']} ({op['duration']})"
                for op in st.session_state.operation_templates
            ],
            key="job_sequence",
        )
        st.session_state.job_sequence = seq
        st.write("Job sequence:", seq)
    else:
        st.info("Add operation templates to the palette to build your job.")

# --- Export ---
if st.button("Export Configuration", key="export_config"):
    st.success("Configuration has been exported successfully.")
