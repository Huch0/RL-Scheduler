import streamlit as st
from rjsp_gui.services.env_service import load_environment, reset_env
import base64
import io

# Use the full browser width instead of Streamlit's default centered layout
st.set_page_config(page_title="Interaction Page", layout="wide")

st.title("Interaction Page")


# Helper to refresh manual action selector options
def _refresh_manual_selectors():
    """
    Populate selector options in the Manual Action panel based on
    the current `env.scheduler` state.
    """
    if "env" not in st.session_state or st.session_state["env"] is None:
        st.session_state["man_machine_options"] = []
        st.session_state["man_job_options"] = []
        st.session_state["man_max_repetition"] = 1
        return

    scheduler = st.session_state["env"].scheduler

    st.session_state["man_machine_options"] = [
        f"Machine {i}" for i, _ in enumerate(scheduler.machine_instances)
    ]

    st.session_state["man_job_options"] = [
        f"Job {i}" for i, _ in enumerate(scheduler.job_instances)
    ]

    # repetitions may vary per job; use max so slider can cover all
    max_rep = max(len(reps) for reps in scheduler.job_instances)
    st.session_state["man_max_repetition"] = max_rep if max_rep > 0 else 1


def _fig_to_base64(fig) -> str:
    """Convert a Matplotlib Figure to a base64 PNG data‑URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# --- Control Plane ---
st.subheader("Control Plane")
env_col, agent_col, manual_col, rand_col, log_col = st.columns([2, 2, 2, 1, 2])

# Environment panel
with env_col:
    st.markdown("**Environment Configuration**")
    machine = st.file_uploader("Machine", type=["json"], key="env_machine")
    job = st.file_uploader("Job", type=["json"], key="env_job")
    operation = st.file_uploader("Operation", type=["json"], key="env_operation")
    contract = st.file_uploader("Contract", type=["json"], key="env_contract")
    use_seed = st.checkbox("Specify random seed", value=False, key="env_use_seed")
    seed = (
        st.number_input("Random Seed", min_value=0, value=0, step=1, key="env_seed")
        if use_seed
        else None
    )
    if st.button("Load", key="env_load"):
        env, err = load_environment(machine, job, operation, contract, seed)
        if err:
            st.error(err)
        else:
            st.session_state["env"] = env
            _refresh_manual_selectors()
            st.success("Environment loaded successfully!")
    if st.button("Reset", key="env_reset"):
        if "env" not in st.session_state or st.session_state["env"] is None:
            st.warning("No environment is loaded yet.")
        else:
            try:
                reset_env(st.session_state["env"], contract_file=contract)
                _refresh_manual_selectors()
                st.success("Environment reset successfully!")
            except Exception as e:
                st.error(f"Failed to reset environment: {e}")

# Agent panel
with agent_col:
    st.markdown("**Agent**")
    path = st.file_uploader(
        "Agent file", type=["pth", "pt", "json", "zip"], key="agent_path"
    )
    st.button("Load", key="agent_load", disabled=True)
    st.button("Do it!", key="agent_do", disabled=True)

# Manual Action panel
with manual_col:
    st.markdown("**Manual Action**")
    m_machine = st.selectbox(
        "Machine",
        st.session_state.get("man_machine_options", []),
        key="man_machine",
    )
    m_job = st.selectbox(
        "Job",
        st.session_state.get("man_job_options", []),
        key="man_job",
    )
    # Determine max repetition for the currently selected job
    if (
        "env" in st.session_state
        and st.session_state["env"] is not None
        and m_job in st.session_state.get("man_job_options", [])
    ):
        job_index = st.session_state["man_job_options"].index(m_job)
        max_rep_for_job = len(
            st.session_state["env"].scheduler.job_instances[job_index]
        )
    else:
        max_rep_for_job = 1

    repetition = st.number_input(
        "Repetition",
        min_value=1,
        max_value=max_rep_for_job,
        value=min(st.session_state.get("man_rept", 1), max_rep_for_job),
        step=1,
        key="man_rept",
    )
    st.button("Do it!", key="man_do", disabled=True)

# Random Action panel
with rand_col:
    st.markdown("**Random Action**")
    st.button("Do it!", key="rand_do", disabled=True)

# Log panel
with log_col:
    st.markdown("**Log**")
    st.button("Load Action Sequence", key="log_load", disabled=True)
    st.button("Save Current Action Sequence", key="log_save", disabled=True)

st.markdown("---")

# --- Job Queue (collapsible) ---
st.subheader("Job Queue")
with st.expander("", expanded=True):
    if "env" in st.session_state and st.session_state["env"] is not None:
        schedulder = st.session_state["env"].scheduler
        print("Job Queue update")

        for job_idx, job_repetitions in enumerate(schedulder.job_instances):
            st.markdown(f"**Job {job_idx}**")

            # Build one long inline‑scroll row of job‑instance plots
            html_parts = []
            for rep_idx, job_instance in enumerate(job_repetitions):
                fig = job_instance.plot()
                data_uri = _fig_to_base64(fig)
                html_parts.append(
                    f"""<div style="display:inline-block; text-align:center;
                    min-width:400px; max-width:600px; margin-right:8px;">
<img src="data:image/png;base64,{data_uri}" style="width:400px;"/>
<div style="font-size:0.8rem;">Job{job_idx}-{rep_idx}</div>
</div>"""
                )

            st.markdown(
                "<div style='overflow-x:auto; white-space:nowrap; padding-bottom:0.5rem;'>"
                + "".join(html_parts)
                + "</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")
    else:
        st.info("Load an environment to view Job Queue")

    # Disable export until implemented
    st.button("Export", key="export_job_queue", disabled=True)


# --- Machine Schedule ---
st.subheader("Machine Schedule")
st.markdown("Step **23**")  # dynamic step indicator

# Placeholder for Gantt chart
st.text("<< Gantt chart of machine schedule goes here >>")
st.button("Export", key="export_machine_schedule", disabled=True)

st.markdown("---")

# Ensure manual selectors are initialized at startup
_refresh_manual_selectors()
