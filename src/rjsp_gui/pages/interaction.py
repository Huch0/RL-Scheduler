import streamlit as st
from rjsp_gui.services.env_service import load_environment, reset_env
import base64
import io
import numpy as np
from rl_scheduler.envs.registry import get_action_handler
from rl_scheduler.renderer import PlotlyRenderer
from rjsp_gui.services.env_service import step_env

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
        max_rep_for_job = (
            len(st.session_state["env"].scheduler.job_instances[job_index]) - 1
        )
    else:
        max_rep_for_job = 1

    repetition = st.number_input(
        "Repetition",
        min_value=0,
        max_value=max_rep_for_job,
        value=min(st.session_state.get("man_rept", 0), max_rep_for_job),
        step=1,
        key="man_rept",
    )
    if st.button("Do it!", key="man_do"):
        if "env" not in st.session_state or st.session_state["env"] is None:
            st.warning("Load an environment first.")
        else:
            # Convert the selector strings "Machine N" / "Job N" to integer indices
            try:
                machine_idx = int(m_machine.split()[1])
                job_idx = int(m_job.split()[1])
            except (IndexError, ValueError, AttributeError):
                st.error("Invalid manual input — please select a machine and job.")
                st.stop()  # Abort the rest of the callback

            user_input = (machine_idx, job_idx, int(repetition))

            env = st.session_state["env"]
            action_handler_id = "mjr"
            # Call step_env
            obs, reward, terminated, truncated, info = step_env(
                env,
                action_handler_id=action_handler_id,
                action=user_input,
            )
            if info.get("invalid_action", False):
                st.error(f"Invalid action: {info['error']}")
            else:
                st.success("Step executed!")
                # Refresh visuals
                _refresh_manual_selectors()


# Random Action panel
with rand_col:
    st.markdown("**Random Action**")

    # Persistent RNG and handler for random sampling
    if "rand_rng" not in st.session_state:
        st.session_state["rand_rng"] = np.random.default_rng()

    if (
        "env" in st.session_state
        and st.session_state["env"] is not None
        and (
            "rand_handler" not in st.session_state
            or st.session_state["rand_handler"].scheduler
            is not st.session_state["env"].scheduler
        )
    ):
        # (Re)build the handler when environment changes
        st.session_state["rand_handler"] = get_action_handler(
            "mjr", st.session_state["env"].scheduler
        )

    if st.button("Random Step", key="rand_step"):
        if "env" not in st.session_state or st.session_state["env"] is None:
            st.warning("Load an environment first.")
        else:
            env = st.session_state["env"]
            handler = st.session_state.get("rand_handler")

            if handler is None:
                st.error("Random action handler not initialised.")
                st.stop()

            try:
                env.action_handler = handler
                env.action_space = handler.action_space
                action = handler.sample_valid_action(st.session_state["rand_rng"])
                # Show action details
                st.markdown(f"**Action**: {action}")
                obs, reward, terminated, truncated, info = env.step(action)

                if info.get("invalid_action", False):
                    st.error(f"Random action invalid: {info['error']}")
                else:
                    st.success(f"Random action executed (reward={reward:.2f})")
                    _refresh_manual_selectors()
            except RuntimeError as exc:
                st.error(f"Failed to sample or execute random action: {exc}")

    # ------------------------------------------------------------
    # Run random actions until the environment terminates
    # ------------------------------------------------------------
    if st.button("Run Random Episode", key="rand_run"):
        if "env" not in st.session_state or st.session_state["env"] is None:
            st.warning("Load an environment first.")
        else:
            env = st.session_state["env"]
            handler = st.session_state.get("rand_handler")
            rng = st.session_state["rand_rng"]

            if handler is None:
                st.error("Random action handler not initialised.")
            else:
                total_reward = 0.0
                try:
                    env.action_handler = handler
                    env.action_space = handler.action_space
                    terminated = False
                    truncated = False
                    # safety cap to avoid infinite loops
                    while not (terminated or truncated) and env.timestep < 10_000:
                        action = handler.sample_valid_action(rng)
                        obs, reward, terminated, truncated, info = env.step(action)
                        if info.get("invalid_action", False):
                            # skip invalid—it shouldn’t happen with sample_valid_action
                            continue
                        total_reward += reward
                    st.success(
                        f"Episode finished in {env.timestep} steps, total reward={total_reward:.2f}"
                    )
                    _refresh_manual_selectors()
                except RuntimeError as exc:
                    st.error(f"Failed during random run: {exc}")

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


# --- Machine Schedule (collapsible) ---
st.subheader("Machine Schedule")
with st.expander("", expanded=True):
    if "env" in st.session_state and st.session_state["env"] is not None:
        scheduler = st.session_state["env"].scheduler
        gantt_fig = PlotlyRenderer.render(scheduler)

        st.plotly_chart(gantt_fig, use_container_width=True)
    else:
        st.info("Load an environment to view the machine schedule.")

st.markdown("---")

# Ensure manual selectors are initialized at startup
_refresh_manual_selectors()
