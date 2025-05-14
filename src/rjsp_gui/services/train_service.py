import io
import json
import pickle
import tempfile
from pathlib import Path

from rl_scheduler.envs.utils import make_env
from rl_scheduler.envs.rjsp_env import RJSPEnv

__all__ = ["build_env"]


def build_env(scheduler_buf: io.BytesIO,
              contract_file,  # file-like object or path to contract JSON
              action_handler,
              observation_handler,
              reward_handler,
              info_handler) -> RJSPEnv:
    """
    Build and initialize the RJSP environment from scheduler pickle buffer and contract JSON.

    Parameters
    ----------
    scheduler_buf : io.BytesIO
        Pickle buffer containing the Scheduler object.
    contract_file : Uploaded file or file path for contracts JSON.
    action_handler : ActionHandler instance
    observation_handler : ObservationHandler instance
    reward_handler : RewardHandler instance
    info_handler : InfoHandler instance

    Returns
    -------
    env : RJSPEnv
        Initialized Gym environment ready for reset/step.
    """
    # Create temporary directory
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)

    # Write scheduler pickle to file
    sched_path = tmp_path / "scheduler.pkl"
    with sched_path.open("wb") as f:
        f.write(scheduler_buf.getvalue())

    # Write contract JSON to file
    if hasattr(contract_file, "read"):
        contract_data = json.load(contract_file)
    else:
        contract_data = json.load(open(contract_file, "r"))
    contract_path = tmp_path / "contracts.json"
    with contract_path.open("w", encoding="utf-8") as f:
        json.dump(contract_data, f, indent=4)

    # Determine which contract generator to use (sampling vs deterministic)
    cg_name = "stochastic" if "sampling" in contract_data else None
    config_path = contract_path

    # Load scheduler and build environment via factory
    scheduler = pickle.load(open(sched_path, "rb"))
    env = make_env(
        scheduler=scheduler,
        contract_generator=cg_name,
        action_handler=action_handler,
        observation_handler=observation_handler,
        reward_handler=reward_handler,
        info_handler=info_handler,
    )
    # Initialize environment with contracts JSON
    env.reset(options={"contract_path": str(config_path)})
    return env
