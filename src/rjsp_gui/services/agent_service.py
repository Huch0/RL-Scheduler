import os
import sys
import tempfile
import traceback
import zipfile
import json
from pathlib import Path
from rl_scheduler.trainer import load_sb3_algo
from .utils import dump_to_temp


def load_agent_and_env(agent_zip):
    """
    Unzip the given agent ZIP, load the saved RJSPEnv and RL model, and
    return them as (env, model).
    """

    # Obtain a filesystem path to the ZIP (handles UploadedFile via
    # dump_to_temp)
    agent_zip_path = dump_to_temp(agent_zip, suffix=".zip")

    # Create a temporary working directory
    tmpdir = tempfile.TemporaryDirectory()
    tmpdir_path = Path(tmpdir.name)

    try:
        # Extract all files
        with zipfile.ZipFile(agent_zip_path, "r") as zf:
            zf.extractall(tmpdir_path)
            # Adjust if the ZIP extracted into a single top-level folder
            entries = list(tmpdir_path.iterdir())
            if len(entries) == 1 and entries[0].is_dir():
                base_path = entries[0]
            else:
                base_path = tmpdir_path

        # Load the environment
        env_file = base_path / "env.pkl"
        if not env_file.exists():
            raise FileNotFoundError(f"env.pkl not found in archive {env_file}")
        from rl_scheduler.envs.rjsp_env import RJSPEnv

        env = RJSPEnv.load(env_file)
        # reset the environment to load the contract
        env.reset(seed=0)

        # Load train_config.json to get algorithm name
        config_file = base_path / "train_config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"train_config.json not found in archive {config_file}"
            )
        with config_file.open("r") as f:
            config = json.load(f)
        algo_name = config.get("ALGO")
        if not algo_name:
            raise ValueError("Algorithm name 'algo_name' missing in train_config.json")
        algo_class = load_sb3_algo(algo_name)
        if algo_class is None:
            raise ValueError(f"Algorithm '{algo_name}' not found in Stable Baselines3")

        # Locate the model file (final_model.zip or any other .zip)
        model_file = base_path / "final_model.zip"
        if not model_file.exists():
            zips = [p for p in base_path.glob("*.zip") if p.name != "env.pkl"]
            if not zips:
                raise FileNotFoundError(
                    f"No model .zip found in archive {agent_zip_path}"
                )
            model_file = zips[0]

        # Load the RL model with the environment
        model = algo_class.load(str(model_file), env=env)

        return model, env, None

    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return None, None, (f"Failed to load agent and environment: {e}")

    finally:
        # Clean up temporary directory and ZIP file
        tmpdir.cleanup()
        try:
            os.remove(agent_zip_path)
        except Exception:
            pass


def sample_agent_action(
    agent,
    env,
    deterministic: bool = False,
):
    """
    Sample an action from the agent using the environment.
    """
    try:
        action, _ = agent.predict(
            env.observation_handler.get_observation(), deterministic=deterministic
        )
        return action
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return None, (f"Failed to sample action from agent: {e}")
