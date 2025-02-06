# test_rl_scheduler.py

import os

import pytest
from sb3_contrib import MaskablePPO

from rl_scheduler.RJSPEnv.Env import RJSPEnv


@pytest.fixture
def setup_environment():
    # Paths to the configuration files
    instance = "12x8"
    machine_config_path = f"rl_scheduler/instances/Machines/v0-{instance}.json"
    job_config_path = f"rl_scheduler/instances/Jobs/v0-{instance}-12.json"

    # Job repeats parameters (mean, std) for each job
    job_repeats_params = [(3, 1)] * 12

    # Create the environment
    env = RJSPEnv(
        machine_config_path=machine_config_path,
        job_config_path=job_config_path,
        job_repeats_params=job_repeats_params,
        max_time=50,
    )

    return env


def test_environment_initialization(setup_environment):
    env = setup_environment
    assert env is not None, "Failed to initialize environment"


def test_model_loading(setup_environment):
    model_path = "rl_scheduler/models/paper/0-paper-8x12-18m/MP_Single_Env4_gamma_1_obs_v4_clip_1_lr_custom_expv1_18000000.zip"

    # Validate model file existence
    assert os.path.exists(model_path), "Model path does not exist"

    # Load the model
    model = MaskablePPO.load(model_path, env=setup_environment)
    assert model is not None, "Failed to load the PPO model"


def test_single_step_in_environment(setup_environment):
    env = setup_environment

    # Reset the environment
    obs, _ = env.reset()
    assert obs is not None, "Failed to reset environment"

    # Perform a single step
    action_masks = env.action_masks()
    model_path = "rl_scheduler/models/paper/0-paper-8x12-18m/MP_Single_Env4_gamma_1_obs_v4_clip_1_lr_custom_expv1_18000000.zip"
    model = MaskablePPO.load(model_path, env=env)
    action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
    assert action is not None, "Model failed to predict action"

    # Check the step function
    obs, reward, terminated, truncated, info = env.step(action)
    assert (
        obs is not None
        and isinstance(reward, float)
        and isinstance(terminated, bool)
        and isinstance(truncated, bool)
    )
