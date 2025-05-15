import os
import json
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from rl_scheduler.envs.rjsp_env import RJSPEnv
from rl_scheduler.envs.utils import make_env
from rl_scheduler.config_path import INSTANCES_DIR
from pathlib import Path
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import copy
import time
from gymnasium.wrappers import TimeLimit
import shutil

_SUPPORTED_ALGOS = {
    "PPO": PPO,
    "MaskablePPO": MaskablePPO,
    # Add other algorithms here as needed
}


def load_sb3_algo(algo_name: str):
    """
    Load the specified algorithm from Stable Baselines3 or SB3 Contrib.
    """
    algo_name = algo_name.strip()
    if algo_name in _SUPPORTED_ALGOS:
        return _SUPPORTED_ALGOS[algo_name]
    else:
        print(f"Algorithm '{algo_name}' not found in Stable Baselines3.")
        return None


def train_agent(
    env: RJSPEnv,
    save_dir: str = "logs",
    ALGO: BaseAlgorithm = MaskablePPO,
    total_timesteps: int = 1e6,
    checkpoint_freq: int = 10000,
    num_eval_episodes: int = 5,
    eval_freq: int = 5000,
    hyperparameters: dict | None = None,
    resume_from: Path | None = None,
    num_envs: int = 1,
    max_episode_steps: int = 1000,
):
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Apply episode length limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # Training environment: vectorized if requested
    if num_envs > 1:

        def make_env_fn():
            return Monitor(copy.deepcopy(env))

        train_env = SubprocVecEnv([make_env_fn for _ in range(num_envs)])
    else:
        train_env = DummyVecEnv([lambda: Monitor(env)])

    # Evaluation environment: single monitored copy
    eval_env = Monitor(copy.deepcopy(env))

    # Initialize or load model
    if resume_from:
        print(f"Resuming training from {resume_from}")
        model = ALGO.load(resume_from, env=train_env)
    else:
        model = ALGO(env=train_env, **hyperparameters)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_dir,
        name_prefix="model_checkpoint",
    )

    new_logger = configure(save_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    if ALGO == MaskablePPO:
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            log_path=save_dir,
            n_eval_episodes=num_eval_episodes,
            eval_freq=eval_freq,  # Evaluate every 5,000 steps
            deterministic=True,
            render=False,
            verbose=1,
        )
    else:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            log_path=save_dir,
            n_eval_episodes=num_eval_episodes,
            eval_freq=eval_freq,  # Evaluate every 5,000 steps
            deterministic=True,
            render=False,
            verbose=1,
        )

    # Train the model
    print("Training starting...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")

    # Save final model
    model_path = os.path.join(save_dir, "final_model.zip")
    model.save(model_path)

    # zip the training directory
    save_train_zip(save_dir)
    print("Training finished.")


def prepare_training(
    save_dir: str,
    env_config: dict,
    train_config: dict,
):
    # copy env_config and train_config to avoid modifying the original
    train_kwargs = train_config
    env_config = copy.deepcopy(env_config)
    train_config = copy.deepcopy(train_config)

    # save_dir + agent name + current_time
    current_time = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(save_dir, train_config["agent_name"], current_time)

    env = make_env(**env_config)
    # Save env.pkl
    env.save(save_dir)

    # Load all json files from env_config
    machine_config_path = env_config["scheduler"]["machine_config_path"]
    job_config_path = env_config["scheduler"]["job_config_path"]
    operation_config_path = env_config["scheduler"]["operation_config_path"]
    contract_config_path = env_config["contract_path"]

    # Update env_config with dict objects
    with open(machine_config_path, "r") as f:
        machine_config = json.load(f)
    with open(job_config_path, "r") as f:
        job_config = json.load(f)
    with open(operation_config_path, "r") as f:
        operation_config = json.load(f)
    with open(contract_config_path, "r") as f:
        contract_config = json.load(f)
    env_config["scheduler"]["machine_config"] = machine_config
    env_config["scheduler"]["job_config"] = job_config
    env_config["scheduler"]["operation_config"] = operation_config
    env_config["contract"] = contract_config

    # Update path to str for serialization
    env_config["scheduler"]["machine_config_path"] = str(machine_config_path)
    env_config["scheduler"]["job_config_path"] = str(job_config_path)
    env_config["scheduler"]["operation_config_path"] = str(operation_config_path)
    env_config["contract_path"] = str(contract_config_path)

    # Save env_config.json
    env_config_path = os.path.join(save_dir, "env_config.json")
    with open(env_config_path, "w") as f:
        json.dump(env_config, f, indent=4)

    # Update ALGO to str for serialization
    train_config["ALGO"] = str(train_config["ALGO"].__name__)

    # Save training config
    train_config_path = os.path.join(save_dir, "train_config.json")
    with open(train_config_path, "w") as f:
        json.dump(train_config, f, indent=4)

    # return training kwargs
    train_kwargs["env"] = env
    train_kwargs["save_dir"] = save_dir
    train_kwargs.pop("agent_name")
    return train_kwargs


def save_train_zip(save_dir: str):
    """
    Save the training directory as a zip file.
    """
    # Determine the parent directory and the target folder name
    parent_dir, dir_name = os.path.split(save_dir)
    # Handle case if save_dir ends with a trailing slash
    if not dir_name:
        parent_dir, dir_name = os.path.split(parent_dir)

    # Base name (path without extension) for the zip file
    base_name = os.path.join(parent_dir, dir_name)
    # Create a zip archive of the directory
    zip_file_path = shutil.make_archive(
        base_name=base_name,
        format="zip",
        root_dir=parent_dir,
        base_dir=dir_name,
    )

    print(f"Training files saved to {zip_file_path}")


def main():
    # Example usage
    m_path = INSTANCES_DIR / "machines" / "M-example0-3.json"
    j_path = INSTANCES_DIR / "jobs" / "J-example0-5.json"
    o_path = INSTANCES_DIR / "operations" / "O-example0.json"
    c_path = INSTANCES_DIR / "contracts" / "C-example1-5.json"

    save_dir = os.path.join(os.getcwd(), "logs")

    env_config = {
        "scheduler": {
            "machine_config_path": m_path,
            "job_config_path": j_path,
            "operation_config_path": o_path,
        },
        "contract_generator": "deterministic",
        "contract_path": c_path,
        "action_handler": "mj",
        "action_handler_kwargs": {"priority_rule_id": "etd"},
        "observation_handler": "mlp",
        "observation_handler_kwargs": {"time_horizon": 1000},
        "reward_handler": "profit_cost",
        "reward_handler_kwargs": {
            "weights": {"C_max": 10.0, "C_mup": 2.0, "C_mid": 1.0},
        },
    }

    train_config = {
        "agent_name": "AGENT_NAME",
        "ALGO": PPO,
        "total_timesteps": int(1e5),
        "checkpoint_freq": 10000,
        "num_eval_episodes": 5,
        "eval_freq": 10000,
        "max_episode_steps": 500,
        "hyperparameters": {
            "policy": "MultiInputPolicy",
            "device": "auto",
            "seed": 42,
            "verbose": 1,
        },
        "num_envs": 1,
        "resume_from": None,
    }

    train_kwargs = prepare_training(save_dir, env_config, train_config)
    train_agent(**train_kwargs)


if __name__ == "__main__":
    main()
