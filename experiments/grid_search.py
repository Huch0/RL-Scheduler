from itertools import product
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from scheduler_env.customEnv_repeat import SchedulingEnv
from stable_baselines3.common.vec_env import VecEnv

class MaskableDummyVecEnv(DummyVecEnv):
    def action_masks(self):
        return np.array([env.get_wrapper_attr('action_masks')() for env in self.envs])


def grid_search(params,
                env_timesteps=1_000_000,
                eval_freq=10_000,
                n_eval_envs=10,
                eval_episodes=1,
                log_path="./experiments/tmp/1"):

    def make_env():
        def _init():
            env = SchedulingEnv(machine_config_path="instances/Machines/v0-8.json",
                                job_config_path="instances/Jobs/v0-12-repeat-hard.json",
                                job_repeats_params=[(3, 1)] * 12,
                                test_mode=False)
            return Monitor(env)
        return _init

    train_env = MaskableDummyVecEnv([make_env()])
    eval_envs = MaskableDummyVecEnv([make_env() for _ in range(n_eval_envs)])

    best_reward = float('-inf')
    best_params = None
    results = []

    # Generate all combinations of parameters
    param_keys = list(params.keys())
    param_values = list(params.values())
    param_combinations = list(product(*param_values))

    for i, combination in enumerate(param_combinations):
        current_params = dict(zip(param_keys, combination))

        # Set up logger for this run
        run_log_path = f"{log_path}/run_{i}-{current_params}"
        logger = configure(run_log_path, ["stdout", "csv", "tensorboard"])

        eval_callback = MaskableEvalCallback(eval_envs, best_model_save_path=run_log_path,
                                             log_path=run_log_path, eval_freq=eval_freq,
                                             deterministic=True, render=False)

        # Create and train the model
        model = MaskablePPO("MultiInputPolicy", train_env, **current_params, verbose=1)

        # Log and print the model parameters
        print(f"\n--- Starting Run {i+1}/{len(param_combinations)} ---")
        print("Requested parameters:")
        for key, value in current_params.items():
            print(f"{key}: {value}")
        print("\nActual model parameters:")
        for key in current_params:
            actual_value = getattr(model, key)
            if callable(actual_value):
                # For scheduled parameters, get the initial value
                actual_value = actual_value(0)
            logger.record(f"model_parameters/{key}", actual_value)
        logger.dump(step=0)

        model.set_logger(logger)
        model.learn(env_timesteps, callback=eval_callback)

        # Clean up
        train_env.close()
        del model

    return best_params, best_reward


if __name__ == "__main__":
    params = {
        "clip_range": [0.1, 0.2, 0.3],
        "learning_rate": [0.0005, 0.0003, 0.0001]
    }
    log_path = "./experiments/tmp/2"

    best_params, best_reward = grid_search(params,
                                        #    env_timesteps=100,
                                        #    eval_freq=100,
                                        #    n_eval_envs=10,
                                        #    eval_episodes=1,
                                           log_path=log_path)
    print(f"Best parameters: {best_params}")
    print(f"Best mean reward: {best_reward:.2f}")
