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
        return [env.action_masks() for env in self.envs]

def grid_search(params,
                env_timesteps=1_000_000,
                eval_freq=10_000,
                n_eval_envs=10,
                eval_episodes=10,
                log_path="./experiments/tmp/1"):

    def make_env():
        def _init():
            env = SchedulingEnv(machine_config_path="instances/Machines/v0-8.json",
                                job_config_path="instances/Jobs/v0-12-repeat-hard.json",
                                job_repeats_params=[(3, 1)] * 12,
                                test_mode=False)
            return Monitor(env)
        return _init

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
        env = make_env()()
        model = MaskablePPO("MultiInputPolicy", env, **current_params, verbose=1)

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

        # Evaluate the model on the fixed set of environments
        episode_rewards = []
        obs = eval_envs.reset()
        for _ in range(eval_episodes):
            done = [False for _ in range(n_eval_envs)]
            episode_reward = [0 for _ in range(n_eval_envs)]
            while not all(done):
                action_masks = eval_envs.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
                obs, reward, done, _ = eval_envs.step(action)
                episode_reward = [r + reward[i] for i, r in enumerate(episode_reward)]
            episode_rewards.extend(episode_reward)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Save the results
        results.append({
            **current_params,
            'mean_reward': mean_reward,
            'std_reward': std_reward
        })

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_params = current_params

        # Clean up
        env.close()
        del model
        
    # Print the best parameters and reward
    for result in results:
        print(result)

    print("\nBest parameters:")
    print(best_params)
    print(f"Best mean reward: {best_reward:.2f}")

    # Clean up evaluation environments
    eval_envs.close()

    return best_params, best_reward


if __name__ == "__main__":
    params = {
        "n_steps": [1024, 2048, 4096],
        "clip_range": [0.1],
        "learning_rate": [0.0005]
        # "clip_range": [0.1, 0.2, 0.3],
        # "learning_rate": [0.0005, 0.0003, 0.0001]
    }
    log_path = "./experiments/tmp/2"

    best_params, best_reward = grid_search(params, log_path=log_path)
    print(f"Best parameters: {best_params}")
    print(f"Best mean reward: {best_reward:.2f}")
