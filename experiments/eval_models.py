import numpy as np
from sb3_contrib import MaskablePPO
from scheduler_env.customEnv_repeat import SchedulingEnv
import os
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
import matplotlib.pyplot as plt


def eval_models(model_path='./experiments/tmp/2', n_eval_episodes=10, n_envs=20):
    model_dirs = sorted([d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))])
    # Load the trained models
    models = []
    model_names = []

    for model_dir in model_dirs:
        model = MaskablePPO.load(os.path.join(model_path, model_dir, 'best_model.zip'))
        models.append(model)
        model_names.append(model_dir)  # Use the directory name as the model name

    def make_env():
        def _init():
            env = SchedulingEnv(machine_config_path="instances/Machines/v0-8.json",
                                job_config_path="instances/Jobs/v0-12-repeat-hard.json",
                                job_repeats_params=[(3, 1)] * 12,
                                test_mode=False)
            return Monitor(env)
        return _init

    # Create a fixed set of environments
    eval_env = DummyVecEnv([make_env() for _ in range(n_envs)])

    # Set random seeds for reproducibility
    np.random.seed(0)
    eval_env.seed(0)

    # Evaluate the models
    results = []
    for model, model_name in zip(models, model_names):
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )
        results.append({
            'model': model_name,
            'mean_reward': mean_reward,
            'std_reward': std_reward
        })

    # Sort results by mean reward (descending order)
    results.sort(key=lambda x: x['mean_reward'], reverse=True)

    # Print results
    print("Evaluation Results:")
    for result in results:
        print(f"{result['model']}: Mean Reward = {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")

    return results


def plot_results(results, output_file='evaluation_results.png'):
    # Sort results by mean reward (descending order)
    results.sort(key=lambda x: x['mean_reward'], reverse=True)

    # Extract data for plotting
    model_names = [result['model'] for result in results]
    mean_rewards = [result['mean_reward'] for result in results]
    std_rewards = [result['std_reward'] for result in results]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(model_names)), mean_rewards, yerr=std_rewards, capsize=5)

    # Customize the plot
    ax.set_xlabel('Models')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Evaluation Results of MaskablePPO Models')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    # Adjust layout and save as PDF
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=100)
    print(f"Plot saved as {output_file}")
    plt.show()
    plt.close(fig)


# Call the evaluation function
eval_results = eval_models()

# Plot the results
plot_results(eval_results)
