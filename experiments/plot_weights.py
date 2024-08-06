from sb3_contrib import MaskablePPO
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from gymnasium import spaces
from scheduler_env.customEnv_repeat import SchedulingEnv


def plot_input_weights(model_path, env, save_img=False):
    # Load the model
    model = MaskablePPO.load(model_path)

    # Get the policy network
    policy_net = model.policy.mlp_extractor

    # Find the first linear layer (input layer)
    input_layer = next(layer for layer in policy_net.policy_net if isinstance(layer, nn.Linear))

    # Get the weights of the input layer
    weights = input_layer.weight.detach().numpy()

    # Calculate the average absolute weight for each input feature
    feature_importance = np.mean(np.abs(weights), axis=0)

    # Create labels for each input feature group and their aggregate importance
    feature_groups = {}
    for key, space in env.observation_space.spaces.items():
        if isinstance(space, spaces.Box):
            num_features = np.prod(space.shape)
            feature_index = len(feature_groups)
            feature_groups[key] = (feature_index, feature_index + num_features)
            feature_importance[feature_index:feature_index +
                               num_features] = np.mean(feature_importance[feature_index:feature_index + num_features])

    # Aggregate feature importance and standard deviation by group
    group_importance = {}
    group_std_dev = {}
    for group, (start_idx, end_idx) in feature_groups.items():
        group_importance[group] = np.mean(feature_importance[start_idx:end_idx])
        group_std_dev[group] = np.std(feature_importance[start_idx:end_idx])

    sorted_group_importance = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)

    sorted_labels = [item[0] for item in sorted_group_importance]
    sorted_importance = [item[1] for item in sorted_group_importance]
    sorted_std_dev = [group_std_dev[item[0]] for item in sorted_group_importance]

    # Function to plot and print group importance with error bars
    def plot_and_print_groups(importance, std_dev, labels, title, filename, num_to_plot=20):
        num_to_plot = min(num_to_plot, len(importance))
        plt.figure(figsize=(15, 10))
        bars = plt.bar(range(num_to_plot), importance[:num_to_plot], yerr=std_dev[:num_to_plot], capsize=5)
        plt.xlabel('Input Feature Groups')
        plt.ylabel('Average Absolute Weight')
        plt.title(title)
        plt.xticks(range(num_to_plot), labels[:num_to_plot], rotation=90)

        # Add value labels on top of each bar
        for rect in bars:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom', rotation=0)

        plt.tight_layout()

        if save_img:
            plt.savefig(filename)
        plt.show()

        print(f"\n{title}:")
        for i in range(num_to_plot):
            print(f"{i+1}. {labels[i]}: {importance[i]:.4f} (std: {std_dev[i]:.4f})")

    # Plot and print all feature groups
    plot_and_print_groups(sorted_importance, sorted_std_dev,
                          sorted_labels, 'Importance of All Input Feature Groups (Sorted)',
                          'all_groups_importance.png')


if __name__ == "__main__":
    model_path = "./experiments/tmp/2/run_0-{'clip_range': 0.1, 'learning_rate': 0.0005}/best_model.zip"
    env = SchedulingEnv(machine_config_path="instances/Machines/v0-8.json",
                        job_config_path="instances/Jobs/v0-12-repeat.json",
                        job_repeats_params=[(1, 1)] * 12,
                        test_mode=True)
    plot_input_weights(model_path, env, save_img=False)
