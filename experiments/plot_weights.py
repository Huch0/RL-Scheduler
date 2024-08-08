from sb3_contrib import MaskablePPO
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from gymnasium import spaces
from scheduler_env.customEnv_repeat import SchedulingEnv
import torch


def plot_input_weights(model_path, env, save_img=False):
    # Load the model
    model = MaskablePPO.load(model_path)
    # print(dir(model))

    # Get the policy network
    policy = model.policy
    # print(f"Policy network: {policy}")

    mlp_extractor = policy.mlp_extractor
    print(f"MLP extractor: {mlp_extractor}")

    features_extractor = policy.features_extractor
    print(f"Features extractor: {features_extractor}")

    model_feature_names = list(features_extractor.extractors.keys())
    print(f"Model feature names: {model_feature_names}")

    # Create labels for each input feature group and their aggregate importance
    feature_groups = {}
    feature_index = 0
    previous_end_index = -1
    for key, space in env.observation_space.spaces.items():
        # Make sure the feature names match the model feature names
        assert model_feature_names[feature_index] == key
        feature_index += 1

        if isinstance(space, spaces.Box):
            num_features = np.prod(space.shape)
            start_index = previous_end_index + 1
            end_index = start_index + num_features - 1
            feature_groups[key] = (start_index, end_index)
            print(f"{key} | range - {start_index}:{end_index}")
            previous_end_index = end_index

    # Find the first linear layer (input layer)
    input_layer = next(layer for layer in mlp_extractor.policy_net if isinstance(layer, nn.Linear))

    # Get the weights of the input layer
    weights = input_layer.weight.detach().numpy()
    print(f"Input shape: {weights.shape}")

    # Calculate the average absolute weight for each input feature
    feature_importance = np.mean(np.abs(weights), axis=0)
    print(f"Feature importance shape: {feature_importance.shape}")

    # Make sure the number of input features matches the number of weights
    assert previous_end_index == weights.shape[1] - 1

    # Function to plot and print feature importance
    def plot_and_print_all_features(importance, feature_groups, title, filename):
        plt.figure(figsize=(15, 10))
        plt.bar(range(len(importance)), importance)
        plt.xlabel('Input Features')
        plt.ylabel('Average Absolute Weight')
        plt.title(title)

        # Set xticks and xtick labels
        xticks = []
        xtick_labels = []
        for group, (start_idx, end_idx) in feature_groups.items():
            xticks.append((start_idx + end_idx) / 2)  # Position the label in the middle of the group
            xtick_labels.append(group)
            # Group x-axis labels by feature group
            plt.axvline(x=end_idx, color='gray', linestyle='--')

        plt.xticks(xticks, xtick_labels, rotation=90)  # Rotate labels for better readability

        plt.tight_layout()

        if save_img:
            plt.savefig(filename)
        plt.show()

    # Plot and print the feature importance
    plot_and_print_all_features(feature_importance, feature_groups,
                                "Input Feature Importance", "input_feature_importance.png")

    # Aggregate feature importance and standard deviation by group
    group_stats = {}
    for group, (start_idx, end_idx) in feature_groups.items():
        group_stats[group] = {
            'avg': np.mean(feature_importance[start_idx:end_idx]),
            'std': np.std(feature_importance[start_idx:end_idx]),
            'min': np.min(feature_importance[start_idx:end_idx]),
            'max': np.max(feature_importance[start_idx:end_idx])
        }

    # Function to plot and print group importance with error bars
    def plot_and_print_groups(group_stats, title, filename, num_to_plot=20):
        # Sort groups by average importance
        sorted_groups = sorted(group_stats.items(), key=lambda x: x[1]['avg'], reverse=True)

        # Extract data for plotting
        groups = [group for group, _ in sorted_groups[:num_to_plot]]
        avg_importance = [stats['avg'] for _, stats in sorted_groups[:num_to_plot]]
        errors = [[stats['avg'] - stats['min'], stats['max'] - stats['avg']]
                  for _, stats in sorted_groups[:num_to_plot]]

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(groups)), avg_importance, yerr=np.array(errors).T, capsize=5)

        # Customize the plot
        ax.set_xlabel('Input Feature Groups')
        ax.set_ylabel('Average Absolute Weight')
        ax.set_title(title)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha='right')

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

        # Adjust layout and save as PNG
        plt.tight_layout()
        if save_img:
            plt.savefig(filename, bbox_inches='tight', dpi=100)
        plt.show()

    # Plot and print the group importance
    plot_and_print_groups(group_stats, "Input Feature Group Importance", "input_feature_group_avg.png")

    return


if __name__ == "__main__":
    # model_path = "./experiments/tmp/2/run_0-{'clip_range': 0.1, 'learning_rate': 0.0005}/best_model.zip"
    model_path = "./logs/tmp/1/best_model.zip"
    env = SchedulingEnv(machine_config_path="instances/Machines/v0-8.json",
                        job_config_path="instances/Jobs/v0-12-repeat-hard.json",
                        job_repeats_params=[(3, 1)] * 12,
                        test_mode=True)
    env.reset()
    plot_input_weights(model_path, env, save_img=False)
