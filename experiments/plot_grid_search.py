import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from PIL import Image

def plot_results(log_path='./experiments/tmp/1'):
    log_dirs = sorted([d for d in os.listdir(log_path) if os.path.isdir(os.path.join(log_path, d))])
    metrics = [
        'rollout/ep_rew_mean', 'train/clip_range', 'train/explained_variance',
        'train/value_loss', 'train/approx_kl', 'train/entropy_loss',
        'train/n_updates', 'train/policy_gradient_loss', 'train/clip_fraction',
        'train/loss', 'eval/mean_reward'
    ]

    fig, axes = plt.subplots(4, 3, figsize=(20, 24))
    axes = axes.flatten()

    colors = sns.color_palette("husl", len(log_dirs))

    for metric, ax in zip(metrics, axes):
        for idx, log_dir in enumerate(log_dirs):
            csv_path = os.path.join(log_path, log_dir, 'progress.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df = df.replace([np.inf, -np.inf], np.nan)
                
                if metric == 'eval/mean_reward':
                    # For eval/mean_reward, use line plot with markers
                    valid_data = df[df[metric].notna()]
                    ax.plot(valid_data['time/total_timesteps'], valid_data[metric],
                            label=log_dir, color=colors[idx % len(colors)],
                            marker='o', markersize=4, linestyle='-')
                else:
                    # For other metrics, use regular line plot
                    ax.plot(df['time/total_timesteps'], df[metric],
                            label=log_dir, color=colors[idx % len(colors)],
                            linestyle='-.')

        ax.set_title(metric)
        ax.set_xlabel('Total Timesteps')
        ax.set_ylabel(metric)
        if metric == 'eval/mean_reward':
            ax.legend(title='Evaluation every 10k steps')
        else:
            ax.legend()

    for i in range(len(metrics), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    save_path = os.path.join(log_path, 'results.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()

    img = Image.open(save_path)
    img.show()

if __name__ == '__main__':
    log_path = './experiments/tmp/1'
    plot_results(log_path=log_path)