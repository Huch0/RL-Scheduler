import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from PIL import Image
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def plot_results(log_path='./experiments/tmp/1'):
    log_dirs = [d for d in os.listdir(log_path) if os.path.isdir(os.path.join(log_path, d))]
    metrics = [
        'rollout/ep_rew_mean', 'train/clip_range', 'train/explained_variance',
        'train/value_loss', 'train/approx_kl', 'train/entropy_loss',
        'train/n_updates', 'train/policy_gradient_loss', 'train/clip_fraction',
        'train/loss', 'eval/mean_reward'
    ]

    all_data = []
    for log_dir in log_dirs:
        csv_path = os.path.join(log_path, log_dir, 'progress.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['model'] = log_dir
            # Replace infinity values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            all_data.append(df)

    combined_df = pd.concat(all_data)
    melted_df = combined_df.melt(id_vars=['time/total_timesteps', 'model'],
                                 value_vars=metrics,
                                 var_name='metric', value_name='value')

    # Set up the FacetGrid
    g = sns.FacetGrid(melted_df, col="metric", row="model", height=4, aspect=1.5, sharex=False, sharey=False)

    # Plot the data
    g.map(sns.lineplot, "time/total_timesteps", "value")

    # Customize the plot
    g.set_titles("{row_name} - {col_name}")
    g.set_axis_labels("Total Timesteps", "Value")

    # Rotate x-axis labels for better readability
    g.set_xticklabels(rotation=45, ha='right')

    # Adjust the layout
    g.tight_layout()

    save_path = os.path.join(log_path, 'results.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()  # Close the matplotlib figure

    # Open the saved image
    img = Image.open(save_path)
    img.show()  # This will open the image in the default image viewer


if __name__ == '__main__':
    log_path = './experiments/tmp/1'
    plot_results(log_path=log_path)
