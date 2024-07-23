import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_results(log_path=None, save_path=None):
    if log_path is None:
        # Find the last log file in the directory ./experiments/tmp/
        log_dirs = [f for f in os.listdir('./experiments/tmp/')]
        log_dirs.sort()
        # Get one file in the last directory
        log_files = [f for f in os.listdir(f'./experiments/tmp/{log_dirs[-1]}')]
        log_path = f'./experiments/tmp/{log_dirs[-1]}/{log_files[0]}'

    print(f'Plotting results from {log_path}')

    df = pd.read_csv(log_path, sep='\t')

    fig, axs = plt.subplots(2, 3, figsize=(24, 12))  # Adjusting for a 2x3 grid

    # Plot Average, Min, Max, and Std of Episode Return
    axs[0, 0].plot(df['Epoch'], df['AverageEpRet'], label='Average Episode Return', color='blue')
    axs[0, 0].fill_between(df['Epoch'], df['AverageEpRet'] - df['StdEpRet'], df['AverageEpRet'] +
                           df['StdEpRet'], color='blue', alpha=0.2, label='Std. Dev.')
    axs[0, 0].plot(df['Epoch'], df['MaxEpRet'], label='Max Episode Return', color='green', linestyle='--')
    axs[0, 0].plot(df['Epoch'], df['MinEpRet'], label='Min Episode Return', color='red', linestyle='--')
    # Add ValMeanEpRet to the same plot
    axs[0, 0].plot(df['Epoch'], df['ValMeanEpRet'], label='Validation Mean Episode Return', color='purple', linestyle='-.', linewidth=2)

    axs[0, 0].set_title('Episode Return Statistics')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Episode Return')
    axs[0, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))


    # Plot Policy Loss
    axs[0, 1].plot(df['Epoch'], df['LossPi'], label='Policy Loss', color='red')
    axs[0, 1].set_title('Policy Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()

    # Plot Value Loss
    axs[0, 2].plot(df['Epoch'], df['LossV'], label='Value Loss', color='green')
    axs[0, 2].set_title('Value Loss')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('Loss')
    axs[0, 2].legend()

    # Plot Entropy
    axs[1, 0].plot(df['Epoch'], df['Entropy'], label='Entropy', color='purple')
    axs[1, 0].set_title('Entropy')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Entropy')
    axs[1, 0].legend()

    # Plot KL Divergence
    axs[1, 1].plot(df['Epoch'], df['KL'], label='KL Divergence', color='orange')
    axs[1, 1].set_title('KL Divergence')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('KL Divergence')
    axs[1, 1].legend()

    # Adjust layout and remove the empty subplot
    fig.delaxes(axs[1][2])
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    plot_results()
