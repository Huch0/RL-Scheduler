import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from scipy import stats


def plot_performance_distributions(csv_file_path='experiments/tmp/1/results.csv'):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Replace inf values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Plot Makespan distributions
    makespan_columns = ['Makespan/OR-dc', 'Makespan/OR-ms', 'Makespan/RL']
    colors = ['r', 'g', 'b']

    for column, color in zip(makespan_columns, colors):
        data = df[column].dropna()
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        ax1.plot(x_range, kde(x_range), label=column.split('/')[-1], color=color)
        ax1.fill_between(x_range, kde(x_range), alpha=0.3, color=color)

    ax1.set_title('Makespan Distribution')
    ax1.set_xlabel('Makespan')
    ax1.set_ylabel('Density')
    ax1.set_xlim(left=0)  # Set the lower limit of x-axis to 0
    ax1.legend()

    # Plot Deadline Compliance distributions
    dc_columns = ['DeadlineCompliance/OR-dc', 'DeadlineCompliance/OR-ms', 'DeadlineCompliance/RL']

    for column, color in zip(dc_columns, colors):
        data = df[column].dropna()
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(max(0, data.min()), min(1, data.max()), 100)
        ax2.plot(x_range, kde(x_range), label=column.split('/')[-1], color=color)
        ax2.fill_between(x_range, kde(x_range), alpha=0.3, color=color)

    ax2.set_title('Deadline Compliance Distribution')
    ax2.set_xlabel('Deadline Compliance')
    ax2.set_ylabel('Density')
    ax2.set_xlim(0, 1)  # Deadline compliance is typically between 0 and 1
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_performance_distributions(csv_file_path='./experiments/MSE/results.csv')
