"""
Functions for Trajectory visualization.

Available Functions
-------------------
[Public]
plot_segmented_trajectory(...): Plots a 2D trajectory segmented by detected walking intervals, coloring each movement segment distinctly.

plot_trajectory_by_interval(...): Creates subplots for each walking interval's 2D trajectory segment, showing movement intervals separately.

plot_density(...): Visualizes 2D trajectory density with a scatter plot colored by density and a KDE heatmap weighted by density values.
------------------

"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# internal imports
from constants import ACTIVITY , WALKING , BLOCK_ID
from .plot_utils import _handle_plot

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def plot_segmented_trajectory(data, save_dir: str, show=True, save=False)-> None:
    """
    Plots a 2D trajectory of positions (x, y) segmented by detected movement periods.

    Each continuous walking segment is plotted in a different color. If no walking
    data is found or 'activity' column is missing, the entire trajectory is plotted
    without segmentation.

    :param data: pandas DataFrame with columns 'x', 'y', 'activity'.
    :param save_dir: String indicating the directory where plots should be saved if `save=True`.
    :param show: Boolean indicating whether to display the plot on screen. Default is True.
    :param save: Boolean indicating whether to save the plot as a PNG file. Default is False.

    """
    plt.figure(figsize=(8, 8))
    if ACTIVITY in data.columns and not data.empty:

        # Split data into continuous walking segments
        segments = [segment for _, segment in data[data[ACTIVITY] == WALKING].groupby(BLOCK_ID)]

        cmap = plt.get_cmap('tab10') # Color map to differentiate segments

        # Plot each walking segment with different color and label
        for segment_index, segment in enumerate(segments):
            plt.plot(segment['x'], segment['y'], marker='o', color=cmap(segment_index % 10), label=f'Movement {segment_index + 1}', lw=1.5)
        plt.legend(loc='upper right')
    else:
        # If no activity column or empty data, plot all points with one color
        plt.plot(data['x'], data['y'], marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Trajectory (colored by movement)')
    plt.axis('equal')
    plt.grid()

    _handle_plot(show=show, save=save, save_dir=save_dir, filename="segmented_trajectory_plot.png")


def plot_trajectory_by_interval(data, save_dir: str, n_cols_subplots=3, show=True, save=False)-> None:
    """
    Plots separate subplots of 2D trajectory segments divided by walking intervals.

    Each walking segment is plotted individually in its own subplot with the time interval
    shown in the title. Supports multiple columns of subplots for better visualization.
    If no walking data or segments are found, an informative message is printed.

    :param data: pandas DataFrame containing columns 'x', 'y', 't', and 'activity'.
    :param save_dir: String indicating the directory where plots should be saved if `save=True`.
    :param n_cols_subplots: Number of subplot columns to arrange the plots in. Default is 3.
    :param show: Boolean indicating whether to display the plot on screen. Default is True.
    :param save: Boolean indicating whether to save the plot as a PNG file. Default is False.


    """
    if ACTIVITY in data.columns and not data.empty:

        # Split data into continuous walking segments
        segments = [segment for _, segment in data[data[ACTIVITY] == WALKING].groupby(BLOCK_ID)]

        if not segments:
            print("No trajectory segments found.")
            return
        n_segments = len(segments)
        n_rows = int(np.ceil(n_segments / n_cols_subplots))

        # Create subplot grid with specified number of rows and columns
        fig, axes = plt.subplots(n_rows, n_cols_subplots, figsize=(5 * n_cols_subplots, 5 * n_rows), squeeze=False)

        # Plot each walking segment in its own subplot
        for segment_index, traj in enumerate(segments):
            ax = axes[segment_index // n_cols_subplots, segment_index % n_cols_subplots]
            ax.plot(traj['x'], traj['y'], marker='o', ms=2, lw=0.8)
            ax.set_title(f"Movement {segment_index + 1} | {traj['t'].iloc[0]:.1f}-{traj['t'].iloc[-1]:.1f}s")
            ax.axis('equal')
            ax.grid(True)

        # Remove empty subplots if total subplots exceed number of segments
        for empty_index in range(n_segments, n_rows * n_cols_subplots):
            fig.delaxes(axes[empty_index // n_cols_subplots, empty_index % n_cols_subplots])

        plt.tight_layout()

        _handle_plot(show=show, save=save, save_dir=save_dir, filename="segmented_trajectory_plot.png")
    else:
        print("Dataframe is empty or activity column is missing.")


def plot_density(xs, ys, densities,save_dir:str, show=True, save=False)-> None:
    """
    Visualizes 2D trajectory point density with a scatter plot colored by density and
    a KDE (Kernel Density Estimate) heatmap weighted by density values.

    :param xs: Iterable of x-coordinates.
    :param ys: Iterable of y-coordinates.
    :param densities: Iterable of density values associated with each (x, y) point.
    :param save_dir: String indicating the directory where plots should be saved if `save=True`.
    :param show: Boolean indicating whether to display the plot on screen. Default is True.
    :param save: Boolean indicating whether to save the plot as a PNG file. Default is False.

    """
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(xs, ys, c=densities, cmap='viridis', s=40, alpha=0.8)
    plt.colorbar(sc, label='Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Trajectory Density')
    plt.axis('equal')
    plt.grid(True)
    _handle_plot(show=show, save=save, save_dir=save_dir, filename="trajectory_density_scatter.png")

    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=xs, y=ys, weights=densities, cmap="hot", fill=True, bw_adjust=0.2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Density Heatmap (KDE)')
    plt.axis('equal')
    plt.grid(True)
    _handle_plot(show=show, save=save, save_dir=save_dir, filename="trajectory_density_kde.png")

