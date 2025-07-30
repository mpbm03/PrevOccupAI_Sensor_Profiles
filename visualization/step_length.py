"""

This function provides a visualization of the estimated step length

Available Functions
-------------------
[Public]
plot_step_length(...): Plots a histogram of the estimated step lengths.

-------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import numpy as np
from .plot_utils import _handle_plot

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def plot_step_length(lengths: np.array, save_dir: str, show=True, save=False)-> None:
    """
        Plots a histogram of the estimated step lengths.

        This function visualizes the distribution of step lengths using a histogram,
        allowing for quick inspection of the data spread and frequency.

        :param lengths: Array containing estimated step lengths in meters.
        :param save_dir: String indicating the directory where plots should be saved if `save=True`.
        :param show: Boolean indicating whether to display the plot on screen. Default is True.
        :param save: Boolean indicating whether to save the plot as a PNG file. Default is False.
    """

    if len(lengths) == 0:
        print("Warning: No step length data to plot.")
        return

    plt.hist(lengths, bins=30, label="Step lengths")
    plt.title("Distribution of Step Lengths")
    plt.xlabel("Step Length (m)")
    plt.ylabel("Frequency")
    plt.legend()

    _handle_plot(show=show, save=save, save_dir=save_dir, filename="step_length_histogram.png")