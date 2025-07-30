"""
Functions for Rotation visualization.

Available Functions
-------------------
[Public]
plot_rotation_angle(...): Plots rotation angle over time with highlighted movement periods.

plot_angular_velocity(...): Plots angular velocity (derivative of rotation) and overlays threshold indicators.
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

import matplotlib.pyplot as plt

# internal imports
from constants import TIME , ROT_COL, ANGULAR_VEL
from .plot_utils import _handle_plot , _highlight_movement_regions


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def plot_rotation_angle(data, save_dir: str, show=True, save=False)-> None:
    """
    Plots the rotation angle (`rot_diff`) over time and highlights periods of movement.

    Movement periods are identified by the `activity` column in the DataFrame,
    and these periods are highlighted as green shaded regions on the plot.

    :param data: pandas DataFrame containing at least the columns:
                 - 't': time values
                 - 'rot_diff': rotation angle in degrees
                 - 'activity': column in the DataFrame that indicates the activity performed in each sample (e.g., 0 = sitting, 1 = standing, 2 = walking).
    :param save_dir: String indicating the directory where plots should be saved if `save=True`.
    :param show: Boolean indicating whether to display the plot on screen. Default is True.
    :param save: Boolean indicating whether to save the plot as a PNG file. Default is False.

    """
    plt.figure(figsize=(14, 5))

    # Plot the rotation angle over time
    plt.plot(data[TIME], data[ROT_COL], label='Rotation (degrees)', color='blue')

    # Highlight movement periods using the 'activity' column
    _highlight_movement_regions(data)

    plt.title('Rotation Over Time with Movement Periods')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotation (degrees)')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    _handle_plot(show=show, save=save, save_dir=save_dir, filename="rotation_angle_plot.png")


def plot_angular_velocity(data, threshold, save_dir: str, show=True, save=False)-> None:
    """
    Plots the derivative of the rotation angle (`rot_diff`) over time, representing angular velocity,
    and overlays threshold lines indicating significant rotation changes.

    :param data: pandas DataFrame containing at least the column 't' for time values.
    :param threshold: Float value representing the angular velocity threshold. Horizontal lines are drawn at ±threshold.
    :param save_dir: String indicating the directory where plots should be saved if `save=True`. Default is "plots".
    :param show: Boolean indicating whether to display the plot on screen. Default is True.
    :param save: Boolean indicating whether to save the plot as a PNG file. Default is False.


    """
    angular_velocity = data[ANGULAR_VEL]
    plt.figure(figsize=(10, 4))

    # Plot angular velocity over time
    plt.plot(data[TIME], angular_velocity, label='Angular velocity', color='blue')

    plt.axhline(threshold, color='red', linestyle='--')
    plt.axhline(-threshold, color='red', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity')
    plt.title('Angular Velocity with Threshold')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    _handle_plot(show=show, save=save, save_dir=save_dir, filename="angular_velocity_plot.png")

