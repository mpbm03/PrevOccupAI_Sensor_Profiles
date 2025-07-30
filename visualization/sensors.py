"""
Functions for visualizing sensor data and highlighting movement periods.

Available Functions
-------------------
[Public]
plot_sensor(...): Plots time series data for a specified sensor axis, highlighting detected movement periods.
plot_signal_with_envelope(...): Visualizes filtered acceleration signal with its envelope and marks detected valid peaks.
------------------

"""


# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import matplotlib.pyplot as plt

# internal imports
from constants import ACC_ENVELOPE , TIME , ACC_Y_COL
from .plot_utils import _handle_plot , _highlight_movement_regions

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
VALID_SENSORS = {
    'ACC': 'Accelerometer',
    'GYR': 'Gyroscope',
    'MAG': 'Magnetometer',
    'ROT': 'Rotation'
}

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def plot_sensor(data, save_dir: str, sensor='ACC', axis='y', show=True, save=False)-> None:
    """
    Plots the time series data for a specific sensor and axis, highlighting detected movement periods.

    Movement periods are highlighted as green shaded regions based on the 'is_walking' boolean column.
    The function checks if the specified sensor and axis exist in the DataFrame.

    :param data: pandas DataFrame containing sensor data with columns in the format '{axis}_{sensor}',
                 e.g., 'y_ACC', 'x_GYR', etc., as well as 't' for time and 'is_walking' boolean column.
    :param save_dir: String indicating the directory where plots should be saved if `save=True`.
    :param sensor: String indicating the sensor to plot (e.g., 'ACC', 'GYR', 'MAG', 'ROT'). Case-insensitive.
    :param axis: String indicating the axis to plot (e.g., 'x', 'y', 'z'). Case-insensitive.
    :param show: Boolean indicating whether to display the plot on screen. Default is True.
    :param save: Boolean indicating whether to save the plot as a PNG file. Default is False.


    :raises ValueError: If the sensor is not in VALID_SENSORS or the constructed column name is not in the DataFrame.

    """
    # Standardize sensor name to uppercase and axis to lowercase
    sensor = sensor.upper()
    axis = axis.lower()

    # Check if the provided sensor is in the predefined list of valid sensors
    if sensor not in VALID_SENSORS:
        raise ValueError(f"Sensor '{sensor}' is not valid. Choose from: {list(VALID_SENSORS.keys())}")

    # Construct the column name based on the axis and sensor
    col_name = f"{axis}_{sensor}"

    # Check if the constructed column name exists in the DataFrame
    if col_name not in data.columns:
        raise ValueError(f"Column '{col_name}' not found in the DataFrame.")

    plt.figure(figsize=(15, 4))

    # Plot the sensor data over time
    plt.plot(data[TIME], data[col_name], label=col_name)

    # Highlight regions where movement is detected
    _highlight_movement_regions(data)

    plt.title(f"Detected Movement Periods - {VALID_SENSORS[sensor]} ({axis.upper()}-axis)")
    plt.xlabel('Time (s)')
    plt.ylabel(f"{VALID_SENSORS[sensor]}")
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()

    _handle_plot(show=show, save=save, save_dir=save_dir, filename=f"{axis}_{sensor}_plot.png")


def plot_signal_with_envelope(data, valid_peaks, save_dir: str, show=True, save=False)-> None:
    """
    Plots the filtered Y-axis acceleration signal with its envelope and marks valid peaks.

    :param data: pandas DataFrame containing columns:
                 - 't': time values
                 - 'y_ACC': filtered Y-axis acceleration
                 - envelope column as specified by `envelope_col` (default: 'envelope_ACC')
    :param valid_peaks: Indexes or boolean mask identifying the valid peak locations within the data.
    :param save_dir: String indicating the directory where plots should be saved if `save=True`.
    :param show: Boolean indicating whether to display the plot on screen. Default is True.
    :param save: Boolean indicating whether to save the plot as a PNG file. Default is False.


    """
    # Extract the filtered acceleration signal from the Y-axis
    acc_y_filtered = data[ACC_Y_COL]

    plt.figure(figsize=(14, 5))

    # Plot the filtered acceleration signal over time
    plt.plot(data[TIME], acc_y_filtered, label='Filtered Y_ACC', alpha=0.6)

    # Plot the envelope signal
    plt.plot(data[TIME], data[ACC_ENVELOPE], label='Envelope', color='orange', linewidth=2)

    # Highlight valid peaks using red 'x' markers
    plt.plot(data.loc[valid_peaks, TIME], data.loc[valid_peaks, ACC_Y_COL], 'rx', label='Valid Peaks', markersize=8)

    # Highlight movement periods using the 'activity' column
    _highlight_movement_regions(data)

    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.title('Signal with Envelope and Peaks')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    _handle_plot(show=show, save=save, save_dir=save_dir, filename="signal_with_envelope_plot.png")







