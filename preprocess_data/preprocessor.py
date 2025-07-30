"""
Functions for loading and pre-processing the Smartphone data.

Available Functions
-------------------
[Public]
data_preprocessing(...): Loads and pre-processes sensor data from a given folder, and adds a time column based on the sampling frequency.
------------------

"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline


# internal imports
from .preprocessing import pre_process_sensors
from constants import FS , TIME

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def data_preprocessing(sensor_data: pd.DataFrame, fs=FS, scalar_first=False) -> pd.DataFrame:
    """
    Loads and pre-processes sensor data from a given folder, and adds a time column based on the sampling frequency.

    :param sensor_data: DataFrame containing the sensor data
    :param fs: Sampling frequency of the sensors in Hz. Default is FS.
    :param scalar_first: Boolean flag used in quaternion-based processing (e.g., SLERP). If True, scalar component
                         of quaternions comes first. Default is False.

    :return: A pandas DataFrame containing the pre-processed sensor data, with columns for each sensor axis and an
             additional time column ('t') computed from the sampling frequency.
    """

    # Get the sensor names (excluding the first column, which is the time)
    sensor_names = sensor_data.columns.values[1:]

    # Remove the time column and convert to NumPy array
    processed_sensor_data = sensor_data.values[:, 1:]

    # Apply sensor pre-processing
    processed_sensor_data = pre_process_sensors(processed_sensor_data, sensor_names, fs, scalar_first)

    # Remove initial impulse response (250 samples)
    processed_sensor_data = processed_sensor_data[250:, :]

    # Reconstruct DataFrame
    df = pd.DataFrame(processed_sensor_data, columns=sensor_names)

    # Add artificial time column
    n = len(df)
    df[TIME] = np.arange(n) / fs

    return df