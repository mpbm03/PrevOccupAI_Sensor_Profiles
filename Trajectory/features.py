"""
Functions for extracting features from quaternion and acceleration signals.

Available Functions
-------------------
[Public]
calculate_dominant_frequency(...): Estimates dominant frequency from filtered Y-axis acceleration using FFT.
calculate_relative_rotation(...): Computes relative rotation projected onto the XZ plane using quaternions.

-------------------
[Private]
project_rotation_difference_correct(...): Projects relative rotation vectors onto the XZ plane and returns angles in degrees.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from constants import FS, ACC_Y_COL, ROT_COL, ACTIVITY, WALKING

from typing import Tuple
import numpy as np
from scipy.fft import fft, fftfreq
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import detrend
import matplotlib.pyplot as plt



# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
quaternion_columns = ['x_ROT','y_ROT','z_ROT','w_ROT']


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def calculate_dominant_frequency_walking_only(data: pd.DataFrame, fs: int = FS) -> int:
    """
    Estimates the dominant frequency of a filtered acceleration signal using FFT,
    restricted to periods when the person is walking (activity == 2).

    Uses the Fourier transform on the 'y_ACC_filt' column to find the frequency
    with the highest magnitude during walking segments only. Also calculates a
    minimum interval between peaks based on this dominant frequency.

    :return:
        - dominant_freq: Estimated dominant frequency in Hz.
        - minimum_interval: Minimum interval between peaks in samples (int).
    """
    # Filter walking segments
    walking_data = data[data[ACTIVITY] == WALKING]

    if len(walking_data) < fs * 2:
        print("Not enough walking data to compute frequency.")
        return int(fs * 0.5)  # Default fallback (0.5s = ~2Hz)

    # Detrend the walking signal
    acc_y = detrend(walking_data[ACC_Y_COL].values)

    n = len(acc_y)
    t = 1 / fs

    # FFT
    yf = fft(acc_y)
    xf = fftfreq(n, t)
    xf = xf[:n // 2]
    yf = 2.0 / n * np.abs(yf[:n // 2])

    # Ignore frequency = 0 Hz
    max_index = np.argmax(yf[1:]) + 1
    dominant_freq = xf[max_index]
    magnitude_max = yf[max_index]

    print(f"Dominant frequency during walking: {dominant_freq:.2f} Hz (Magnitude: {magnitude_max:.2f})")

    # Convert to minimum interval between peaks
    interval = fs / dominant_freq
    minimum_interval = int(interval * 0.5)

    return minimum_interval

def calculate_relative_rotation(data: pd.DataFrame) -> pd.DataFrame:
    """
    calculate_relative_rotation:
    Calculates the relative rotation projected onto the XZ plane from quaternions.

    Adds a new column 'rot_diff' to the DataFrame with the angular variation in degrees,
    projected onto the XZ plane using a reference vector.

    :param data: DataFrame containing the quaternion columns (x, y, z, w).

    :return:
        - pd.DataFrame: DataFrame with an additional 'rot_diff' column.
    """
    # Convert quaternions to Rotation objects
    rotations = R.from_quat(data[quaternion_columns].values)

    # Use the first quaternion as the reference
    initial_rotation = R.from_quat([data[quaternion_columns].values[0]])

    # Compute relative rotations: inv(initial) * each rotation
    relative_rotations = initial_rotation.inv() * rotations

    # Project the relative rotations onto the XZ plane (vectorized)
    ref_vector = np.array([[1, 0, 0]])  # shape (1, 3)
    rotated_vectors = relative_rotations.apply(ref_vector)  # shape (N, 3)

    # Extract x and z components
    x = rotated_vectors[:, 0]
    z = rotated_vectors[:, 2]

    # Compute angles in radians and convert to degrees
    angles_rad = np.arctan2(z, x)
    angles_deg = -np.rad2deg(angles_rad)

    # Add result to DataFrame
    data[ROT_COL] = angles_deg

    return data


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

# Function to project relative rotation onto the XZ plane
def _project_rotation_difference_correct(rotations) -> np.array:
    """
        Projects relative rotations onto the XZ plane.

        :param rotations: Array of Rotation objects.
        :return: Numpy array of projected angles in degrees.
        """
    ref_vector = np.array([1, 0, 0])  # reference vector along the X axis

    projected_angles = []  # List to store projected angles in degrees

    # Iterate over each rotation object in the input array
    for rotation in rotations:
        # Apply the rotation to the reference vector
        rotated_vector = rotation.apply(ref_vector)

        # Extract the X and Z components of the rotated vector
        x = rotated_vector[0]
        z = rotated_vector[2]

        # Compute the angle of the vector projection onto the XZ plane
        # using the arctangent of Z over X (returns radians)
        angle_rad = np.arctan2(z, x)

        # Convert the angle from radians to degrees
        angle_deg = np.rad2deg(angle_rad)

        # Append the angle in degrees to the list
        projected_angles.append(angle_deg)
    return np.array(projected_angles)

