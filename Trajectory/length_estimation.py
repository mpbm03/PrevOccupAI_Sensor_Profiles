"""
This module provides utilities for processing acceleration data from inertial sensors,
specifically focusing on transforming device-frame acceleration into the world frame
using orientation information (quaternions), and estimating step lengths based on
transformed acceleration signals.

Available Functions
-------------------
[Public]
rotate_acceleration(...): Rotates acceleration data from the device frame to the world frame using a quaternion-based rotation matrix.
compute_step_lengths_by_blocks(...): Estimates step lengths for walking activity by analyzing acceleration data within segmented movement blocks.
    It integrates acceleration signals to compute average velocity and uses an adaptive scaling model to estimate each step length.

-------------------
[Private]
_adaptive_k(...): Adaptive scaling coefficient K based on average estimated velocity.

"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from constants import WORLD_ACC, ACC_Y_COL, ACC_Z_COL, BLOCK_ID, ACTIVITY, WALKING
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.spatial.transform import Rotation as R
from typing import Tuple


# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
ACC_X_COL = 'x_ACC'
ROT_X_COL = 'x_ROT'
ROT_Y_COL = 'y_ROT'
ROT_Z_COL = 'z_ROT'
ROT_W_COL = 'w_ROT'

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def rotate_acceleration(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds world-frame acceleration columns to the DataFrame by rotating the local acceleration vectors
    using quaternion-based orientation.

    This function uses the device's quaternion rotation data to transform acceleration vectors
    from the local (device) reference frame into the global (world) reference frame.

    :param data: A DataFrame that includes:
        - 'x_ACC', 'y_ACC', 'z_ACC': local acceleration components
        - 'x_ROT', 'y_ROT', 'z_ROT', 'w_ROT': quaternion components
    :return: The same DataFrame with new columns:
        - 'a_rX', 'a_rY', 'a_rZ': world-frame acceleration components
    """
    # Extract quaternions and acceleration data as NumPy arrays
    quats = data[[ROT_X_COL, ROT_Y_COL, ROT_Z_COL, ROT_W_COL]].values
    accs = data[[ACC_X_COL, ACC_Y_COL, ACC_Z_COL]].values

    # Create rotation objects from quaternions
    rotations = R.from_quat(quats)

    # Rotate all acceleration vectors to world frame
    acc_world = rotations.apply(accs)

    # Add rotated acceleration columns to the original DataFrame
    data[WORLD_ACC[0]] = acc_world[:, 0]
    data[WORLD_ACC[1]] = acc_world[:, 1]
    data[WORLD_ACC[2]] = acc_world[:, 2]

    return data

def compute_step_lengths_by_blocks(data: pd.DataFrame, peaks_idx: np.ndarray, valleys_idx: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate step lengths within walking blocks using block_id to avoid errors caused
    by step calculations across different movements.

    :param data: DataFrame with acceleration columns and 'block_id' and 'activity'
    :param peaks_idx: Indices of detected peaks
    :param valleys_idx: Indices of detected valleys
    :param fs: Sampling frequency in Hz

    :return:
        - Array of estimated step lengths
        - Array of valid peak indices used
    """
    step_lengths = []
    used_peaks = []

    # Loop through each unique block defined by block_id
    for block_id, block_data in data.groupby(BLOCK_ID):
        # Only process blocks where the activity is 'walking'
        if block_data[ACTIVITY].iloc[0] != WALKING:
            continue

        # Get index range for the current block
        start_idx = block_data.index.min()
        end_idx = block_data.index.max()

        # Filter peaks and valleys that fall within the current block
        peaks_block = [p for p in peaks_idx if start_idx <= p <= end_idx]
        valleys_block = [v for v in valleys_idx if start_idx <= v <= end_idx]

        # Skip block if insufficient peaks or valleys
        if len(peaks_block) < 1 or len(valleys_block) < 2:
            continue

        N = min(len(peaks_block), len(valleys_block))

        # Compute step lengths for each interval defined by consecutive valleys
        for i in range(N - 1):
            start = valleys_block[i]
            end = valleys_block[i + 1]

            # Find peaks between the two valleys
            peaks_in_interval = [p for p in peaks_block if start < p < end]
            if not peaks_in_interval:
                continue

            peak = peaks_in_interval[0]
            used_peaks.append(peak)

            # Extract world-frame acceleration data for the step interval
            aX = data.loc[start:end, 'a_rX'].values
            aY = data.loc[start:end, 'a_rY'].values
            aZ = data.loc[start:end, 'a_rZ'].values
            t = np.linspace(0, len(aX) / fs, len(aX))

            # Numerically integrate acceleration to estimate velocity
            vX = cumtrapz(aX, t, initial=0)
            vY = cumtrapz(aY, t, initial=0)
            vZ = cumtrapz(aZ, t, initial=0)

            # Compute average velocity components
            v_stepX = np.mean(vX)
            v_stepY = np.mean(vY)
            v_stepZ = np.mean(vZ)

            # Calculate total step velocity magnitude
            v_step = np.sqrt(v_stepX**2 + v_stepY**2 + v_stepZ**2)

            # Calculate adaptive velocity factor K_vel (Equation 12)
            K_vel = _adaptive_k(v_step)

            # Calculate acceleration magnitude during the step
            a_mag = np.sqrt(aX**2 + aY**2 + aZ**2)

            A_max = np.max(a_mag)
            A_min = np.min(a_mag)

            # Estimate step length (Equation 15)
            L_step = K_vel * np.power(A_max - A_min, 0.25)
            step_lengths.append(L_step)

    return np.array(step_lengths), np.array(used_peaks)


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _adaptive_k(v_avg: float) -> float:
    """
    Calculates the adaptive coefficient K based on the estimated average step velocity.

    Instead of using a constant K value for all steps, this method derives K as a
    polynomial function of the average step velocity (v_avg) to improve the accuracy
    of velocity estimation from inertial measurements.

    The process used to determine the polynomial coefficients was as follows:

    1. An N-fold Cross-Validation was performed on a dataset containing ground truth
       acceleration and velocity data.

    2. The dataset was randomly split into training and testing sets, with different
       ratios ranging from 20% to 90%. Based on the Root Mean Square Errors (RMSE) of
       actual vs. estimated velocities, it was decided to use 70% of the data for training
       and 30% for testing.

    3. Polynomial models of different degrees were fitted between step average velocity
       and K. The variance of the Residual Sum of Squares (vRSS) was used as the main
       evaluation metric.

    4. This process was repeated 100 times (N = 100) to ensure robustness. The degree
       that minimized the variance on the test set was selected.

    5. As a result, a second-degree polynomial was chosen. The final linear regression
       model used both the first- and second-degree terms of v_avg:

       K(v_avg) = 0.68 - 0.37 * v_avg + 0.15 * v_avg²

    :param v_avg: Average acceleration component
    :return: Adaptive coefficient K
    """
    # Adaptive coefficient K based on estimated average velocity
    return 0.68 - 0.37 * v_avg + 0.15 * v_avg**2