"""

Functions for processing and analyzing sensor data to detect and measure walking steps.

Available Functions
-------------------
[Public]
step_detection(...): Combines all processing stages to detect valid steps and estimate step lengths.
detect_steps_moving_average(...): Detects steps and valleys using moving average (envelope) of rectified ACC_Y signal.
filter_steps_during_direction_changes(...): Filters out steps that happen during abrupt direction changes.
filter_steps_by_gyro(...): Filters steps based on gyroscope threshold on X-axis.
regularize_valleys_and_peaks(...): Ensures at least one peak exists between every two valleys.
filter_valid_peaks(...): Filters peaks that occur between two consecutive valleys.

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from constants import FS, ACC_Y_COL, ROT_COL, ACTIVITY, ACC_ENVELOPE, WALKING, ANGULAR_VEL

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Tuple
from numba import njit



from .length_estimation import rotate_acceleration, compute_step_lengths_by_blocks
# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #

STEP_LENGTH = 'step_length'
GYR_X_COL = 'x_GYR'
GYRO_THRESHOLD = 0


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def step_detection(data: pd.DataFrame, min_dist: int, peak_threshold: float = 0.3, valley_threshold: float =0.3,
                   window_mov_average: int=100, rotation_change_threshold : int = 60, fs: int = FS) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Function to detect walking steps and estimate step lengths from inertial sensor data. This function combines
    multiple processing stages including peak/valley detection, filtering based on rotation and gyroscope signals,
    and step length estimation using the Z-axis acceleration. The output includes the estimated step lengths and
    the indices of the valid detected steps.

    Some of the implemented methods were based on this article: https://www.mdpi.com/1424-8220/16/9/1423
    Such as: regularize_valleys_and_peaks, filter_valid_peaks, compute_step_lengths_by_blocks

    The default values for the peak and valley thresholds, as well as for the moving average window,
    were selected based on empirical observations across multiple subjects.

    :param data: pandas DataFrame containing the sensor data (e.g., accelerometer, gyroscope, rotation).
    :param min_dist: minimum distance (in samples) between detected peaks/valleys.
    :param peak_threshold: Threshold for peak detection, expressed as a fraction (0 to 1) of the envelope's maximum value.
                       Peaks in the original signal envelope must be at least this fraction of the maximum to be detected.
    :param valley_threshold: Threshold for valley detection, expressed as a fraction (0 to 1) of the envelope's maximum value.
                        Since valleys are detected by finding peaks in the inverted envelope,
                        this threshold corresponds to the minimum height these inverted peaks must have,
                        i.e., valleys in the original signal must be at least this fraction below the maximum envelope value.
    :param window_mov_average: window size (in samples) for computing the moving average (envelope).
    :param rotation_change_threshold : Threshold, in deg/s, applied to the derivative of the rotation (angular velocity) signal. It is used to filter out
        steps detected during abrupt orientation changes (e.g., sharp turns)
    :param fs: sampling frequency in Hz.

    :return: a tuple containing:
            - lengths: Estimated step lengths for each detected step.
            - peaks_idx: Indices of detected steps.
            - angular_velocity: Derivative of the rotation signal used to detect abrupt motion.
    """

    # Detect peaks and valleys in the Y-axis acceleration signal during walking periods,
    # using a moving average (envelope) of the rectified signal.
    # Also returns the DataFrame updated with the envelope column.
    peaks_idx, valleys_idx, data = detect_steps_moving_average(data, peak_threshold, valley_threshold,
                                                                min_dist, window_mov_average)

    # Filter out peaks (steps) that occur during abrupt direction changes,
    # based on the derivative of the rotation signal (angular velocity).
    # Also returns the derivative of the rotation for optional analysis.
    peaks_idx, data = filter_steps_during_direction_changes(data, peaks_idx, fs, rotation_change_threshold)

    # Filters the peaks using the X-axis gyroscope signal,
    # removing steps where the gyroscope value exceeds the defined threshold.
    peaks_idx = filter_steps_by_gyro(data, peaks_idx)

    # Ensures that each pair of valleys contains at least one peak.
    # If no peak exists between a valley pair, both valleys are replaced with a new one.
    valleys_idx = regularize_valleys_and_peaks(data, peaks_idx, valleys_idx, min_dist)

    # Keeps only valid peaks — those that lie between two consecutive valleys.
    peaks_idx = filter_valid_peaks(peaks_idx, valleys_idx)

    # Rotates the acceleration data from the sensor frame to the world frame
    data = rotate_acceleration(data)

    # Returns the estimated step lengths and the valid peaks used in the computation.
    # Step lengths are calculated by integrating the world-frame acceleration between consecutive valleys,
    # estimating average velocities during each step interval, and applying an empirical model
    # that relates acceleration magnitude changes and velocity to step length
    # Returns the estimated step lengths and the valid peaks used in the computation.
    lengths, peaks_idx = compute_step_lengths_by_blocks(data, peaks_idx, valleys_idx, fs)

    # Add step_length column: step lengths at peak indices, 0.0 elsewhere
    data[STEP_LENGTH] = pd.Series(lengths, index=peaks_idx).reindex(data.index, fill_value=0.0)

    return lengths, peaks_idx, data


def detect_steps_moving_average(data: pd.DataFrame, peak_threshold: float, valley_threshold: float,
                                min_dist: int, window_mov_average: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Processes the rectified signal using a moving average (envelope), identifies peaks and valleys
    during walking periods, and returns the indices of these events.

    :param data: DataFrame containing the acceleration signal and 'activity' column.
    :param peak_threshold: Threshold for peak detection, expressed as a fraction (0 to 1) of the envelope's maximum value.
                       Peaks in the original signal envelope must be at least this fraction of the maximum to be detected.
    :param valley_threshold: Threshold for valley detection, expressed as a fraction (0 to 1) of the envelope's maximum value.
                        Since valleys are detected by finding peaks in the inverted envelope,
                        this threshold corresponds to the minimum height these inverted peaks must have,
                        i.e., valleys in the original signal must be at least this fraction below the maximum envelope value.
    :param min_dist: Minimum distance (in samples) between detected peaks/valleys.
    :param window_mov_average: Size of the moving average window (number of samples).

    :return:
        - peaks_idx: Indices of the detected peaks in the original DataFrame.
        - valleys_idx: Indices of the detected valleys in the original DataFrame.
        - data: Updated DataFrame with a new 'acc_envelope' column (moving average of the signal).
    """

    # Rectify the entire signal
    acc_y = data[ACC_Y_COL]
    acc_y_rect = abs(acc_y)

    # Moving average (envelope)
    acc_y_smooth = acc_y_rect.rolling(window=window_mov_average, center=True).mean().fillna(0)

    # Store the envelope in the DataFrame
    data[ACC_ENVELOPE] = acc_y_smooth

    # Define thresholds
    peak_limit = acc_y_smooth.max() * peak_threshold
    valley_limit = acc_y_smooth.max() * valley_threshold

    # Mask: only consider periods when the person is walking
    mask = data[ACTIVITY] == WALKING
    acc_y_walk = acc_y[mask].values

    # Detect peaks and valleys in the envelope during walking
    peaks, _ = find_peaks(acc_y_walk, height=peak_limit, distance=min_dist)
    valleys, _ = find_peaks(-acc_y_walk, height=valley_limit, distance=min_dist)

    # Map to original indices
    idxs_walk = data.index[mask]
    peaks_idx = idxs_walk[peaks]
    valleys_idx = idxs_walk[valleys]

    return peaks_idx, valleys_idx, data

def filter_steps_during_direction_changes(data: pd.DataFrame, peaks_idx: np.ndarray,
    fs: int, angular_velocity_threshold: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Removes step events (peaks) that occur during abrupt direction changes (high rotation).

    :param data: DataFrame containing the rotation signal column.
    :param peaks_idx: Indices of the detected peaks (e.g., from step detection).
    :param fs: Sampling frequency in Hz.
    :param angular_velocity_threshold: Threshold for detecting abrupt rotational changes based on derivative of rotation (deg/s).

    :return:
        - filtered_peaks: Array of peaks that do not occur during abrupt rotations.
        - data: DataFrame containing the computed derivative of the rotation signal.
    """
    rot = data[ROT_COL].values
    dt = 1 / fs

    # Compute time derivative of rotation
    rotation_derivative = np.gradient(rot, dt)

    # Remove unrealistic angular jumps (e.g., wrapping from 180° to -180°)
    artificial_limit = 1000
    rotation_derivative[np.abs(rotation_derivative) > artificial_limit] = 0

    # Create boolean mask where abrupt rotation is detected
    is_abrupt_change  = np.abs(rotation_derivative) > angular_velocity_threshold

    # Identify continuous intervals of abrupt rotation
    abrupt_change_intervals  = []
    in_abrupt_segment = False

    for sample_index, abrupt  in enumerate(is_abrupt_change):
        if abrupt  and not in_abrupt_segment:
            start = sample_index
            in_abrupt_segment = True
        elif not abrupt  and in_abrupt_segment:
            end = sample_index
            in_abrupt_segment = False
            abrupt_change_intervals.append((start, end))
    if in_abrupt_segment:
        abrupt_change_intervals.append((start, len(is_abrupt_change)))

    # Create exclusion mask for the entire signal
    exclusion = np.zeros(len(data), dtype=bool)
    for start, end in abrupt_change_intervals:
        exclusion[start:end] = True

    # Filter out peaks that occur during abrupt rotation intervals
    filtered_peaks = [peak_index  for peak_index  in peaks_idx if not exclusion[peak_index]]

    data[ANGULAR_VEL] = rotation_derivative
    return np.array(filtered_peaks), data


def filter_steps_by_gyro(data: pd.DataFrame, peaks_idx: np.ndarray) -> np.ndarray:
    """
    Filters steps based on the X-axis gyroscope value.

    Keeps only steps where the gyroscope value is below zero.

    :param data: DataFrame containing the signal columns.
    :param peaks_idx: Indices of detected peaks (candidate steps).

    :return:
        - filtered_steps: Array of step indices that passed the gyroscope filter.
    """
    gyro = data[GYR_X_COL].values
    # Filter peaks (candidate steps) where the gyroscope value is below the threshold
    # This helps discard steps likely caused by noise or non-walking movements
    filtered_steps = [peak_index  for peak_index  in peaks_idx if gyro[peak_index] < GYRO_THRESHOLD]
    return np.array(filtered_steps)


def regularize_valleys_and_peaks(data: pd.DataFrame, peaks: np.ndarray, valleys: np.ndarray, min_dist: int) -> np.ndarray:
    """
    Ensures that each pair of consecutive valleys contains at least one peak.
    If no peak exists between two valleys, both valleys are removed and replaced
    by a new valley located at the point closest to the average of their values.

    :param peaks: Array or list of indices of detected peaks.
    :param valleys: Array or list of indices of detected valleys.
    :param min_dist: minimum distance (in samples) between detected peaks/valleys.

    :return:
        - np.ndarray: New array of regularized valley indices.

    """
    new_valleys = []
    removed_valleys = []

    signal = data[ACC_Y_COL]

    valley_index = 0
    while valley_index  < len(valleys) - 1:
        v1 = valleys[valley_index ]
        v2 = valleys[valley_index  + 1]

        # Peaks between the two valleys
        peaks_between = [p for p in peaks if v1 < p < v2]

        # Case 1: No peak → insert a new valley in the middle
        if len(peaks_between) == 0:
            removed_valleys.extend([v1, v2])
            # Average value between both valleys
            avg_val = (signal[v1] + signal[v2]) / 2
            # Segment between valleys
            segment = signal[v1:v2 + 1]
            # Local index of the value closest to the average
            local_idx = np.argmin(np.abs(segment - avg_val))
            # Global index in the DataFrame
            new_valley = v1 + local_idx

            new_valleys.append(new_valley)
            valley_index += 2  # Skip both valleys

        # Case 2: Two peaks and distant valleys → insert a new valley between the two peaks
        elif len(peaks_between) == 2 and (v2 - v1) > 2 * min_dist:
            p1, p2 = peaks_between
            segment = signal[p1:p2 + 1]
            local_idx = np.argmin(segment)
            new_valley = p1 + local_idx
            new_valleys.append(new_valley)
            valley_index += 2

        # Common case → keep the first valley
        else:
            new_valleys.append(v1)
            valley_index += 1

    return np.array(new_valleys)

@njit
def filter_valid_peaks(peaks: np.ndarray, valleys: np.ndarray) -> np.ndarray:
    """
    Filters peaks that occur between consecutive pairs of valleys.

    :param peaks: Indices of detected peaks.
    :param valleys: Indices of detected valleys.

    :return:
        - np.ndarray: Array of valid peak indices, i.e., those that occur between two consecutive valleys.
    """
    valid_peaks = []

    # Iterate over consecutive pairs of valleys
    for valley_index  in range(len(valleys) - 1):
        v1 = valleys[valley_index]
        v2 = valleys[valley_index + 1]

        # Find peaks that occur strictly between the current pair of valleys
        peaks_between = [p for p in peaks if v1 < p < v2]

        # Add the found peaks to the list of valid peaks
        valid_peaks.extend(peaks_between)

    return np.array(valid_peaks)









