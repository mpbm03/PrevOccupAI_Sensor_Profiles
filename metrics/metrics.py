"""
Functions for analyzing walking and stationary behavior using 2D displacements and rotation data.

Available Functions
-------------------
[Public]
analyze_total_movement(...): Analyzes continuous walking segments, computing distance, duration, and average speed per movement block.
analyze_stationary_segments(...): Analyzes stationary periods, detecting significant rotational movement within fixed time windows.
calculate_activity_proportions(...): Calculates the proportion of time spent in Walking, Sitting, and Standing activities.
-------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from constants import ROT_COL, TIME, ACTIVITY, WALKING, BLOCK_ID, WALKING_NAME, SITTING_NAME, STANDING_NAME
from Trajectory.step_detection import STEP_LENGTH
import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
TIME_DIFF = 'delta_t'
STEP_DIST = 'step_dist'
TOTAL_DURATION = 'total_duration'
# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def analyze_total_movement(data: pd.DataFrame, valid_peaks: np.ndarray, sampling_freq: float) -> dict:
    """
    Analyzes walking activity in sensor trajectory data and computes movement statistics.

    The function identifies continuous walking segments from the activity column,
    then calculates for each segment:
        - Distance (meters)
        - Duration (seconds)
        - Average speed (m/s)

    It also returns overall statistics for the entire walking period:
        - Total distance (meters)
        - Total duration of walking (seconds)
        - Average walking speed (m/s) = total_distance / total_duration
        - Total number of steps
        - Total number of walking segments
        - Distance per total recording duration (m/s) = total_distance / total_recording_duration
        - Steps per total recording duration (steps/s) = n_steps / total_recording_duration

    :param data: DataFrame containing position, time, and activity state.
    :param valid_peaks: Array of indices corresponding to detected step events.
    :param sampling_freq: Sampling frequency of the recording in Hz.

    :return: dict with:
        - total_distance: float
        - total_duration: float
        - avg_speed: float
        - n_steps: int
        - n_segments: int
        - total_recording_duration: float
        - distance_per_total_duration: float
        - steps_per_total_duration: float
        - movements: list of dicts with keys: distance, duration, avg_speed
    """
    print("Computing movement metrics")

    # Compute time difference between samples
    data[TIME_DIFF] = data[TIME].diff().fillna(0)

    # Group continuous walking segments using a segment/block ID
    walking_groups = data[data[ACTIVITY] == WALKING].groupby(BLOCK_ID)

    # Initialize accumulators
    total_distance = 0
    total_duration = 0
    movements = []

    # Loop over walking segments to compute metrics
    for movement_id, (_, walking_group) in enumerate(walking_groups, start=1):
        distance = walking_group[STEP_LENGTH].sum()
        duration = walking_group[TIME_DIFF].sum()
        avg_speed = distance / duration if duration > 0 else 0

        print(f"Walking {movement_id}:")
        print(f"   - Distance: {distance:.2f} m")
        print(f"   - Duration: {duration:.2f} s")
        print(f"   - Average speed: {avg_speed:.2f} m/s\n")

        total_distance += distance
        total_duration += duration

        movements.append({
            "distance": distance,
            "duration": duration,
            "avg_speed": avg_speed
        })

    # Compute total average speed
    total_avg_speed = total_distance / total_duration if total_duration > 0 else 0

    # Compute total recording duration and derived metrics
    total_recording_duration = len(data) / sampling_freq
    distance_per_total_duration = total_distance / total_recording_duration if total_recording_duration > 0 else 0
    steps_per_total_duration = len(valid_peaks) / total_recording_duration if total_recording_duration > 0 else 0

    print(f"Total walking distance: {total_distance:.2f} m")
    print(f"Total duration (walking): {total_duration:.2f} s")
    print(f"Average walking speed: {total_avg_speed:.2f} m/s")
    print(f"Total detected steps: {len(valid_peaks)}")
    print(f"Total walking segments: {len(movements)}")
    print(f"Distance per total duration: {distance_per_total_duration:.4f} m/s")
    print(f"Steps per total duration: {steps_per_total_duration:.4f} steps/s\n")

    return {
        "total_distance": total_distance,
        "total_duration": total_duration,
        "avg_speed": total_avg_speed,
        "n_steps": len(valid_peaks),
        "n_segments": len(movements),
        "distance_per_total_duration": distance_per_total_duration,
        "steps_per_total_duration": steps_per_total_duration,
        "movements": movements
    }


def analyze_stationary_segments(
        data: pd.DataFrame,
        activity_code: int,
        rotation_threshold: float = 50,
        window_s: float = 1.5,
        fs: int = 100,
        overlap: float = 0.0
):
    """
    Analyzes significant rotational movement during stationary periods (sitting or standing).

    Identifies continuous blocks of non-walking intervals and slides a time window over them
    to detect significant torso rotations. Useful for quantifying meaningful posture changes
    even while the subject appears still.

    :param data: DataFrame containing time, activity state, and rotation.
    :param activity_code: Integer activity label (0 = sitting, 1 = standing).
    :param rotation_threshold: Minimum total angular change (degrees) to count as significant.
    :param window_s: Window size in seconds.
    :param fs: Sampling frequency (Hz).
    :param overlap: Overlap between windows (from 0.0 to 1.0).

    :return: dict with:
        - n_segments: int, number of continuous stationary segments
        - total_duration: float, total time spent in this activity
        - total_windows: int, total number of analysis windows
        - total_significant_windows: int, number of windows with significant rotation
        - rotation_percent: float, percentage of windows with significant rotation
        - segments: list of dicts, each with:
            - segment_id: int
            - duration_s: float
            - n_windows: int
            - significant_windows: int
    """
    print(f"Computing {'Sitting' if activity_code == 0 else 'Standing'} metrics")

    if not (0 <= overlap <= 1):
        raise ValueError(f"Invalid overlap: {overlap}. It must be between 0 and 1.")

    window_size = int(window_s * fs)
    step_size = int(window_size * (1 - overlap))

    # Filter data for the selected stationary activity
    stationary_data = data[data[ACTIVITY] == activity_code]

    # Group into continuous stationary blocks
    stationary_blocks = stationary_data.groupby(BLOCK_ID)

    total_duration = 0
    total_significant_windows = 0
    total_windows = 0
    n_segments = 0
    segments = []

    for segment_id, (_, block) in enumerate(stationary_blocks, start=1):
        n_samples = len(block)
        if n_samples < window_size:
            continue

        significant_windows = 0
        num_windows = 0

        for start in range(0, n_samples - window_size + 1, step_size):
            end = start + window_size
            window = block.iloc[start:end]

            angles = np.deg2rad(window[ROT_COL].values)
            unwrapped = np.unwrap(angles)
            cumulative_change = np.rad2deg(unwrapped.max() - unwrapped.min())

            if cumulative_change > rotation_threshold:
                significant_windows += 1
            num_windows += 1

        duration = n_samples / fs
        total_duration += duration
        total_significant_windows += significant_windows
        total_windows += num_windows
        n_segments += 1

        segments.append({
            "segment_id": segment_id,
            "duration_s": duration,
            "n_windows": num_windows,
            "significant_windows": significant_windows
        })

        print(f"{'Sitting' if activity_code == 0 else 'Standing'} Segment {segment_id}:")
        print(f"   - Duration: {duration:.2f} s")
        print(f"   - Significant rotation windows: {significant_windows} / {num_windows}\n")

    rotation_percent = (total_significant_windows / total_windows * 100) if total_windows > 0 else 0.0

    print(f"==== Summary for {'Sitting' if activity_code == 0 else 'Standing'} ====")
    print(f"Total segments: {n_segments}")
    print(f"Total duration: {total_duration:.2f} s")
    print(f"Significant rotation: {total_significant_windows}/{total_windows} windows ({rotation_percent:.1f}%)\n")

    return {
        "n_segments": n_segments,
        "total_duration": total_duration,
        "total_windows": total_windows,
        "total_significant_windows": total_significant_windows,
        "segments": segments,
        "rotation_percent": total_significant_windows/total_windows
    }


def calculate_activity_proportions(
    movement_metrics: dict,
    sitting_metrics: dict,
    standing_metrics: dict
) -> dict:
    """
    Calculates the proportion of time spent in Walking, Sitting, and Standing activities.

    :param movement_metrics: Dictionary containing metrics for walking activity.
    :param sitting_metrics: Dictionary containing metrics for sitting activity.
    :param standing_metrics: Dictionary containing metrics for standing activity.
    :return: Dictionary with proportions of activities: {'walking_proportion': ..., 'sitting_proportion': ..., 'standing_proportion': ...}
    """
    durations = {
        WALKING_NAME: movement_metrics.get(TOTAL_DURATION, 0),
        SITTING_NAME: sitting_metrics.get(TOTAL_DURATION, 0),
        STANDING_NAME: standing_metrics.get(TOTAL_DURATION, 0)
    }

    total = sum(durations.values())
    if total == 0:
        return {
            "walking_proportion": 0.0,
            "sitting_proportion": 0.0,
            "standing_proportion": 0.0
        }

    return {
        "walking_proportion": durations[WALKING_NAME] / total* 100,
        "sitting_proportion": durations[SITTING_NAME] / total* 100,
        "standing_proportion": durations[STANDING_NAME] / total* 100
    }