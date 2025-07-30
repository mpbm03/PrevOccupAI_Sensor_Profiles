"""
Functions for computing and analyzing 2D trajectories using step data and orientation information.

Available Functions
-------------------
[Public]
compute_adaptive_trajectory(...): Computes 2D estimated trajectory from step timing, step lengths, and orientation.
compute_position_density(...): Computes spatial density of positions based on movement and pauses.

-------------------
[Private]
_extract_indices(...): Extracts sample indices from a boolean mask, ensuring `min_interval` distance between each.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from constants import ROT_COL, ANGLE, DISP_X, DISP_Y, TIME, TRAJECTORY_X, TRAJECTORY_Y, ACTIVITY, SITTING, STANDING
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def compute_adaptive_trajectory(data: pd.DataFrame, step_times: np.ndarray, step_lengths: np.ndarray) -> pd.DataFrame:
    """
    Computes a 2D estimated trajectory using step timing, step lengths,
    and adaptive orientation from rotational data.

    :param data: DataFrame with columns: 't' (time), and 'rot_diff' (in degrees).
    :param step_times: Timestamps of detected steps (seconds).
    :param step_lengths: Step lengths (meters).

    :return: Updated DataFrame with the following additional columns:
             - 'angle_rad': 'rot_diff' angle converted to radians.
             - 'dx': Step-wise displacement in the X direction (meters).
             - 'dy': Step-wise displacement in the Y direction (meters).
             - 'x': Cumulative X position (meters).
             - 'y': Cumulative Y position (meters).
    """
    # Convert rotation angle to radians
    data[ANGLE] = np.deg2rad(data[ROT_COL])

    # Initialize displacement columns with zeros
    data[DISP_X] = 0.0
    data[DISP_Y] = 0.0

    # Ensure timestamps are sorted for searchsorted
    timestamps = data[TIME].values
    step_times = np.array(step_times)

    # Use searchsorted to find closest indices efficiently
    indices = np.searchsorted(timestamps, step_times, side='left')
    indices = np.clip(indices, 0, len(data) - 1)  # prevent out-of-bounds

    # Discard last step to match original logic
    indices = indices[:-1]
    step_lengths = step_lengths[:len(indices)]

    # Get corresponding angles at each step index
    angles = data[ANGLE].values[indices]

    # Compute displacement vectors
    dx = step_lengths * np.cos(angles)
    dy = step_lengths * np.sin(angles)

    # Store displacements at correct rows
    data.loc[indices, DISP_X] = dx
    data.loc[indices, DISP_Y] = dy

    # Cumulative position from displacements
    data[TRAJECTORY_X] = data[DISP_X].cumsum()
    data[TRAJECTORY_Y] = data[DISP_Y].cumsum()

    return data


def compute_position_density(data: pd.DataFrame, step_times: np.ndarray, min_interval: int,
                              precision: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes density of positions using one position per step and one every `min_interval` samples during pauses.

    :param data: DataFrame containing trajectory and timestamp information.
    :param step_times: Array of timestamps (in seconds) when steps occur.
    :param min_interval: Minimum number of samples between stored positions during inactivity.
    :param precision: Decimal precision for rounding positions.

    :return:
        - xs: Array of rounded X coordinates.
        - ys: Array of rounded Y coordinates.
        - densities: Array with the frequency of each unique (x, y) position.
    """

    # Round trajectory coordinates to reduce precision and improve frequency counting performance
    rounded_x = np.round(data[TRAJECTORY_X].values, precision)
    rounded_y = np.round(data[TRAJECTORY_Y].values, precision)

    # List to hold all rounded positions to be counted
    rounded_positions = []

    # Part 1: Add one position per step using np.searchsorted (fast timestamp lookup)
    # Retrieve the timestamp values from the dataset
    timestamps = data[TIME].values

    # Find the index in the dataset where each step time fits (left insertion)
    step_indices = np.searchsorted(timestamps, step_times, side='left')

    # Ensure no index exceeds the bounds of the DataFrame
    step_indices = np.clip(step_indices, 0, len(data) - 1)

    # Extract the (x, y) positions corresponding to step times
    step_positions = list(zip(rounded_x[step_indices], rounded_y[step_indices]))
    rounded_positions.extend(step_positions)

    # Part 2: Add positions during sitting or standing every `min_interval` samples
    # Extract activity labels
    activity = data[ACTIVITY].values

    # Create boolean masks for sitting and standing states
    sitting_mask = (activity == SITTING)
    standing_mask = (activity == STANDING)

    # Get sampled indices during pauses (sitting and standing)
    sitting_indices = _extract_indices(sitting_mask, min_interval)
    standing_indices = _extract_indices(standing_mask, min_interval)
    pause_indices = np.array(sitting_indices + standing_indices, dtype=int)

    # Extract (x, y) positions for pauses
    pause_positions = list(zip(rounded_x[pause_indices], rounded_y[pause_indices]))
    rounded_positions.extend(pause_positions)

    # Count position occurrences
    counter = Counter(rounded_positions)

    # Unpack the counter into separate lists for X, Y, and count
    xs, ys, densities = zip(*[(x, y, count) for (x, y), count in counter.items()])

    return np.array(xs), np.array(ys), np.array(densities)


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _extract_indices(mask: np.array, min_interval: int) -> List[int]:
    """
       Selects indices from a boolean mask ensuring at least `min_interval` samples between selections.
    """
    # Get the indices where the mask is True
    indices = np.where(mask)[0]

    # If there are no True values, return an empty list
    if len(indices) == 0:
        return []

    # List to store the selected indices
    selected = []

    # Initialize the index of the last selected element far enough in the past
    # This allows the first eligible index to always be included
    last_idx = -min_interval

    # Iterate through all valid indices
    for current_index  in indices:
        # Only select the current index if it's far enough from the last one
        if current_index - last_idx >= min_interval:
            selected.append(current_index )
            last_idx = current_index # Update the last selected index
    # Return the list of selected indices that meet the spacing condition
    return selected








