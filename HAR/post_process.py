"""
These functions provide a set of  post-processing tools for refining activity classification outputs, such as
adjusting ambiguous predictions, correcting short or fragmented activity segments, and expanding predictions
to match the original sampling rate.

Available Functions
-------------------
[Public]
threshold_tuning(...): Adjusts predictions where the probability difference between 'stand' and 'sit' is low, reducing misclassifications.
find_class_segments(...): Identifies contiguous segments of a specific class label from a sequence of predictions.
correct_short_segments(...): Replaces short-duration segments of a given class with the most likely neighboring class.
expand_classification(...): Expands window-level classification predictions to match the original sample-level timeline.
heuristics_correction(...): Applies minimum-duration filters to clean up short segments for each class.

"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from constants import SITTING, STANDING
import numpy as np
from typing import Tuple, List, Dict, Union
from collections import Counter

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def threshold_tuning(probabilities: np.ndarray, y_pred: Union[np.ndarray, list],
                     sit_label: int = SITTING, stand_label: int = STANDING, threshold: float = 0.1) -> np.ndarray:
    """
    Adjusts predictions for a classifier by reducing confusion between 'stand' and 'sit'.

    If the model predicts 'stand' (class 1) and the difference in predicted probability
    between 'stand' and 'sit' (class 0) is less than the given threshold, the prediction
    is changed to 'sit'.

    :param probabilities: 2D array of predicted probabilities with shape (n_samples, n_classes).
    :param y_pred: Original predicted class labels.
    :param sit_label: Integer label representing the 'sit' class.
    :param stand_label: Integer label representing the 'stand' class.
    :param threshold: Minimum difference between 'stand' and 'sit' probabilities to keep 'stand' prediction.
    :return: Array of adjusted class label predictions.
    """
    adjusted = []

    # Iterate over each sample's predicted probabilities
    for sample_index, probs in enumerate(probabilities):
        # Get original prediction for this sample
        pred = y_pred[sample_index]

        # Apply adjustment only if model predicted 'stand'
        if pred == stand_label:
            p_stand = probs[stand_label]
            p_sit = probs[sit_label]

            # If difference between 'stand' and 'sit' probabilities is below threshold,
            # change prediction to 'sit' to reduce confusion
            if (p_stand - p_sit) < threshold:
                pred = sit_label  # Change prediction to 'sit'

        adjusted.append(pred)

    return np.array(adjusted)

def find_class_segments(predictions: np.ndarray, target_class: int) -> List[Tuple[int, int]]:
    """
    Find start and end indices of contiguous segments belonging to the target class.

    :param predictions: 1D array of predicted class labels.
    :param target_class: Class label to find segments for.
    :return: List of tuples, where each tuple is (start_idx, end_idx) of a segment.
    """

    # list for holding the segments
    segments = [] # List to hold tuples of (start_idx, end_idx) for each segment

    # init variables
    in_segment = False # Flag to indicate whether we're currently inside a segment
    start = 0

    # Iterate over the predictions array, with index and prediction value
    for index, pred in enumerate(predictions):

        # check whether current prediction corresponds to the target class
        if pred == target_class:

            # If we are not already tracking a segment, start a new one
            if not in_segment:
                in_segment = True
                start = index # Mark the start of this segment
        else:
            # If current prediction is different, and we were tracking a segment,
            # this marks the end of the segment
            if in_segment:
                segments.append((start, index - 1)) # Add segment to list
                in_segment = False # Reset flag since segment ended

    # when reaching the end, assign the end of the prediction array as the stop
    if in_segment:
        segments.append((start, len(predictions) - 1))

    return segments


def correct_short_segments(predictions: np.ndarray, class_id: int, min_duration: float, window_size: float) -> np.ndarray:
    """
    Replace segments of a specific class that are shorter than a given duration.

    The replacement class is chosen as the most frequent class from neighboring values.

    :param predictions: 1D array of predicted class labels.
    :param class_id: Class to check for short-duration segments.
    :param min_duration: Minimum acceptable duration for a segment in seconds.
    :param window_size: Duration of each prediction window in seconds.
    :return: Updated prediction array with short segments replaced.
    """

    # get the segments for the class
    segments = find_class_segments(predictions, class_id)
    corrected = predictions.copy()

    # cycle over the segments
    for start, end in segments:

        # calculate the segment lengths in seconds
        duration = (end - start + 1) * window_size

        # check whether the segment needs to be corrected (too short)
        if duration < min_duration:
            # Get neighbor classes
            left = predictions[start - 1] if start > 0 else None
            right = predictions[end + 1] if end < len(predictions) - 1 else None

            neighbors = [neighbor_class  for neighbor_class  in (left, right) if neighbor_class  is not None]
            if neighbors:

                # in case the left and the right neighbor are from different
                # classes then the left neighbor (i.e., the previous activity - chronologically) is chosen
                replacement = Counter(neighbors).most_common(1)[0][0]
                corrected[start:end + 1] = replacement

    return corrected


def expand_classification(clf_result: Union[List[int], np.ndarray],
                          w_size: float,
                          fs: int) -> List[int]:
    """
    Expands window-level classification results to sample-level predictions.

    Repeats each window prediction to match the number of samples in the corresponding window.

    :param clf_result: List of predicted class labels, one per window.
    :param w_size: Window size in seconds used during classification.
    :param fs: Sampling rate (samples per second) of the input signal.
    :return: List of predicted class labels, expanded to match the sample-level resolution.
    """

    expanded_clf_result = []

    # cycle over the classification results list
    for predicted_class in clf_result:

        expanded_clf_result += [predicted_class] * int(w_size * fs)

    return expanded_clf_result


def heuristics_correction(predictions: np.ndarray,
                                     window_size: float,
                                     min_durations: Dict[int, float]) -> np.ndarray:
    """
    Apply post-processing to correct short activity segments for each class.

    :param predictions: 1D array of predicted class labels.
    :param window_size: Duration of each prediction window in seconds.
    :param min_durations: Dictionary mapping each class label to its minimum segment duration in seconds.
    :return: Post-processed prediction array with short segments corrected.
    """
    corrected = predictions.copy()

    # Apply correction for each class using the specified minimum duration
    for class_id, min_duration in min_durations.items():
        corrected = correct_short_segments(corrected, class_id, min_duration, window_size)

    return corrected

