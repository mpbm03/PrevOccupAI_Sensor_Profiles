"""
This function provides a complete classification pipeline for human activity recognition (HAR) based on
windowed features and a trained Random Forest model. It includes steps for performing raw classification,
refining predictions with threshold tuning and heuristic corrections, and expanding results to match the
original signal resolution.

Available Functions
-------------------
[Public]
apply_classification_pipeline(...)): Executes the full classification pipeline, including prediction,
threshold tuning, heuristic correction, and signal-length expansion.

-------------------

"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

from constants import ACTIVITY, BLOCK_ID
from typing import List , Dict , Tuple
import pandas as pd
import numpy as np
from .post_process import threshold_tuning , heuristics_correction , expand_classification
from sklearn.ensemble import RandomForestClassifier


# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def apply_classification_pipeline(data: pd.DataFrame, features: np.ndarray, har_model: RandomForestClassifier,
                                  w_size: int,fs: int, threshold: float,
                                  min_durations: Dict[int, int]) -> (Tuple)[pd.DataFrame, List[int], List[int]]:
    """
    Applies classification pipeline. The classification pipeline consists of:

    1. Perform classification using a Random Forest
    2. Apply threshold tuning label correction
    3. Apply heuristics-based label correction


    :param data: Pandas DataFrame containing the raw data.
    :param features: numpy.array of shape (n_samples, n_features) containing the features.
    :param har_model: object from RandomForestClassifier.
    :param w_size: window size in seconds.
    :param fs: the sampling frequency (samples per second).
    :param threshold: The probability margin threshold for adjusting predictions. Default is 0.1.
    :param min_durations: Dictionary mapping each class label to its minimum segment duration in seconds.
    :return: A tuple containing:
        - DataFrame containing the predictions
        - List[int]: Labels for each window.
        - List[int]: Labels expanded to the original sampling frequency
    """
    print("Applying HAR model: classifying into walking, standing, and sitting")

    # classify the data - vanilla model
    y_pred = har_model.predict(features)

    # get class probabilities
    y_pred_proba = har_model.predict_proba(features)

    # apply threshold tuning
    y_pred_tt = threshold_tuning(y_pred_proba, y_pred, 0, 1, threshold)

    # combine tt with heuristics
    y_pred_tt_heur = heuristics_correction(y_pred_tt, w_size, min_durations)

    # expand the predictions to the size of the original signal
    y_pred_tt_heur_expanded = expand_classification(y_pred_tt_heur, w_size=w_size, fs=fs)

    # assign the expanded predictions to the ACTIVITY colum
    data[ACTIVITY] = y_pred_tt_heur_expanded

    # assign block IDs based on changes in activity
    data[BLOCK_ID] = (data[ACTIVITY] != data[ACTIVITY].shift()).cumsum()

    return data, y_pred_tt_heur, y_pred_tt_heur_expanded

