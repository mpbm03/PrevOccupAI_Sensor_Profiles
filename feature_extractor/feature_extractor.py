"""
Functions for extracting features and windowing signal.

Available Functions
-------------------
[Public]
feature_extractor(...): Extracts features from time-series data based on a given configuration and model.

[Private]
_trim_data(...): Trims the data to ensure it can be evenly divided into fixed-size windows.
-------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import tsfel
import pandas as pd
from typing import Tuple

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def extract_features(data: pd.DataFrame, cfg: dict, model, model_sensors: Tuple[str, ...], window_size: float, fs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts features from time-series data based on a given configuration and model.

    This function trims the input data to ensure complete windows, then extracts time-series
    features using the TSFEL library, considering only the sensors used during model training.

    :param data: DataFrame containing the data.
    :param cfg: Dictionary with the TSFEL feature extraction configuration.
    :param model: Trained model object, which must have a 'feature_names_in_' attribute.
    :param model_sensors: Tuple of sensor names (column names) used during model training.
    :param window_size: Window size in seconds for feature extraction.
    :param fs: Sampling rate (samples per second).
    :return: Tuple containing:
        - Trimmed DataFrame with the original data.
        - DataFrame with the extracted features used by the model.
    """
    print("Extracting features for HAR model")

    # Get the features that the model was trained with
    model_features = model.feature_names_in_

    # Trim the data to accommodate all windows
    data, _ = _trim_data(data, w_size=window_size, fs=fs)

    # Extract the features
    features = tsfel.time_series_features_extractor(cfg, data[list(model_sensors)],
                                                    window_size=int(window_size * fs), fs=fs,
                                                    header_names=list(model_sensors))
    return data, features[model_features]


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _trim_data(data, w_size, fs) -> Tuple[pd.DataFrame, int]:
    """
    Trims the data to ensure it can be evenly divided into fixed-size windows.

    :param data: DataFrame containing the data.
    :param w_size: Window size in seconds.
    :param fs: Sampling rate (samples per second).
    :return: Tuple containing:
        - DataFrame with the trimmed data.
        - Integer indicating the number of samples trimmed.
    """
    to_trim = int(data.shape[0] % (w_size * fs))
    return data.iloc[:-to_trim, :], to_trim
