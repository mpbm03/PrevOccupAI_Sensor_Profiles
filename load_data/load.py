"""
Functions to load the Smartphone data.

Available Functions
-------------------
[Public]
load_data_from_same_recording(...): Loads and aligns Android sensor data from a single recording.
    Resamples all signals to a common frequency and ensures temporal alignment using padding.

[Private]
_validate_sensor_names(...): Validates user-provided sensor names against a predefined list of supported sensors.
_load_raw_data(...): Loads raw sensor data files and performs basic cleaning (e.g., removing NaNs, duplicates).
_get_file_by_sensor(...): Locates the filename corresponding to a specific sensor within a folder.
_load_sensor_file(...): Loads and cleans an individual sensor file, applies special handling for rotation vector data.
_remove_non_unit_quaternion(...): Filters out invalid rotation vector samples that are not unit quaternions.
_clean_df(...): Cleans a DataFrame by removing NaNs, duplicates, and resetting the index.
_pad_data(...): Pads sensor recordings so that all signals begin and end at the same timestamp.
_create_padding(...): Generates padding rows with repeated values or zeros to match a desired time range.
_re_sample_data(...): Resamples sensor data to a target frequency using interpolation (cubic or SLERP).
------------------
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import pandas as pd
import numpy as np
from typing import Tuple, List , Dict , Any , Union , Optional
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp



# internal imports
from constants import VALID_SENSORS , TIME , SENSOR_MAP , ROT , IMU_SENSORS

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #

# padding types
PADDING_SAME = 'same'
PADDING_ZERO = 'zero'
VALID_PADDING_TYPES = ['same', 'zero']

# sensor data dictionary keys
LOADED_SENSORS = 'loaded sensors'
STARTING_TIMES = 'starting times'
STOPPING_TIMES = 'stopping times'

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def load_data_from_same_recording(folder_path: str, sensor_names: List[str] = None,
                                  fs: int = 100, padding_type: str = PADDING_SAME) -> pd.DataFrame:
    """
    Function to load Android sensor data from the same recording into a single DataFrame. The function assumes that
    files stored at the provided path belong to the same recording. Alignment (in time) of the files is done based on
    the last sensor to start and the first to stop, meaning that data is only considered while all sensors are recording
    at the same time. The data is re-sampled to the sampling rate given by 'fs'. The resampling is necessary as the
    Android OS does not ensure equidistant sampling at a fixes rate.

    :param folder_path: the path to the folder containing the sensor files.
    :param sensor_names: list of sensors (as strings) indicating which sensors should be loaded. The following sensors
                         can be loaded: ['ACC', 'GYR', 'MAG', 'ROT']
                         Default: None (all sensors are loaded)
    :param fs: the sampling rate to which all sensors should be re-sampled to. Default: 100 (Hz)
    :param padding_type: padding which should be used to ensure that all sensors start and stop at the same time. The
                         following padding types are supported: 'same', 'zero'. Default: 'same'
    :return: pandas.DataFrame containing all sensors aligned in time and re-sampled to the same sampling rate.
    """
    # in case no sensor names list is given, load all valid sensors
    if sensor_names is None:
        sensor_names = VALID_SENSORS

    # check whether the provided sensor names are valid sensors
    _validate_sensor_names(sensor_names)

    # check for valid padding type
    if padding_type not in VALID_PADDING_TYPES:
        raise ValueError(f"The padding type you chose is not valid. Chosen padding type: {padding_type}."
                         f"\nPlease choose one of the following: {', '.join(VALID_PADDING_TYPES)}.")

    # load the data
    sensor_data, report = _load_raw_data(folder_path, sensor_names)

    # align the data
    # (1) pad the data (all sensors start and stop at the same timestep)
    padded_data = _pad_data(sensor_data, report, padding_type)

    # (2) resample the data to 100 Hz
    interpolated_data = _re_sample_data(padded_data, report, fs=fs)

    # (3) create a DataFrame containing all the data
    aligned_sensor_df = pd.concat([interpolated_data[0]] + [df.drop(columns=['t']) for df in interpolated_data[1:]],
                                  axis=1)

    return aligned_sensor_df

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _validate_sensor_names(sensor_names: list) -> None:
    """
    Validates the list of sensor names against a list of valid sensors.

    :param sensor_names: List of sensor names provided by the user.
    :raises ValueError: If any sensor in the list is not valid.
    """

    # get the names of the sensors that do not match the VALID SENSORS
    invalid_sensors = [sensor for sensor in sensor_names if sensor not in VALID_SENSORS]

    # check if there were any found and raise an error
    if invalid_sensors:
        raise ValueError(
            f"The following sensor(s) that was/were provided as input is/are not supported: {', '.join(invalid_sensors)}. "
            f"\nPlease choose from the list of supported sensors: {', '.join(VALID_SENSORS)}.")


def _load_raw_data(folder_path: str, sensor_names: List[str]) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """
    Loads sensor data contained in 'folder_path' into a list of pandas DataFrames. Each element in the list corresponds
    to a sensor's data.A dictionary is also returned containing the loaded sensors and the timestamps when each sensor
    started and stopped recording.

    General data cleaning includes:
    (1) Removal of NaN values
    (2) Removal of duplicates
    (3) Resetting of DataFrame index

    :param folder_path: The path to the folder containing sensor data files.
    :param sensor_names: A list of sensor names to load the data for.
    :return: A tuple where the first element is a list of pandas DataFrames for each sensor's data, and the second
             element is a dictionary containing sensor start/stop timestamps and order information.
    """

    # list for holding the loaded DataFrames
    sensor_data = []

    # list for holding the sensor names
    loaded_sensors = []

    # list for holding starting and stopping timestamps
    start_times = []
    stop_times = []

    try:
        # list the files in folder_path
        files = os.listdir(folder_path)
    except FileNotFoundError:

        # raise error in case the folder path is invalid
        raise ValueError(f"The folder at path {folder_path} was not found.")

    # cycle over the sensor names
    for sensor in tqdm(sensor_names, desc="--> Loading data"):

        # get the file corresponding to the provided sensor
        sensor_file = _get_file_by_sensor(sensor, files)

        # check if a sensor file was found
        if sensor_file:

            # load the data
            sensor_df = _load_sensor_file(folder_path, sensor_file, sensor)

            # append the data to sensor_data
            sensor_data.append(sensor_df)

            # append the sensor to loaded_sensors
            loaded_sensors.append(sensor)

            # append the start and stop times
            start_times.append(sensor_df[TIME].iloc[0])
            stop_times.append(sensor_df[TIME].iloc[-1])

        # give warning to user, in the case no file was found
        else:
            print(f"Warning: No file found for sensor {sensor}. Skipping this sensor.")

    # create dictionary
    report = {
        LOADED_SENSORS: loaded_sensors,
        STARTING_TIMES: start_times,
        STOPPING_TIMES: stop_times,
    }

    return sensor_data, report


def _get_file_by_sensor(sensor_name: str, files: List[str]) -> Optional[str]:
    """
    Returns the file name corresponding to the sensor name provided.

    :param sensor_name: Sensor name abbreviation ('ACC', 'GYR', 'MAG', 'ROT')
    :param files: List of files in the folder
    :return: File name if found, otherwise None
    """

    # Extract the corresponding identifier
    file_identifier = SENSOR_MAP[sensor_name]

    # Search for the file in the list
    for file in files:
        if file_identifier in file:
            return file

    # If no file is found
    print(f"No file found for sensor: {sensor_name}.")
    return None


def _load_sensor_file(folder_path: str, file_name: str, sensor_name: str) -> pd.DataFrame:
    """
    Load a sensor file into a pandas DataFrame and cleans it.

    This function reads a sensor data file located in the specified folder, performs initial cleanup
    by removing unnecessary columns, and assigns appropriate column names. For rotation vector data,
    additional steps are taken to ensure that only valid unit quaternions are kept.

    :param folder_path: The directory where the sensor file is located.
    :param file_name: The name of the sensor file to be loaded.
    :param sensor_name: The name of the sensor, used to define appropriate column names and handle
                        sensor-specific preprocessing.
    :return: A cleaned pandas DataFrame containing the sensor data with appropriate column names.
    """

    # create full file path
    file_path = os.path.join(folder_path, file_name)

    # read the file
    sensor_df = pd.read_csv(file_path, delimiter='\t', header=None, skiprows=3)

    # remove nan column (the loading of the opensignals sensor file through read_csv(...) generates a nan column
    sensor_df.dropna(axis=1, how='all', inplace=True)

    # define column names depending on sensor name
    col_names = [TIME, f'x_{sensor_name}', f'y_{sensor_name}', f'z_{sensor_name}']

    # perform extra steps for rotation vector
    if sensor_name == ROT:
        # add fourth column name
        col_names.append(f'w_{sensor_name}')

        # remove samples that are not unit vectors
        sensor_df = _remove_non_unit_quaternion(sensor_df)

    # add column names
    sensor_df.columns = col_names

    # remove nan values and duplicates + reset index
    sensor_df = _clean_df(sensor_df)

    return sensor_df


def _remove_non_unit_quaternion(rotvec_df: pd.DataFrame, tol: float = 0.1) -> pd.DataFrame:
    """
    Remove corrupted samples from a DataFrame containing Android rotation vector data.
    Android rotation vector data are expected to be unit quaternions (i.e., their norm should be close to 1).
    This function removes samples where the quaternion norm deviates from 1 beyond a given tolerance.

    :param rotvec_df: A DataFrame where the first column represents timestamps, and the remaining columns
                      contain quaternion components (x, y, z, w).
    :param tol: optional (default=0.1). The tolerance for deviation from a unit quaternion. Samples
                with a norm less than `1 - tol` are considered corrupted and removed.
    :return: The cleaned DataFrame containing only valid unit quaternions.
    """

    # get number of samples before removal
    num_samples_pre = len(rotvec_df)

    # calculate the norm of the vector
    vector_norm = np.linalg.norm(rotvec_df.iloc[:, 1:], axis=1)

    # remove samples that do not adhere to the norm (keep samples that adhere to the vector norm)
    rotvec_df = rotvec_df[vector_norm >= 1 - tol]

    # calculate the number of removed samples
    num_samples_removed = num_samples_pre - len(rotvec_df)

    if num_samples_removed > 0:
        print(f"Removed {num_samples_removed} samples that were not normal from Rotation Vector")

    return rotvec_df


def _clean_df(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs general cleaning of the data frame.
    (1) remove nan values
    (2) remove duplicates
    (3) reset index
    Parameters

    :param sensor_df: The data frame that was loaded from the sensor file.
    :return: pandas.DataFrame containing the cleaned data.
    """

    # remove any nan values and duplicates
    sensor_df = sensor_df.dropna()
    sensor_df = sensor_df.drop_duplicates(subset=[TIME])

    # reset the index to start at zero
    sensor_df = sensor_df.reset_index(drop=True)

    return sensor_df


def _pad_data(sensor_data: List[pd.DataFrame], report: Dict[str, Any], padding_type: str = PADDING_SAME)\
        -> List[pd.DataFrame]:
    """
    Pads the sensor data so that all sensors start and end at the same timestep. Padding is done based on the
    sensor that starts and stops the latest and earliest, respectively. Only data where all sensors are collected
    simultaneously are considered. By default, 'same' padding type is used.

    :param sensor_data: A list of pandas DataFrames containing the sensor data.
    :param report: A dictionary containing metadata such as 'STARTING_TIMES', 'STOPPING_TIMES', and 'LOADED_SENSORS'.
    :param padding_type: The padding type to use. 'same' uses the first and last valid sensor data values for padding,
                         while 'zero' uses zero padding. Default: 'same'.
    :return: A list of pandas.DataFrames containing the padded sensor data.
    """

    # list for holding the padded sensor data
    padded_data = []

    # get the index of the latest start and the earliest stopping times
    start_index = report[STARTING_TIMES].index(max(report[STARTING_TIMES]))
    stop_index = report[STOPPING_TIMES].index(min(report[STOPPING_TIMES]))

    # get the start and stop timestamps
    start_timestamp = report[STARTING_TIMES][start_index]
    end_timestamp = report[STOPPING_TIMES][stop_index]

    # get the time axis of the start and stop sensor
    time_axis_start = sensor_data[start_index][TIME]
    time_axis_end = sensor_data[stop_index][TIME]

    # loop over the sensors
    for num, sensor_name in tqdm(enumerate(report[LOADED_SENSORS]), total=len(report[LOADED_SENSORS]),
                                 desc="Padding data to ensure all data begins and ends on the same timestamp."):

        # get the data of the sensor
        sensor_df = sensor_data[num]

        # (1) padding at the beginning
        if start_timestamp > sensor_df[TIME].iloc[
            0]:  # start_timestamp after current signal start --> crop signal

            # crop the DataFrame
            sensor_df = sensor_df[sensor_df[TIME] >= start_timestamp]

        # get the timestamp values that need to be padded at the beginning of the DataFrame
        timestamps_start_pad = time_axis_start[time_axis_start < sensor_df[TIME].iloc[0]]

        # (2) padding at the end
        if end_timestamp < sensor_df[TIME].iloc[
            -1]:  # end_timestamp before current signal end --> crop signal

            # crop the time axis
            sensor_df = sensor_df[sensor_df[TIME] <= end_timestamp]

        # get the timestamp values that need to be padded at the end of the DataFrame
        timestamps_end_pad = time_axis_end[time_axis_end > sensor_df[TIME].iloc[-1]]

        if padding_type == 'same':
            # create padding for beginning and end
            padding_start = _create_padding(timestamps_start_pad, sensor_df.iloc[0, 1:].values)
            padding_end = _create_padding(timestamps_end_pad, sensor_df.iloc[-1, 1:].values)
        else:
            # create zero padding
            padding_start = _create_padding(timestamps_start_pad, np.zeros(len(sensor_df.columns) - 1))
            padding_end = _create_padding(timestamps_end_pad, np.zeros(len(sensor_df.columns) - 1))

        # get the columns of the DataFrame
        column_names = sensor_df.columns

        # create padded array
        padded_df = np.concatenate((padding_start, sensor_df.values, padding_end))

        # append the padded data
        padded_data.append(pd.DataFrame(padded_df, columns=column_names))

    return padded_data


def _create_padding(timestamps: List[Union[int, float]], values: np.ndarray):
    """
    Create padding for the given timestamps using specified values.
    This function replicates the provided `values` for each timestamp in `timestamps`,
    creating a padded array where each row consists of a timestamp followed by the repeated values.

    :param timestamps: A list of timestamp values.
    :param values: A 1D array containing the values to be repeated for each timestamp.
    :return: A 2D array where each row contains a timestamp followed by the replicated values.
    """

    # get the number of timestamps
    n_timestamps = len(timestamps)

    # tile the padding
    padding = np.tile(values, (n_timestamps, 1))

    return np.column_stack((timestamps, padding))


def _re_sample_data(sensor_data: List[pd.DataFrame], report:  Dict[str, Any], fs=100) -> List[pd.DataFrame]:
    """
    Resamples the sensor data to the specified sampling frequency.
    This function takes a list of sensor data DataFrames and resamples each sensor's data to the desired
    sampling frequency (`fs`). For IMU-based sensors (ACC, GYR, MAG), cubic spline interpolation is used,
    and for Rotation Vector data, SLERP interpolation is performed.

    :param sensor_data: A list of DataFrames, each containing sensor data. It is assumed that the first contains
                        the time axis, while the other columns contain sensor data.
    :param report: A dictionary containing metadata, including the sensor names under the key 'LOADED_SENSORS'.
    :param fs: The target sampling frequency for the resampled data. Default: 100 (Hz)
    :return: A list of DataFrames containing the resampled sensor data.
    """

    # list to hold the re-sampled data
    re_sampled_data = []

    # cycle over the sensors
    for sensor_df, sensor_name in tqdm(zip(sensor_data, report[LOADED_SENSORS]), total=len(sensor_data),
                                       desc=f"Ensurig equidistant sampling by resampling data to {fs} Hz"):

        # DataFrame for holding the interpolated data
        interpolated_sensor_df = pd.DataFrame()

        # interpolation for IMU (ACC, GYR, MAG)
        if sensor_name in IMU_SENSORS:

            # perform cubic spline interpolation
            interpolated_sensor_df = cubic_spline_interpolation(sensor_df, fs=fs)

        # interpolation for rotation vector (ROT)
        elif sensor_name == ROT:

            # perform SLERP interpolation
            interpolated_sensor_df = slerp_interpolation(sensor_df, fs=fs)

        else:

            # add more interpolation methods for other sensors
            # (in the future: e.g., heart rate data from smartwatch & noise sensor)
            print(f"There is no interpolation implemented for the sensor you have chosen. Chosen sensor: {sensor_name}.")

        # append interpolated data to list
        re_sampled_data.append(interpolated_sensor_df)

    return re_sampled_data

# ------------------------------------------------------------------------------------------------------------------- #
# Interpolation functions: Public
# ------------------------------------------------------------------------------------------------------------------- #
def cubic_spline_interpolation(sensor_df: pd.DataFrame, fs: int = 100) -> pd.DataFrame:
    """
    Apply cubic spline interpolation to resample sensor data at a given frequency.
    This function interpolates time-series sensor data using cubic splines. The first column of `sensor_df` is assumed
    to be the time axis, while the remaining columns contain sensor measurements.

    :param sensor_df: A DataFrame containing timestamps in the first column and sensor data in the remaining columns.
    :param fs: The target sampling frequency in Hz. Default: 100 (Hz)
    :return: A DataFrame containing the resampled time axis and interpolated sensor values.
    """

    # extract time axis
    time_axis = sensor_df.iloc[:, 0]

    # convert time axis to seconds (and cast to numpy.array)
    time_axis = _convert_android_timestamp_to_seconds(time_axis).values

    # extract signals (and cast to numpy.array
    signals = sensor_df.iloc[:, 1:].values

    # define the new time axis
    time_axis_inter = np.arange(time_axis[0], time_axis[-1], 1 / fs)

    # list for holding the interpolated signals
    interpolated_signals = []

    # cycle over the signal channels
    for channel in range(signals.shape[1]):
        # init cubic spline interpolator
        cubic_spline_interpolator = CubicSpline(time_axis, signals[:, channel], bc_type='natural')

        # interpolate the signal and append to interpolated signals
        interpolated_signals.append(cubic_spline_interpolator(time_axis_inter))

    # create interpolated DataFrame
    interpolated_df = pd.DataFrame(np.column_stack([time_axis_inter] + interpolated_signals), columns=sensor_df.columns)

    return interpolated_df


def slerp_interpolation(rotvec_df: pd.DataFrame, fs: int = 100) -> pd.DataFrame:
    """
    Perform SLERP (Spherical Linear Interpolation) over a quaternion time series.
    This function interpolates a given time series of quaternions using SLERP, resampling it
    to match a specified target sampling frequency.

    :param rotvec_df: A DataFrame containing timestamp values in the first column and quaternion
                      components (x, y, z, w) in the subsequent columns.
    :param fs: The target sampling frequency in Hz. Default: 100 (Hz)
    :return: A DataFrame containing the interpolated timestamps and quaternions.
    """

    # extract time axis
    time_axis = rotvec_df.iloc[:, 0]

    # convert time axis to seconds (and cast to numpy.array)
    time_axis = _convert_android_timestamp_to_seconds(time_axis).values

    # get the quaterion data
    quaternion_data = rotvec_df.iloc[:, 1:].values

    # convert quaternions to Rotation objects
    rotations = R.from_quat(quaternion_data)

    # define new time axis
    time_axis_inter = np.arange(time_axis[0], time_axis[-1], 1 / fs)

    # init SLERP interpolator
    slerp_interpolator = Slerp(time_axis, rotations)

    # interpolate the rotations
    interpolated_rotations = slerp_interpolator(time_axis_inter)

    # create interpolated DataFrame (stack time axis and quaternion data + convert result back to quaterions)
    rotvec_interpolated_df = pd.DataFrame(np.column_stack((time_axis_inter, interpolated_rotations.as_quat())),
                                          columns=rotvec_df.columns)

    return rotvec_interpolated_df


# ------------------------------------------------------------------------------------------------------------------- #
# Interpolation Fuctions: private
# ------------------------------------------------------------------------------------------------------------------- #
def _convert_android_timestamp_to_seconds(time_column: pd.Series) -> pd.Series:
    """
    Converts the time column from the android timestamp which is in nanoseconds to seconds.

    :param time_column: The column of the sensor dataframe containing the time axis
    :return: the converted time axis as pd.Series.
    """

    # get the first entry of the column
    time_0 = time_column.iloc[0]

    # subtract the first entry from the entire column
    time_column = time_column - time_0

    # convert from nanoseconds to seconds
    time_column = time_column * 1e-9

    return time_column