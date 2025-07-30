"""
Functions to pre-processing the Smartphone data.

Available Functions
-------------------
[Public]
pre_process_sensors(...): Pre-processes the sensors contained in data_array according to their sensor type.
pre_process_inertial_data(...): Applies the pre-processing pipeline of "A Public Domain Dataset for Human Activity Recognition Using Smartphones"
median_and_lowpass_filter(...): Applies a median filter followed by a butterworth lowpass filter. The lowpass filter is 3rd order with a cutoff frequency of 20 Hz .
gravitational_filter(...): Function to filter out the gravitational component of ACC signals using a 3rd order butterworth lowpass filter with a cuttoff frequency of 0.3 Hz
slerp_smoothing(...): Smooths a quaternion time series using spherical linear interpolation (SLERP).
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
from typing import List
from pyquaternion import Quaternion
from tqdm import tqdm
from scipy import signal

# internal imports
from constants import VALID_SENSORS, ACC, GYR, MAG

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #



def pre_process_sensors(data_array: np.array, sensor_names: List[str], fs: int =100, scalar_first = False) -> np.array:
    """
    Pre-processes the sensors contained in data_array according to their sensor type.
    :param data_array: the loaded data
    :param sensor_names: the names of the sensors contained in the data array
    :return: numpy.array containing the pre-processed sensor data
    """

    # make a copy to not override the original data
    processed_data = data_array.copy()

    # process each sensor
    for valid_sensor in VALID_SENSORS:

        # get the positions of the sensor in the sensor_names
        sensor_cols = [col for col, sensor_name in enumerate(sensor_names) if valid_sensor in sensor_name]

        if sensor_cols:

            print(f"--> pre-processing {valid_sensor} sensor")
            # acc pre-processing
            if valid_sensor == ACC:

                processed_data[:, sensor_cols] = pre_process_inertial_data(processed_data[:, sensor_cols], is_acc=True,
                                                                           fs=fs, f_c = 5)

            # gyr and mag pre-processing
            elif valid_sensor in [GYR, MAG]:

                processed_data[:, sensor_cols] = pre_process_inertial_data(processed_data[:, sensor_cols], is_acc=False,
                                                                           fs=fs)

            # rotation vector pre-processing
            else:

                processed_data[:, sensor_cols] = slerp_smoothing(processed_data[:, sensor_cols], 0.3,
                                                                 scalar_first,
                                                                 return_numpy=True, return_scalar_first=False)
        else:

            print(f"The {valid_sensor} sensor is not in the loaded data. Skipping the pre-processing of this sensor.")

    return processed_data



def pre_process_inertial_data(sensor_data: np.array, is_acc: bool = False, fs: int = 100, normalize: bool = False, f_c: int=20) -> np.array:
    """
    Applies the pre-processing pipeline of "A Public Domain Dataset for Human Activity Recognition Using Smartphones"
    (https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf). The pipeline consists of:
    (1) applying a median filter
    (2) applying a 3rd order low-pass filter with a cut-off at 20 Hz

    in case the sensor data belongs to an ACC sensor the following additional steps are performed.
    (3) applying a 3rd order low-pass filter with a cut-off at 0.3 Hz to obtain gravitational component
    (4) subtract gravitational component from ACC signal

    :param sensor_data: the sensor data.
    :param is_acc: boolean indicating whether the sensor is an accelerometer.
    :param fs: the sampling frequency of the sensor data (in Hz).
    :param normalize: boolean to indicate whether the data should be normalized (division by the max)
    :param f_c: the cutoff frequency (in Hz).
    :return: numpy.array containing the pre-processed data.
    """
     # apply median and lowpass filter
    filtered_data = median_and_lowpass_filter(sensor_data, fs=fs, f_c = f_c)


    # check if signal is supposed to be normalized
    if normalize:
        # normalize the signal
        filtered_data = filtered_data / np.max(filtered_data)

    # check if sensor is ACC (additional steps necessary)
    if is_acc:
        # print('Applying additional processing steps')

        # get the gravitational component
        gravitational_component = gravitational_filter(filtered_data, fs=fs)


        # subtract the gravitational component
        filtered_data = filtered_data - gravitational_component

    return filtered_data

def median_and_lowpass_filter(sensor_data: np.ndarray, fs: int, f_c: int, medfilt_window_length=11) -> np.ndarray:
    """
     Applies a median filter followed by a Butterworth low-pass filter. By default, the low-pass filter is 3rd order,
    and the cutoff frequency can be specified by the user (e.g., 20 Hz or 5 Hz depending on the application).

    This processing scheme is based on:
    "A Public Domain Dataset for Human Activity Recognition Using Smartphones"
    https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf

    For accelerometer (ACC) data specifically, a cutoff frequency of 5 Hz may be used instead of 20 Hz
    in order to better preserve motion patterns related to human gait and step detection.

    - The **5 Hz cutoff** is appropriate because most of the relevant signal components for walking and running
      fall below this frequency.
    - Higher frequencies often correspond to noise or mechanical vibrations that are not related to steps.

    :param sensor_data: a 1-D or (MxN) array, where M is the signal length in samples and
                        N is the number of signals / channels.
    :param f_c: the cutoff frequency (in Hz).
    :param fs: the sampling frequency of the acc data (in Hz).
    :param medfilt_window_length: the length of the median filter (has to be odd). Default: 11
    :return: the filtered data
    """

    # define the filter
    order = 3
    filt = signal.butter(order, f_c, fs=fs, output='sos')

    # copy the array
    filtered_data = sensor_data.copy()

    # check the dimensionality of the input
    if filtered_data.ndim > 1:  # (MxN) array

        # cycle of the channels contained in data
        for channel in range(filtered_data.shape[1]):
            # get the channel
            sig = sensor_data[:, channel]

            # apply the median filter
            sig = signal.medfilt(sig, medfilt_window_length)

            # apply butterworth filter
            filtered_data[:, channel] = signal.sosfilt(filt, sig)

    else:  # 1-D array

        # apply median filter
        med_filt = signal.medfilt(sensor_data, medfilt_window_length)

        # apply butterworth filter
        filtered_data = signal.sosfilt(filt, med_filt)

    return filtered_data


def gravitational_filter(acc_data: np.ndarray, fs: int) -> np.ndarray:
    """
    Function to filter out the gravitational component of ACC signals using a 3rd order butterworth lowpass filter with
    a cuttoff frequency of 0.3 Hz
    The implementation is based on:
    "A Public Domain Dataset for Human Activity Recognition Using Smartphones"
    https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf
    :param acc_data: a 1-D or (MxN) array, where M is the signal length in samples and
                 N is the number of signals / channels.
    :param fs: the sampling frequency of the acc data.
    :return: the gravitational component of each signal/channel contained in acc_data
    """

    # define the filter
    order = 3
    f_c = 0.3
    filter = signal.butter(order, f_c, fs=fs, output='sos')

    # copy the array
    gravity_data = acc_data.copy()

    # check the dimensionality of the input
    if gravity_data.ndim > 1:  # (MxN) array

        # cycle of the channels contained in data
        for channel in range(gravity_data.shape[1]):
            # get the channel
            sig = acc_data[:, channel]

            # apply butterworth filter
            gravity_data[:, channel] = signal.sosfilt(filter, sig)

    else:  # 1-D array

        gravity_data = signal.sosfilt(filter, acc_data)

    return gravity_data


def slerp_smoothing(quaternion_array: np.array, smooth_factor: float = 0.5, scalar_first: bool = False,
                    return_numpy: bool = True, return_scalar_first: bool = False) -> np.array:
    """
    Smooths a quaternion time series using spherical linear interpolation (SLERP).

    This function applies SLERP to smooth a sequence of quaternions by interpolating
    between consecutive quaternions with a specified smoothing factor. The method follows
    the approach described in:
    https://www.mathworks.com/help/fusion/ug/lowpass-filter-orientation-using-quaternion-slerp.html

    :param quaternion_array: 2D numpy.array of shape (N, 4) containing a sequence of quaternions. The quaternions can
                             be represented in either scalar-first (w, x, y, z) or scalar-last (x, y, z, w) notation.
    :param smooth_factor: the interpolation factor for SLERP, controlling how much smoothing is applied. The value must
                          be between [0, 1]. Values closer to 0 increase smoothing, while values closer to 1 retain the
                          original sequence.
    :param scalar_first: boolean indicating the notation that is used. Default: False
    :param return_numpy: boolean indicating, whether a numpy.array should be returned. If false an array containing
                         pyquaternion.Quaternion objects are returned.
    :param return_scalar_first: boolean indicating the notation for the return type. Default: False
    :return: returns quaternions in either scalar first (w, x, y, z) or scalar last notation (x, y, z, w), depending on
             the parameter settings of the boolean parameters.
    """

    # check range of smooth factor
    if not (0 <= smooth_factor <= 1):
        raise ValueError(f"The smooth factor has to be between [0, 1]. Provided smooth factor: {smooth_factor}")

    # change quaternion notation to scalar first notation (w, x, y, z)
    # this is needed as pyquaternion assumes this notation
    if not scalar_first:

        quaternion_array = np.hstack((quaternion_array[:, -1:], quaternion_array[:, :-1]))

    # get the number of rows
    num_rows = quaternion_array.shape[0]

    # array for holding the result
    smoothed_quaternion_array = np.zeros(num_rows, dtype=object)

    # initialize the first quaternion
    smoothed_quaternion_array[0] = Quaternion(quaternion_array[0])

    # cycle over the quaternion series
    for row in tqdm(range(1, num_rows), ncols=50, bar_format="{l_bar}{bar}| {percentage:3.0f}% {elapsed}"):

        # get the previous and the current quaternion
        q_prev = smoothed_quaternion_array[row - 1]
        q_curr = Quaternion(quaternion_array[row])

        # perform SLERP
        q_slerp = Quaternion.slerp(q_prev, q_curr, smooth_factor)

        # add the quaternion to the smoothed series
        smoothed_quaternion_array[row] = q_slerp

    # return as numpy array
    if return_numpy:

        # transform the output into a 2D numpy array
        smoothed_quaternion_series_numpy = np.zeros_like(quaternion_array)

        for row, quat in enumerate(smoothed_quaternion_array):

            smoothed_quaternion_series_numpy[row] = quat.elements

        # return in (x, y, z, w) notation
        if not return_scalar_first:

            smoothed_quaternion_series_numpy = np.hstack((smoothed_quaternion_series_numpy[:, 1:],
                                                          smoothed_quaternion_series_numpy[:, :1]))

        return smoothed_quaternion_series_numpy

    return smoothed_quaternion_array




