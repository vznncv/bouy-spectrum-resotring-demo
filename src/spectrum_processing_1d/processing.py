"""
The module contains function to get spectrum estimation using sensor data,
and some other helper functions
"""

import numpy as np
import scipy.ndimage as ndimage
from scipy import signal
from scipy.integrate import cumtrapz
from scipy.signal import welch


def calculate_acceleration(x, spacing=1.0, axis=0):
    """
    Calculate acceleration using trajectory.

    :param x: array with trajectory points
    :param spacing: distance between samples
    :param axis: if ``x`` is multidimensional array, then acceleration will be calculated along specified axis
    :return: array with acceleration data
    """
    return np.gradient(
        np.gradient(
            x,
            spacing,
            axis=axis
        ),
        spacing,
        axis=axis
    )


def calculate_trajectory(ax, spacing=0.5, axis=0, correlation_distance='auto'):
    """
    Calculate trajectory using acceleration data.

    :param ax: acceleration data
    :param spacing: distance between two samples
    :param axis: if ``x`` is multidimensional array, then trajectory will be calculated along specified axis
    :param correlation_distance: approximage correlation distance between samples
                                (it's used to suppress low frequency noises)
    :return: restored trajectory
    """
    if correlation_distance == 'auto':
        correlation_distance = 1 / (spacing * 0.05)

    if correlation_distance is not None:
        if correlation_distance < 0:
            raise ValueError("Correlation should be positive")

        correlation_distance_n_samples = int(correlation_distance * (0.5 / spacing))

        b = [spacing]
        exp_mean_k = 2 / (1 + correlation_distance_n_samples)
        a = [1, -1 + exp_mean_k]

        vx = signal.lfilter(b=b, a=a, x=ax)
        x = signal.lfilter(b=b, a=a, x=vx)

        low_pass_fir_b = signal.get_window('hamming', correlation_distance_n_samples)
        low_pass_fir_b /= low_pass_fir_b.sum()
        x -= ndimage.convolve1d(x, low_pass_fir_b, axis=axis, mode='mirror')

    else:
        vx = cumtrapz(ax, dx=spacing, axis=axis, initial=0)
        x = cumtrapz(vx, dx=spacing, axis=axis, initial=0)

    return x


def calculate_particle_trajectory(sensor_ax, sensor_ay, sensor_alpha, *,
                                  spacing=1.0,
                                  corr_dist='auto', corr_dist_ax='auto', corr_dist_ay='auto', corr_dist_alpha='auto',
                                  axis=0):
    """
    Calculate trajectory, using sensor data.

    :param sensor_alpha: sensor angle acceleration data
    :param sensor_ax: sensor x acceleration data
    :param sensor_ay: sensor y acceleration data
    :param spacing: sampling period
    :param corr_dist: see ``correlation_distance`` in the :func:`~calculate_trajectory` for more details
    :param corr_dist_ax: if it's ``'auto'``, the ``corr_dist`` is used
    :param corr_dist_ay: if it's ``'auto'``, the ``corr_dist`` is used
    :param corr_dist_alpha: if it's ``'auto'``, the ``corr_dist`` is used
    :param axis: if ``sensor_*`` is multidimensional arrays, then trajectory will be resorted
                 using data along specified axis
    :return:
    """
    sensor_ax = np.asfarray(sensor_ax)
    sensor_ay = np.asfarray(sensor_ay)
    sensor_alpha = np.asfarray(sensor_alpha)

    corr_dist_ax = corr_dist if corr_dist_ax == 'auto' else corr_dist_ax
    corr_dist_ay = corr_dist if corr_dist_ay == 'auto' else corr_dist_ay
    corr_dist_alpha = corr_dist if corr_dist_alpha == 'auto' else corr_dist_alpha

    # restore angle
    angle = calculate_trajectory(sensor_alpha, spacing=spacing, axis=axis, correlation_distance=corr_dist_alpha)
    # restore accelerations in the original coordinates
    tmp_angle = -angle
    ax = sensor_ax * np.cos(tmp_angle) - sensor_ay * np.sin(tmp_angle)
    ay = sensor_ax * np.sin(tmp_angle) + sensor_ay * np.cos(tmp_angle)

    # restore x and y
    x = calculate_trajectory(ax, spacing=spacing, axis=axis, correlation_distance=corr_dist_ax)
    y = calculate_trajectory(ay, spacing=spacing, axis=axis, correlation_distance=corr_dist_ay)

    return x, y, angle


def estimate_spectrum(ax, ay, alpha, fs=2.0, *, transition_interval=0.1,
                      corr_dist='auto', corr_dist_ax='auto', corr_dist_ay='auto', corr_dist_alpha='auto',
                      return_trajectory=False, nperseg=128, nfft=None):
    """
    Estimate spectrum using acceleration data from sensors.

    :param ax: acceleration data from X sensor axis
    :param ay: acceleration data from Y sensor axis
    :param alpha: angular acceleration data in the XY plane (Z axis)
    :param fs: sampling frequency
    :param transition_interval: sensor transition interval (it will be removed from estimation)
    :param nperseg:
    :param nfft:
    :param corr_dist: see :func:`~calculate_particle_trajectory` for more details
    :param corr_dist_ax: see :func:`~calculate_particle_trajectory` for more details
    :param corr_dist_ay: see :func:`~calculate_particle_trajectory` for more details
    :param corr_dist_alpha: see :func:`~calculate_particle_trajectory` for more details
    :param return_trajectory:
    :return: (<spectrum frequency>, <spectrum_values>) if ``return_trajectory`` is ``False``.
             (<spectrum frequency>, <spectrum_values>, (<x>, <y>, <angle>) if ``return_trajectory`` is ``True``.
    """
    if isinstance(transition_interval, (np.floating, float)):
        if not (0 <= transition_interval < 1):
            raise ValueError("If transition_interval is float, it should be in range (0, 1)")
        transition_range = int(len(ax) * transition_interval)
    elif isinstance(transition_interval, (np.integer, int)):
        if 0 < transition_interval:
            raise ValueError("If transition_interval is integer, it should be non-negative")
        transition_range = int(transition_interval)
    else:
        raise ValueError("Unknown transition_interval type: {} ({})"
                         .format(type(transition_interval), transition_interval))

    dt = 1 / fs

    x, y, angle = calculate_particle_trajectory(
        sensor_ax=ax, sensor_ay=ay, sensor_alpha=alpha,
        corr_dist=corr_dist, corr_dist_ax=corr_dist_ax, corr_dist_ay=corr_dist_ay, corr_dist_alpha=corr_dist_alpha,
        spacing=dt
    )

    signal = x[transition_range:] + 1j * y[transition_range:]

    f, s = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        nfft=nfft,
        scaling='density',
        return_onesided=False,
        axis=0,
        detrend=None
    )
    f = np.fft.fftshift(f, axes=0)
    s = np.fft.fftshift(s, axes=0)
    s *= 1 / np.sqrt(nperseg)

    omega = f * 2 * np.pi

    if return_trajectory:
        return omega, s, (x, y, angle)
    else:
        return omega, s
