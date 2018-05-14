import numpy as np
from scipy import signal
from scipy.signal import lfilter


def calculate_acceleration(x, spacing=1.0, axis=0):
    """
    Calculate acceleration using trajectory.

    :param x:
    :param spacing:
    :param axis:
    :return:
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


def calculate_trajectory(ax, alpha=0.01, spacing=1.0, axis=0):
    """
    Calculate trajectory using acceleration data.
    
    :param ax:
    :param alpha:
    :param spacing:
    :param axis:
    :return:
    """
    if not (0 <= alpha < 1):
        raise ValueError("Invalid alpha value")

    a = [1, alpha - 1]
    b = [spacing]

    v = lfilter(b, a, ax, axis=axis)
    x = lfilter(b, a, v, axis=axis)

    return x


def calculate_particle_trajectory(sensor_ax, sensor_ay, sensor_alpha, *,
                                  spacing=1.0, alpha=0.01, alpha_ax=None, alpha_ay=None, alpha_alpha=None,
                                  axis=0):
    """
    Calculate trajectory, using sensor data.

    :param sensor_alpha:
    :param sensor_ax:
    :param sensor_ay:
    :param spacing:
    :param alpha:
    :param alpha_ax:
    :param alpha_ay:
    :param alpha_alpha:
    :param axis:
    :return:
    """
    alpha_ax = alpha if alpha_ax is None else alpha_ax
    alpha_ay = alpha if alpha_ay is None else alpha_ay
    alpha_alpha = alpha if alpha_alpha is None else alpha_alpha

    # restore angle
    angle = calculate_trajectory(sensor_alpha, alpha=alpha_alpha, spacing=spacing, axis=axis)
    # restore accelerations in the original coordinates
    tmp_angle = -angle
    ax = sensor_ax * np.cos(tmp_angle) - sensor_ay * np.sin(tmp_angle)
    ay = sensor_ax * np.sin(tmp_angle) + sensor_ay * np.cos(tmp_angle)

    # restore x and y
    x = calculate_trajectory(ax, alpha=alpha_ax, spacing=spacing, axis=axis)
    y = calculate_trajectory(ay, alpha=alpha_ay, spacing=spacing, axis=axis)

    # additional high pass filter
    # TODO: add parameter to function
    mean_len = 100
    window = np.ones(mean_len) / mean_len

    def high_pass_filter(val):
        return val - signal.convolve(val, window, mode='same')

    x = high_pass_filter(x)
    y = high_pass_filter(y)

    return x, y, angle
