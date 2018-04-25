import numpy as np
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
