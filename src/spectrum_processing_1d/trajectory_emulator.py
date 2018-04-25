"""
Helper module with function and classes to emulate 1d wave surface and buoy trajectory.
"""
from collections import namedtuple

import numpy as np
from abc import abstractmethod, ABC
from scipy.interpolate import interp1d

from spectrum_processing_1d.utils import calculate_acceleration

TrajectoryData = namedtuple('TrajectoryData', [
    't',
    'x',
    'y',
    'angle',
    'sensor_ax',
    'sensor_ay',
    'sensor_alpha'
])


class KOmegaRelation(ABC):
    @abstractmethod
    def calc_omega(self, k):
        pass

    @abstractmethod
    def calc_k(self, omega):
        pass


class KFunRelation(KOmegaRelation):
    _N_POINTS = 4096
    _OMEGA_LIM = 10.0

    def __init__(self, k_fun):
        self._k_fun = k_fun

        omega = np.linspace(0, self._OMEGA_LIM, self._N_POINTS)
        k = k_fun(omega)
        k[0] = 0

        omega = np.hstack((-omega[::-1], omega[1:]))
        k = np.hstack((-k[::-1], k[1:]))

        self._k_to_omega_interp = interp1d(k, omega, bounds_error=True)
        self._omega_to_k_interp = interp1d(omega, k, bounds_error=True)

    def calc_omega(self, k):
        return self._k_to_omega_interp(k)

    def calc_k(self, omega):
        return self._omega_to_k_interp(omega)


_NUM_HARMONICS = 1024


def iter_trajectory(s_fun, k_omega_relation, x_0=0.0, frame_len=100, fn=1.0, fs=None, seed=None):
    """

    :param s_fun: s(omega)
    :param k_omega_relation: k(omega) and omega(k)
    :param x_0:
    :param frame_len:
    :param fn:
    :param fs:
    :param seed:
    :return:
    """
    # default arguments
    x_0 = np.asfarray(x_0)
    if np.isscalar(x_0):
        x_0 = x_0[np.newaxis]
    if fs is None:
        fs = 2 * fn
    x_0 = x_0[..., np.newaxis]

    # calculate helper parameters
    dt = 1 / fs
    random = np.random.RandomState(seed)
    t = 0
    num_points = len(x_0)
    # calculate wave frequencies
    omega = np.linspace(-fn, fn, _NUM_HARMONICS) * 2 * np.pi
    k = k_omega_relation.calc_k(omega)
    # calculate harmonic amplitudes
    harmonic_width = (omega[-1] - omega[0]) / len(omega)
    s = s_fun(omega)
    a = np.sqrt(s)
    a_norm = a * np.sqrt(harmonic_width) * np.sqrt(2)
    # calculate initial phases
    phi_0 = random.rand(*omega.shape) * 2 * np.pi
    # calculate phases
    phi = x_0 * k + phi_0
    d_phi = omega * dt

    # create base frame
    trajectory_data = TrajectoryData(
        t=np.zeros(frame_len),
        x=np.zeros((frame_len, num_points)),
        y=np.zeros((frame_len, num_points)),
        angle=np.zeros((frame_len, num_points)),
        sensor_ax=np.zeros((frame_len, num_points)),
        sensor_ay=np.zeros((frame_len, num_points)),
        sensor_alpha=np.zeros((frame_len, num_points))
    )

    while True:
        t = np.arange(frame_len) * dt + t
        frame_phi = phi[np.newaxis] + d_phi * t[..., np.newaxis, np.newaxis]

        trajectory_data.t[...] = t
        sin_harmonics = a_norm * np.sin(frame_phi)
        cos_harmonics = a_norm * np.cos(frame_phi)
        trajectory_data.x[...] = -sin_harmonics.sum(-1)
        trajectory_data.y[...] = cos_harmonics.sum(-1)

        # estimate object angle
        angle = trajectory_data.angle[...] = np.arctan((-a_norm * sin_harmonics * k).sum(-1))

        # calculate angle angular acceleration
        trajectory_data.sensor_alpha[...] = calculate_acceleration(trajectory_data.angle, spacing=dt, axis=0)

        # calculate x and y acceleration
        ax = calculate_acceleration(trajectory_data.x, spacing=dt, axis=0)
        ay = calculate_acceleration(trajectory_data.y, spacing=dt, axis=0)

        # calculate sensor acceleration
        trajectory_data.sensor_ax[...] = ax * np.cos(angle) + ay * np.sin(angle)
        trajectory_data.sensor_ay[...] = -ax * np.sin(angle) + ay * np.cos(angle)

        t += dt * frame_len
        phi += d_phi * frame_len

        yield trajectory_data


def generate_trajectory(**kwargs):
    """
    Non-stream version of the :fun:`iter_trajectory`.
    """
    if 'frame_len' in kwargs:
        raise ValueError("Unknown argument frame_len. Probably you should use trajectory_len")

    trajectory_len = kwargs.pop('trajectory_len')
    kwargs['frame_len'] = trajectory_len

    return next(iter_trajectory(**kwargs))
