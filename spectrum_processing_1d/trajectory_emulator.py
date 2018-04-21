"""
Helper module with function and classes to emulate 1d wave surface and buoy trajectory.
"""
from abc import abstractmethod, ABC
from collections import namedtuple

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from spectrum_processing_1d.spectrum_functions import build_wave_spectrum_fun

TrajectoryData = namedtuple('TrajectoryData', [
    't',
    'x',
    'y',
    'surface',
    'sensor_ax',
    'sensor_ay',
    'sensor_xi'
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

        self._omega_k_interp = interp1d(omega, k, bounds_error=True)
        self._k_omega_k_interp = interp1d(k, omega, bounds_error=True)

    def calc_omega(self, k):
        return self._omega_k_interp(k)

    def calc_k(self, omega):
        return self._k_omega_k_interp(omega)


def iter_trajectory(s_fun, kw_relation, num_x_points=256, frame_len=100, fs_t=2.0, fs_x=2.0):
    # create amplitude functions
    def a_from_omega(omega):
        f = omega / (2 * np.pi)
        s = s_fun(f)
        return np.sqrt(s)

    # prepare parameters for a calculations
    k = np.linspace(-fs_x, fs_x, num_x_points)
    omega = kw_relation.calc_omega(k)
    a = a_from_omega(omega)
    phi = np.random.randn(omega.shape) * 2 * np.pi
    dt = 1 / fs_t
    d_phi = omega * dt
    t = 0

    # create base frame
    trajectory_data = TrajectoryData(
        t=np.arange(frame_len) * dt,
        x=np.zeros(frame_len),
        y=np.zeros(frame_len),
        surface=np.zeros((frame_len, num_x_points)),
        sensor_ax=np.zeros(frame_len),
        sensor_ay=np.zeros(frame_len),
        sensor_xi=np.zeros(frame_len)
    )

    while True:
        for i in range(frame_len):
            # fill data
            trajectory_data.t[i] = t
            trajectory_data.surface[i, :] = np.fft.ifft(a * np.exp(phi)).real * 2
            trajectory_data.x[i] = (a * np.sin(phi)).sum()
            trajectory_data.y[i] = (a * np.cos(phi)).sum()

            # update parameters
            t += dt
            phi += d_phi

        yield trajectory_data


def get_test_spectrum_fun():
    return build_wave_spectrum_fun(
        omega_m=[0.3, 0.8, -0.5],
        std=[1.0, 0.5, 0.6]
    )


if __name__ == '__main__':
    omega = np.linspace(-np.pi, np.pi, 1000)
    s_fun = get_test_spectrum_fun()
    s = s_fun(omega)

    plt.plot(omega, s)

    plt.show()
