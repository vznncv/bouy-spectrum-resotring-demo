"""
Helper module with function and classes to emulate 1d wave surface and buoy trajectory.
"""
from collections import namedtuple

import numexpr as ne
import numpy as np
from abc import abstractmethod, ABC
from scipy.interpolate import interp1d

from spectrum_processing_1d.processing import calculate_acceleration

TrajectoryData = namedtuple('TrajectoryData', [
    't',
    'x',
    'y',
    'angle',
    'sensor_ax',
    'sensor_ay',
    'sensor_alpha'
])
"""
Information about particle trajectory.
"""


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
_MAX_TMP_POINTS = 8 * 1024 * 1024


def iter_trajectory(s_fun, k_omega_relation, x_0=0.0, frame_len=100.0, frame_num_samples=None,
                    fn=1.0, fs=None, seed=None):
    """
    Iterate blocks with trajectory data of the particles on ocean surface.

    The trajectory data contains:

    - ``x`` and ``y`` coordinates
    - particle ``angle``
    - ``sensor_ax`` and ``sensor_ay`` - acceleration in the particle coordinate system
    - ``sensor_alpha`` particle angular acceleration
    - ``t`` time coordinates

    :param s_fun: wave spectral density function ``s(omega)``
    :param k_omega_relation: relation between wave number ``k`` and frequency ``omega``
    :param x_0: initial particle coordinates. If it's scalar, then emulate trajectory of one article.
                If it's vector, then emulate trajectory of the articles.
    :param frame_len: approximate frame length in length units
    :param frame_num_samples: frame length in the samples
    :param fn: it defines the frequency range [fn, fn] that will be used from ``s_fun`` for modeling
    :param fs: sampling frequency
    :param seed: seed for random generator
    :return: iterator of the :class:``TrajectoryData`` object. Note: the object is reused
             at every iteration to save memory. So if you need to save trajectory, the data
             should be copied explicitly.
    """
    # check arguments
    x_0 = np.asfarray(x_0)
    if x_0.ndim == 0:
        x_0 = x_0[np.newaxis]
    if x_0.ndim != 1:
        raise ValueError("x_0 must be vector or scalar")
    if fs is None:
        fs = 2 * fn
    if (frame_len is not None and frame_num_samples is not None) or \
            (frame_len is None and frame_num_samples is None):
        raise ValueError("Either frame_len or frame_num_samples should be set")
    elif frame_num_samples is None:
        frame_num_samples = int(frame_len * fs)

    # memory optimization
    tmp_frame_len = _MAX_TMP_POINTS // (len(x_0) * _NUM_HARMONICS)
    if tmp_frame_len == 0:
        tmp_frame_len = 1

    if frame_num_samples < tmp_frame_len:
        return _iter_trajectory_internal(
            s_fun=s_fun, k_omega_relation=k_omega_relation, x_0=x_0, frame_num_samples=frame_num_samples,
            fn=fn, fs=fs, seed=seed
        )
    else:
        src_frame_iterator = _iter_trajectory_internal(
            s_fun=s_fun, k_omega_relation=k_omega_relation, x_0=x_0, frame_num_samples=tmp_frame_len,
            fn=fn, fs=fs, seed=seed)
        return _resize_trajectory_data_frame(src_frame_iterator, frame_num_samples)


def _resize_trajectory_data_frame(src_iterator, dst_frame_len):
    src_frame = next(src_iterator)
    src_frame_len = src_frame.x.shape[0]
    num_points = src_frame.x.shape[1]

    dst_frame = TrajectoryData(
        t=np.zeros(dst_frame_len),
        x=np.zeros((dst_frame_len, num_points)),
        y=np.zeros((dst_frame_len, num_points)),
        angle=np.zeros((dst_frame_len, num_points)),
        sensor_ax=np.zeros((dst_frame_len, num_points)),
        sensor_ay=np.zeros((dst_frame_len, num_points)),
        sensor_alpha=np.zeros((dst_frame_len, num_points))
    )
    src_start_pos = 0
    dst_start_pos = 0
    while True:
        copy_range = min(dst_frame_len - dst_start_pos, src_frame_len - src_start_pos)
        dst_end_pos = dst_start_pos + copy_range
        src_end_pos = src_start_pos + copy_range
        src_slice = slice(src_start_pos, src_end_pos)
        dst_slice = slice(dst_start_pos, dst_end_pos)
        for src_array, dst_array in zip(src_frame, dst_frame):
            dst_array[dst_slice, ...] = src_array[src_slice, ...]

        if dst_end_pos == dst_frame_len:
            yield dst_frame
            dst_start_pos = 0
        else:
            dst_start_pos = dst_end_pos
        if src_end_pos == src_frame_len:
            src_frame = next(src_iterator)
            src_start_pos = 0
        else:
            src_start_pos = src_end_pos


def _iter_trajectory_internal(s_fun, k_omega_relation, x_0, frame_num_samples, fn, fs, seed=None):
    # calculate helper parameters
    dt = 1 / fs
    random = np.random.RandomState(seed)
    t_0 = 0
    num_points = len(x_0)
    # calculate wave frequencies
    num_harm = _NUM_HARMONICS
    omega = np.linspace(-fn, fn, num_harm) * 2 * np.pi
    # wave number of the corresponding harmonics
    k = k_omega_relation.calc_k(omega)
    # calculate harmonic amplitudes
    harmonic_width = (omega[-1] - omega[0]) / len(omega)
    s = s_fun(omega)
    a = np.sqrt(s)
    a_norm = a * np.sqrt(harmonic_width) * np.sqrt(2)
    # calculate phases
    # phase dimensions:
    # 0 - time axis
    # 1 - space axis
    # 2 - separate harmonics
    d_phi = np.broadcast_to(omega * dt, (1, 1, num_harm))
    d_phi_noise = d_phi * (1 / np.log2(num_harm))
    frame_phi_harm_0 = (random.rand(num_harm) * 2 * np.pi)[np.newaxis, np.newaxis, :]
    frame_phi_k_offset = k * x_0[np.newaxis, :, np.newaxis]

    frame_phi = np.zeros((frame_num_samples, num_points, num_harm))

    # create base frame
    trajectory_data = TrajectoryData(
        t=np.zeros(frame_num_samples),
        x=np.zeros((frame_num_samples, num_points)),
        y=np.zeros((frame_num_samples, num_points)),
        angle=np.zeros((frame_num_samples, num_points)),
        sensor_ax=np.zeros((frame_num_samples, num_points)),
        sensor_ay=np.zeros((frame_num_samples, num_points)),
        sensor_alpha=np.zeros((frame_num_samples, num_points))
    )

    t = np.arange(frame_num_samples) * dt + t_0

    while True:
        trajectory_data.t[...] = t

        frame_d_phi_harm = d_phi + random.randn(frame_num_samples, 1, num_harm) * d_phi_noise
        frame_phi_harm = frame_phi_harm_0 + frame_d_phi_harm.cumsum(0)
        frame_phi_harm_0 = frame_phi_harm[-1, ...] + d_phi

        frame_phi[...] = frame_phi_harm
        frame_phi += frame_phi_k_offset

        # NOTE: the performance can be improved by ifft usage
        sin_harmonics = ne.evaluate("a_norm * sin(frame_phi)")
        cos_harmonics = ne.evaluate("a_norm * cos(frame_phi)")
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
        trajectory_data.sensor_ax[...] = ax * np.cos(angle) - ay * np.sin(angle)
        trajectory_data.sensor_ay[...] = ax * np.sin(angle) + ay * np.cos(angle)

        t += dt * frame_num_samples

        yield trajectory_data


def generate_trajectory(**kwargs):
    """
    Non-stream version of the :fun:`iter_trajectory`.

    Instead ``frame_len`` and ``frame_num_samples`` the ``trajectory_len`` and
    ``trajectory_num_samples`` should be used.
    """
    if 'frame_len' in kwargs:
        raise ValueError("Unknown argument frame_len. Probably you should use trajectory_len")
    if 'frame_num_samples' in kwargs:
        raise ValueError("Unknown argument frame_len. Probably you should use trajectory_num_samples")

    kwargs['frame_len'] = kwargs.pop('trajectory_len', None)
    kwargs['frame_num_samples'] = kwargs.pop('trajectory_num_samples', None)

    return next(iter_trajectory(**kwargs))
