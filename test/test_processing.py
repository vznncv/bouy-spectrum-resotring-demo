import numpy as np
import scipy.signal as signal
from hamcrest import assert_that, greater_than, less_than
from unittest import TestCase

from spectrum_processing_1d.processing import calculate_acceleration, calculate_trajectory, \
    calculate_particle_trajectory, estimate_spectrum
from spectrum_processing_1d.spectrum_functions import build_wave_spectrum_fun
from spectrum_processing_1d.trajectory_emulator import KFunRelation, generate_trajectory
from testing_utils import compensate_lag, assert_signals_almost_equal, remove_transition_interval


class CalculateAcceleration(TestCase):
    def test_sin(self):
        t = np.linspace(0, 100, 1000)
        dt = (t[-1] - t[0]) / len(t)

        x = np.sin(t)
        expected_ax = np.sin(t + np.pi)

        ax = calculate_acceleration(x, spacing=dt)

        cmp_res = np.isclose(ax, expected_ax, atol=0.005)
        assert_that(cmp_res.sum() / cmp_res.size, greater_than(0.99))


class CalculateTrajectory(TestCase):
    def test_sin(self):
        t = np.linspace(0, 10000, 200000)
        dt = (t[-1] - t[0]) / len(t)

        ax = np.sin(t)
        x = np.sin(t + np.pi)

        x_est = calculate_trajectory(ax=ax, spacing=dt)

        t, x, x_est = remove_transition_interval(t, x, x_est, transition_interval=6000)
        x_est, x = compensate_lag(x_est, x)

        assert_signals_almost_equal(x_est, x, deviation_std=0.1)

        # import matplotlib.pyplot as plt
        # slice_to_show = slice(-1000, None, None)
        # plt.plot(t[slice_to_show], x[slice_to_show])
        # plt.plot(t[slice_to_show], x_est[slice_to_show])
        # plt.legend('desired', 'actual')
        # plt.show()

    def test_noisy_sin(self):
        t = np.linspace(0, 10000, 200000)
        dt = (t[-1] - t[0]) / len(t)

        ax = np.sin(t) + np.random.randn(*t.shape) * 0.01
        x = np.sin(t + np.pi)

        x_est = calculate_trajectory(ax=ax, spacing=dt, correlation_distance=80)

        t, x, x_est = remove_transition_interval(t, x, x_est, transition_interval=6000)
        x_est, x = compensate_lag(x_est, x)

        assert_signals_almost_equal(x_est, x, deviation_std=0.04)

        # import matplotlib.pyplot as plt
        # slice_to_show = slice(-1000, None, None)
        # plt.plot(t[slice_to_show], x[slice_to_show])
        # plt.plot(t[slice_to_show], x_est[slice_to_show])
        # plt.legend(['desired', 'actual'])
        # plt.show()

    def test_sophisticated_signal(self):
        # frequency response
        f = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        g = [0.0, 0.1, 1.2, 0.8, 0.1, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]
        a = [1]
        fs = 2.0
        b = signal.firwin2(numtaps=129, freq=f, gain=g, fs=fs)

        random = np.random.RandomState(0)
        x_len = 20000
        t = np.arange(x_len) * 0.5
        x = signal.lfilter(b=b, a=a, x=random.randn(x_len))

        # calculate acceleration and add noise
        ax = calculate_acceleration(x, spacing=1 / fs)
        noise_std = ax.std() * 0.05
        ax += random.randn(*ax.shape) * noise_std

        x_est = calculate_trajectory(
            ax=ax,
            spacing=1 / fs,
            correlation_distance=30
        )

        x_est, x = compensate_lag(x_est, x)

        assert_signals_almost_equal(x_est, x, deviation_std=0.12)

        # resample_k = 5
        # x_est = signal.resample(x=x_est, num=resample_k * len(x_est))
        # x = signal.resample(x=x, num=resample_k * len(x))
        #
        # import matplotlib.pyplot as plt
        # slice_to_show = slice(-2000, -1500, None)
        # plt.plot(x[slice_to_show])
        # plt.plot(x_est[slice_to_show])
        # # plt.plot(t[slice_to_show], x_est[slice_to_show])
        # plt.legend(['desired', 'actual'])
        # plt.show()


class CalculateParticleTrajectoryTestCase(TestCase):
    def add_noise(self, signal, relative_noise_power=0.05, seed=0):
        noise_std = signal.std() * relative_noise_power
        random_state = np.random.RandomState(seed)
        return signal + random_state.randn(*signal.shape) * noise_std

    def test_single_wave(self):
        t = np.linspace(0, 10000, 200000)
        dt = (t[-1] - t[0]) / len(t)
        omega = 1.0
        angle_noise_power = 0.01
        xy_noise_power = 0.02

        x = np.sin(t * omega)
        y = np.cos(t * omega)
        angle = np.arctan(x) * 0.1
        ax_tmp = calculate_acceleration(x, spacing=dt)
        ay_tmp = calculate_acceleration(y, spacing=dt)
        ax = ax_tmp * np.cos(angle) - ay_tmp * np.sin(angle)
        ay = ax_tmp * np.sin(angle) + ay_tmp * np.cos(angle)
        alpha = calculate_acceleration(angle, spacing=dt)
        ax = self.add_noise(ax, xy_noise_power)
        ay = self.add_noise(ay, xy_noise_power)
        alpha = self.add_noise(alpha, angle_noise_power)

        x_est, y_est, angle_est = calculate_particle_trajectory(
            sensor_ax=ax,
            sensor_ay=ay,
            sensor_alpha=alpha,
            spacing=dt,
            corr_dist=50.0
        )
        # compensate integration delay
        x, x_est = compensate_lag(x, x_est)
        y, y_est = compensate_lag(y, y_est)
        angle, angle_est = compensate_lag(angle, angle_est)

        # remove transition interval
        t, x, y, angle, x_est, y_est, angle_est = remove_transition_interval(
            t, x, y, angle, x_est, y_est, angle_est,
            transition_interval=2000
        )

        # check results
        assert_signals_almost_equal(x_est, x, deviation_std=0.06)
        assert_signals_almost_equal(y_est, y, deviation_std=0.06)
        assert_signals_almost_equal(angle_est, angle, deviation_std=0.03)

        # import matplotlib.pyplot as plt
        # slice_to_show = slice(-2000, -1000, None)
        # for i, (s_est, s, label) in enumerate(zip([x_est, y_est, angle_est], [x, y, angle], ['x', 'y', 'angle'])):
        #     plt.subplot(3, 1, i + 1)
        #     plt.plot(t[slice_to_show], s[slice_to_show])
        #     plt.plot(t[slice_to_show], s_est[slice_to_show])
        #     plt.legend(['desired', 'actual'])
        #     plt.title(label)
        # plt.show()


class EstimateSpectumTestCase(TestCase):
    def _test_simple_spectrum(self, fs):
        s_fun = build_wave_spectrum_fun(
            omega_m=np.array([0.25]) * np.pi * 2,
            var=np.array([2.5]),
            omega_lim=1.0 * 2 * np.pi
        )
        # relation between k and omega
        k_omega_relation = KFunRelation(lambda omega: omega ** 2 / 9.8)
        # other parameters
        fn = 1.0
        trajectory_len = 4000

        trajectory_data = generate_trajectory(
            s_fun=s_fun,
            k_omega_relation=k_omega_relation,
            x_0=0,
            trajectory_len=trajectory_len,
            fn=fn,
            fs=fs,
            seed=0
        )

        omega, s, (x_est, _, _) = estimate_spectrum(
            ax=trajectory_data.sensor_ax[:, 0],
            ay=trajectory_data.sensor_ay[:, 0],
            alpha=trajectory_data.sensor_alpha[:, 0],
            fs=fs,
            nfft=1024,
            nperseg=128,
            corr_dist=14.0,
            return_trajectory=True
        )
        expected_s = s_fun(omega)

        assert_that((s - expected_s).std(), less_than(0.5))
        np.testing.assert_allclose(s, expected_s, atol=0.7)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(omega, s)
        # plt.plot(omega, expected_s)
        # plt.legend(['estimation', 'actual'])
        #
        # slice_to_show = slice(1500, 2000)
        # plt.figure()
        # t = trajectory_data.t
        # x = trajectory_data.x
        # plt.plot(t[slice_to_show], x_est[slice_to_show])
        # plt.plot(t[slice_to_show], x[slice_to_show])
        # plt.legend(['estimation', 'actual'])
        #
        # plt.show()

    def test_simple_spectrum_fs_2(self):
        self._test_simple_spectrum(fs=2.0)

    def test_simple_spectrum_fs_5(self):
        self._test_simple_spectrum(fs=5.0)
