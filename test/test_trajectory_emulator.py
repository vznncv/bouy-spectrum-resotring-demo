import numpy as np
from hamcrest import assert_that
from scipy.signal import welch
from unittest import TestCase

from spectrum_processing_1d.spectrum_functions import build_wave_spectrum_fun
from spectrum_processing_1d.trajectory_emulator import KFunRelation, generate_trajectory
from spectrum_processing_1d.utils import calculate_trajectory


class KFunRelationTestCase(TestCase):
    def test_relation(self):
        k_fun = lambda omega: omega ** 2 / 9.8
        k_fun_exp = lambda omega: (omega ** 2 / 9.8) * np.sign(omega)
        omega_fun_exp = lambda k: np.sqrt(np.abs(k) * 9.8) * np.sign(k)

        k_omega_relation = KFunRelation(k_fun)

        omega = np.linspace(-1, 1, 10)
        k = k_omega_relation.calc_k(omega)
        np.testing.assert_allclose(
            k,
            k_fun_exp(omega),
            atol=0.0001
        )
        print(repr(k))

        k = np.linspace(-3, 3, 10)
        omega = k_omega_relation.calc_omega(k)
        np.testing.assert_allclose(
            omega,
            omega_fun_exp(k),
            atol=0.0001
        )


class TrajectoryEmulatorTestCase(TestCase):

    def setUp(self):
        self.spectrum_fun = build_wave_spectrum_fun(
            omega_m=np.array([0.08, 0.15, -0.1]) * 2 * np.pi,
            std=np.array([0.8, 0.4, 0.6])
        )
        self.k_omega_relation = KFunRelation(lambda omega: omega ** 2 / 9.8)

    def test_iter_trajectory(self):
        dt = 0.5
        trajectory_data = generate_trajectory(
            s_fun=self.spectrum_fun,
            k_omega_relation=self.k_omega_relation,
            x_0=np.linspace(-100, 100, 3),
            trajectory_len=10000,
            fn=1.0,
            fs=1 / dt,
            seed=0
        )

        # check coordinate variance
        x_var = trajectory_data.x.var(0)
        np.testing.assert_allclose(x_var, 1.8, atol=0.1)
        y_var = trajectory_data.y.var(0)
        np.testing.assert_allclose(y_var, 1.8, atol=0.1)

        # check that angle is small
        max_angle = np.abs(trajectory_data.angle).max(0)
        np.testing.assert_allclose(max_angle, 0.0, atol=0.1)

        # check that angle accelerations is small
        max_alpha = np.abs(trajectory_data.sensor_alpha).max(0)
        np.testing.assert_allclose(max_alpha, 0.0, atol=0.05)

        alpha = 0.01
        x_est = calculate_trajectory(trajectory_data.sensor_ax, alpha=alpha, spacing=dt)
        y_est = calculate_trajectory(trajectory_data.sensor_ay, alpha=alpha, spacing=dt)

        # remove transitional process
        transitional_range = 1000
        x_est = x_est[transitional_range:, :]
        y_est = y_est[transitional_range:, :]

        # estimate psd function
        signal = x_est + 1j * y_est
        f, s_est = welch(
            signal,
            fs=1 / dt,
            nperseg=128,
            nfft=512,
            scaling='spectrum',
            return_onesided=False,
            axis=0,
        )
        f = np.fft.fftshift(f, axes=0)
        s_est = np.fft.fftshift(s_est, axes=0)
        # TODO: fix hack
        f *= 2
        s_est *= 2

        s_exp = self.spectrum_fun(f * 2 * np.pi)
        s_exp = np.broadcast_to(s_exp[..., np.newaxis], s_est.shape)

        # check estimation
        cmp_res = np.isclose(s_est, s_exp, atol=0.1)
        mismatch_percent = 1 - cmp_res.sum(0) / len(cmp_res)
        assert_that(np.all(mismatch_percent < 0.08))
