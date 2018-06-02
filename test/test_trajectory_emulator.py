import numpy as np
from hamcrest import assert_that, equal_to, less_than
from unittest import TestCase

from spectrum_processing_1d.spectrum_functions import build_wave_spectrum_fun
from spectrum_processing_1d.trajectory_emulator import KFunRelation, generate_trajectory


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
            var=np.array([0.8, 0.4, 0.6]),
            omega_lim=-0.5 * 2 * np.pi
        )
        self.k_omega_relation = KFunRelation(lambda omega: omega ** 2 / 9.8)

    def test_generate_trajectory(self):
        num_points = 2
        trajectory_data = generate_trajectory(
            s_fun=self.spectrum_fun,
            k_omega_relation=self.k_omega_relation,
            x_0=np.linspace(-100, 100, num_points),
            trajectory_len=4000,
            fn=1.0,
            fs=3.0,
            seed=0
        )

        num_samples = 12000
        for elem in trajectory_data:
            assert_that(elem.shape[0], equal_to(num_samples))

        # check coordinate variance and mean value
        x_vars = trajectory_data.x.var(0)
        np.testing.assert_allclose(x_vars, 1.8, atol=0.1)
        y_vars = trajectory_data.y.var(0)
        np.testing.assert_allclose(y_vars, 1.8, atol=0.1)

        x_mean = trajectory_data.x.mean(0)
        np.testing.assert_allclose(x_mean, 0, atol=0.01)
        y_mean = trajectory_data.y.mean(0)
        np.testing.assert_allclose(y_mean, 0, atol=0.01)

        angle_var = trajectory_data.angle.var(0)
        np.testing.assert_allclose(angle_var, 0.0001, atol=0.001)
        angle_mean = trajectory_data.angle.mean(0)
        np.testing.assert_allclose(angle_mean, 0, atol=0.001)

    def test_fs_insensitivity(self):
        def get_trajectory(fs, seed=0):
            return generate_trajectory(
                s_fun=self.spectrum_fun,
                k_omega_relation=self.k_omega_relation,
                x_0=0.0,
                trajectory_len=30,  # note: small range is chosen to prevent noise influence
                fn=1.0,
                fs=fs,
                seed=seed
            )

        fs_base = 2.0
        fs_k = [1, 3, 5]
        trajectory_datas = [get_trajectory(fs_base * k, 0) for k in fs_k]

        # import matplotlib.pyplot as plt
        # for i, trajectory_data in enumerate(trajectory_datas):
        #     plt.subplot(len(fs_k), 1, i + 1)
        #     plt.plot(trajectory_data.t, trajectory_data.x[:, 0])
        # plt.show()

        vals_to_compare = [trajectory_data.x[::k] for k, trajectory_data in zip(fs_k, trajectory_datas)]
        standard_val = vals_to_compare[0]
        vals_to_compare = vals_to_compare[1:]
        for val in vals_to_compare:
            assert_that((val-standard_val).std(), less_than(0.6))
