import numpy as np
from unittest import TestCase

from spectrum_processing_1d.spectrum_functions import pierson_moskowitz_s_fun, \
    build_wave_spectrum_fun


class SpectrumFunctionTestCase(TestCase):
    def test_wave_spectrum_fun(self):
        omega = np.linspace(-np.pi, np.pi, 10)

        s = pierson_moskowitz_s_fun(omega, omega_m=1.0, var=2.0)

        np.testing.assert_allclose(
            s,
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 2.8083, 0.5396, 0.1109, 0.0323],
            atol=0.0001
        )

    def test_wave_spectrum_fun_mix(self):
        omega = np.linspace(-np.pi, np.pi, 10)

        s = pierson_moskowitz_s_fun(omega, omega_m=[-0.5, 1.0], var=[0.7, 1.2])

        np.testing.assert_allclose(
            s,
            [0.0007, 0.0025, 0.0134, 0.1628, 0.2188, 0.0000, 1.6850, 0.3238, 0.0665, 0.0194],
            atol=0.0001
        )

    def test_build_wave_spectrum_fun(self):
        s_fun = build_wave_spectrum_fun(omega_m=[-0.5, 1.0], var=[0.7, 1.2])

        omega = np.linspace(-np.pi, np.pi, 10)
        s = s_fun(omega)

        np.testing.assert_allclose(
            s,
            [0.0007, 0.0025, 0.0134, 0.1628, 0.2188, 0.0000, 1.6850, 0.3238, 0.0665, 0.0194],
            atol=0.0001
        )

    def test_build_wave_spectrum_fun_var_1(self):
        s_fun = build_wave_spectrum_fun(omega_m=[-0.3, 0.2, 0.4], var=[0.7, 1.2, 0.6])

        omega = np.linspace(-10, 10, 20000)
        s = s_fun(omega)

        std_val = np.trapz(s, omega)
        np.testing.assert_allclose(std_val, 2.5, atol=0.0001)

    def test_build_wave_spectrum_fun_tail(self):
        s_fun = build_wave_spectrum_fun(omega_m=[-0.3, 0.2, 0.4], var=[0.7, 1.2, 0.6])
        omega = np.linspace(-2, 2, 20000)
        s = s_fun(omega)

        std_val = np.trapz(s, omega)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(std_val, 2.5, atol=0.0001)

        s_fun = build_wave_spectrum_fun(omega_m=[-0.3, 0.2, 0.4], var=[0.7, 1.2, 0.6],
                                        omega_lim=1)
        omega = np.linspace(-2, 2, 20000)
        s = s_fun(omega)

        std_val = np.trapz(s, omega)
        np.testing.assert_allclose(std_val, 2.5, atol=0.0001)
