import numpy as np


def wave_spectrum_fun(omega, omega_m, std):
    omega = np.asfarray(omega)

    omega_ri = omega_m / omega
    s = (1.25 / 4) * (omega_ri ** 5 / np.abs(omega_m)) * np.exp(-1.25 * omega_ri ** 4)
    s *= std
    if omega_m > 0:
        s[omega <= 0] = 0
    else:
        s[omega >= 0] = 0

    return s


def wave_spectrum_fun_mix(omega, omega_m, std):
    omega = np.asfarray(omega)
    omega_m = np.asfarray(omega_m)
    std = np.asfarray(std)

    omega = omega[..., np.newaxis]

    omega_ri = omega_m / omega
    s = (1.25 / 4) * (omega_ri ** 5 / np.abs(omega_m)) * np.exp(-1.25 * omega_ri ** 4)
    s *= std

    s[(omega_m > 0) & (omega <= 0)] = 0
    s[(omega_m < 0) & (omega >= 0)] = 0

    return s.sum(axis=-1)


def build_wave_spectrum_fun(omega_m, std):
    def spectrum_fun(omega):
        return wave_spectrum_fun_mix(omega, omega_m, std)

    return spectrum_fun
