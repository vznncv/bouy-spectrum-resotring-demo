from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt


def spectrum_fun(omega, omega_m, std):


    omega_ri = omega_m / omega
    s = (1.25 / 4) * (omega_ri ** 5 / np.abs(omega_m)) * np.exp(-1.25 * omega_ri ** 4)
    s *= std
    if omega_m > 0:
        s[omega <= 0] = 0
    else:
        s[omega >= 0] = 0

    return s


def get_test_spectrum(n_points):
    omega = np.linspace(-3, 3, n_points)
    s = spectrum_fun(omega, 0.3, 1.0) + \
        spectrum_fun(omega, 0.8, 0.5) + \
        spectrum_fun(omega, -0.5, 0.6)
    return omega, s


def main(args=None):
    n = 256

    # test spectrum
    omega, s = get_test_spectrum(n)

    # amplitudes
    a = np.sqrt(s)
    # phases
    phi_0 = np.random.randn(*a.shape) * 2 * np.pi

    # time
    fs = 3
    total_t = 30 * 60
    t = np.linspace(0, total_t, total_t * fs)

    # buoy coordinates
    phi = t[..., np.newaxis] * omega + phi_0
    x = (a * np.sin(phi)).sum(-1)
    y = (a * np.cos(phi)).sum(-1)

    # try to restore spectrum
    res = x - 1j * y

    omega_est, s_est = welch(res, fs=fs, nfft=n, scaling='density')
    omega_est *= 2 * np.pi
    s_est *= (1 / n)

    # plot x and y coordinates
    plt.figure('buoy coordinates')
    plt.subplot(211)
    plt.plot(t, x)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.subplot(212)
    plt.plot(t, y)
    plt.xlabel('t')
    plt.ylabel('y')

    # plot spectra
    plt.figure('spectra')
    plt.plot(omega, s)
    plt.plot(omega_est, s_est)

    plt.show()


if __name__ == '__main__':
    main()
