"""
The demo script of the ocean wave spectrum and its spatio-temporal representation.
"""
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib import cm

from demo_util import get_demo_plot_manager
from spectrum_processing_1d.spectrum_functions import build_wave_spectrum_fun


def main():
    with get_demo_plot_manager() as pm:
        pm.build_figure('spatial_temporal_spectrum')

        omega_lim = 0.5
        k_lim = 0.02
        resolution_coefficient = 10

        omega_1d = np.linspace(-omega_lim, omega_lim, 50 * resolution_coefficient)
        k = np.linspace(-k_lim, k_lim, 50 * resolution_coefficient)
        k, omega = np.meshgrid(k, omega_1d)
        g = 9.8
        k_max = omega ** 2 / g

        s_omega_fun = build_wave_spectrum_fun(omega_m=0.2, var=1.0)

        scale = 0.0005
        h = np.max(np.stack((
            scipy.stats.norm.pdf(k, loc=k_max, scale=scale),
            scipy.stats.norm.pdf(k, loc=-k_max, scale=scale)
        )), axis=0) * s_omega_fun(omega * np.sign(k))

        plt.subplot(121)

        plt.plot(omega_1d, s_omega_fun(omega_1d), 'C1')
        plt.xlabel('$\omega$')
        plt.ylabel('$s$')
        plt.title('$S_t(\omega)$')
        plt.xlim([-omega_lim, omega_lim])

        ax = plt.subplot(122, projection='3d')

        # Plot the surface.
        ax.view_init(35, -65)
        ax.plot_surface(
            k, omega, h,
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1,
            cmap=cm.jet
        )
        ax.set_xlabel('$k$', labelpad=10)
        ax.set_ylabel('$\omega$', labelpad=10)
        ax.set_zlabel('$s$', labelpad=20)
        plt.title('$S(\omega, k)$', pad=50)

        # adjust tick params
        ax.locator_params(nbins=5, axis='x')
        ax.locator_params(nbins=5, axis='y')
        ax.tick_params(pad=10, axis='z')

        pm.save_figure()

        pm.show()


if __name__ == '__main__':
    main()
