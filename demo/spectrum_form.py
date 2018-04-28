# noinspection PyUnresolvedReferences
import numpy as np
import scipy.stats

from demo_util import get_demo_plot_manager
from spectrum_processing_1d.spectrum_functions import build_wave_spectrum_fun


def main():
    with get_demo_plot_manager() as pm:
        fig = pm.build_figure('spatial_temporal_spectrum')

        omega = np.linspace(0, 0.5, 200)
        k = np.linspace(-0.02, 0.02, 400)
        k, omega = np.meshgrid(k, omega)
        g = 9.8
        k_max = omega ** 2 / g

        omega_k = build_wave_spectrum_fun(omega_m=0.2, std=1.0)(omega)

        scale = 0.0005
        h = np.max(np.stack((
            scipy.stats.norm.pdf(k, loc=k_max, scale=scale),
            scipy.stats.norm.pdf(k, loc=-k_max, scale=scale)
        )), axis=0) * omega_k

        ax = fig.gca(projection='3d')

        # Plot the surface.
        ax.plot_surface(
            k, omega, h,
            linewidth=0,
            antialiased=True,
            rstride=2,
            cstride=2
        )
        ax.set_xlabel('$k$')
        ax.set_ylabel('$\omega$')
        ax.set_zlabel('$s$')
        ax.view_init(75, -55)

        pm.save_figure()

        pm.show()


if __name__ == '__main__':
    main()
