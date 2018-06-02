"""
The demo with animation of the ocean wave.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from os.path import abspath, dirname, join

from spectrum_processing_1d.processing import estimate_spectrum

sys.path.append(abspath(join(dirname(__file__), '..', 'src')))

from spectrum_processing_1d.trajectory_emulator import generate_trajectory, KFunRelation
from spectrum_processing_1d.spectrum_functions import build_wave_spectrum_fun
from spectrum_processing_1d.plot_utils import LineAnimation


def main(args=None):
    # spectrum function
    s_fun = build_wave_spectrum_fun(
        omega_m=np.array([0.15, 0.45, -0.2]) * np.pi * 2,
        var=np.array([1.0, 0.5, 0.6]),
        omega_lim=1.0 * np.pi * 2
    )
    # relation between k and omega
    k_omega_relation = KFunRelation(lambda omega: omega ** 2 / 9.8)
    # other parameters
    x_0 = np.linspace(-10, 10, 50)
    trajectory_len = 5000
    dt = 0.25
    fn = 1.0
    fs = 1 / dt

    trajectory_data = generate_trajectory(
        s_fun=s_fun,
        k_omega_relation=k_omega_relation,
        x_0=x_0,
        trajectory_len=trajectory_len,
        fn=fn,
        fs=fs
    )
    t = trajectory_data.t
    point_to_show = len(x_0) // 2
    omega, s, (x, y, angle) = estimate_spectrum(
        ax=trajectory_data.sensor_ax[:, point_to_show],
        ay=trajectory_data.sensor_ay[:, point_to_show],
        alpha=trajectory_data.sensor_alpha[:, point_to_show],
        fs=fs,
        return_trajectory=True,
        corr_dist=20.0
    )
    x_exp = trajectory_data.x[:, point_to_show]
    y_exp = trajectory_data.y[:, point_to_show]

    # plot original and estimated trajectory
    trajectory_to_show = slice(-1000, None, None)
    plt.figure("trajectories")
    plt.subplot(211)
    plt.plot(t[trajectory_to_show], x_exp[trajectory_to_show], label="original")
    plt.plot(t[trajectory_to_show], x[trajectory_to_show], label="estimated")
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.subplot(212)
    plt.plot(t[trajectory_to_show], y_exp[trajectory_to_show], label="original")
    plt.plot(t[trajectory_to_show], y[trajectory_to_show], label="estimated")
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()

    # plot original and estimated spectrum
    plt.figure("spectrum")
    # cut tails
    s_to_show = slice(len(omega) // 3, -len(omega) // 3)
    plt.xlabel('$\omega$')
    plt.ylabel('$s$')
    plt.plot(omega[s_to_show], s_fun(omega)[s_to_show], label="original")
    plt.plot(omega[s_to_show], s[s_to_show], label="estimated")
    plt.legend()

    # wave animation
    line_animation = LineAnimation(
        "wave", x_0, trajectory_data.y,
        interval=50,
        repeat=True,
        blit=True
    )
    line_animation.axes.set_xlabel('t')
    line_animation.axes.set_ylabel('y')
    line_animation.fig.set_size_inches(8, 4)
    line_animation.axes.set_ylim(-5, 5)

    plt.show()


if __name__ == '__main__':
    main()
