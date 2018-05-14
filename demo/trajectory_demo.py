# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
import numpy as np

from demo_util import get_demo_plot_manager, restore_spectrum
from spectrum_processing_1d.spectrum_functions import build_wave_spectrum_fun
from spectrum_processing_1d.trajectory_emulator import KFunRelation, generate_trajectory


def main():
    # spectrum function
    s_fun = build_wave_spectrum_fun(
        omega_m=np.array([0.15, 0.45, -0.2]) * np.pi * 2,
        std=np.array([1.0, 0.5, 0.6])
    )
    # relation between k and omega
    k_omega_relation = KFunRelation(lambda omega: omega ** 2 / 9.8)
    dt = 0.25
    fn = 1.0
    fs = 1 / dt

    # demo to show source spectrum and wave
    x_0 = np.linspace(-10, 10, 1000)

    trajectory_data_wave_demo = generate_trajectory(
        s_fun=s_fun,
        k_omega_relation=k_omega_relation,
        x_0=x_0,
        trajectory_len=3,
        fn=fn,
        fs=fs
    )

    # demo to show spectrum restoration
    trajectory_data_wave_param = generate_trajectory(
        s_fun=s_fun,
        k_omega_relation=k_omega_relation,
        x_0=[0],
        trajectory_len=100000,
        fn=fn,
        fs=fs
    )

    omega_est, s_est, x_est, y_est, angle_est = restore_spectrum(
        ax=trajectory_data_wave_param.sensor_ax[:, 0],
        ay=trajectory_data_wave_param.sensor_ay[:, 0],
        alpha=trajectory_data_wave_param.sensor_alpha[:, 0],
        fs=fs,
        nperseg=512,
        nfft=1024
    )
    # cut spectrum
    omega_lim = np.searchsorted(omega_est, -fn * 2 * np.pi, side='right')
    omega_est = omega_est[omega_lim:-omega_lim]
    s_est = s_est[omega_lim:-omega_lim]

    with get_demo_plot_manager(start_figure_num=2) as pm:
        pm.build_figure('source_spectrum', height='width/3')
        omega = np.linspace(-fn, fn, 1000) * 2 * np.pi
        src_s = s_fun(omega)
        plt.plot(omega, src_s)
        plt.xlabel('$\omega$')
        plt.ylabel('$s$')
        pm.save_figure()

        pm.build_figure('wave_sample', height='width/3')
        y = trajectory_data_wave_demo.y[0]
        plt.plot(x_0, y)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        pm.save_figure()

        pm.build_figure('x_y_coords', height='width/3')
        slice_to_show = slice(-1000, None, None)
        x_est = x_est[slice_to_show]
        x_src = trajectory_data_wave_param.x[slice_to_show]
        y_est = y_est[slice_to_show]
        y_src = trajectory_data_wave_param.y[slice_to_show]
        t = trajectory_data_wave_param.t[slice_to_show]
        plt.subplot(211)
        plt.plot(t, x_src, label='source')
        plt.plot(t, x_est, label='estimation')
        plt.legend()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.subplot(212)
        plt.plot(t, y_src, label='source')
        plt.plot(t, y_est, label='estimation')
        plt.legend()
        plt.xlabel('$t$')
        plt.ylabel('y')
        pm.save_figure()

        pm.build_figure('spectrum estimation', height='width/3')
        plt.plot(omega_est, s_est)
        plt.xlabel('$\omega$')
        plt.ylabel('$s$')
        pm.save_figure()

        pm.show()


if __name__ == '__main__':
    main()
