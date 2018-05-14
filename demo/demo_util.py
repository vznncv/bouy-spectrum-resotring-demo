import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from os.path import join, dirname, exists
from scipy.signal import welch

from spectrum_processing_1d.utils import calculate_particle_trajectory


def get_image_dir():
    image_dir = join(dirname(__file__), 'images')
    if not exists(image_dir):
        os.mkdir(image_dir)
    return image_dir


class PlotManager:
    """
    Helper class for a demo graphics showing and saving.
    """
    _WINDOW_DPI = 72
    _RES_IMAGE_DPI = 150

    def __init__(self, result_dir, interactive=True, start_figure_num=1):
        self._interactive = interactive
        self._result_dir = result_dir

        self._matplotlib_style_context = None
        self._old_backend = None

        self._figure_count = None
        self._start_figure_num = start_figure_num
        self._figures = None

    def __enter__(self):
        self._figure_count = self._start_figure_num
        self._figures = {}

        if not self._interactive:
            self._old_backend = matplotlib.get_backend()
            plt.switch_backend('AGG')

        self._matplotlib_style_context = matplotlib.style.context(self.get_plot_style())
        self._matplotlib_style_context.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # close all figures
        plt.close('all')
        # revert styles
        context = self._matplotlib_style_context
        self._matplotlib_style_context = None
        context.__exit__(exc_type, exc_val, exc_tb)
        # revert backend
        if not self._interactive:
            plt.switch_backend(self._old_backend)

    def _check_context(self):
        if self._matplotlib_style_context is None:
            raise ValueError("Cannot perform operation outside GraphManager context")

    def get_plot_style(self):
        linewidth_line = 2.0
        linewidth_norm = 1.5
        linewidth_small = 0.8

        return {
            'font.size': 12,

            # line properties
            'axes.labelsize': 'x-large',
            'lines.linewidth': linewidth_line,
            'axes.linewidth': linewidth_norm,
            'xtick.major.width': linewidth_norm,
            'xtick.minor.width': linewidth_small,
            'ytick.major.width': linewidth_norm,
            'ytick.minor.width': linewidth_small,
            'grid.linewidth': linewidth_norm,
            'grid.linestyle': '--',

            # other properties
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'figure.subplot.bottom': 0.08,
            'figure.subplot.top': 0.98,

            # figure size
            'figure.dpi': self._WINDOW_DPI,
            'figure.figsize': (16, 8)
        }

    def build_figure(self, name=None, width=None, height=None):
        """

        :param name: figure name
        :param width: figure width in inches
        :param height: figure height in inches
        :return:
        """
        default_width, default_height = matplotlib.rcParams['figure.figsize']

        def eval_size(val, default_val):
            if isinstance(val, str):
                return eval(val, {'width': default_width, 'height': default_height})
            elif val is None:
                return default_val
            else:
                return float(val)

        width = eval_size(width, default_width)
        height = eval_size(height, default_height)

        fig = plt.figure(name, (width, height))
        if name is not None:
            self._figures[fig.number] = name
        return fig

    def save_figure(self, fig_name=None, figure=None, dpi=None,
                    add_count=True, file_prefix='fig_', file_ext='.png'):
        self._check_context()

        if figure is None:
            figure = plt.gcf()
        if dpi is None:
            dpi = self._RES_IMAGE_DPI
        if fig_name is None and figure.number in self._figures:
            fig_name = self._figures[figure.number]
        if fig_name is None:
            raise ValueError("Cannot determine figure name automatically. Please specify it explicitly.")

        fig_name = fig_name + file_ext
        if add_count:
            fig_name = '{}_'.format(self._figure_count) + fig_name
        if file_prefix:
            fig_name = file_prefix + fig_name
        fig_path = join(self._result_dir, fig_name)

        # save figure
        figure.savefig(fig_path, dpi=dpi)

        self._figure_count += 1

    def show(self):
        if self._interactive:
            plt.show()


def get_demo_plot_manager(start_figure_num=1):
    interactive = True if 'INTERACTIVE_DEMO' in os.environ else False
    return PlotManager(result_dir=get_image_dir(), interactive=interactive, start_figure_num=start_figure_num)


def restore_spectrum(ax, ay, alpha, fs=2.0, nperseg=128, nfft=None):
    """
    Restore spectrum using acceleration data from sensors
    """
    transition_range = int(len(ax) * 0.1)
    integration_alpha = 0.01
    dt = 1 / fs

    x, y, angle = calculate_particle_trajectory(
        sensor_ax=ax, sensor_ay=ay, sensor_alpha=alpha,
        alpha=integration_alpha,
        spacing=dt
    )

    signal = x[transition_range:] + 1j * y[transition_range:]
    f, s = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        nfft=nfft,
        scaling='spectrum',
        return_onesided=False,
        axis=0,
        detrend='constant'
    )
    f = np.fft.fftshift(f, axes=0)
    s = np.fft.fftshift(s, axes=0)
    # TODO: fix hack
    f *= fs
    s *= 2 / fs

    omega = f * 2 * np.pi
    return omega, s, x, y, angle
