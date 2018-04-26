import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class LineAnimation:
    def __init__(self, fig_name, x=None, y=None, **kwargs):
        y = np.asarray(y)
        if y.ndim != 2:
            raise ValueError("y should be 2 dimensional array")
        if x is None:
            x = np.arange(y.shape[-1])

        self.x = x
        self.y = y
        self.fig = plt.figure(fig_name)

        # get animation options
        interval = kwargs.pop('interval', 200)
        repeat_delay = kwargs.pop('repeat_delay', None)
        repeat = kwargs.pop('repeat', True)
        blit = kwargs.pop('blit', False)
        # save plot options
        self._plot_options = kwargs

        self.animation_function = FuncAnimation(
            fig=self.fig,
            func=self._update_plot,
            frames=y,
            init_func=self._init_plot,
            interval=interval,
            repeat_delay=repeat_delay,
            repeat=repeat,
            blit=blit
        )

        self._init_plot()

    def _init_plot(self):
        ax = self.fig.gca()
        ax.lines.clear()
        ax.set_prop_cycle(None)
        self._line = ax.plot(self.x, self.y[0], **self._plot_options)[0]
        return self._line,

    def _update_plot(self, frame):
        self._line.set_xdata(self.x)
        self._line.set_ydata(frame)
        return self._line,

    @property
    def axes(self):
        return self.fig.gca()
