import matplotlib.pyplot as plt
import numpy as np
from hamcrest import assert_that, greater_than
from unittest import TestCase

from spectrum_processing_1d.utils import calculate_acceleration, calculate_trajectory


class CalculateAcceleration(TestCase):
    def test_sin(self):
        t = np.linspace(0, 100, 1000)
        dt = (t[-1] - t[0]) / len(t)

        x = np.sin(t)
        expected_ax = np.sin(t + np.pi)

        ax = calculate_acceleration(x, spacing=dt)

        cmp_res = np.isclose(ax, expected_ax, atol=0.005)
        assert_that(cmp_res.sum() / cmp_res.size, greater_than(0.99))


class CalculateTrajectory(TestCase):
    def test_sin(self):
        t = np.linspace(0, 10000, 200000)
        dt = (t[-1] - t[0]) / len(t)

        ax = np.sin(t)
        expected_x = np.sin(t + np.pi)

        x = calculate_trajectory(ax=ax, alpha=0.005, spacing=dt)

        trans_period = 2000
        t = t[trans_period:]
        x = x[trans_period:]
        expected_x = expected_x[trans_period:]

        # slice_to_show = slice(-1000, None, None)
        # plt.plot(t[slice_to_show], expected_x[slice_to_show])
        # plt.plot(t[slice_to_show], x[slice_to_show])
        # plt.legend('expected', 'obtained')
        # plt.show()

    def test_noisy_sin(self):
        t = np.linspace(0, 10000, 200000)
        dt = (t[-1] - t[0]) / len(t)

        ax = np.sin(t) + np.random.randn(*t.shape) * 0.01
        expected_x = np.sin(t + np.pi)

        x = calculate_trajectory(ax=ax, alpha=0.005, spacing=dt)

        trans_period = 2000
        t = t[trans_period:]
        x = x[trans_period:]
        expected_x = expected_x[trans_period:]

        # slice_to_show = slice(-1000, None, None)
        # plt.plot(t[slice_to_show], expected_x[slice_to_show])
        # plt.plot(t[slice_to_show], x[slice_to_show])
        # plt.legend('expected', 'obtained')
        # plt.show()
