import numpy as np

from hamcrest import assert_that, less_than_or_equal_to


def compensate_lag(sig_1, sig_2, max_lag=10):
    lags = np.arange(-max_lag, max_lag + 1)
    corr_max_val = -np.inf
    sig_1_comp = None
    sig_2_comp = None

    for lag in lags:
        if lag < 0:
            sig_1_lagged = sig_1[-lag:]
            sig_2_lagged = sig_2[:lag]
        elif lag == 0:
            sig_1_lagged = sig_1
            sig_2_lagged = sig_2
        else:
            sig_1_lagged = sig_1[:-lag]
            sig_2_lagged = sig_2[lag:]
        corr_val = (sig_1_lagged * sig_2_lagged).sum()
        if corr_val > corr_max_val:
            corr_max_val = corr_val
            sig_1_comp = sig_1_lagged
            sig_2_comp = sig_2_lagged

    return sig_1_comp, sig_2_comp


def assert_signals_almost_equal(actual, desired, deviation_std=0.02):
    deviation = actual - desired
    obtained_deviation_std = deviation.std()
    assert_that(obtained_deviation_std, less_than_or_equal_to(deviation_std))


def remove_transition_interval(*signals, transition_interval):
    return (signal[transition_interval:] for signal in signals)
