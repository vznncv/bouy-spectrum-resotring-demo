import logging
import numpy as np

logger = logging.getLogger(__name__)


def _pierson_moskowitz_fun(omega, omega_m, beta):
    omega_ri = omega_m / omega
    s = (4 * beta) * (omega_ri ** 5 / np.abs(omega_m)) * np.exp(-beta * omega_ri ** 4)
    s[(omega_m > 0) & (omega <= 0)] = 0
    s[(omega_m < 0) & (omega >= 0)] = 0
    return s


def _pierson_moskowitz_int_from_zero_to_omega_lim_fun(omega_lim, omega_m, beta):
    omega_ri = omega_m / omega_lim
    return np.exp(-beta * omega_ri ** 4)


def pierson_moskowitz_s_fun(omega, omega_m, var, beta=1.25, omega_lim=None):
    """
    Mix of the Pierson-Moskowitz power density spectrum functions.

    :param omega: frequency
    :param omega_m: frequency peaks of each Pierson-Moskowitz spectrum function
    :param var: wave dispersion of each Pierson-Moskowitz spectrum function
    :param beta: power spectral density beta parameters
    :param omega_lim: as the ``Pierson-Moskowitz spectrum`` have a "heavy" tail, if ``omega_lim`` is
                      not ``None``, then all frequencies that are greater than ``omega_lim` will be suppressed
    :return: values of the power spectral density for corresponding ``omega``
    """
    omega = np.asfarray(omega)

    omega_m = np.asfarray(omega_m)
    beta = np.asfarray(beta)
    var = np.asfarray(var)
    omega_m, beta, var = np.broadcast_arrays(omega_m, beta, var)

    omega = omega[..., np.newaxis]

    s = _pierson_moskowitz_fun(omega, omega_m, beta)

    if omega_lim is not None:
        omega_lim = np.abs(omega_lim)
        omega_lim = np.broadcast_to(omega_lim, omega_m.shape)

        if np.any(omega_lim < np.abs(omega_m) * 2):
            logger.warning("omega_lim ({}) is small and close to omega_m ({})"
                           .format(omega_lim, np.abs(omega_m)))

        s[(omega < -omega_lim) | (omega > omega_lim)] = 0
        # normalize psd
        s /= _pierson_moskowitz_int_from_zero_to_omega_lim_fun(omega_lim, omega_m, beta)

    s *= var

    return s.sum(axis=-1)


def build_wave_spectrum_fun(omega_m, var, beta=1.25, omega_lim=None):
    """
    Factory of the mix of the Pierson-Moskowitz spectrum functions.
    :param omega_m: frequency peaks
    :param var: wave dispersion
    :param beta: power spectral density beta parameters
    :param omega_lim: as the ``Pierson-Moskowitz spectrum`` have a "heavy" tail, if ``omega_lim`` is
                  not ``None``, then all frequencies that are greater than ``omega_lim` will be suppressed
    :return: power spectral density function
    """

    def spectrum_fun(omega):
        """
        Power spectral density function

        :param omega: frequency
        :return: values of the power spectral density for corresponding ``omega``
        """
        return pierson_moskowitz_s_fun(omega, omega_m, var, beta, omega_lim)

    return spectrum_fun

# if __name__ == '__main__':
#     import sympy
#
#     omega, omega_m, beta = sympy.symbols('omega, omega_m, beta', real=True, positive=True)
#
#     s = (4 * beta) * (omega_m ** 4 / omega ** 5) * sympy.exp(-beta * (omega_m / omega) ** 4)
#     omega_lim = sympy.symbols('omega_lim', real=True, positive=True)
#
#     s_sum = sympy.integrate(s, (omega, 0, omega_lim))
#     print(s_sum)
