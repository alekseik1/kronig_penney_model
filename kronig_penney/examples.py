import numpy as np


def square_well_potential(x: np.ndarray, V0: float, a: float, b: float) -> np.ndarray:
    """
    Square well potential

    :param x: coordinate grid
    :param V0: depth of a well
    :param a: width of a well
    :param b: width of an elemental cell (i.e. one period)
    :return: potential values corresponding to x-grid
    """
    return V0 * np.ones_like(x) * (
                (x % (b / 2) < a / 2).astype(np.float) + (b / 2 - a / 2 < x % (b / 2)).astype(np.float))


def square_well_analytical_ft(k_range: np.ndarray, V0: float, a: float, b: float = 0.) -> np.ndarray:
    """
    Analytical values for F[f](k) where f is a square well potential function and k is
    a frequency

    :param k_range: range of frequencies (neither spatial nor angular, just frequency in 1/t)
    :param V0: depth of a well
    :param a: width of a well
    :param b: not used but accepted for inner compatibilities
    :return: array of amplitudes for corresponding frequencies
    """
    tmp = V0 / (np.pi * k_range) * np.sin(np.pi * k_range * a)
    np.nan_to_num(tmp, copy=False, nan=V0*a)
    return tmp
