import numpy as np
import pytest
from kronig_penney.fft_helpers import FourierTransformer


@pytest.fixture()
def ft():
    yield FourierTransformer()


@pytest.fixture()
def square_well_potential():
    def pot_direct(x, V0: float, a: float, b: float):
        return V0 * np.ones_like(x) * ((x % (b/2) < a/2).astype(np.float) + (b/2 - a/2 < x % (b/2)).astype(np.float))
    yield pot_direct


@pytest.fixture()
def analytical_ft_func():
    def analytical_ft(k_range, V0: float, a: float, b: float):
        tmp = V0 / (np.pi * k_range) * np.sin(np.pi * k_range * a)
        np.nan_to_num(tmp, copy=False, nan=V0*a)
        return tmp
    yield analytical_ft

