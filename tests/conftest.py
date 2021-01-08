import pytest
from kronig_penney.fft_helpers import FourierTransformer
from kronig_penney.examples import square_well_potential as swp, square_well_analytical_ft as aft


@pytest.fixture()
def ft():
    yield FourierTransformer()


@pytest.fixture()
def square_well_potential():
    yield swp


@pytest.fixture()
def analytical_ft_func():
    yield aft

