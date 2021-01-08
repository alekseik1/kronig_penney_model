import numpy as np
import hypothesis.strategies as st
import matplotlib.pyplot as plt
from hypothesis import given, assume
from kronig_penney.fft_helpers import FourierTransformer


@given(
    V0=st.floats(min_value=-100., max_value=-1.),
    a=st.floats(min_value=0.1, max_value=10.),
    b=st.floats(min_value=0.1, max_value=10.),
    n_dots=st.integers(min_value=10**4, max_value=10**5)
)
def test_square_well_values(
        V0: float, a: float, b: float, n_dots: int,
        ft: FourierTransformer, square_well_potential, analytical_ft_func):
    assume(V0 < 0)
    assume(a < b)
    kwargs = {'V0': V0, 'a': a, 'b': b}
    ft.\
        add_grid(b/2, n_dots).\
        add_potential(square_well_potential, **kwargs)
    expected = analytical_ft_func(ft.freq_grid, **kwargs)
    actual = np.real(ft.freq_coeffs)
    np.testing.assert_allclose(actual, expected, atol=abs(V0*0.005))
