import matplotlib.pyplot as plt
from kronig_penney.fft_helpers import FourierTransformer
from kronig_penney.examples import square_well_potential as swp


if __name__ == '__main__':
    kwargs = {'V0': -10., 'a': 1., 'b': 5.}
    tr = FourierTransformer()
    tr.add_grid(kwargs['b']/2, 10**5).add_potential(swp, **kwargs)
    fig, ax = plt.subplots(1, 1)
    ax.plot(tr.x_grid, tr.convert_back(tr.freq_coeffs), label='numeric')
    ax.plot(tr.x_grid, swp(tr.x_grid, **kwargs), label='analytical')
    ax.legend()
    ax.grid()
    fig.show()
