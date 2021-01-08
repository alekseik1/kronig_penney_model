import numpy as np
import matplotlib.pyplot as plt
from typing import List
from kronig_penney.fft_helpers import FourierTransformer
from kronig_penney.examples import square_well_potential as pot_direct, square_well_analytical_ft as analytical_ft

V0 = -10
a = 1
b = 10
kwargs = {'V0': V0, 'a': a, 'b': b}




if __name__ == '__main__':
    tr = FourierTransformer()
    fig, axes = plt.subplots(4, 1, figsize=(9, 16), dpi=300)    # type: plt.Figure, List[plt.Axes]

    tr.\
        add_grid(b/2, 10**5).\
        add_potential(pot_direct, **kwargs)

    freq = tr.freq_grid
    coeffs = np.real(tr.freq_coeffs)
    anal = analytical_ft(tr.freq_grid, **kwargs)
    # Fourier
    for ax in axes[:2]:
        ax.plot(freq, coeffs, label='numeric', alpha=0.3)
        ax.scatter(freq, anal, label='analytical', alpha=0.3, s=2)
        ax.legend()
        ax.grid()
    #axes[0].set_xlim(0.8*np.max(tr.freq_grid), np.max(tr.freq_grid))
    #axes[0].set_ylim(-0.5, 0.5)
    axes[0].set_xlim(0, 20.)
    axes[0].set_ylim(-10, 5.)
    axes[0].set_title('Near the border')
    axes[1].set_title('Fourier images')
    # Difference
    #axes[2].scatter(freq, np.abs((coeffs - anal) / anal), label='relative difference', alpha=0.3, s=2)
    axes[2].scatter(freq, np.abs((coeffs - anal) / anal), label='relative difference')
    axes[2].set_xlim(0, 100)
    axes[2].legend()
    axes[2].grid()
    # Direct space
    axes[-1].set_title('Potential in direct space')
    axes[-1].plot(tr.x_grid, pot_direct(tr.x_grid, **kwargs))
    axes[-1].grid()

    fig.show()
    plt.close(fig)
