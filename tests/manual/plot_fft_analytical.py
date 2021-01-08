import numpy as np
import matplotlib.pyplot as plt
from typing import List
from kronig_penney.fft_helpers import FourierTransformer

V0 = -10
a = 1
b = 10
kwargs = {'V0': V0, 'a': a, 'b': b}


def analytical_ft(k_range):
    return (k_range != 0).astype(np.float) * V0 / (np.pi * k_range) * np.sin(np.pi * k_range * a) + \
            (k_range == 0).astype(np.float) * V0 * a


def pot_direct(x):
    return V0 * np.ones_like(x) * ((x % (b/2) < a/2).astype(np.float) + (b/2 - a/2 < x % (b/2)).astype(np.float))


if __name__ == '__main__':
    tr = FourierTransformer()
    fig, axes = plt.subplots(4, 1, figsize=(9, 16), dpi=300)    # type: plt.Figure, List[plt.Axes]

    tr.\
        add_grid(b/2, 10**5).\
        add_potential(pot_direct)

    freq = tr.freq_grid
    coeffs = np.real(tr.freq_coeffs)
    anal = analytical_ft(tr.freq_grid)
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
    axes[-1].plot(tr.x_grid, pot_direct(tr.x_grid))
    axes[-1].grid()

    fig.show()
    plt.close(fig)
