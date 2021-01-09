import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kronig_penney.fft_helpers import FourierTransformer
from kronig_penney.examples import square_well_potential as swp


def getPlaneWave(x, a, kvec_j):
    assert type(np.array([0])) == type(x)
    return 1/np.sqrt(a) * (np.cos(kvec_j * x) + 1j * np.sin(kvec_j * x))


def getPsiFunc(kvec, phi_0, a):
    x = np.linspace(-a/2, a/2, 1000, endpoint=False)
    psi = 1j * np.zeros(len(x))
    plane_waves = np.array([getPlaneWave(x, a, kvec_j) for kvec_j in kvec])
    for i in range(len(x)):
        psi[i] = np.dot(phi_0, plane_waves[:, i])
    return x, psi


if __name__ == '__main__':
    kwargs = {'V0': -1., 'a': 2., 'b': 50.}
    for n_dots in [10, 15, 20, 40, 100, 500]:
        tr = FourierTransformer()
        tr.add_grid(kwargs['b'], n_dots).add_potential(swp, **kwargs)

        # fig, axes = plt.subplots(3, 1)
        energies, coeffs = np.linalg.eig(tr.get_hamilton_matrix())
        plt.scatter(range(len(energies)), energies)
        plt.show()
        continue
        coeffs = coeffs
        elow = np.sort(energies, axis=None)[:3]
        print(elow)
        psi_coeffs = coeffs[:, np.argmin(energies)]
        x, psi = getPsiFunc(tr.full_freq_grid * 2 * np.pi, psi_coeffs, kwargs['b'])
        plt.plot(x, psi**2)
        plt.show()
