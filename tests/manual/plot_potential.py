import numpy as np
import matplotlib.pyplot as plt
from kronig_penney.examples import square_well_potential as swp

if __name__ == '__main__':
    kwargs = {'V0': -1000., 'a': 1., 'b': 15.}
    x_grid = np.linspace(0, kwargs['b'], 10**3)
    plt.plot(x_grid, swp(x_grid, **kwargs))
    plt.show()
