import numpy as np
from examples import square_well_analytical_ft as aft


class FourierTransformer:
    _freq_coeffs_cache: np.ndarray = None

    def add_grid(self, period: float, n_dots: int) -> 'FourierTransformer':
        """
        Creates direct grid

        :param period:
        :param n_dots:
        :return:
        """
        self.x_grid = np.linspace(0, period, n_dots, endpoint=False)
        delta = self.x_grid[1] - self.x_grid[0]
        assert len(self.x_grid) > 1
        self._delta = delta
        self.freq_grid = np.fft.rfftfreq(self.x_grid.size, d=delta)
        self._freq_delta = self.freq_grid[1] - self.freq_grid[0]
        self.freq_grid_negative = -self.freq_grid[:0:-1]
        self.full_freq_grid = np.concatenate((self.freq_grid_negative, self.freq_grid))
        return self

    def add_potential(self, potential_func, **kwargs) -> 'FourierTransformer':
        self.kwargs = kwargs
        self.potential_function_direct = lambda x: potential_func(x, **kwargs)
        return self

    @property
    def freq_coeffs(self) -> np.ndarray:
        if self._freq_coeffs_cache is None:
            self._freq_coeffs_cache = np.fft.rfft(self.potential_function_direct(self.x_grid)) * self._delta
        return self._freq_coeffs_cache

    def convert_back(self, values: np.ndarray) -> np.ndarray:
        return np.fft.irfft(values / self._delta, self.x_grid.size)

    def get_potential_matrix(self) -> np.ndarray:
        result = np.zeros((self.full_freq_grid.size, self.full_freq_grid.size))
        # Need to double k-matrix for building V-matrix
        extended_grid = np.arange(2*min(self.full_freq_grid), 2*max(self.full_freq_grid) + self._freq_delta, self._freq_delta)
        an = aft(extended_grid, **self.kwargs)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i][j] = an[len(extended_grid)//2 + i - j]
        return result

    def get_kinetic_matrix(self) -> np.ndarray:
        return np.diag([0.5 * (2 * np.pi * i)**2 for i in self.full_freq_grid])

    def get_hamilton_matrix(self) -> np.ndarray:
        return self.get_kinetic_matrix() + self.get_potential_matrix()
