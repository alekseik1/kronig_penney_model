import numpy as np


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
