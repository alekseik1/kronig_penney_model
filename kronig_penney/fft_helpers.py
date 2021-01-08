import numpy as np


class FourierTransformer:
    def add_grid(self, period: float, n_dots: int) -> 'FourierTransformer':
        """
        Creates direct grid

        :param period:
        :param n_dots:
        :return:
        """
        self.x_grid = np.linspace(0, period, n_dots)
        delta = self.x_grid[1] - self.x_grid[0]
        assert len(self.x_grid) > 1
        self._delta = delta
        unsorted_grid = np.fft.rfftfreq(self.x_grid.size, d=delta)
        # self._freq_order = np.argsort(unsorted_grid)
        self._freq_order = np.arange(unsorted_grid.size)
        self.freq_grid = unsorted_grid#[self._freq_order]
        return self

    def add_potential(self, potential_func, **kwargs) -> 'FourierTransformer':
        self.potential_function_direct = lambda x: potential_func(x, **kwargs)
        return self

    @property
    def freq_coeffs(self) -> np.ndarray:
        return np.fft.rfft(self.potential_function_direct(self.x_grid)) * self._delta
