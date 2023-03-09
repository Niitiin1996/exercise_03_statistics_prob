"""Test the autocorrelation function."""

import numpy as np

from src.sample_mean_corr_rhein import auto_corr


def test_autocorr() -> None:
    """Test the user implementation src/auto_corr."""
    x = np.random.uniform(0, 1, size=100)

    my_corr = auto_corr(x)
    np_corr = np.correlate(x, x, mode="full")
    assert np.allclose(my_corr, np_corr)
