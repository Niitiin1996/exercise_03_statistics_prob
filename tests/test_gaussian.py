"""Test the autocorrelation function."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "./src/")

from src.plot_gaussian import gaussian_pdf

testdata = [
    (
        np.linspace(0, 1, 10),
        np.array(
            [
                0.39894,
                0.39649,
                0.38921,
                0.37738,
                0.36142,
                0.34189,
                0.31945,
                0.29481,
                0.26874,
                0.24197,
            ]
        ),
    ),
    (np.linspace(0.5, 0.8, 4), np.array([0.35207, 0.33322, 0.31225, 0.28969])),
]


@pytest.mark.parametrize("x, result", testdata)
def test_gau_pdf(x, result) -> None:
    """Test the user implementation src/auto_corr."""
    res = gaussian_pdf(x, mu=0.0, sigma=1.0)
    res = np.round(res, 5)
    assert np.allclose(res, result)
