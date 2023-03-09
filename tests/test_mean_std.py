"""Test the python function from src."""

import sys

import numpy as np

sys.path.insert(0, "./src/")

from src.sample_mean_corr_rhein import my_mean, my_std


def test_mean() -> None:
    """See if the mean is correct."""
    rand_dat = np.random.uniform(0, 1, size=1000)
    assert np.isclose(np.mean(rand_dat), my_mean(rand_dat))


def test_std() -> None:
    """See if the std is correct."""
    rand_dat = np.random.uniform(0, 1, size=1000)
    assert np.isclose(np.std(rand_dat, ddof=1), my_std(rand_dat))
