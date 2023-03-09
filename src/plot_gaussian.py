"""Look at a 1d gaussian pdf and its cdf."""

from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def gaussian_pdf(x: np.ndarray, mu=0.0, sigma=1.0) -> np.ndarray:
    """Return a Gaussian probability density function.

    Args:
        x (np.ndarray): The input array of shape [points].
        mu (float, optional): The mean. Defaults to 0.0.
        sigma (float, optional): The standard deviation. Defaults to 1.0.

    Returns:
        np.ndarray: The gaussian pdf with shape [points].
    """
    return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)


def forward_euler(x: np.ndarray, fun: Callable, int_0: float = 0.0) -> np.ndarray:
    """Numerically integrate a function using eulers method.

    https://en.wikipedia.org/wiki/Euler_method


    Args:
        x (np.ndarray): The sampling points of the function we want
            to integrate. Shape [points].
        fun (Callable): The function to integrate.
        int_0: The initial condition.

    Returns:
        np.ndarray: Approximation of the integral with shape [points].
    """
    ints = [int_0]
    dx = (max(x) - min(x)) / len(x)
    for pos_x in x:
        ints.append(ints[-1] + dx * fun(pos_x))
    return np.stack(ints[1:])


if __name__ == "__main__":
    params = (
        (0.0, np.sqrt(0.2)),
        (0.0, np.sqrt(1.0)),
        (0.0, np.sqrt(5.0)),
        (-2.0, np.sqrt(0.5)),
    )
    for param in params:
        x = np.linspace(-5.0, 5.0, 500)
        plt.plot(
            x,
            gaussian_pdf(x, param[0], param[1]),
            label=f"mu = {param[0]}, sigma**2 = {param[1]**2:2.1f}",
        )
        plt.title("Gaussian Probability Density Function")
        plt.xlabel("x")
        plt.ylabel("pdf(x)")
    plt.legend()
    plt.show()

    for param in params:
        gaussian_pdf = partial(gaussian_pdf, mu=param[0], sigma=param[1])
        X = forward_euler(x, gaussian_pdf)
        plt.plot(x, X, label=f"mu = {param[0]}, sigma**2 = {param[1]**2:2.1f}")
        plt.title("Gaussian Cumulative distribution function")
        plt.xlabel("x")
        plt.ylabel("cdf(x)")
    plt.legend()
    plt.show()
