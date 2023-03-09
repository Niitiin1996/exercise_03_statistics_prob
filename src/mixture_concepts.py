"""This module explores the core ideas of gaussian mixture modelling."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from util import write_movie


def twod_gaussian_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return a two dimensional gaussian distribution.

    Args:
        x (np.ndarray): A grid of shape (grid_height, grid_width, 2).
        mu (np.ndarray, optional): Mean parameter of shape (2,).
        sigma (np.ndarray, optional): Covariance matrix of shape (2, 2).

    Returns:
        np.ndarray: The two dimensional gaussian distribution.
    """
    # TODO: Implement me.
    return np.zeros_like(x)



def get_classification(points: np.ndarray, params_list: List) -> np.ndarray:
    """Classify each point based on the Gaussian distributions.

    Args:
        points (np.ndarray): Point array of shape [point_no, 2].
        params_list (List): A list of lists with global probability,
            mean values and standard deviations.

    Returns:
        np.ndarray: The probability with which a point was drawn for every
            gaussian in the mix.
    """
    # TODO: OPTIONALLY, Implement me.
    return np.zeros_like(x)



def fit_gmm(points: np.ndarray, init_params_list: List) -> List:
    """Fit a GMM to the points using expectation maximization.

    Args:
        points (np.ndarray): Point array of shape [point_no, 2].
        init_params_list (List): A list of lists with global probability,
            mean values and standard deviations.

    Returns:
        List: A list with the new parameters the points and their labels.
    """
    # get labels
    point_count = len(points)
    zig = get_classification(points, init_params_list).squeeze()
    labels = np.argmax(zig, 0)

    # update params
    new_params_list = []
    # TODO: OPTIONALLY, Implement me.

    return [new_params_list, points, labels]


if __name__ == "__main__":
    np.random.seed(42)
    dist1 = np.random.normal(loc=(2, 2), scale=(1.0, 1.0), size=(100, 2))
    dist2 = np.random.normal(loc=(-2, -2), scale=(1.0, 1.0), size=(100, 2))

    points = np.concatenate([dist1, dist2], axis=0)

    plt.plot(dist1[:, 0], dist1[:, 1], ".")
    plt.plot(dist2[:, 0], dist2[:, 1], ".")
    plt.show()

    mu1 = np.array([-1.5, 2])
    mu2 = np.array([1.5, -2])

    sigma1 = np.array([[1, 0.0], [0.0, 1]])
    sigma2 = np.array([[1, 0.0], [0.0, 1]])

    rho = 0.5
    params = [(rho, mu1, sigma1), (rho, mu2, sigma2)]

    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    xv, yv = np.meshgrid(x, y)
    samples1 = twod_gaussian_pdf(np.stack([xv, yv], -1), mu1, sigma1)
    samples2 = twod_gaussian_pdf(np.stack([xv, yv], -1), mu2, sigma2)

    plt.imshow(samples1.squeeze())
    plt.colorbar()
    plt.show()
    plt.imshow(samples2.squeeze())
    plt.colorbar()
    plt.show()

    step_list = None
    for _ in range(20):
        if step_list:
            step_list.append(fit_gmm(points, step_list[-1][0]))
        else:
            zig = get_classification(points, params).squeeze()
            labels = np.argmax(zig, 0)
            step_list = [[params, points, labels]]
            step_list.append(fit_gmm(points, params))

    write_movie(step_list)
