"""Explore gaussian mixtures on medical data."""
import matplotlib.pyplot as plt
import numpy as np
import pandas

from mixture_concepts import fit_gmm, get_classification, twod_gaussian_pdf
from util import write_movie

if __name__ == "__main__":
    diabetes_data = pandas.pandas.read_csv("./data/diabetes.csv")

    def get_by_diagnosis(diagnosis: str) -> np.ndarray:
        """Return data points based on their diagnosis label."""
        patients = [
            row[:-1]
            for row in diabetes_data.to_numpy()
            if row[-1].lstrip() == diagnosis
        ]
        return np.stack(patients)

    normal_patients = get_by_diagnosis("Normal")
    overt = get_by_diagnosis("Overt_Diabetic")

    plt.title("glufast vs. glutest")
    plt.plot(normal_patients[:, 1], normal_patients[:, 2], ".")
    plt.plot(overt[:, 1], overt[:, 2], ".")
    plt.show()

    healthy = np.stack([normal_patients[:, 1], normal_patients[:, 2]], axis=-1)
    diabetes = np.stack([overt[:, 1], overt[:, 2]], axis=-1)
    points = np.concatenate([healthy, diabetes], axis=0).astype(np.float32)
    point_mean = np.mean(points, axis=0)
    point_std = np.std(points, axis=0)

    points = (points - point_mean) / point_std

    plt.plot(points[:, 0], points[:, 1], ".")
    plt.xlabel("glufast")
    plt.ylabel("glutest")
    plt.show()

    mu1 = np.array([3.0, 3])
    mu2 = np.array([-2.0, -2])

    sigma1 = np.array([[1, 0.0], [0.0, 1]])
    sigma2 = np.array([[1, 0.0], [0.0, 1]])

    rho = 0.5
    params = ((rho, mu1, sigma1), (rho, mu2, sigma2))

    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    xv, yv = np.meshgrid(x, y)
    samples1 = twod_gaussian_pdf(np.stack([xv, yv], -1), mu1, sigma1)
    samples2 = twod_gaussian_pdf(np.stack([xv, yv], -1), mu2, sigma2)

    step_list = None
    for _ in range(10):
        if step_list:
            step_list.append(fit_gmm(points, step_list[-1][0]))
        else:
            zig = get_classification(points, params).squeeze()
            labels = np.argmax(zig, 0)
            step_list = [[params, points, labels]]
            step_list.append(fit_gmm(points, params))

    write_movie(step_list, name="diabetes", xlabel="glutest", ylabel="glutest")
