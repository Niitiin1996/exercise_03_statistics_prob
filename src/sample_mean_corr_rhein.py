"""Implement sample analysis methods."""
from datetime import datetime
from functools import reduce
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas


def my_mean(data_sample) -> float:
    """Implement a function to find the mean of an input List."""
    return sum(data_sample) / len(data_sample)


def my_std(data_sample) -> float:
    """Implement a function to find the standard deviation of a sample in a List."""
    mean = my_mean(data_sample)
    sigma_squared = reduce(
        lambda x, y: x + y, (map(lambda z: (z - mean) ** 2.0, data_sample))
    )
    sigma_squared = sigma_squared / (len(data_sample) - 1)
    return sqrt(sigma_squared)


def auto_corr(x: np.ndarray) -> np.ndarray:
    """Impement a function to compute the autocorrelation of x."""
    ac = []
    for i in range(1, len(x)):
        acv = sum(x[:i] * x[-i:])
        ac.append(acv)
    ac.append(sum(x * x))
    for i in range(1, len(x)):
        acv = sum(x[i:] * x[:-i])
        ac.append(acv)
    return np.stack(ac)


if __name__ == "__main__":
    rhein = pandas.read_csv("./data/pegel.tab", sep=" 	")
    levels = np.array([int(pegel.split(" ")[0]) for pegel in rhein["Pegel"]])

    timestamps = [ts[:-4] for ts in rhein["Zeit"]]
    datetime_list = []
    for ts in timestamps:
        ts_date, ts_time = ts.split(",")
        day, month, year = ts_date.split(".")
        hour, minute = ts_time.split(":")
        datetime_list.append(datetime(int(year), int(month), int(day)))

    before_2000 = [
        level
        for level, timepoint in zip(levels, datetime_list)
        if timepoint < datetime(2000, 1, 1)
    ]
    after_2000 = [
        level
        for level, timepoint in zip(levels, datetime_list)
        if timepoint > datetime(2000, 1, 1)
    ]

    print(f"pre 2000. Mean {my_mean(before_2000):2.2f}, std {my_std(before_2000):2.2f}")
    print(f"post 2000. Mean {my_mean(after_2000):2.2f}, std {my_std(after_2000):2.2f}")

    rhine_after_2000_down = np.array(after_2000)
    rhine_after_2000_down = (
        rhine_after_2000_down - np.mean(rhine_after_2000_down)
    ) / np.std(rhine_after_2000_down)
    corr = np.correlate(rhine_after_2000_down, rhine_after_2000_down, mode="full")
    plt.plot(corr / len(corr), label="rhine")
    random_uniform = np.random.randn(*rhine_after_2000_down.shape)
    random_uniform = (random_uniform - np.mean(random_uniform)) / np.std(random_uniform)
    rcorr = np.correlate(random_uniform, random_uniform, mode="full")
    plt.plot(rcorr / len(rcorr), label="random")
    plt.legend()
    plt.show()
