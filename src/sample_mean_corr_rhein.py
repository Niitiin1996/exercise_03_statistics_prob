"""Implement sample analysis methods."""
from datetime import datetime
from functools import reduce
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas


def my_mean(data_sample) -> float:
    """Implement a function to find the mean of an input List."""
    mean = 0.
    for i in range(len(data_sample)):
        mean += data_sample[i]
    mean = mean / len(data_sample)
    return mean


def my_std(data_sample) -> float:
    """Implement a function to find the standard deviation of a sample in a List."""
    # TODO: Implement me.

    mean = my_mean(data_sample)
    std = 0.
    for i in range(len(data_sample)):
        std +=(data_sample[i]-mean)**2
    std = std/(len(data_sample)-1)
    std = sqrt(std)
    return std


def auto_corr(x: np.ndarray) -> np.ndarray:
    """Impement a function to compute the autocorrelation of x.
    
    Args:
        x (np.ndarray): Normalized input signal array of shape (signal_length,).

    Returns:
        np.ndarray: Autocorrelation of input signal of shape (signal_length*2 - 1,).
    """
    N=x.shape[0]
    c=np.zeros(2*N-1)
    mean = my_mean(x)
    std = my_std(x)

    #x = (x-mean)/std 
    for k in range(-N+1,N):
        for t in range (N-abs(k)):
            c[k+N-1]+=x[t]*x[t+abs(k)]
    # TODO: Implement me.
    return c


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


    # TODO: compute the mean and standard deviation before and after 2000.

    mean = my_mean(before_2000)
    std = my_std(before_2000)

    # TODO: Compare the autocorrelation functions of the rhine data and of a random signal.
    autocor = auto_corr(rhein)

    plt.plot(autocor)
    plt.show()
