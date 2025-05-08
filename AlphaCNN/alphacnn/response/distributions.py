import numpy as np


def normal(x, scale, std):
    return scale * np.exp(-0.5 * ((x / std) ** 2))


def truncated_normal(x, scale, std, xcut):
    y = scale * np.exp(-0.5 * ((x / std) ** 2))
    y[x >= xcut] = 0.
    return y


def loss_normal(theta, x, y):
    scale, std = theta
    fit_y = normal(x, scale=scale, std=std)
    return np.mean((fit_y - y) ** 2)


def loss_truncated_normal(theta, x, y):
    scale, std, xcut = theta
    fit_y = truncated_normal(x, scale=scale, std=std, xcut=xcut)
    return np.mean((fit_y - y) ** 2)
