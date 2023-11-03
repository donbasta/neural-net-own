import numpy as np


def sigmoid(x: np.ndarray):
    x = np.clip(x, -1000, 1000)
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray):
    return np.maximum(0, x)


def linear(x: np.ndarray):
    return x
