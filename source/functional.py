from scipy.special import expit
import numpy as np

def sigmoid(x):
    return expit(x)

def mse(input, target):
    error = (input - target).flatten()
    return np.dot(error, error) / (2 * len(error))
