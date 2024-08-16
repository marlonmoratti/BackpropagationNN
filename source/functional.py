from scipy.special import expit
import numpy as np

def sigmoid(x, derivative=False, sigmoid_value=False, *args):
    if derivative:
        if sigmoid_value:
            return x * (1 - x)
        
        sig = expit(x)
        return sig * (1 - sig)

    return expit(x)

def ReLU(x, derivative=False, *args):
    if derivative:
        return np.where(x >= 0, 1, 0)
    return np.maximum(0, x)

def ce_loss(input, target, derivative=False, eps=1e-15, *args):
    target = np.eye(input.shape[1])[target.flatten()]

    if derivative:
        return (input - target) / ((input * (1 - input)) + eps)

    return -np.mean(np.sum(target * np.log(input + eps), axis=1))
