import numpy as np

def inv_sq(x, eps=0.0001):
    return 1./(x+eps)
def constant(x):
    return np.ones(x.shape)

kernels = {
    'inv_sq': inv_sq,
    'constant': constant,
}

