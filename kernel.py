import numpy as np

def inv_sq(dist_sq, eps=0.0001): # 1 / dist_sq
    return 1./(dist_sq+eps)

def constant(dist_sq): # 1
    return np.ones(dist_sq.shape)

def triangle(dist_sq): # 1 - dist/max(dist)
    dist = np.sqrt(dist_sq)
    normalized = dist/np.max(dist)
    return 1 - normalized

def tricube(dist_sq): # (1 - (dist/max(dist))^3)^3
    dist = np.sqrt(dist_sq)
    normalized = dist_sq/np.max(dist_sq)
    cubed = 1 - normalized*normalized*normalized

    return cubed*cubed*cubed

kernels = {
    'inv_sq': inv_sq,
    'constant': constant,
    'triangle': triangle,
    'tricube': tricube,
}

