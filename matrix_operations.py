import numpy as np
from copy import deepcopy

def any_finite(x):
    
    return np.any(np.isfinite(x))

def all_finite(x):
    
    return np.all(np.isfinite(x))

def count_finite(x):
    
    return np.sum(np.isfinite(x))/np.prod(x.shape)

def finitize(x):
    
    return x[np.isfinite(x)]

def symmetrize(mx):
    
    return np.nanmean([mx, np.swapaxes(mx, 0, 1)], axis=0)