import numpy as np
import scipy.stats

def rectangularize(m):
    
    m = np.expand_dims(m, axis=0)
    return np.vstack([np.cos(m), np.sin(m)])

def circ_mean(x, axis):

    rect_x = rectangularize(x)
    if axis >= 0: axis += 1
    mean = np.nanmean(rect_x, axis=axis)
    return np.arctan2(mean[1, ...],mean[0, ...])

def circ_diff(x, y):

    return regularize(x - y)

def regularize(x):
    
    return np.mod(x + np.pi, 2*np.pi) - np.pi

def mean_resultant_vector_length(x, axis):
    
    rect_x = rectangularize(x)
    if axis >= 0: axis += 1
    mean = np.nanmean(rect_x, axis=axis)
    return np.linalg.norm(mean, axis=0)

def dstat(phase_wavelet_recalled_diff, phase_wavelet_not_recalled_diff, axis):

    cstd_recalled = scipy.stats.circstd(phase_wavelet_recalled_diff, axis=axis)
    cstd_not_recalled = scipy.stats.circstd(phase_wavelet_not_recalled_diff, axis=axis)
    PLVs_recalled = np.power(np.e, np.divide(np.power(cstd_recalled, 2), -2))
    PLVs_not_recalled = np.power(np.e, np.divide(np.power(cstd_not_recalled, 2), -2))

    dstat = PLVs_recalled - PLVs_not_recalled
    return dstat

def ppc(phase):
    
    n = phase.shape[0]
    sin_phase, cos_phase = np.sin(phase), np.cos(phase)
    distance_sum = []
    for j in np.arange(n-1):
        d = np.sum(cos_phase[j:(j+1), ...]*cos_phase[(j+1):, ...] + sin_phase[j:(j+1), ...]*sin_phase[(j+1):, ...], axis=0)
        distance_sum.append(d)

    distance_sum = np.sum(distance_sum, axis=0)
    
    return (2/(n*(n-1)))*distance_sum