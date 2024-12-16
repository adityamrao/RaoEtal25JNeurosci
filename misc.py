# Basic
import numpy as np
import scipy
import scipy.stats
import os
from os.path import join
import itertools
import warnings
import sys
from copy import deepcopy
from matrix_operations import *

# Data Handling
import pickle
import os
import h5py

# Data Analysis
import pandas as pd
import xarray as xr


def ftag(dfrow):
    
    '''
    Returns a session label as a string that can be used for file names.
    
    Parameters:
        dfrow : pandas.Series
            Session label.
            
    Returns:
        str
            Session label as a string for file names.
    '''
    
    sub, exp, sess, loc, mon = dfrow[['sub', 'exp', 'sess', 'loc', 'mon']]
    if (int(loc) == 0) and (int(mon) == 0):
        return f'{sub}_{exp}_{sess}'
    else:
        return f'{sub}_{exp}_{sess}_{loc}_{mon}'

def npl(fname):
    
    '''
    Convenience function for numpy.load.
    '''
    
    return np.load(join('/scratch/amrao', fname), allow_pickle=True)
    
def load_mat(path):
    
    try:
        return scipy.io.loadmat(path)
    except:
        f = h5py.File(path, 'r') #can use .keys() and .get('{key}') methods to get data
        return f
    
def get_dfrow(dfrow):
    
    '''
    Converts a session label typed as a tuple, list, or numpy.array into a session label as a pandas.Series.
    
    Parameters:
        dfrow : tuple, list, numpy.array
            Session label in the form (sub, exp, sess, loc, mon) --- i.e., (subject, experiment, session, localization, montage).
            
    Returns:
        dfrow : pandas.Series
            Session label with 'sub', 'exp', 'sess', 'loc', and 'mon' keys.
    '''
    
    if len(dfrow) == 3:
        sub, exp, sess = dfrow
        return pd.Series({'sub': sub, 
                          'exp': exp, 
                          'sess': sess})
    else:
        sub, exp, sess, loc, mon = dfrow
        return pd.Series({'sub': sub, 
                          'exp': exp, 
                          'sess': sess, 
                          'loc': loc, 
                          'mon': mon})

def load_pickle(path):
    
    '''
    Convenience function for loading pickle files.
    '''
    
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, obj):
    
    '''
    Convenience function for saving pickle files.
    '''
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        
def iterative_avg(avg, update, index_tracker):

    if np.sum(index_tracker) == 0:
        avg = update
        update_mask = np.isfinite(update)
        index_tracker[update_mask] += 1
    else:
        avg_mask = np.isfinite(avg) #all non-nan values in cumulative average mx
        update_mask = np.isfinite(update) #all non-nan values in most recent subject's mx

        mask_bothf = avg_mask & update_mask #where both cumavg mx and recent subject's mx are non-nan
        mask_bothnf = (~avg_mask) & (~update_mask) #where both cumavg mx and recent subject's mx are nan

        avg[mask_bothf] = np.multiply(avg[mask_bothf], np.divide(index_tracker[mask_bothf], (index_tracker[mask_bothf]+1))) #subavgregsymx * n/(n+1)
        update[mask_bothf] = np.multiply(update[mask_bothf], np.divide(1, (index_tracker[mask_bothf]+1))) #regsymx * 1/(n+1)

        avg = np.vstack([avg[np.newaxis, ...], update[np.newaxis, ...]])
        avg = np.nansum(avg, axis = 0)
        avg[mask_bothnf] = np.nan #because np.nansum says sum of nans is 0 

        index_tracker[update_mask] += 1 #update index tracker
            
    return avg, index_tracker

def display_all():
    
    '''
    Display all rows and columns of a pandas.DataFrame in the cell of a Jupyter notebook.
    '''
    
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    np.set_printoptions(threshold=np.inf)
    
def print_header(text):
    print(f'---------{text}---------')

def print_whether_significant(p, alpha=0.05):
    
    if p < alpha: print(f'Statistically Significant (p < {alpha})')
    else: print(f'NOT Statistically Significant (p >= {alpha})')
    
def print_ttest_1samp(vals, header=None, alternative='two-sided', popmean=0):
    
    vals = finitize(vals)
    t, p = scipy.stats.ttest_1samp(vals, popmean=popmean, alternative=alternative)
    if header is not None: print_header(header)
    print_whether_significant(p)
    print(f't_{len(vals)-1} = {t:.3}, p = {p:.3}, Mean: {np.mean(vals):.3} ± {scipy.stats.sem(vals):.3}')
    return t, p
    
def print_ttest_ind(a, b, header=None, alternative='two-sided'):
    
    a = finitize(a)
    b = finitize(b)
    t, p = scipy.stats.ttest_ind(a, b, alternative=alternative)
    if header is not None: print_header(header)
    print_whether_significant(p)
    print(f't_{len(a)+len(b)-2} = {t:.3}, p = {p:.3}, Mean_A: {np.mean(a):.3} ± {scipy.stats.sem(a):.3}, Mean_B: {np.mean(b):.3} ± {scipy.stats.sem(b):.3}')
    return t, p

def print_ttest_rel(a, b, header=None, alternative='two-sided'):
    
    where_finite = np.isfinite(a) & np.isfinite(b)
    a = a[where_finite]
    b = b[where_finite]
    t, p = scipy.stats.ttest_rel(a, b, alternative=alternative)
    if header is not None: print_header(header)
    print_whether_significant(p)
    print(f't_{len(a)-1} = {t:.3}, p = {p:.3}, Mean_A: {np.mean(a):.3} ± {scipy.stats.sem(a):.3}, Mean_B: {np.mean(b):.3} ± {scipy.stats.sem(b):.3}, Mean_Diff: {np.mean(a-b):.3} ± {scipy.stats.sem(a-b):.3}')
    return t, p

def print_pearsonr(a, b, header=None):
    
    where_finite = np.isfinite(a) & np.isfinite(b)
    r = scipy.stats.pearsonr(a[where_finite], b[where_finite])[0]
    
    if header is not None: print_header(header)
    print(f'Pearson\'s r: {r:.3}')
    return r

def jzs_bayes_factor(t, N):
    
    '''
    Returns the JZS Bayes factor.
    
    Parameters:
        t : float
            t-statistic.
        N : int
            Number of samples.
            
    Returns:
        B : float
            JZS Bayes factor.
    '''
    
    from scipy.integrate import quad as integral
    v = N - 1
    numerator = (1 + t**2/v)**(-(v+1)/2)
    integrand = lambda g: (1 + N*g)**(-1/2)*(1 + t**2/((1 + N*g)*v))**(-(v+1)/2)*(2*np.pi)**(-1/2)*(g**(-3/2))*((np.e)**(-1/(2*g)))
    denominator = integral(integrand, 0, np.inf)[0]                                                        
    B = numerator/denominator
    return B

def one_stage_linear_step_up(ps, alpha=0.05):
    '''
    Implementation of the Benjamini-Hochberg (1995) procedure.
    '''
    
    m = len(ps)
    ranks = np.argsort(np.argsort(ps))+1
    ps_ = np.asarray([(m*p)/j for j, p in zip(ranks, ps)])
    ps_corr = np.asarray([np.min(ps_[ranks >= i]) for i in ranks])
    ps_corr[ps_corr > 1] = 1
    
    return pd.Series({'ps_corr': ps_corr,
                      'rejected': ps_corr < alpha})

def two_stage_linear_step_up(ps, alpha=0.05):
    '''
    Implementation of the Benjamini-Yekutieli (2006) FDR control procedure,
    as described in Defintion 6 (two-stage linear step-up procedure). 
    '''    
    
    m = len(ps)
    ranks = np.argsort(np.argsort(ps))+1
    ps_ = np.asarray([(m*p*(1+alpha))/j for j, p in zip(ranks, ps)])
    ps_corr = np.asarray([np.min(ps_[ranks >= i]) for i in ranks])
    ps_corr[ps_corr > 1] = 1
    r1 = np.sum(ps_corr < alpha)
    m0 = m - r1
    
    if r1 not in [0, m]:
        ps_ = np.asarray([(m0*p*(1+alpha))/j for j, p in zip(ranks, ps)])
        ps_corr = np.asarray([np.min(ps_[ranks >= i]) for i in ranks])
        ps_corr[ps_corr > 1] = 1
        
    return pd.Series({'ps_corr': ps_corr,
                      'rejected': ps_corr < alpha})

def tfce(ts):
    
    '''
    Threshold-free cluster enhancement.
    
    Parameters:
        ts : numpy.array
            A two-dimensional array of t-statistics.
            
    Returns:
        ts_tfce : numpy.array
            A two-dimmensional array of the TFCE-enhanced t-statistics.
    '''
    
    import skimage
    
    def one_sided_tfce(ts, thresholds, comparison_function):
        
        ts_tfce = np.zeros_like(ts)
        
        for threshold in thresholds:

            ts_beyond_threshold = comparison_function(ts, threshold)
            ts_clustered = skimage.measure.label(ts_beyond_threshold, connectivity=2)
            
            cluster_sizes = np.zeros_like(ts_clustered)
            for iCluster in np.arange(1, np.max(ts_clustered) + 1):
                cluster_sizes[ts_clustered == iCluster] = np.sum(ts_clustered == iCluster)
                
            for m in np.arange(ts.shape[0]):
                for n in np.arange(ts.shape[1]):
                    ts_tfce[m, n] = ts_tfce[m, n] + cluster_sizes[m, n]**0.5 * threshold**2
        
        return ts_tfce
    
    positive_thresholds = np.arange(0, np.max(ts) + 0.05, 0.05)
    negative_thresholds = np.arange(np.min(ts), 0.00 + 0.05, 0.05)
    positive_ts_tfce = one_sided_tfce(ts, positive_thresholds, np.greater)
    negative_ts_tfce = one_sided_tfce(ts, negative_thresholds, np.less)
    ts_tfce = positive_ts_tfce + negative_ts_tfce
    
    return ts_tfce

def duration_to_samples(duration_s, sample_rate_Hz):
    return int(duration_s * sample_rate_Hz) + 1

def get_time_offset(phase_offset, frequency):
    return phase_offset / (2 * np.pi * frequency)


def get_username_from_working_directory(index=2):
  """Extracts the username from the current working directory."""
  try:
    working_directory = os.getcwd()
    path_parts = working_directory.split(os.sep)
    # Assuming the username is in a fixed position, extract it. 
    # For example, if the path is /home/user/some/directory, 
    # then the username is at index 2.
    username = path_parts[index]
    return username
  except IndexError:
    raise ValueError("Unable to extract username from working directory.")
