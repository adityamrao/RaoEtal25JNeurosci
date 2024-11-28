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
    
    sub, exp, sess, loc, mon = dfrow[['sub', 'exp', 'sess', 'loc', 'mon']]
    if (int(loc) == 0) and (int(mon) == 0):
        return f'{sub}_{exp}_{sess}'
    else:
        return f'{sub}_{exp}_{sess}_{loc}_{mon}'

def npl(fname):
    
    return np.load(join('/scratch/amrao', fname), allow_pickle=True)
    
def load_mat(path):
    
    try:
        return scipy.io.loadmat(path)
    except:
        f = h5py.File(path, 'r') #can use .keys() and .get('{key}') methods to get data
        return f
    
def get_dfrow(dfrow):
    
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
    
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, obj):
    
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
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    np.set_printoptions(threshold=np.inf)
    
def print_header(text):
    print(f'---------{text}---------')

def print_whether_significant(p, alpha=0.05):
    
    if p < alpha: print(f'Statistically Significant (p < {alpha})')
    else: print(f'NOT Statistically Significant (p >= {alpha})')
    
def print_ttest_1samp(vals, header=None, alternative='two-sided'):
    
    vals = finitize(vals)
    t, p = scipy.stats.ttest_1samp(vals, popmean=0, alternative=alternative)
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
    
    from scipy.integrate import quad as integral
    v = N - 1
    numerator = (1 + t**2/v)**(-(v+1)/2)
    integrand = lambda g: (1 + N*g)**(-1/2)*(1 + t**2/((1 + N*g)*v))**(-(v+1)/2)*(2*np.pi)**(-1/2)*(g**(-3/2))*((np.e)**(-1/(2*g)))
    denominator = integral(integrand, 0, np.inf)[0]                                                        
    B = numerator/denominator
    return B

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
