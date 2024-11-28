import numpy as np
import xarray as xr
from time import time
from ptsa.data.timeseries import TimeSeries
import matplotlib.pyplot as plt

from cstat import ppc_to_wrapped_normal_sigma, sample_wrapped_multivariate_normal, ppc
from matrix_operations import apply_function_expand_dims, symmetrize, is_square, is_symmetric, is_positive_definite, strict_triu, upper_tri_values, off_diagonal_values, sort_array_across_order
from eeg_processing import get_phase, timebin_phase_timeseries
from misc import duration_to_samples, get_time_offset
from wavelet import Wavelet


AVAILABLE_SIMULATIONS = [
            'standard', '', None,  # use empirical EEG (no simulation)
            'null_connectivity',  # no functional connectivity (FC) with EEG simulated with pink noise
            'strong_oscillation_only',  # idealized FC with noiseless wavelet oscillations giving non-physiologically strong FC
            'oscillation_only',  # idealized FC with noiseless wavelet oscillations having realistic FC
            'noisy_oscillation',  # wavelet oscillations plus additive pink noise
]

simulation_parameters = {
            # use empirical EEG (no simulation)
            'standard': None,
            '': None, 
            None: None,
            # no functional connectivity (FC) with EEG simulated with pink noise
            'null_connectivity': {
                'wavelet_amplitude': 0,
                'phase_covariance_function': None,
                'oscillation_frequency': 5,  # Hz
                'morlet_reps': 5,
                'pinknoise_amplitude': 100,
                'pinknoise_exponent': 1,
            },
            # idealized FC with noiseless wavelet oscillations giving non-physiologically strong FC
            'strong_oscillation_only': {
                'wavelet_amplitude': 100,
                'phase_covariance_function': 'within_region_group',
                'global_ppc': 0.05,
                'within_group_ppc': 0.2,
                'within_region_ppc': 0.4,
                'oscillation_frequency': 5,  # Hz
                'morlet_reps': 5,
                'pinknoise_amplitude': 0,
                'pinknoise_exponent': 1,
            },
            # idealized FC with noiseless wavelet oscillations having realistic FC
            'oscillation_only': {
                'wavelet_amplitude': 100,
                'phase_covariance_function': 'within_region_group',
                'global_ppc': 0.0001,
                'within_group_ppc': 0.01,
                'within_region_ppc': 0.02,
                'oscillation_frequency': 5,  # Hz
                'morlet_reps': 5,
                'pinknoise_amplitude': 0,
                'pinknoise_exponent': 1,
            },
            # wavelet oscillations plus additive pink noise
            'noisy_oscillation': {
                'wavelet_amplitude': 100,
                'phase_covariance_function': 'within_region_group',
                'global_ppc': 0.0001,
                'within_group_ppc': 0.01,
                'within_region_ppc': 0.02,
                'oscillation_frequency': 5,  # Hz
                'morlet_reps': 5,
                'pinknoise_amplitude': 100,
                'pinknoise_exponent': 1,
            },
}


def generate_pink_noise(N_samples, pinknoise_amplitude, pinknoise_exponent):
    out_N = N_samples
    N = N_samples
    if (N % 2) == 1:
        N += 1
    scales = np.linspace(0, 0.5, N//2+1)[1:]
    scales = scales**(-pinknoise_exponent/2)
    pinkf = (np.random.normal(scale=scales) * 
             np.exp(2j*np.pi*np.random.random(N//2)))
    fdata = np.concatenate([[0], pinkf])
    sigma = np.sqrt(2*np.sum(scales**2)) / N
    data = pinknoise_amplitude * np.fft.irfft(fdata)/sigma
    return data[:out_N]


def get_random_state_by_time():
    return int(time() * 1e6) % (2**32 - 1)


def sample_phases(mean, cov, samples=1, random_state=None):
    # kappa = plv_to_kappa(population_ppc)
    # phases = vonmises.rvs(kappa=kappa, loc=phase_offset, size=samples,
    #                       random_state=random_state if random_state is not None else get_random_state_by_time())  # random_state=None is deterministic

    if random_state is not None:
        np.random.seed(random_state)
    phases = sample_wrapped_multivariate_normal(mean=mean, cov=cov, size=samples)
    return phases


def generate_event_wavelets(phases, duration_s=1, frequency=5, phase_offset=0, morlet_reps=5, sampling_rate=1000, amplitude=None, wavelet_type='real', normalize=True):
    # generate wavelets centered at middle of epoch
    wavelet = Wavelet(fmin=2,
                      fmax=200,
                      fnum=1,
                      sampling_rate=sampling_rate,
                      morlet_reps=morlet_reps,
                      amplitude=amplitude,
                      tmin=-duration_s / 2,
                      tmax=duration_s / 2)

    offsets = get_time_offset(phases, frequency)
    def get_offset_wavelet(offset):
        w = wavelet.Morlet(wavelet.tvals - offset, wavelet.morlet_reps, frequency)
        if wavelet_type == 'real':
            w = w.real
        elif wavelet_type == 'imaginary':
            w = w.imag
        elif wavelet_type == 'full':
            pass
        else:
            raise ValueError
        if normalize and (wavelet_type in ['real', 'imaginary']):
            # normalize to unity power (dropping real or imaginary component halves power)
            w *= np.sqrt(2)
        return w
    wavelets = apply_function_expand_dims(offsets, get_offset_wavelet)
    return wavelets

def sample_eeg(n_events,
               n_channels,
               phase_mean,
               phase_covariance,
               sample_rate_Hz,
               start_time_ms,
               duration_ms,
               wavelet_amplitude=1,
               connectivity_frequency_Hz=5,
               morlet_reps=5,
               pinknoise_amplitude=1,
               pinknoise_exponent=1):
    duration_s = duration_ms / 1000
    if wavelet_amplitude > 0:
        assert n_channels == len(phase_mean)
        assert phase_covariance.shape == (n_channels, n_channels)
    N_samples_per_epoch = duration_to_samples(duration_s, sample_rate_Hz)
    end_time_ms = start_time_ms + duration_ms
    times = np.linspace(start_time_ms, end_time_ms, N_samples_per_epoch)
    coords = {'event': ['event_' + str(i) for i in range(n_events)],
              'channel': ['CH' + str(i) for i in range(n_channels)],
              'time': times
    }
    data = np.full(shape=(n_events, n_channels, len(times)), fill_value=np.nan)
    
    for event in range(n_events):
        if wavelet_amplitude > 0:
            phases = sample_phases(mean=phase_mean, cov=phase_covariance, samples=1)[0]
            wavelets = generate_event_wavelets(phases=phases,
                                               duration_s=duration_s,
                                               frequency=connectivity_frequency_Hz,
                                               sampling_rate=sample_rate_Hz,
                                               amplitude=wavelet_amplitude,
                                               morlet_reps=morlet_reps,
                                               # wavelet_type='full'
                                              )
        for channel in range(n_channels):
            if wavelet_amplitude > 0:
                data[event, channel] = wavelets[channel]
            if pinknoise_amplitude > 0:
                noise = generate_pink_noise(N_samples=N_samples_per_epoch,
                                            pinknoise_amplitude=pinknoise_amplitude,
                                            pinknoise_exponent=pinknoise_exponent)
                if wavelet_amplitude > 0:
                    data[event, channel] += noise
                else:
                    data[event, channel] = noise
    if np.isnan(data).any():
        raise ValueError('NaNs detected in EEG!')
    eeg = TimeSeries.create(
        data,
        samplerate=sample_rate_Hz,
        coords=coords,
        dims=list(coords.keys()),
    )

    return eeg


def compute_ppc_matrix_nonoverlapping(phase, sample_rate, epoch_width_ms=200, time_axis=-1):
    if not isinstance(phase, xr.DataArray):
        raise ValueError()
    electrode_count = len(phase.channel)
    freq_count = len(phase.frequency)
    epoch_size = int(sample_rate * epoch_width_ms / 1000)
    epoch_count = int(np.round(phase.shape[time_axis] / epoch_size))
    ppcs = np.full((electrode_count, electrode_count, freq_count, epoch_count), np.nan)
    for iElec in np.arange(electrode_count):
        for jElec in np.arange(electrode_count):
            # if (jElec > iElec) or ((iElec, jElec) in overlapping_pairs):
            #     continue
            diff = (phase.isel(channel = jElec) - phase.isel(channel = iElec)).data
            diff = timebin_phase_timeseries(diff, sample_rate, bin_width_ms=epoch_width_ms)
            ppcs[iElec, jElec] = ppc(diff)
    
    ppcs = symmetrize(ppcs)
    ppcs = xr.DataArray(data=ppcs,
                        dims=('channel1', 'channel2', 'frequency', 'epoch'),
                        coords={'channel1': phase.channel.values,
                                'channel2': phase.channel.values,
                                'frequency': phase.frequency,
                                'epoch': np.arange(epoch_count) * epoch_size,
                        }
                       )
    
    return ppcs


def ppc_matrix_to_wrapped_normal_covariance(ppc_matrix):
    is_square(ppc_matrix, require=True)
    ppc_variances = ppc_to_wrapped_normal_sigma(ppc_matrix) ** 2
    ppc_variances = strict_triu(ppc_variances)
    max_var = ppc_variances.max()
    covar = np.ones(ppc_matrix.shape) * max_var
    for i in range(covar.shape[0]):
        for j in range(i + 1, covar.shape[0]):
            ppc_delta_var = ppc_variances[i, j] / 2
            covar[i, j] -= ppc_delta_var
            covar[j, i] -= ppc_delta_var
    is_symmetric(covar, require=True)
    is_positive_definite(covar, require=True)
    return covar


def get_block_diagonal_ppc_matrix(n_channels=16,
                                  n_regions=4,
                                  n_region_groups=2,
                                  regions=None,
                                  region_groups=None,
                                  global_ppc=0.05,
                                  within_group_ppc=0.2,
                                  within_region_ppc=0.4,
                                  verbose=False):
    # global_ppc models global connectivity between all channels or lack thereof
    # Note: all PPC values must be positive given a PPC of zero corresponds to a wrapped normal with 
    # infinite linear variance (which could be implemented with a separate outer uniform random 
    # phase sample for disjoint connectivity subgraphs), so use a minimum PPC well below 
    # empirically observed values to model zero connectivity
    assert global_ppc > 0
    assert within_group_ppc > 0
    assert within_region_ppc > 0
    
    
    if regions is None:
        ppc_matrix = np.ones((n_channels, n_channels)) * global_ppc
        assert n_channels % n_regions == 0
        n_channels_per_region = n_channels // n_regions

        assert n_channels % n_region_groups == 0
        n_channels_per_group = n_channels // n_region_groups

        # model functional connectivity as connected groups of internally connected regions
        for region in range(0, n_channels, n_channels_per_group):
            ppc_matrix[region:region + n_channels_per_group, 
                       region:region + n_channels_per_group] = within_group_ppc

        for channel in range(0, n_channels, n_channels_per_region):
            ppc_matrix[channel:channel + n_channels_per_region, 
                       channel:channel + n_channels_per_region] = within_region_ppc
    else:
        assert isinstance(regions, list) and all([isinstance(region, str) for region in regions])
        assert within_group_ppc <= within_region_ppc
        n_channels = len(regions)
        ppc_matrix = np.ones((n_channels, n_channels)) * global_ppc
        unique_regions, region_counts = np.unique(regions, return_counts=True)
        n_regions = len(unique_regions)
        
        for idx1, (region1, group1) in enumerate(zip(regions, region_groups)):
            for idx2, (region2, group2) in enumerate(zip(regions, region_groups)):
                if group1 == group2:
                    ppc_matrix[idx1, idx2] = within_group_ppc
                if region1 == region2:
                    ppc_matrix[idx1, idx2] = within_region_ppc
        
    np.fill_diagonal(ppc_matrix, 1)

    if verbose:
        ppcs = strict_triu(ppc_matrix).ravel()
        ppcs = ppcs[ppcs > 0]
        print(f'Mean PPC: {ppcs.mean()}')
        plt.figure()
        plt.imshow(ppc_matrix)
        plt.colorbar()
        plt.title('PPC Matrix')
        if regions:
            # from matrix_operations import sort_array_across_order
            ppc_matrix_sorted = sort_array_across_order(ppc_matrix, regions, axis=[0, 1])
            plt.figure()
            plt.imshow(ppc_matrix_sorted)
            plt.colorbar()
            plt.title('PPC Matrix Sorted by Region')
        
    return ppc_matrix
