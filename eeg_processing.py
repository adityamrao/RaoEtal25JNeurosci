import numpy as np
from ptsa.data.filters import MorletWaveletFilter

from cstat import circ_mean


def get_phase(eeg, freqs, width=5):
    wavelet_filter = MorletWaveletFilter(freqs=freqs, width=width, output='phase', complete=True, verbose=False)
    phase = wavelet_filter.filter(timeseries=eeg)
    phase = phase.transpose('event', 'channel', 'frequency', 'time')
    return phase


# def process_phase(dfrow, events, freqs):
#     eeg, mask = get_beh_eeg(dfrow, events)
#     phase = get_phase(eeg, freqs)
    
#     sample_rate = float(eeg.samplerate)
#     buffer_length = int(sample_rate/1000*1000)
#     phase = clip_buffer(phase, buffer_length)
    
#     return phase, mask, sample_rate


def timebin_phase_timeseries(timeseries, sample_rate, bin_width_ms=200):
    return timebin_timeseries(timeseries, sample_rate, circ_mean, bin_width_ms=bin_width_ms)


def timebin_power_timeseries(timeseries, sample_rate, bin_width_ms=200):
    return timebin_timeseries(timeseries, sample_rate, np.mean, bin_width_ms=bin_width_ms)

def timebin_timeseries(timeseries, sample_rate, average_function, bin_width_ms=200, time_axis=-1):
    bin_size = int(sample_rate * bin_width_ms / 1000)
    bin_count = int(np.round(timeseries.shape[time_axis] / bin_size))
    
    timebinned_timeseries = []
    for iBin in range(bin_count):
        left_edge = iBin*bin_size
        right_edge = (iBin+1)*bin_size if iBin < bin_count - 1 else None
        this_epoch = average_function(timeseries[..., left_edge:right_edge], axis=time_axis)
        timebinned_timeseries.append(this_epoch)

    timebinned_timeseries = np.asarray(timebinned_timeseries)
    timebinned_timeseries = np.moveaxis(timebinned_timeseries, 0, time_axis)
    
    return timebinned_timeseries


def clip_buffer(timeseries, buffer_length):
    return timeseries.isel(time=np.arange(buffer_length, len(timeseries['time'])-buffer_length))

