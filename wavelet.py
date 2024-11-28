import numpy as np
import scipy as sp
import scipy.fft
import scipy.signal
import matplotlib.pyplot as plt

from misc import duration_to_samples


class Wavelet:
    def __init__(self, fmin, fmax, fnum, tmin=-4, tmax=4, sampling_rate=1000, morlet_reps=5, amplitude=None):
        self.tmin = tmin
        self.tmax = tmax
        self.sampling_rate = sampling_rate

        self.fmin = fmin
        self.fmax = fmax
        self.fnum = fnum
        self.morlet_reps = morlet_reps
        self.amplitude = amplitude
        self.freqs = np.logspace(np.log10(self.fmin), np.log10(self.fmax),
            self.fnum)
        self.tvals = np.linspace(self.tmin, self.tmax,
            duration_to_samples(self.tmax-self.tmin, self.sampling_rate))
        self.tlen = len(self.tvals)

    def get_morlet_width(self, f):
        return self.morlet_reps/(2*np.pi*f)
    
    def get_morlet_envelope(self, t, sig):
        # Ryan Colyer's implementation (not normalized to unity power...)
        # A = 1/(sig*np.sqrt(2*np.pi))
        # taken from https://cp.copernicus.org/preprints/cp-2019-105/cp-2019-105-supplement.pdf
        A = 1/np.sqrt(sig*np.sqrt(np.pi))
        if self.amplitude is not None:
            A *= self.amplitude
        return A * np.exp(-t**2 / (2 * sig**2))

    def Morlet(self, t, morlet_reps, f):
        sig = self.get_morlet_width(f)
        return self.get_morlet_envelope(t, sig) * np.exp(1j*2*np.pi * f * t)

    def PlotF(self, f, plot_wavelet=False):
        mvals = np.array([self.Morlet(t, self.morlet_reps, f) for t in self.tvals])

        ftlen = (self.tlen+1)//2
        ftlen = self.tlen
        fvals = sp.fft.fftfreq(self.tlen, self.tvals[1]-self.tvals[0])[0:ftlen]
        ft_morlet = sp.fft.fft(mvals)[0:ftlen]

        # To see the wavelets generated.
        if plot_wavelet:
            sigma = self.get_morlet_width(f)
            envelope = self.get_morlet_envelope(self.tvals, sigma)
            max_morlet = envelope.max()
            envelope /= max_morlet
            mvals /= max_morlet
            plt.plot(self.tvals, mvals.real, label='Real')
            plt.plot(self.tvals, mvals.imag, label='Imaginary')
            plt.plot(self.tvals,
                     envelope,
                     label='Envelope',
                     alpha=0.5)
            plt.plot(self.tvals,
                     envelope ** 2,
                     label='Envelope-Squared',
                     alpha=0.5)
            n_sigma = 3
            plt.vlines(x=[i * sigma for i in range(-n_sigma, n_sigma + 1) if i != 0],
                       ymin=0, ymax=mvals.real.max(), colors='c', alpha=0.3, label='Sigma multiples from t = 0')
            plt.legend(loc=(1.02, 0))
            plt.title(f'Max-Normalized Wavelet: f = {f} Hz', fontsize=25)
            plt.xlabel('Time (s)')
            plt.ylabel('Wavelet')
        else:
            start_f = np.argwhere(fvals >= 1).ravel()[0]
            end_f = np.argwhere(fvals > 300).ravel()[0]
            ft_mor_rel_power = np.abs(ft_morlet)**2
            ft_mor_rel_power /= np.max(ft_mor_rel_power)
            plt.semilogx(fvals[start_f:end_f], ft_mor_rel_power[start_f:end_f],
                label=f'{f:.2f}Hz')


    def PlotButterworth(self, bw_min, bw_max):
        yvals = np.zeros(self.tlen)
        yvals[self.tlen//2] = 1

        nyq = self.sampling_rate / 2
        b, a = sp.signal.butter(4, [bw_min/nyq, bw_max/nyq], 'stop')
        yvals = sp.signal.filtfilt(b, a, yvals, axis=0)

        ftlen = (self.tlen+1)//2
        ftlen = self.tlen
        fvals = sp.fft.fftfreq(self.tlen, self.tvals[1]-self.tvals[0])[0:ftlen]
        ft_bw = sp.fft.fft(yvals)[0:ftlen]

        start_f = np.argwhere(fvals >= 1).ravel()[0]
        end_f = np.argwhere(fvals > 300).ravel()[0]
        ft_bw_rel_power = np.abs(ft_bw)**2
        ft_bw_rel_power /= np.max(ft_bw_rel_power)
        plt.semilogx(fvals[start_f:end_f], ft_bw_rel_power[start_f:end_f],
            color='darkgrey', label=f'BW {bw_min}-{bw_max}')


    def PlotValidate(self, f):
        '''Generates (slowly) a comparable plot to PlotF but using PTSA and
           measuring the power it produces for sinusoids of each frequency.
           This validates that the analytical approach in PlotF correctly
           shows the power extracted by a Morlet Transform done by PTSA.'''
        from ptsa.data.timeseries import TimeSeries
        from ptsa.data.filters import morlet

        cvals = []
        xfreqs = np.logspace(np.log10(f/10), np.log10(f*10), 200)
        for fx in xfreqs:
            cvals.append(np.cos(2*np.pi*fx*self.tvals))

        cos_timeseries = TimeSeries(cvals, {'samplerate':self.sampling_rate},
                ['testfreq', 'time'])
        wf = morlet.MorletWaveletFilter(timeseries=cos_timeseries,
                freqs=[f], width=self.morlet_reps, output=['power'], complete=True)
        power = wf.filter()

        pow_plot = np.mean(power[0, :, self.tlen//12:-self.tlen//12],
                axis=1)
        pow_plot /= np.max(pow_plot)
        plt.semilogx(xfreqs, pow_plot, label=f'PTSA')


    def MakePlot(self, show=False):
        self.fig = plt.figure()
        plt.rcParams.update({'font.size': 12})

        for f in self.freqs:
            self.PlotF(f)

        self.PlotButterworth(58, 62)
        # if self.fmax > 130:
        self.PlotButterworth(118, 122)
        # if self.fmax > 190:
        self.PlotButterworth(178, 182)

        #self.PlotValidate(6)

        plt.vlines(60, 0, 1, linestyles='dashdot', colors='black', label='60Hz')
        plt.vlines(120, 0, 1, linestyles='dashdot', colors='black', label='120Hz')
        plt.vlines(180, 0, 1, linestyles='dashdot', colors='black', label='180Hz')
        plt.legend(loc=(1.02, 0))
        plt.ylabel('Relative power')
        plt.xlabel('Frequency (Hz)')
        plt.title(f'Morlet power {self.fnum} freqs {self.fmin}Hz to {self.fmax}Hz, wavenum {self.morlet_reps}')
        basename = f'morlet_power_{self.fnum}_{self.fmin}_{self.fmax}'
        if self.morlet_reps != 5:
            basename += f'_{self.morlet_reps}'
        # plt.savefig(f'{basename}.png')
        # plt.savefig(f'{basename}.pdf')
        # if show:
        plt.show()
