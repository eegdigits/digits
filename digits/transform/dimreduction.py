# -*- coding: utf-8 -*-
"""
This is the main package for feature transformation and selection implementations.

"""
from ..data import select

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

from scipy import fftpack
from scipy.signal import hanning, ricker, cwt, decimate

from mne.filter import band_pass_filter
from mne import set_log_level

import logging
from itertools import combinations
from  warnings import catch_warnings, simplefilter, showwarning, warn
from joblib import Parallel, delayed

# this module is not available with pip and we are not using it anyway, so
# ignore silently if it's not installed from source
#   https://github.com/aaren/wavelets
try:
    import wavelets
except:
    pass

# again, don't fail if keras is not working, because it's not been super
# important
try:
    from keras.layers import Input, Dense
    from keras.models import Model
except:
    pass



def repack(X, samples, y, targets):

    new_samples = pd.DataFrame(data=X,
                               index=samples.index,
                               columns=samples.columns,
                               dtype=samples.values.dtype)
    new_targets = pd.DataFrame(data=y,
                               index=targets.index,
                               columns=targets.columns,
                               dtype=targets.values.dtype)
    return (new_samples, new_targets)


class Transform(object):
    """Transformation base class.

    Parameters
    ----------
    verbose : boolean (default=True)

    """

    def __init__(self, verbose=True, n_jobs=1):
        self.verbose = verbose
        self.update_index = 1
        self.n_jobs = n_jobs

    #def __del__(self):
    #    if self.verbose:
    #        print("")

    def transform(self, samples=None, targets=None):
        if samples is not None:
            self.channels = select.getchannelnames(samples)
            self.samplen = len(samples.iloc[0].loc[self.channels[0]])
            self.chanlen = len(self.channels)

    def update_status(self, n_max, n_inc=1, extra=None):
        if self.verbose:
            msg = "\rrunning {} for sample".format(self.__class__.__name__)
            msg += ' {}/{}'.format(self.update_index, n_max)
            if extra:
                msg += ' [{}]'.format(extra)
            print(msg, end='', flush=True)
            self.update_index += n_inc

    def fit_transform(self, X, **kwargs):
        self.fit(**kwargs)
        return self.transform(X, **kwargs)


class WaveletTransform(Transform):
    """
    Not actively being used yet, but it should work.
    """

    def __init__(self, timepoint=None, dj=None, **kwargs):
        super(WaveletTransform, self).__init__(**kwargs)
        self.wavelet = None
        self.timepoint = timepoint
        self.dj = dj

    def transform(self, samples):
        #samples = samples.swaplevel(0, 3, axis=0)
        super(WaveletTransform, self).transform(samples=samples)

        if not self.timepoint:
            self.timepoint = int(self.samplen/2)

        # pre-compute a transformation to get the shape
        data = samples.iloc[0].loc[self.channels[0]].values
        wavetransform = wavelets.WaveletTransform(data, dj=self.dj,
                                                  wavelet=self.wavelet)
        (scales, timeres) = wavetransform.wavelet_power.shape

        zpadlen = int(np.floor(np.log10(scales)) + 1)
        wavnames = ['s_'+str(int(x)).zfill(zpadlen) for x in np.arange(scales)]
        subcolix = pd.MultiIndex.from_product([self.channels, wavnames],
                                               names=['channel','wavelet'])
        wave_samples = pd.DataFrame(index=samples.index, columns=subcolix,
                                    dtype=samples.values.dtype)

        # TODO:
        #  + don't use iloc[] but actual indexer to be consistent
        #  + use joblib as this is embarrassingly parallel
        with Parallel(n_jobs=self.n_jobs) as parallel:
            for (index, sample) in samples.iterrows():
                self.update_status(len(samples))
                waves = parallel(delayed(self.channel_waves)(sample, chix, channel)
                                 for (chix,channel) in enumerate(self.channels))
                for chix, channel in enumerate(self.channels):
                    wave_samples.loc[index].loc[channel].values[:] = waves[chix]
                #for channel in self.channels:
                #    x = sample.loc[channel].values
                #    xw = wavelets.WaveletTransform(x, dj=self.dj,
                #                                   wavelet=self.wavelet)
                #    y = xw.wavelet_power[:, self.timepoint]
                #    wave_samples.loc[index].loc[channel].values[:] = y

        return wave_samples

    def channel_waves(self, sample, chix, channel):
        x = sample.loc[channel].values
        xw = wavelets.WaveletTransform(x, dj=self.dj, wavelet=self.wavelet)
        return xw.wavelet_power[:, self.timepoint]


class MorletTransform(WaveletTransform):

    def __init__(self, **kwargs):
        super(MorletTransform, self).__init__(**kwargs)
        self.wavelet = wavelets.Morlet()


class SubsampleTransform(Transform):
    """ Subsample time series by some width.
    Pads sample if sample size is not a multiple of width.

    Parameters
    ----------
    width: integer (default 5)

        Number of consecutive time points to average.

    Returns
    ----
    Re-indexed Dataframe with subsampled data for each sample.
    """

    def __init__(self, width=5, **kwargs):
        super(SubsampleTransform, self).__init__(**kwargs)
        self.width = width


    def transform(self, samples):
        super(SubsampleTransform, self).transform(samples)

        if self.width == 1:
            print("subsampling width is 1, skipping")
            return samples

        timelen = len(select.getsamplingnames(samples))
        sublen = np.ceil(timelen/self.width).astype('int')
        if timelen % self.width != 0:
            self.padlen = self.width - (timelen % self.width)
        else:
            self.padlen = 0

        # initialize return arrays
        zpadlen = int(np.log10(sublen)) + 1
        timenames = ['t_'+str.zfill(x, zpadlen) for x in np.arange(0, sublen).astype('str')]
        subcolix = pd.MultiIndex.from_product([self.channels, timenames],
                                              names=['channel','sample'])
        sub_samples = pd.DataFrame(index=samples.index, columns=subcolix,
                                   dtype=samples.values.dtype)


        for ix, sample in samples.iterrows():
            self.update_status(len(samples))
            for chix, channel in enumerate(self.channels):
                data = sample[channel].values
                data = np.pad(data, (0, self.padlen), mode='edge')
                #data = data.reshape(-1, self.width).mean(axis=1)
                data = decimate(data, self.width, zero_phase=True)
                sub_samples.loc[ix].loc[channel].values[:] = data

        return sub_samples


class AverageTransform(Transform):
    """
    Average randomly sampled samples to create a new sample with potentially lower SNR.
    Needs samples and targets data frames.

    Parameters
    ----------
    averae: integer (default 3)

        Number of samples to average.

    Returns
    ----
    Re-indexed Dataframes with averaged data for each sample and target.
    """

    def __init__(self, average=3, **kwargs):
        super(AverageTransform, self).__init__(**kwargs)
        self.average = average


    def transform(self, samples, targets):

        # create a full copy with NaN, will be sparse after the loop
        avg_samples = pd.DataFrame(index=samples.index, columns=samples.columns,
                                   dtype=samples.values.dtype)

        for target in np.unique(targets.values):
            self.update_index = 1
            target_samples = samples[ targets.label == target ]
            # randomize order by sampling N out of N
            target_samples = target_samples.sample(n=len(target_samples))
            strideix = 1
            data = np.zeros(samples.shape[1])
            for (index, sample) in target_samples.iterrows():
                data = sample.values + data
                if strideix < self.average:
                    strideix += 1
                else:
                    self.update_status(len(samples), n_inc=self.average,
                                       extra='target {}'.format(target))
                    # implicitly drop remaining (average-1) samples
                    avg_samples.loc[index].values[:] = data/self.average
                    data = np.zeros(samples.shape[1])
                    strideix = 1

        # drop rows containing NaN
        avg_samples = avg_samples.dropna()

        # construct new target df with indices from avg_samples
        avg_targets = pd.DataFrame(index=avg_samples.index, columns=targets.columns,
                                   dtype=avg_samples.values.dtype)
        for (index, _) in avg_samples.iterrows():
            avg_targets.loc[index].label = targets.loc[index].label

        return avg_samples.astype('float'), avg_targets.astype('int')


class ICATransform(Transform):
    """
    Not actively being used. Uses FastICA to project sample to a number of new
    ICA components.

    Parameters
    ----------
    components: integer (default 8)

        Number of transformation components.

    maxiter: integer (default 2000)

        Maximum number iterations after which FastICA should stop.

    Returns
    ----
    New dataframe with ica components.
    """

    def __init__(self, components=8, maxiter=2000, **kwargs):
        if components > 99:
            raise ValueError("number of components is too large")
        super(ICATransform, self).__init__(**kwargs)
        self.components = components
        self.maxiter = maxiter

    def transform(self, samples):

        components = ['C'+str.zfill(x, 2) for x in np.arange(self.components).astype('str')]
        timenames = select.getsamplingnames(samples)
        subcolix = pd.MultiIndex.from_product([components, timenames],
                                              names=['component','sample'])
        ica_samples = pd.DataFrame(index=samples.index, columns=subcolix,
                                   dtype=samples.values.dtype)

        chanlen = len(select.getchannelnames(samples))

        with catch_warnings(record=True) as w:
            simplefilter('ignore', UserWarning)
            for index, sample in samples.iterrows():
                self.update_status(len(samples))

                data = sample.reshape(chanlen, -1)
                ica = FastICA(max_iter=self.maxiter, n_components=self.components)
                ica.fit(data)

                # FIXME:

                # how do we get the same ordering (EEGlab is sorting by
                # "mean projected variance")

                # also we need to check the correct sign of the components
                # this is the point where I realized ICA might not be useful at all
                for compindex, component in enumerate(components):
                    ica_samples.loc[index].loc[component].values[:] = ica.components_[compindex]

            if w:
                showwarning("recorded {0} warnings (non-convergence)".format(len(w)))

        return ica_samples.astype('float')

class DCTWaveletTransform(Transform):
    """
    Not actively being used.
    Transforms sample data from WaveletTransform with a discrete consine transformation.

    Returns
    ----
    New dataframe with dct coefficients.
    """
    def __init__(self, **kwargs):
        super(DCTWaveletTransform, self).__init__(**kwargs)

    def transform(self, samples):
        super(DCTWaveletTransform, self).transform(samples)
        thresh = 100

        dctnames = ['d_'+str(x) for x in np.arange(thresh)]
        subcolix = pd.MultiIndex.from_product([self.channels, dctnames],
                                              names=['channel','sample'])
        dct_samples = pd.DataFrame(index=samples.index, columns=subcolix,
                                    dtype=samples.values.dtype)

        for ix, sample in samples.iterrows():
            self.update_status(len(samples))
            for chix, channel in enumerate(self.channels):
                x = sample.loc[channel].values
                y = cwt(x, ricker, np.arange(1,15))
                y = fftpack.dct(y)[:,:thresh].mean(axis=0)
                dct_samples.loc[ix].loc[channel].values[:] = y

        return dct_samples

class IFFTransform(Transform):
    """
    Not actively being used.
    Undo a FFTransform. Mainly used for testing.

    Returns
    ----
    New dataframe in time domain.
    """
    def __init__(self, rate=1/1000, **kwargs):
        super(IFFTransform, self).__init__(**kwargs)
        self.rate = rate

    def transform(self, samples):
        super(IFFTransform, self).transform(samples)
        fnames = samples.iloc[0].loc[self.channels[0]].index.tolist() # duplicates..
        if fnames[0] != fnames[1] or fnames[0][0] != 'f':
            raise ValueError('samples must be in non-power frequency domain')
        fmin = int(fnames[0].split('_',1)[1])
        fmax = int(fnames[-1].split('_',1)[1])
        freqs = np.arange(1/self.rate/2)

        # FIXME: add actual time names
        timenames = ['t_'+str(x) for x in np.arange(len(freqs))]
        subcolix = pd.MultiIndex.from_product([self.channels, timenames],
                                              names=['channel','sample'])
        ifft_samples = pd.DataFrame(index=samples.index, columns=subcolix,
                                    dtype=samples.values.dtype)
        for ix, sample in samples.iterrows():
            self.update_status(len(samples))
            for chix, channel in enumerate(self.channels):
                data = sample.loc[channel].values.reshape(-1, 2)
                z = np.zeros(2*len(freqs)).reshape(-1, 2)
                fs = [ int(x.split('f_',1)[1]) for x in np.unique(fnames)]
                for fix, f in enumerate(fs):
                    z[f] = data[fix]

                z = z.flatten()
                z = np.roll(z, -1)
                iz = fftpack.irfft(z)
                ifft_samples.loc[ix].loc[channel].values[:] = iz[:int(len(iz)/2)]

        return ifft_samples



class FFTransform(Transform):
    """Fast Fourier Transform wrapper.

    Parameters
    ----------
    rate : float (required)
        sampling rate

    window : function handle (default scipy.signal.hanning)
        a windowing function

    bins : integer(default=40)
        desired bin count

    fmin: float (default=None)
        lowest frequency to include in the filtering

    fmax: float (default=None)
        highest frequency to include in the filtering

    logdistance : boolean (default=True)
        whether or not to use exponential distance with increasing frequency values

    power : boolean (default=True)
        whether or not to compute the power spectrum

    average: boolean (default=True)
        whether or not to average inter-bin values to the next bin

    multiplesof: integer (default=2)
        include only frequency values that are a multiple of this value

    logtransform: boolean (default=False)
        whether or not to logarithmically transform frequency values

    Returns
    -------
    Samples dataframe in frequency domain. Second level index starts with 'f_'.

    """
    def __init__(self, rate, window=hanning, bins=40, fmin=None, fmax=None,
                 logdistance=True, power=True, average=True, multiplesof=2,
                 logtransform=False, **kwargs):
        super(FFTransform, self).__init__(**kwargs)
        self.rate = rate
        self.fmin = fmin
        self.fmax = fmax
        self.logdistance = logdistance
        self.power = power
        self.average = average
        self.multiplesof = multiplesof
        self.logtransform = logtransform
        if self.power:
            self.bins = bins
        else:
            self.bins = int(bins/2)
        if window is None:
            self.window = self.ident
        else:
            self.window = window

    def ident(self, length):
        return 1

    def mask_freqs(self):
        # drop f[0] and f[max] if length is even
        # fs = R[0], R[1], Im[1], R[2], Im[2], ... , R[n/2-1], Im[n/2-1], R[n/2]
        if self.samplen % 2 == 0:
            freqs = fftpack.rfftfreq(self.samplen, self.rate)[1:-1:2]
        else:
            freqs = fftpack.rfftfreq(self.samplen, self.rate)[1::2]

        # target spectrum window
        if self.fmin is not None:
            freqs = freqs[ freqs >= self.fmin ]
        if self.fmax is not None:
            if 2*self.fmax < freqs[-1]:
                warn('highest frequency bin ({}) is 2x larger than '
                     'cutoff frequency ({}), consider subsampling'.format(freqs[-1], self.fmax),
                     stacklevel=2)
            freqs = freqs[ freqs <= self.fmax ]

        # initial mask is everything
        maskix = np.arange(len(freqs))

        # drop all freqs that are not a multiple of k
        if self.multiplesof is not None:
            rests = divmod(freqs, self.multiplesof)[1]
            # TODO: handle empty list
            tmpix = np.where(np.isclose(rests, 0))[0]
            maskix = np.array([x for x in maskix if x in tmpix])

        self.validmask = maskix
        num_f = len(maskix)

        if self.bins is not None and num_f < self.bins:
            warn('number of bins ({}) exceeds valid frequency count ({}). Is the sample size a multiple of {}?'.
                    format(self.bins, num_f, self.multiplesof))
            self.bins = num_f

        if self.logdistance is not None:
            # FIXME
            # there is probably an actual formula for this 8)
            # also handle self.distance here in case we want an equidistant mask
            i = 1
            bins = 0
            while ( bins < self.bins ):
                tmpix = np.logspace(0, np.log10(len(maskix)), i)
                tmpix = np.round(tmpix).astype('int') - 1
                tmpix = np.unique(tmpix)
                bins = len(tmpix)
                i += 1
            maskix = np.array([x for x in maskix if x in tmpix])

        self.distmask = maskix
        return np.array([freqs[x] for x in maskix])


    def transform(self, samples):
        super(FFTransform, self).transform(samples)
        self.xs_window = self.window(self.samplen)
        freqs = self.mask_freqs()

        zpadlen = int(np.floor(np.log10(freqs[-1])) + 1)
        if not self.power:
            freqs = np.repeat(freqs, 2)
        freqnames = ['f_'+str(int(x)).zfill(zpadlen) for x in freqs]
        subcolix = pd.MultiIndex.from_product([self.channels, freqnames],
                                               names=['channel','sample'])
        fft_samples = pd.DataFrame(index=samples.index, columns=subcolix,
                                   dtype=np.float64)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            for ix, sample in samples.iterrows():
                self.update_status(len(samples))
                ys = parallel(delayed(self._aggregate)(
                              sample, channel)
                              for _,channel in enumerate(self.channels))
                for chix, channel in enumerate(self.channels):
                    if self.logtransform:
                        # in case we don't create the power spectrum and still
                        # want to logtransform, handle negative values
                        fft_samples.loc[ix].loc[channel].values[:] = np.sign(ys[chix]) * np.log(np.abs(ys[chix]))
                    else:
                        fft_samples.loc[ix].loc[channel].values[:] = ys[chix]

        return fft_samples

    def _aggregate(self, sample, channel):
        y_window = self.xs_window*sample.loc[channel].values
        y = fftpack.rfft(y_window)[1:]
        if self.samplen % 2 == 0:
            y = y[:-1]
        y = y.reshape(-1,2)
        ysq = np.array([y[ix] for ix in self.validmask])
        if self.power:
            # get magnitude as sqrt(im**2 + re**2)
            ysq = np.apply_along_axis(lambda x: np.sqrt(x[0]**2 + x[1]**2), 1, ysq)
            if self.average:
                y = list(np.zeros(len(self.distmask)))
                for mix in np.arange(len(self.distmask)-1):
                    y[mix] = ysq[self.distmask[mix]:self.distmask[mix+1]].mean()
                y[-1] = ysq[self.distmask[-1]:].mean()
            else:
                y = [ysq[x] for x in self.distmask]
        else:
            # not sure if this is making any sense
            # I am averaging real and imaginary parts each
            if self.average:
                y = list(np.zeros(len(self.distmask)))
                for mix in np.arange(len(self.distmask)-1):
                    y[mix] = ysq[self.distmask[mix]:self.distmask[mix+1]].mean(axis=0)
                y[-1] = ysq[self.distmask[-1]:].mean(axis=0)
                y = np.squeeze(y).reshape(1, -1)
            else:
                y = np.array([ysq[x] for x in self.distmask]).reshape(1,-1).squeeze()
        return y




class BandPassTransform(Transform):
    """
    Not actively being used.
    Filter data in time domain using a scipy band pass filter.

    Parameters
    ----------
    rate : float (default=1000)
        sampling rate
    min:   float (default=7)
        lower end
    max: float (default=30)
        higher end
    """
    def __init__(self, rate=1000, min=7, max=30, **kwargs):
        super(BandPassTransform, self).__init__(**kwargs)
        self.rate = rate
        self.max = max
        self.min = min

    def transform(self, samples):

        pass_samples = pd.DataFrame(index=samples.index, columns=samples.columns,
                                    dtype=samples.values.dtype)
        channels = select.getchannelnames(samples)
        set_log_level(logging.ERROR)

        for ix, sample in samples.iterrows():
            self.update_status(len(samples))
            for channel in channels:
                vals = band_pass_filter(sample.loc[channel].values.astype('float64'),
                                        self.rate, self.min, self.max)
                pass_samples.loc[ix].loc[channel].values[:] = vals

        return pass_samples

class STDTransform(Transform):
    """
    Not actively being used, just a test.
    Blockwise transform for each channel to its standard deviation.

    Parameters
    ----------
    blocks : integer (default 1)

        Number of blocks to split a channel in

    Returns
    ----
    Dataframe with *channel* x *blocks* values
    """

    def __init__(self, blocks=1, **kwargs):
        super(STDTransform, self).__init__(**kwargs)
        self.blocks = blocks

    def transform(self, samples):
        channels = select.getchannelnames(samples)
        samplen = samples.iloc[0].loc[channels[0]].size
        sublen = int(np.ceil(samplen/self.blocks))
        padlen = int(self.blocks*sublen - samplen)
        if padlen != 0:
            warn("sample size not dividable by {}, need to pad".
                 format(self.blocks))
        zpadlen = int(np.log10(self.blocks)) + 1
        names = ['s_'+str.zfill(x, zpadlen) for x in np.arange(0, self.blocks).astype('str')]
        subcolix = pd.MultiIndex.from_product([channels, names],
                                              names=['channel','stdblock'])
        std_samples = pd.DataFrame(index=samples.index, columns=subcolix,
                                    dtype=samples.values.dtype)

        for ix, sample in samples.iterrows():
            self.update_status(len(samples))
            for channel in channels:
                data = sample.loc[channel]
                data = np.pad(data, (0, padlen), mode='mean')
                data = data.reshape(self.blocks, sublen).std(axis=1)
                std_samples.loc[ix].loc[channel].values[:] = data

        return std_samples

class AEDenoise(Transform):
    """
    Not actively being used yet. Initial tests for autoencoder denoising using Keras.

    """

    def __init__(self, **kwargs):
        super(AEDenoise, self).__init__(**kwargs)
        self.comp_factor = 4
        self.fitted = False

    def fit(self, X):
        dim = X.shape[1]
        enc_dim = int(dim/self.comp_factor)
        input_eeg = Input(shape=(dim,), name='raw_eeg')
        encoded = Dense(enc_dim, activation='relu')(input_eeg)
        decoded = Dense(dim, activation='sigmoid')(encoded)
        autoencoder = Model(input=input_eeg, output=decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        x_source = np.vstack((x[0] for x in combinations(X.values[0:100], 2)))
        x_target = np.vstack((x[1] for x in combinations(X.values[0:100], 2)))
        autoencoder.fit(x_source, x_target, verbose=self.verbose,
                        nb_epoch=100, batch_size=1000, shuffle=False,
                        validation_data=None)
        #self.weights = autoencoder.get_weights()
        self.fitted = True
        self.autoencoder = autoencoder

    def transform(self, X):
        if not self.fitted:
            raise ValueError("instance has not been fitted yet")
        #return self.weights[-1]*X
        return self.autoencoder.predict(X)
