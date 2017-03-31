
import numpy as np
from math import sqrt
from scipy import linalg
from scipy.fftpack import fft, ifft

from matplotlib import pyplot as plt

def my_morlet(sfreq, freq, x_shift, x, n_cycles=7, sigma=None, zero_mean=False):
    """As morlet_mne, but for one frequency, and with possibility to
    specify length and center latency.

    Parameters
    ----------
    sfreq : float
        The sampling Frequency.
    freq : float
        frequency of interest
    n_cycles: float | array of float, defaults to 7.0
        Number of cycles. Fixed number or one per frequency.    
    x_shift: float
        centre latency of wavelet (ms)
    x : array of float
        x values for output wavelet
    sigma : float, defaults to None
        It controls the width of the wavelet ie its temporal
        resolution. If sigma is None the temporal resolution
        is adapted with the frequency like for all wavelet transform.
        The higher the frequency the shorter is the wavelet.
        If sigma is fixed the temporal resolution is fixed
        like for the short time Fourier transform and the number
        of oscillations increases with the frequency.
    zero_mean : bool, defaults to False
        Make sure the wavelet has a mean of zero.

    Returns
    -------
    W : array
        The wavelet time series.
        Note: if desired wavelet longer than computed one, it will be
        padded with zeros
    """

    n = x.shape # length of output wavelet

    if not type(freq)==list:
        freq = [freq]
    elif len(freq)>1:
        print "Only first list item used for 'freq'."
        freq = [freq[0]]

    if not type(n_cycles)==list:
        n_cycles = [n_cycles]

    # from morlet_mne(), in order to get latencies t
    if len(n_cycles) != 1:
            this_n_cycles = n_cycles[k]
    else:
        this_n_cycles = n_cycles[0]
    # fixed or scale-dependent window
    if sigma is None:
        sigma_t = this_n_cycles / (2.0 * np.pi * freq[0])
    else:
        sigma_t = this_n_cycles / (2.0 * np.pi * sigma)
    # this scaling factor is proportional to (Tallon-Baudry 98):
    # (sigma_t*sqrt(pi))^(-1/2);
    t = np.arange(0., 5. * sigma_t, 1.0 / sfreq)
    t = np.r_[-t[::-1], t[1:]]
    
    # Compute original wavelet
    W0 =  morlet_mne(sfreq, freq, n_cycles=n_cycles, sigma=sigma, zero_mean=zero_mean)
    W0 = W0[0]

    # initialise output wavelet
    W = np.zeros(n)

    # indices to elements in x that are close to latencies in t
    t = 1000*t + x_shift # wavelet latencies plus shift

    discr = 1000./(2*sfreq) # discrepancy allowed between sample points in x and t

    tx_idx = [] # record pairs of matching indices between x and t
    tt_idx = []
    for [ti,tt] in enumerate(t):
        idx = np.argmin(np.abs(x-tt))
        d = np.abs(x[idx]-tt)
        if (d<=discr):
            tx_idx.append(idx)
            tt_idx.append(ti)

    W[tx_idx] = W0[tt_idx]

    return W
    
    


### The following stuff is from MNE-Python, v0.14, tfr.py
# I intend not to modify it at all
def morlet_mne(sfreq, freqs, n_cycles=7.0, sigma=None, zero_mean=False):
    """Compute Morlet wavelets for the given frequency range.

    Parameters
    ----------
    sfreq : float
        The sampling Frequency.
    freqs : array
        frequency range of interest (1 x Frequencies)
    n_cycles: float | array of float, defaults to 7.0
        Number of cycles. Fixed number or one per frequency.
    sigma : float, defaults to None
        It controls the width of the wavelet ie its temporal
        resolution. If sigma is None the temporal resolution
        is adapted with the frequency like for all wavelet transform.
        The higher the frequency the shorter is the wavelet.
        If sigma is fixed the temporal resolution is fixed
        like for the short time Fourier transform and the number
        of oscillations increases with the frequency.
    zero_mean : bool, defaults to False
        Make sure the wavelet has a mean of zero.

    Returns
    -------
    Ws : list of array
        The wavelets time series.
    """
    Ws = list()
    n_cycles = np.atleast_1d(n_cycles)

    if (n_cycles.size != 1) and (n_cycles.size != len(freqs)):
        raise ValueError("n_cycles should be fixed or defined for "
                         "each frequency.")
    for k, f in enumerate(freqs):
        if len(n_cycles) != 1:
            this_n_cycles = n_cycles[k]
        else:
            this_n_cycles = n_cycles[0]
        # fixed or scale-dependent window
        if sigma is None:
            sigma_t = this_n_cycles / (2.0 * np.pi * f)
        else:
            sigma_t = this_n_cycles / (2.0 * np.pi * sigma)
        # this scaling factor is proportional to (Tallon-Baudry 98):
        # (sigma_t*sqrt(pi))^(-1/2);
        t = np.arange(0., 5. * sigma_t, 1.0 / sfreq)
        t = np.r_[-t[::-1], t[1:]]
        oscillation = np.exp(2.0 * 1j * np.pi * f * t)
        gaussian_enveloppe = np.exp(-t ** 2 / (2.0 * sigma_t ** 2))
        if zero_mean:  # to make it zero mean
            real_offset = np.exp(- 2 * (np.pi * f * sigma_t) ** 2)
            oscillation -= real_offset
        W = oscillation * gaussian_enveloppe
        W /= sqrt(0.5) * linalg.norm(W.ravel())
        Ws.append(W)
    return Ws


def cwt(X, Ws, use_fft=True, mode='same', decim=1):
    """Compute time freq decomposition with continuous wavelet transform.

    Parameters
    ----------
    X : array, shape (n_signals, n_times)
        The signals.
    Ws : list of array
        Wavelets time series.
    use_fft : bool
        Use FFT for convolutions. Defaults to True.
    mode : 'same' | 'valid' | 'full'
        Convention for convolution. 'full' is currently not implemented with
        `use_fft=False`. Defaults to 'same'.
    decim : int | slice
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note:: Decimation may create aliasing artifacts.

        Defaults to 1.

    Returns
    -------
    tfr : array, shape (n_signals, n_frequencies, n_times)
        The time-frequency decompositions.

    See Also
    --------
    mne.time_frequency.cwt_morlet : Compute time-frequency decomposition
                                    with Morlet wavelets
    """
    decim = _check_decim(decim)
    n_signals, n_times = X[:, decim].shape

    coefs = _cwt(X, Ws, mode, decim=decim, use_fft=use_fft)

    tfrs = np.empty((n_signals, len(Ws), n_times), dtype=np.complex)
    for k, tfr in enumerate(coefs):
        tfrs[k] = tfr

    return tfrs


def _cwt(X, Ws, mode="same", decim=1, use_fft=True):
    """Compute cwt with fft based convolutions or temporal convolutions.

    Parameters
    ----------
    X : array of shape (n_signals, n_times)
        The data.
    Ws : list of array
        Wavelets time series.
    mode : {'full', 'valid', 'same'}
        See numpy.convolve.
    decim : int | slice, defaults to 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note:: Decimation may create aliasing artifacts.

    use_fft : bool, defaults to True
        Use the FFT for convolutions or not.

    Returns
    -------
    out : array, shape (n_signals, n_freqs, n_time_decim)
        The time-frequency transform of the signals.
    """
    if mode not in ['same', 'valid', 'full']:
        raise ValueError("`mode` must be 'same', 'valid' or 'full', "
                         "got %s instead." % mode)
    if mode == 'full' and (not use_fft):
        # XXX JRK: full wavelet decomposition needs to be implemented
        raise ValueError('`full` decomposition with convolution is currently' +
                         ' not supported.')
    decim = _check_decim(decim)
    X = np.asarray(X)

    # Precompute wavelets for given frequency range to save time
    n_signals, n_times = X.shape
    n_times_out = X[:, decim].shape[1]
    n_freqs = len(Ws)

    Ws_max_size = max(W.size for W in Ws)
    size = n_times + Ws_max_size - 1
    # Always use 2**n-sized FFT
    fsize = 2 ** int(np.ceil(np.log2(size)))

    # precompute FFTs of Ws
    if use_fft:
        fft_Ws = np.empty((n_freqs, fsize), dtype=np.complex128)
    for i, W in enumerate(Ws):
        if len(W) > n_times:
            raise ValueError('At least one of the wavelets is longer than the '
                             'signal. Use a longer signal or shorter '
                             'wavelets.')
        if use_fft:
            fft_Ws[i] = fft(W, fsize)

    # Make generator looping across signals
    tfr = np.zeros((n_freqs, n_times_out), dtype=np.complex128)
    for x in X:
        if use_fft:
            fft_x = fft(x, fsize)

        # Loop across wavelets
        for ii, W in enumerate(Ws):
            if use_fft:
                ret = ifft(fft_x * fft_Ws[ii])[:n_times + W.size - 1]
            else:
                ret = np.convolve(x, W, mode=mode)

            # Center and decimate decomposition
            if mode == "valid":
                sz = int(abs(W.size - n_times)) + 1
                offset = (n_times - sz) // 2
                this_slice = slice(offset // decim.step,
                                   (offset + sz) // decim.step)
                if use_fft:
                    ret = _centered(ret, sz)
                tfr[ii, this_slice] = ret[decim]
            else:
                if use_fft:
                    ret = _centered(ret, n_times)
                tfr[ii, :] = ret[decim]
        yield tfr


def _check_decim(decim):
    """Aux function checking the decim parameter."""
    if isinstance(decim, int):
        decim = slice(None, None, decim)
    elif not isinstance(decim, slice):
        raise(TypeError, '`decim` must be int or slice, got %s instead'
                         % type(decim))
    return decim


def _centered(arr, newsize):
    """Aux Function to center data."""
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def shift_kernel(filt_kern, x_kernel, x_shift, sfreq):
    """Shift kernel (array) by specified latency. Keep array length, pad with zeros.

    Parameters
    ----------
    filt_kern : array of float
        Kernel to be shifted.
    x_kernel : array of float
        latencies of kernel (ms).
    x_shift: array of float
        Latency by which to shift kernel (ms).
    sfreq: float
        sampling frequency (Hz)

    Returns
    -------
    Ws : array of float
        Shifted kernel, same length as filt_kern (padded with zeros if necessary)
    """

    n = filt_kern.shape

    shift_kern = np.zeros(n)

    x_kernel2 = x_kernel - x_shift # latencies of shifted kernel

    # find overlapping elements between kernels
    discr = 1000./(2*sfreq) # discrepancy allowed between sample points in x and t

    k_idx = [] # record pairs of matching indices between shifted (sh) and oritinal (k) kernel
    sh_idx = []
    for [idx_k,tt] in enumerate(x_kernel):
        idx_sh = np.argmin(np.abs(x_kernel2-tt))
        d = np.abs(x_kernel2[idx_sh]-tt)
        if (d<=discr):
            k_idx.append(idx_k)
            sh_idx.append(idx_sh)

    shift_kern[sh_idx] = filt_kern[k_idx]

    return shift_kern


# # random signal
# x = np.linspace(0,2000,2000)
# n = x.shape[0]

# data = np.zeros([1,n])
# # data[0,np.round(n/2)] = 1 # peak
# f = 100
# # data[0,] = np.sin(x*2*np.pi*f/1000)
# data[0,] = np.sin((x**2/10)*2*np.pi/1000)

# freqs = np.arange(5,500,1)
# n_cycles = np.zeros(freqs.shape)
# n_cycles[freqs<=5] = 2
# n_cycles[(freqs>5) & (freqs<=20)] = 3
# n_cycles[freqs>20] = 7

# Ws = morlet_mne(sfreq=1000., freqs=freqs, n_cycles=n_cycles, zero_mean=True)

# out = cwt(data, Ws)

# plt.ion()
# plt.imshow(np.abs(out[0,:,:]), aspect='auto')