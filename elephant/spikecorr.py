# -*- coding: utf-8 -*-
"""
Spike train correlation

This modules provides functions to calculate correlations between spike trains.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division
import numpy as np
import quantities as pq
import neo
import elephant.conversion as rep


def cch(
        st1, st2, window=None, normalize=False, border_correction=False,
        binary=False, kernel=None):
    """
    Computes the cross-correlation histogram (CCH) between two binned spike
    trains st1 and st2.

    Parameters
    ----------
    st1,st2 : BinnedSpikeTrain
        Binned spike trains to cross-correlate.
    window : int or None (optional)
        histogram half-length. If specified, the cross-correlation histogram
        has a number of bins equal to 2*window+1 (up to the maximum length).
        If not specified, the full crosscorrelogram is returned
        Default: None
    normalize : bool (optional)
        whether to normalize the central value (corresponding to time lag
        0 s) to 1; the other values are rescaled accordingly.
        Default: False
    border_correction : bool (optional)
        whether to correct for the border effect. If True, the value of the
        CCH at bin b (for b=-H,-H+1, ...,H, where H is the CCH half-length)
        is multiplied by the correction factor:
                            (H+1)/(H+1-|b|),
        which linearly corrects for loss of bins at the edges.
        Default: False
    binary : bool (optional)
        whether to binary spikes from the same spike train falling in the
        same bin. If True, such spikes are considered as a single spike;
        otherwise they are considered as different spikes.
        Default: False.
    kernel : array or None (optional)
        A one dimensional array containing an optional smoothing kernel applied
        to the resulting CCH. The length N of the kernel indicates the
        smoothing window. The smoothing window cannot be larger than the
        maximum lag of the CCH. The kernel is normalized to unit area before
        being applied to the resulting CCH. Popular choices for the kernel are
          * normalized boxcar kernel: numpy.ones(N)
          * hamming: numpy.hamming(N)
          * hanning: numpy.hanning(N)
          * bartlett: numpy.bartlett(N)
        If None is specified, the CCH is not smoothed
        Default: None

   Returns
   -------
      returns the cross-correlation histogram between st1 and st2. The central
      bin of the histogram represents correlation at zero delay. Offset bins
      correspond to correlations at a delay equivalent to the difference
      between the spike times of st1 and those of st2: an entry at positive
      lags corresponds to a spike in st2 following a spike in st1 bins to the
      right, and an entry at negative lags corresponds to a spike in st1
      following a spike in st2.
      To illustrate, consider the two spike trains:
      st1: 0 0 0 0 1 0 0 0 0 0 0
      st2: 0 0 0 0 0 0 0 1 0 0 0
      Here, the CCH will have an entry of 1 at lag h=+3.

    Example
    -------
    TODO: make example!

    TODO:
    * output as AnalogSignal
    * make function faster?
    * more unit tests
    * variable renaming?
    * doc string completion
    *
    """
    # Retrieve unclipped matrix
    st1_spmat = st1.to_sparse_array()
    st2_spmat = st2.to_sparse_array()

    # For each row, extract the nonzero column indices and the corresponding
    # data in the matrix (for performance reasons)
    st1_bin_idx_unique = st1_spmat.nonzero()[1]
    st2_bin_idx_unique = st2_spmat.nonzero()[1]
    if binary:
        st1_bin_counts_unique = np.array(st1_spmat.data > 0, dtype=int)
        st2_bin_counts_unique = np.array(st2_spmat.data > 0, dtype=int)
    else:
        st1_bin_counts_unique = st1_spmat.data
        st2_bin_counts_unique = st2_spmat.data

    # Define the half-length of the full crosscorrelogram.
    Len = st1.num_bins + st2.num_bins - 1
    Hlen = Len // 2
    Hbins = Hlen if window is None else min(window, Hlen)

    # Initialize the counts to an array of zeroes, and the bin ids to
    counts = np.zeros(2 * Hbins + 1)
    bin_ids = np.arange(-Hbins, Hbins + 1)

    # Compute the CCH at lags in -Hbins,...,Hbins only
    for r, i in enumerate(st1_bin_idx_unique):
        timediff = st2_bin_idx_unique - i
        timediff_in_range = np.all(
            [timediff >= -Hbins, timediff <= Hbins], axis=0)
        timediff = (timediff[timediff_in_range]).reshape((-1,))
        counts[timediff + Hbins] += st1_bin_counts_unique[r] * \
            st2_bin_counts_unique[timediff_in_range]

    # Correct the values taking into account lacking contributes at the edges
    if border_correction is True:
        correction = float(Hlen + 1) / np.array(
            Hlen + 1 - abs(np.arange(-Hlen, Hlen + 1)), float)
        counts = counts * correction

    # Define the kernel for smoothing as an ndarray
    if hasattr(kernel, '__iter__'):
        if len(kernel) > Len:
            raise ValueError(
                'The length of the kernel cannot be larger than the '
                'length %d of the resulting CCH.' % Len)
        kernel = np.array(kernel, dtype=float)
        kernel = 1. * kernel / sum(kernel)
    elif kernel is not None:
        raise ValueError('Invalid smoothing kernel.')

    # Smooth the cross-correlation histogram with the kernel
    if kernel is not None:
        counts = np.convolve(counts, kernel, mode='same')

    # Rescale the histogram so that the central bin has height 1, if requested
    if normalize:
        counts = np.array(counts, float) / float(counts[Hlen])

    # Return only the Hbins bins and counts before and after the central one
    return counts, bin_ids


def ccht(x, y, w, window=None, start=None, stop=None, corrected=False,
         smooth=None, clip=False, normed=False, xaxis='time', kernel='boxcar'):
    """
    Computes the cross-correlation histogram (CCH) between two spike trains.

    Given a reference spike train x and a target spike train y, their CCH
    at time lag t is computed as the number of spike pairs (s1, s2), with s1
    from x and s2 from y, such that s2 follows s1 by a time lag \tau in the
    range - or time bin - [t, t+w):

        CCH(\tau; x,y) := #{(s1,s2) \in x \times y:  t= < s2-s1 < t+w}

    Therefore, the times associated to the CCH are the left ends of the
    corresponding time bins.

    Note: CCH(\tau; x,y) = CCH(-\tau; y,x).

    This routine computes CCH(x,y) at the times
                     ..., -1.5*w, -0.5*w, 0.5*w, ...
    corresponding to the time bins
            ..., [-1.5*w, -0.5*w), [-0.5*w, 0.5*w), [0.5*w, 1.5*w), ...
    The second one is the central bin, corresponding to synchronous spiking
    of x and y.

    Parameters
    ----------
    x,y : neo.SpikeTrains or lists of neo.SpikeTrains
        If x and y are both SpikeTrains, computes the CCH between them.
        If x and y are both lists of SpikeTrains (with same length, say l),
        computes for each i=1,2,...,l the CCH between x[i] and y[i] , and
        returns their average CCH.
        All input SpikeTrains must have same t_start and t_stop.
    w : Quantity
        time width of the CCH time bin.
    lag : Quantity (optional).
        positive time, specifying the range of the CCH: the CCH is computed
        in [-lag, lag]. This interval is automatically extended to contain
        an integer, odd number of bins. If None, the range extends to
        the maximum lag possible, i.e.[start-stop, stop-start].
        Default: None
    start, stop : Quantities (optional)
        If not None, the CCH is computed considering only spikes from x and y
        in the range [start, stop]. Spikes outside this range are ignored.
        If start (stop) is None, x.t_start (x.t_stop) is used instead
        Default: None
    corrected : bool (optional)
        whether to correct for the border effect. If True, the value of the
        CCH at bin b (for b=-H,-H+1, ...,H, where H is the CCH half-length)
        is multiplied by the correction factor:
                            (H+1)/(H+1-|b|),
        which linearly corrects for loss of bins at the edges.
        Default: False
    smooth : Quantity or None (optional)
        if smooth is a positive time, each bin in the raw cross-correlogram
        is averaged over a window (-smooth/2, +smooth/2) with the values in
        the neighbouring bins. If smooth <= w, no smoothing is performed.
        Default: None
    clip : bool (optional)
        whether to clip spikes from the same spike train falling in the
        same bin. If True, such spikes are considered as a single spike;
        otherwise they are considered as different spikes.
        Default: False.
    normed : bool (optional)
        whether to normalize the central value (corresponding to time lag
        0 s) to 1; the other values are rescaled accordingly.
        Default: False
    kernel : str or array (optional)
        kernel used for smoothing (see parameter smooth above). Can be:
        * list or array of floats defining the kernel weights
        * one of the following strings:
          * 'boxcar' : normalized boxcar window;
          * 'hamming': normalized hamming window;
          * 'hanning': normalized hanning window;
          * 'bartlett': normalized bartlett window;
        Default: 'boxcar'
    xaxis : str (optional)
        whether to return the times or the bin ids as the first output.
        Can be one of:
        * 'time' (default): returns the actul times of the cch_all_pairs
        * 'ids': returns the bin ids of the cch_all_pairs.
        Default: 'time'

    Returns
    -------
    counts : array of float
        array of lagged coincidences counts, representing the number of
        spike pairs (t1, t2), with t1 from x and t2 from y, such that
        t2-t1 lies in a given range (possibly smoothed or normalized).
    times : Quantity array or array of floats
        array of spike times (or array of bin ids) associated to the
        calculated counts in CCH.

    Example
    -------
    >>> import neo, quantities as pq
    >>> st1 = neo.SpikeTrain([1.2, 3.5, 8.7, 10.1] * pq.ms, t_stop=15*pq.ms)
    >>> st2 = neo.SpikeTrain([1.9, 5.2, 8.4] * pq.ms, t_stop=15*pq.ms)
    >>> print ccht(st1, st2, w=3*pq.ms)
    (array([ 0.,  0.,  1.,  2.,  3.,  3.,  2.,  1.,  0.]),
     array([-12.,  -9.,  -6.,  -3.,   0.,   3.,   6.,   9.,  12.]) * ms)
    >>> print ccht(st1, st2, w=3*pq.ms, window=3*pq.ms)
    (array([ 2.,  3.,  3.]), array([-3.,  0.,  3.]) * ms)
    >>> print ccht(st1, st2, w=3*pq.ms, window=3*pq.ms, xaxis='ids')
    (array([ 2.,  3.,  3.]), array([-1,  0,  1]))

    """

    # Raise errors if x.t_start != y.t_start or x.t_stop != y.t_stop
    if x.t_start != y.t_start:
        raise ValueError('x and y must have the same t_start attribute')
    if x.t_stop != y.t_stop:
        raise ValueError('x and y must have the same t_stop attribute')

    # Set start. Only spike times >= start will be considered for the CCH
    if start is None:
        start = x.t_start

    # Set stop to end of spike trains if None
    if stop is None:
        if len(x) * len(y) == 0:
            stop = 0 * x.units if window is None else 2 * window
        else:
            stop = x.t_stop if window is None else max(x.t_stop, window)

    # By default, set smoothing to 0 ms (no smoothing)
    if smooth is None:
        smooth = 0 * pq.ms

    # Set the window for the CCH
    win = (stop - start) if window is None else min(window, (stop - start))

    # Cut the spike trains, keeping the spikes between start and stop only
    x_cut = x if len(x) == 0 else x.time_slice(t_start=start, t_stop=stop)
    y_cut = y if len(y) == 0 else y.time_slice(t_start=start, t_stop=stop)

    # Bin the spike trains
    x_binned = rep.Binned(x_cut, t_start=start, t_stop=stop, binsize=w)
    y_binned = rep.Binned(y_cut, t_start=start, t_stop=stop, binsize=w)

    # Evaluate the CCH for the binned trains with cch()
    counts, bin_ids = cch(
        x_binned, y_binned, border_correction=corrected, binary=clip, normalize=normed,
        smooth=int((smooth / w).rescale(pq.dimensionless)), kernel=kernel,
        window=int((win / w).rescale(pq.dimensionless).magnitude))

    # Convert bin ids to times if the latter were requested
    if xaxis == 'time':
        bin_ids = bin_ids * w

    # Return the CCH and the bins used to compute it
    return counts, bin_ids


def cch_all_pairs(x, y, w, lag=None, start=None, stop=None, corrected=False,
                  smooth=None, clip=False, normed=False, kernel='boxcar'):
    """
    Computes the cross-correlation histogram (CCH) between two spike trains,
    or the average CCH between the spike trains in two spike train lists.

    Given a reference spike train x and a target spike train y, their CCH
    at time lag t is computed as the number of spike pairs (s1, s2), with s1
    from x and s2 from y, such that s2 follows s1 by a time lag \tau in the
    range - or time bin - [t, t+w):

        CCH(\tau; x,y) := #{(s1,s2) \in x \times y:  t <= s2-s1 < t+w}

    Therefore, the times associated to the CCH are the left ends of the
    corresponding time bins.

    Note: CCH(\tau; x,y) = CCH(-\tau; y,x).

    This routine computes CCH(x,y) at the times
                     ..., -1.5*w, -0.5*w, 0.5*w, ...
    corresponding to the time bins
            ..., [-1.5*w, -0.5*w), [-0.5*w, 0.5*w), [0.5*w, 1.5*w), ...
    The second one is the central bin, corresponding to synchronous spiking
    of x and y.

    Parameters
    ----------
    x,y : neo.SpikeTrains or lists of neo.SpikeTrains
        If x and y are both SpikeTrains, computes the CCH between them.
        If x and y are both lists of SpikeTrains (with same length, say l),
        computes for each i=1,2,...,l the CCH between x[i] and y[i] , and
        returns their average CCH.
        All input SpikeTrains must have same t_start and t_stop.
    w : Quantity
        time width of the CCH time bin.
    lag : Quantity (optional).
        positive time, specifying the range of the CCH: the CCH is computed
        in [-lag, lag]. This interval is automatically extended to contain
        an integer, odd number of bins. If None, the range extends to
        the maximum lag possible, i.e.[start-stop, stop-start].
        Default: None
    start, stop : Quantities (optional)
        If not None, the CCH is computed considering only spikes from x and y
        in the range [start, stop]. Spikes outside this range are ignored.
        If start (stop) is None, x.t_start (x.t_stop) is used instead
        Default: None
    corrected : bool (optional)
        whether to correct for the border effect. If True, the value of the
        CCH at bin b (for b=-H,-H+1, ...,H, where H is the CCH half-length)
        is multiplied by the correction factor:
                            (H+1)/(H+1-|b|),
        which linearly corrects for loss of bins at the edges.
        Default: False
    smooth : Quantity or None (optional)
        if smooth is a positive time, each bin in the raw cross-correlogram
        is averaged over a window (-smooth/2, +smooth/2) with the values in
        the neighbouring bins. If smooth <= w, no smoothing is performed.
        Default: None
    clip : bool (optional)
        whether to clip spikes from the same spike train falling in the
        same bin. If True, such spikes are considered as a single spike;
        otherwise they are considered as different spikes.
        Default: False.
    normed : bool (optional)
        whether to normalize the central value (corresponding to time lag
        0 s) to 1; the other values are rescaled accordingly.
        Default: False
    kernel : str or array (optional)
        kernel used for smoothing (see parameter smooth above). Can be:
        * list or array of floats defining the kernel weights
        * one of the following strings:
          * 'boxcar' : normalized boxcar window;
          * 'hamming': normalized hamming window;
          * 'hanning': normalized hanning window;
          * 'bartlett': normalized bartlett window;
        Default: 'boxcar'

    Returns
    -------
    AnalogSignal
        returns an analog signal, representing the CCH between the spike
        trains x and y at different time lags. AnalogSignal.times represents
        the left edges of the time bins. AnalogSignal.sampling_period
        represents the bin width used to compute the CCH.

    Example
    -------
    >>> import neo, quantities as pq
    >>> t1 = neo.SpikeTrain([1.2, 3.5, 8.7, 10.1] * pq.ms, t_stop=20*pq.ms)
    >>> t2 = neo.SpikeTrain([1.9, 5.2, 8.4] * pq.ms, t_stop=20*pq.ms)
    >>> CCH = cch_all_pairs(t1, t2, 3*pq.ms)
    >>> print CCH
    [ 0.  0.  0.  1.  2.  3.  3.  2.  1.  0.  0.] dimensionless
    >>> print CCH.times
    [-16.5 -13.5 -10.5  -7.5  -4.5  -1.5   1.5   4.5   7.5  10.5  13.5] ms
    """
    if isinstance(x, neo.core.SpikeTrain) and isinstance(y, neo.core.SpikeTrain):
        CCH, bins = ccht(x, y, w, window=lag, start=start, stop=stop,
                         corrected=corrected, smooth=smooth, clip=clip, normed=normed,
                         xaxis='time', kernel=kernel)

        if not isinstance(CCH, pq.Quantity):
            CCH = CCH * pq.dimensionless

        return neo.AnalogSignal(
            CCH, t_start=bins[0] - w / 2., sampling_period=w)

    else:
        CCH_exists = False
        for xx, yy in zip(x, y):
            if CCH_exists == False:
                CCH = cch_all_pairs(xx, yy, w, lag=lag, start=start, stop=stop,
                                    corrected=corrected, smooth=smooth, clip=clip,
                                    normed=normed, kernel=kernel)
                CCH_exists = True
            else:
                CCH += cch_all_pairs(xx, yy, w, lag=lag, start=start, stop=stop,
                                     corrected=corrected, smooth=smooth, clip=clip,
                                     normed=normed, kernel=kernel)
        CCH = CCH / float(len(x))

        return CCH


def cov(spiketrains, binsize, clip=True):
    '''
    Matrix of pairwise covariance coefficients for a list of spike trains.

    For each spike trains i,j in the list, the coavriance coefficient
    C[i, j] is given by the covariance between the vectors obtained by
    binning i and j at the desired bin size. Called b_i, b_j such vectors
    and m_i, m_j their respective averages:

                    C[i,j] = <b_i-m_i, b_j-m_j>

    where <.,.> is the scalar product of two vectors.
    If spiketrains is a list of n spike trains, a n x n matrix is returned.
    If clip is True, the spike trains are clipped before computing the
    covariance coefficients, so that the binned vectors b_i, b_j are binary.

    Parameters
    ----------
    spiketrains : list
        a list of SpikeTrains with same t_start and t_stop values
    binsize : Quantity
        the bin size used to bin the spike trains
    clip : bool, optional
        whether to clip spikes of the same spike train falling in the same
        bin (True) or not (False). If True, the binned spike trains are
        binary arrays

    Output
    ------
    M : ndarrray
        the square matrix of correlation coefficients. M[i,j] is the
        correlation coefficient between spiketrains[i] and spiketrains[j]

    '''

    # Check that all spike trains have same t_start and t_stop
    tstart_0 = spiketrains[0].t_start
    tstop_0 = spiketrains[0].t_stop
    assert(all([st.t_start == tstart_0 for st in spiketrains[1:]]))
    assert(all([st.t_stop == tstop_0 for st in spiketrains[1:]]))

    # Bin the spike trains
    t_start = spiketrains[0].t_start
    t_stop = spiketrains[0].t_stop
    binned_sts = rep.Binned(
        spiketrains, binsize=binsize, t_start=t_start, t_stop=t_stop)

    # Create the binary matrix M of binned spike trains
    if clip is True:
        M = binned_sts.matrix_clipped()
    else:
        M = binned_sts.matrix_unclipped()

    # Return the matrix of correlation coefficients
    return np.cov(M)


def ccht2(x, y, binsize, corrected=False, smooth=0, normed=False,
          xaxis='time', **kwargs):
    """

    .. note::
        Same as ccht() [in turn deprecated].
        Should be faster, but is slower...deprecated. Use cch_all_pairs() instead

    .. See also::
        cch_all_pairs()

    """
    import numpy as np

    # Convert the inputs to arrays
    if type(x) == list:
        x = np.array(x)
    if type(y) == list:
        y = np.array(y)
    if 'clip' not in kwargs.keys():
        clip = True
    else:
        clip = kwargs['clip']
    #
    #*************************************************************************
    # setting the starting and stopping times for the cch_all_pairs; cutting the spike trains accordingly
    #*************************************************************************
    if 'start' not in kwargs.keys():
        start = min(0, min(x), min(y))
    else:
        start = kwargs['start']
    if 'stop' not in kwargs.keys():
        stop = max(max(x), max(y))
    else:
        stop = kwargs['stop']
    x_cut = x[np.all((start <= x, x <= stop), axis=0)]
    y_cut = y[np.all((start <= y, y <= stop), axis=0)]
    #
    #*************************************************************************
    # binning the (cut) trains and computing the bin difference
    #*************************************************************************
    if clip == True:
        x_filledbins = np.unique(np.array((x_cut - start) / binsize, int))
        y_filledbins = np.unique(np.array((y_cut - start) / binsize, int))
    else:
        x_filledbins = np.array((x_cut - start) / binsize, int)
        y_filledbins = np.array((y_cut - start) / binsize, int)
    bindiff = np.concatenate([i - y_filledbins for i in x_filledbins], axis=0)
    #
    #*************************************************************************
    # computing the number of bins and initializing counts and bin ids. Cutting the bin difference
    #*************************************************************************
    if 'window' in kwargs.keys():
        win = min(kwargs['window'], (stop - start) / 2.)
    else:
        win = (stop - start) / 2.
    Hlen = min(int((stop - start) / (2 * binsize)),
               int(np.ceil((win + smooth / 2.) / binsize)))
    Len = 2 * Hlen + 1
    Hbins = min(int(win / binsize), Hlen)
    counts = np.zeros(Len)
    bin_ids = np.arange(-Hlen, Hlen + 1)
    bindiff_cut = bindiff[np.all((bindiff >= -Hlen, bindiff <= Hlen), axis=0)]
    #
    #*************************************************************************
    # computing the counts
    #*************************************************************************
    for i in bindiff_cut:
        counts[Hlen + i] += 1
    #
    #*************************************************************************
    # correcting, smoothing and normalizing the counts, if requested
    #*************************************************************************
    if corrected == True:
        correction = float(
            Hlen + 1) / np.array(Hlen + 1 - abs(np.arange(-Hlen, Hlen + 1)), float)
        counts = counts * correction
    if smooth > binsize:
        if 'kernel' not in kwargs.keys():
            kerneltype = 'boxcar'
        else:
            kerneltype = kwargs['kernel']
        smooth_Nbin = min(int(smooth / binsize), Len)
        if kerneltype == 'hamming':
            win = np.hamming(smooth_Nbin)
        elif kerneltype == 'bartlett':
            win = np.bartlett(smooth_Nbin)
        elif kerneltype == 'hanning':
            win = np.hanning(smooth_Nbin)
        elif kerneltype == 'boxcar':
            win = np.ones(smooth_Nbin)
        kernel = win / sum(win)
        counts = np.convolve(counts, kernel, mode='same')
    if normed == True:
        counts = np.array(counts, float) / float(counts[Hlen])
    #
    #*************************************************************************
    # returning the CCH; the first output is the array of bin ids if xaxis == 'binid', and the array
    # of times (for the bin centers) if xaxis == 'time'.
    #*************************************************************************
    if xaxis == 'time':
        return bin_ids[Hlen - Hbins:Hlen + Hbins + 1] * binsize + start, counts[Hlen - Hbins:Hlen + Hbins + 1]
    else:
        return bin_ids[Hlen - Hbins:Hlen + Hbins + 1], counts[Hlen - Hbins:Hlen + Hbins + 1]
