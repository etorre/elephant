# -*- coding: utf-8 -*-
"""
docstring goes here.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function

import numpy as np
import quantities as pq
import scipy.stats
import neo
import neo.core
import warnings
import conversion


def isi(spiketrain, axis=-1):
    """
    Return an array containing the inter-spike intervals of the SpikeTrain.

    Accepts a Neo SpikeTrain, a Quantity array, or a plain NumPy array.
    If either a SpikeTrain or Quantity array is provided, the return value will
    be a quantities array, otherwise a plain NumPy array. The units of
    the quantities array will be the same as spiketrain.

    Parameters
    ----------

    spiketrain : Neo SpikeTrain or Quantity array or NumPy ndarray
                 The spike times.
    axis : int, optional
           The axis along which the difference is taken.
           Default is the last axis.

    Returns
    -------

    NumPy array or quantities array.

    """
    if axis is None:
        axis = -1
    intervals = np.diff(spiketrain, axis=axis)
    if hasattr(spiketrain, 'waveforms'):
        intervals = pq.Quantity(intervals.magnitude, units=spiketrain.units)
    return intervals


def mean_firing_rate(spiketrain, t_start=None, t_stop=None, axis=None):
    """
    Return the firing rate of the SpikeTrain.

    Accepts a Neo SpikeTrain, a Quantity array, or a plain NumPy array.
    If either a SpikeTrain or Quantity array is provided, the return value will
    be a quantities array, otherwise a plain NumPy array. The units of
    the quantities array will be the inverse of the spiketrain.

    The interval over which the firing rate is calculated can be optionally
    controlled with `t_start` and `t_stop`

    Parameters
    ----------

    spiketrain : Neo SpikeTrain or Quantity array or NumPy ndarray
                 The spike times.
    t_start : float or Quantity scalar, optional
              The start time to use for the inveral.
              If not specified, retrieved from the``t_start`
              attribute of `spiketrain`.  If that is not present, default to
              `0`.  Any value from `spiketrain` below this value is ignored.
    t_stop : float or Quantity scalar, optional
             The stop time to use for the time points.
             If not specified, retrieved from the `t_stop`
             attribute of `spiketrain`.  If that is not present, default to
             the maximum value of `spiketrain`.  Any value from
             `spiketrain` above this value is ignored.
    axis : int, optional
           The axis over which to do the calculation.
           Default is `None`, do the calculation over the flattened array.

    Returns
    -------

    float, quantities scalar, NumPy array or quantities array.

    Notes
    -----

    If `spiketrain` is a Quantity or Neo SpikeTrain and `t_start` or `t_stop`
    are not, `t_start` and `t_stop` are assumed to have the same units as
    `spiketrain`.

    Raises
    ------

    TypeError
        If `spiketrain` is a NumPy array and `t_start` or `t_stop`
        is a quantity scalar.

    """
    if t_start is None:
        t_start = getattr(spiketrain, 't_start', 0)

    found_t_start = False
    if t_stop is None:
        if hasattr(spiketrain, 't_stop'):
            t_stop = spiketrain.t_stop
        else:
            t_stop = np.max(spiketrain, axis=axis)
            found_t_start = True

    # figure out what units, if any, we are dealing with
    if hasattr(spiketrain, 'units'):
        units = spiketrain.units
    else:
        units = None

    # convert everything to the same units
    if hasattr(t_start, 'units'):
        if units is None:
            raise TypeError('t_start cannot be a Quantity if '
                            'spiketrain is not a quantity')
        t_start = t_start.rescale(units)
    elif units is not None:
        t_start = pq.Quantity(t_start, units=units)
    if hasattr(t_stop, 'units'):
        if units is None:
            raise TypeError('t_stop cannot be a Quantity if '
                            'spiketrain is not a quantity')
        t_stop = t_stop.rescale(units)
    elif units is not None:
        t_stop = pq.Quantity(t_stop, units=units)

    if not axis or not found_t_start:
        return np.sum((spiketrain >= t_start) & (spiketrain <= t_stop),
                      axis=axis) / (t_stop-t_start)
    else:
        # this is needed to handle broadcasting between spiketrain and t_stop
        t_stop_test = np.expand_dims(t_stop, axis)
        return np.sum((spiketrain >= t_start) & (spiketrain <= t_stop_test),
                      axis=axis) / (t_stop-t_start)


# we make `cv` an alias for scipy.stats.variation for the convenience
# of former NeuroTools users
##cv = scipy.stats.variation


# Define function to compute FF in one window only
def ff(x):
    '''
    Evaluates the empirical Fano Factor (FF) of the spike counts of
    a list of spike trains.

    Given the vector v containing the observed spike counts (one per
    spike train) in the time window [t0, t1], the FF in [t0, t1] is:

                        FF := var(v)/mean(v).

    The FF is usually computed for spike trains representing the activity
    of the same neuron over different trials. The higher the FF, the larger
    the cross-trial non-stationarity.
    For a time-stationary Poisson process, the theoretical FF is 1.

    Parameters
    ----------
    x : list of SpikeTrain
        a list of spike trains for which to compute the FF of spike counts.

    Returns
    -------
    float
        the Fano Factor of the spike counts of the input spike trains
    '''
    # Build array of spike counts (one per spike train)
    x_counts = np.array([len(t) for t in x])

    # Compute FF
    if all([count == 0 for count in x_counts]):
        ff = 0
    else:
        ff = x_counts.var() / x_counts.mean()

    return ff


def ff_timeresolved(x, win=None, start=None, stop=None, step=None):
    '''
    Evaluates the empirical Fano Factor (FF) of the spike counts of
    a list of spike trains.
    By default computes the FF over the full time span of the data.
    However, it can compute the FF time-resolved as well.

    Given the vector v containing the observed spike counts (one per
    spike train) in the time window [t0, t1], the FF in [t0, t1] is:

                        FF := var(v)/mean(v).

    The FF is usually computed for spike trains representing the activity
    of the same neuron over different trials. The higher the FF, the larger
    the cross-trial non-stationarity.
    For a time-stationary Poisson process, the theoretical FF is 1.

    Parameters
    ----------
    x : list of SpikeTrain
        a list of spike trains for which to compute the FF of spike counts.
    win : Quantity or None (optional)
        Length of each time window over which to compute the FF.
        If None, the FF is computed over the largest window possible;
        otherwise, the window slides along time (see parameter step).
        Default: None
    start : Quantity or None (optional)
        starting time for the computation of the FF. If None, the largest
        t_start among those of the spike trains in x is used.
        Default: None
    stop : Quantity or None (optional)
        ending time for the computation of the FF. If None, the smallest
        t_stop among those of the spike trains in x is used.
        Default: None
    step : Quantity or None (optional)
        time shift between two consecutive sliding windows. If None,
        successive windows are adjacent.
        Default: None

    Returns
    -------
    values: array
        array of FF values computed over consecutive time windows
    windows: array of shape (..., 2)
        array of time windows over which the  FF has been computed

    '''

    # Compute max(t_start) and min(t_stop) and check consistency
    max_tstart = min([t.t_start for t in x])
    min_tstop = max([t.t_stop for t in x])

    if not (all([max_tstart == t.t_start for t in x]) and
        all([min_tstop == t.t_stop for t in x])):
        warnings.warning('spike trains have different t_start or t_stop'
            ' values. FF computed for inner values only')

    # Set start, stop, window length and step for the default cases
    t_start = max_tstart if start == None else start
    t_stop = min_tstop if stop == None else stop
    wlen = t_stop - t_start if win == None else win
    wstep = wlen if step == None else step

    # Convert all time quantities in dimensionless (_dl) units (meant in s)
    start_dl = float(t_start.simplified.base)
    stop_dl = float(t_stop.simplified.base)
    wlen_dl = float(wlen.simplified.base)
    step_dl = float(wstep.simplified.base)

    # Define the centers of the sliding windows where the FF must be computed
    ff_times = np.arange(wlen_dl / 2. + start_dl,
        stop_dl - wlen_dl / 2. + step_dl / 2, step_dl)

    # Define the windows within which the FF must be computed (as Nx2 array)
    windows = pq.s * np.array([np.max([ff_times - wlen_dl / 2.,
        start_dl * np.ones(len(ff_times))], axis=0), np.min([ff_times \
        + wlen_dl / 2., stop_dl * np.ones(len(ff_times))], axis=0)]).T
    windows = windows.rescale(x[0].units)

    # Compute the FF in each window define above
    ff_values = np.zeros(len(ff_times))
    for i, w in enumerate(windows):
        x_sliced = [t.time_slice(w[0], w[1]) for t in x]
        ff_values[i] = ff(x_sliced)

    return ff_values, windows


def isi_pdf(x, bins=10, range=None, density=False):
    '''
    Evaluate the empirical inter-spike-interval (ISI) probability density
    function (pdf) from a list of spike trains.

    Parameters:
    ----------
    x : neo.SpikeTrain or list of neo.SpikeTrain
        one or more spike trains for which to compute the ISI pdf

    bins : int or time Quantity. Optional, default is 10
        If int, number of bins of the pdf histogram.
        If single-value time Quantity, width of each time bin.

    range : Quantity array or None. Optional, default is None
        range (in time unit) over which to compute the histogram:
        * if None (default) computes the histogram in the full range
        * if pair of Quantities [r0, r1], ignores ISIs outside the range. r0
          or r1 can also be None (min and max ISI used instead, respectively)

    density : bool. Optional, default is False
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the pdf at the bin, normalized
        such that the *integral* over the range is 1.
        Note that the sum of the histogram values will not be equal to 1
        unless bins of unity width are chosen.

    Returns
    -------
    AnalogSignal:
        an analog signal containing the values of the ISI distribution.
        AnalogSignal[j] represents the ISI's pdf computed between
        t_start + j * sampling_period and t_start + (j + 1) * sampling_period

    '''

    # Convert x to a list if it is a SpikeTrain
    if type(x) == neo.core.SpikeTrain:
        x = [x]

    # Collect all ISIs from all SpikeTrains in x (as dimensionless array,
    # meant in seconds)
    ISIs = []
    for st in x:
        ISIs = np.hstack([ISIs, np.diff(st.simplified.magnitude)])

    # Set histogram range [isi_min, isi_max]
    if range == None:
        isi_min, isi_max = min(ISIs), max(ISIs)
    elif len(range) == 2:
        if range[0] == None:
            isi_min = min(ISIs)
        else:
            try:
                isi_min = range[0].rescale('s').magnitude
            except:
                raise ValueError('range[0] must be a time Quantity')
        if range[1] == None:
            isi_max = max(ISIs)
        else:
            try:
                isi_max = range[1].rescale('s').magnitude
            except:
                raise ValueError('range[1] must be a time Quantity')
    else:
        raise ValueError('range can only be None or sequence of length two')

    # If bins is a Quantity, convert it to a dimensionless array
    # of bin edges (meant in seconds)
    if type(bins) == pq.quantity.Quantity:
        binsize = bins.simplified.magnitude
        # If bins has length 1, interpret it as binsize and create bin array.
        # Otherwise return error
        if binsize.ndim == 0:
            bins = np.arange(isi_min, isi_max + binsize / 2, binsize)
        else:
            raise ValueError(
                'bins can be either int or single-value Quantity. Quantity '
                'array of shape ' + bins.shape + ' given instead')

    vals, edges = np.histogram(ISIs, bins, density=density)

    # Add unit (inverse of time) to the histogram values if normalized
    if density == True:
        vals = (vals / pq.s).rescale(1. / (x[0].units))
    else:
        vals = vals * pq.dimensionless

    # Rescale edges to the 1st spike train's unit and compute bin size
    edges = (edges * pq.s).rescale(x[0].units)
    w = edges[1] - edges[0]

    # Return histogram values and bins; the latter are r
    return neo.AnalogSignal(signal=vals, sampling_period=w, t_start=edges[0])


def cv(x):
    '''
    Evaluate the empirical coefficient of variation (CV) of the inter-spike
    intervals (ISIs) of one spike train or a list of spike trains.

    Given the vector v containing the observed ISIs of one spike train,
    the CV is defined as
                    CV := std(v)/mean(v).
    The CV of a list of spike trains is computed collecting the ISIs of all
    spike trains.

    The CV represents a measure of irregularity in the spiking activity. For
    For a time-stationary Poisson process, the theoretical CV is 1.

    Arguments
    ---------
    x: SpikeTrain or list of SpikeTrains
        a neo.SpikeTrain object (or a list of), for which to compute the CV.

    Returns
    -------
    float
        the CV of all (ISIs) in the input SpikeTrain(s).  If no ISI can be
        calculated (less than 2 spikes in all SpikeTrains) the CV is 0.
    '''

    # Convert x to a list if it is a SpikeTrain
    if type(x) == neo.core.SpikeTrain:
        x = [x]

    # Collect the ISIs of all trains in x, and return their CV
    isis = np.array([])
    for st in x:
        if len(st) > 1:
            isis = np.hstack([isis, np.diff(st.simplified.base)])

    # Compute CV of ISIs
    CV = 0. if len(isis) == 0 else isis.std() / isis.mean()

    return CV


def cv_timeresolved(x, win=None, start=None, stop=None, step=None):
    '''
    Evaluate the empirical coefficient of variation (CV) of the inter-spike
    intervals (ISIs) of one spike train (or a list of spike trains).
    By default computes the CV over the full time span of the data. However,
    it can compute the CV time-resolved as well.

    Given the vector v containing the observed ISIs of one spike train in
    the time window [t0, t1], the CV in [t0, t1] is defined as
                    CV := std(v)/mean(v).
    The CV of a list of spike trains is computed collecting the ISIs of all
    spike trains.

    The CV represents a measure of irregularity in the spiking activity. For
    For a time-stationary Poisson process, the theoretical CV is 1.

    Arguments
    ---------
    x: SpikeTrain or list of SpikeTrains
        a neo.SpikeTrain object (or a list of), for which to compute the CV.
    win: Quantity (optional)
        the length of the time windows over which to compute the CV.
        If None, the CV is computed over the largest window possible;
        otherwise, the window slides along time (see argument 'step')
        Default: None
    start Quantity (optional)
        initial time for the computation of the CV. If None, the largest
        t_start among those of the input spike trains in x is used.
        Default: None
    stop: Quantity (optional)
        last time for the computation of the CV. If None, the smallest
        t_stop among those of the input spike trains in x is used.
        Default: None
    step: Quantity (optional)
        time shift between two consecutive sliding windows. If None,
        successive windows are adjacent.
        Default: None

    Returns
    -------
    values: array
        array of CV values computed over consecutive time windows
    windows: array
        nx2 array of time windows over which the CV has been computed

    '''

    # Convert x to a list if it is a SpikeTrain
    if type(x) == neo.core.SpikeTrain:
        x = [x]

    max_tstart = min([t.t_start for t in x])
    min_tstop = max([t.t_stop for t in x])

    if not (all([max_tstart == t.t_start for t in x]) and
        all([min_tstop == t.t_stop for t in x])):
        warnings.warning('spike trains have different t_start or t_stop'
        ' values. CV computed for inner values only')

    t_start = max_tstart if start == None else start
    t_stop = min_tstop if stop == None else stop
    wlen = t_stop - t_start if win == None else win
    wstep = wlen if step == None else step

    # Convert all time quantities in dimensionless (_dl) units (meant in s)
    start_dl = float(t_start.simplified.base)
    stop_dl = float(t_stop.simplified.base)
    wlen_dl = float(wlen.simplified.base)
    step_dl = float(wstep.simplified.base)

    # Define the centers of the sliding windows where the CV must be computed
    cv_times = np.arange(wlen_dl / 2. + start_dl,
        stop_dl - wlen_dl / 2. + step_dl / 2, step_dl)

    # Define the nx2 array of time windows within which to compute the CV
    windows = pq.s * np.array([np.max([cv_times - wlen_dl / 2.,
        start_dl * np.ones(len(cv_times))], axis=0), np.min([cv_times +
        wlen_dl / 2., stop_dl * np.ones(len(cv_times))], axis=0)]).T

    # Compute the CV in each window defined above
    cv_values = np.zeros(len(cv_times))  # Initialize CV values to 0
    for i, w in enumerate(windows):
        x_sliced = [t.time_slice(w[0], w[1]) for t in x]
        cv_values[i] = cv(x_sliced)

    return cv_values, windows


def peth(sts, w, t_start=None, t_stop=None, output='counts'):
    '''
    Peri-Event Time Histogram (PETH) of a list of spike trains.

    Parameters
    ----------
    sts : list of SpikeTrain
        spike trains with a common time axis (same t_start and t_stop)
    w : Quantity
        width of the histogram's time bins.
    t_start, t_stop : Quantity (optional)
        Start and stop time of the histogram. Only events in the input
        spike trains falling between t_start and t_stop (both included) are
        considered in the histogram. If t_start and/or t_stop are not
        specified, the maximum t_start of all Spiketrains is used as t_start,
        and the minimum t_stop is used as t_stop.
        Default: t_start=t_stop=None
    output : str (optional)
        Normalization of the histogram. Can be one of:
        * 'counts': spike counts at each bin (as integer numbers)
        * 'mean': mean spike counts per spike train
        * 'rate': mean spike rate per spike train. Like 'mean', but the
          counts are additionally normalized by the bin width.

    Returns
    -------
    AnalogSignal
        analog signal containing the PETH values. AnalogSignal[j] is the
        PETH computed between t_start + j * w and t_start + (j + 1) * w.

    '''
    max_tstart = max([t.t_start for t in sts])
    min_tstop = min([t.t_stop for t in sts])

    if t_start is None:
        t_start = max_tstart
        if not all([max_tstart == t.t_start for t in sts]):
            warnings.warn(
                "Spiketrains have different t_start values -- "
                "using maximum t_start as t_start.")

    if t_stop is None:
        t_stop = min_tstop
        if not all([min_tstop == t.t_stop for t in sts]):
            warnings.warn(
                "Spiketrains have different t_stop values -- "
                "using minimum t_stop as t_stop.")

    sts_cut = [st.time_slice(t_start=t_start, t_stop=t_stop) for st in sts]

    bs = conversion.binned_st(
        sts_cut, t_start=t_start, t_stop=t_stop, binsize=w)
    bin_hist = np.sum(bs.matrix_clipped(), axis=0)

    if output == 'counts':
        # Raw
        bin_hist = bin_hist * pq.dimensionless
    elif output == 'mean':
        # Divide by number of input spike trains
        bin_hist = bin_hist * 1. / len(sts) * pq.dimensionless
    elif output == 'rate':
        # Divide by number of input spike trains and bin width
        bin_hist = bin_hist * 1. / len(sts) / w
    else:
        raise ValueError('Parameter output is not valid.')

    return neo.AnalogSignal(
        signal=bin_hist, sampling_period=w, t_start=t_start)


def fanofactor(spiketrains):
    """
    Evaluates the empirical Fano factor F of the spike counts of
    a list of `neo.core.SpikeTrain` objects.

    Given the vector v containing the observed spike counts (one per
    spike train) in the time window [t0, t1], F is defined as:

                        F := var(v)/mean(v).

    The Fano factor is typically computed for spike trains representing the
    activity of the same neuron over different trials. The higher F, the larger
    the cross-trial non-stationarity. In theory for a time-stationary Poisson
    process, F=1.

    Parameters
    ----------
    spiketrains : list of neo.core.SpikeTrain objects, quantity array,
                  numpy array or list
        Spike trains for which to compute the Fano factor of spike counts.

    Returns
    -------
    fano : float or nan
        The Fano factor of the spike counts of the input spike trains. If an
        empty list is specified, or if all spike trains are empty, F:=nan.
    """
    # Build array of spike counts (one per spike train)
    spike_counts = np.array([len(t) for t in spiketrains])

    # Compute FF
    if all([count == 0 for count in spike_counts]):
        fano = np.nan
    else:
        fano = spike_counts.var() / spike_counts.mean()
    return fano
