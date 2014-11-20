import numpy
import numpy as np
import neo
import warnings
import quantities as pq
import elephant.rep as rep


def fanofactor(spiketrains):
    """
    Evaluates the empirical Fano factor F of the spike counts of a list of
    `neo.core.SpikeTrain` objects.

    Given the vector v containing the observed spike counts (one per spike
    train) in the time window [t0, t1], F is defined as:

                        F := var(v)/mean(v).

    The Fano factor is typically computed for spike trains representing the
    activity of the same neuron over different trials. The higher F, the larger
    the cross-trial non-stationarity. In theory for a time-stationary Poisson
    process, F=1.

    Parameters
    ----------
    spiketrains : list of neo.core.SpikeTrain objects
        Spike trains for which to compute the F of spike counts.

    Returns
    -------
    fano : float
        The Fano factor of the spike counts of the input spike trains. If an
        empty list is specified, or if all spike trains are empty, F:=0.

    Remarks
    -------
    This routine does not check or warn if spike trains have the unequal length
    in time. However, be advised that the computation of F may not be sensible
    in such a situation.
    """
    # Build array of spike counts (one per spike train)
    spike_counts = numpy.array([len(t) for t in spiketrains])

    # Compute fano factor
    if all([count == 0 for count in spike_counts]):
        fano = 0.
    else:
        fano = spike_counts.var() / spike_counts.mean()

    return fano


def ff_timeresolved(x, win=None, start=None, stop=None, step=None):
    """
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

    """

    # Compute max(t_start) and min(t_stop) and check consistency
    max_tstart = min([t.t_start for t in x])
    min_tstop = max([t.t_stop for t in x])

    if not (all([max_tstart == t.t_start for t in x]) and
                all([min_tstop == t.t_stop for t in x])):
        warnings.warning('spike trains have different t_start or t_stop'
                         ' values. FF computed for inner values only')

    # Set start, stop, window length and step for the default cases
    t_start = max_tstart if start is None else start
    t_stop = min_tstop if stop is None else stop
    wlen = t_stop - t_start if win is None else win
    wstep = wlen if step is None else step

    # Convert all time quantities in dimensionless (_dl) units (meant in s)
    start_dl = float(t_start.simplified.base)
    stop_dl = float(t_stop.simplified.base)
    wlen_dl = float(wlen.simplified.base)
    step_dl = float(wstep.simplified.base)

    # Define the centers of the sliding windows where the FF must be computed
    ff_times = numpy.arange(wlen_dl / 2. + start_dl,
                            stop_dl - wlen_dl / 2. + step_dl / 2, step_dl)

    # Define the windows within which the FF must be computed (as Nx2 array)
    windows = pq.s * numpy.array([numpy.max([ff_times - wlen_dl / 2.,
                                             start_dl * numpy.ones(
                                                 len(ff_times))], axis=0),
                                  numpy.min([ff_times
                                             + wlen_dl / 2.,
                                             stop_dl * numpy.ones(
                                                 len(ff_times))], axis=0)]).T
    windows = windows.rescale(x[0].units)

    # Compute the FF in each window define above
    ff_values = numpy.zeros(len(ff_times))
    for i, w in enumerate(windows):
        x_sliced = [t.time_slice(w[0], w[1]) for t in x]
        ff_values[i] = fanofactor(x_sliced)

    return ff_values, windows


def isi_pdf(spiketrain, bins=10, rng=None, density=False):
    """
    Evaluate the empirical inter-spike-interval (ISI) probability density
    function (pdf) from a list of spike trains.

    Parameters:
    ----------
    spiketrain : neo.core.SpikeTrain or list of neo.core.SpikeTrain objects
        One or more spike trains for which to compute the ISI pdf

    bins : int or time Quantity. (Optional)
        If int, number of bins of the pdf histogram.
        If single-value time Quantity, width of each time bin.
        Default is 10.

    rng : Quantity array or None. (Optional)
        Range (in time unit) over which to compute the histogram:
        * if None (default) computes the histogram in the full rng
        * if pair of Quantities [r0, r1], ignores ISIs outside the rng. r0
          or r1 can also be None (min and max ISI used instead, respectively)
        Default is False.

    density : bool. (Optional)
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the pdf at the bin, normalized
        such that the *integral* over the rng is 1.
        Note that the sum of the histogram values will not be equal to 1
        unless bins of unity width are chosen.
        Default is False.

    Returns
    -------
    analogSignal: neo.core.AnalogSignal
        A neo.core.AnalogSignal containing the values of the ISI distribution.
        AnalogSignal[j] represents the ISI's pdf computed between
        `t_start + j * sampling_period and t_start + (j + 1) * sampling_period`

    """

    # Convert spiketrain to a list if it is a SpikeTrain
    if type(spiketrain) == neo.core.SpikeTrain:
        spiketrain = [spiketrain]

    # Collect all ISIs from all SpikeTrains in spiketrain
    # (as dimensionless array, meant in seconds)
    isis = []
    for st in spiketrain:
        isis = numpy.hstack([isis, numpy.diff(st.simplified.magnitude)])

    # Set histogram rng [isi_min, isi_max]
    if rng is None:
        isi_min, isi_max = min(isis), max(isis)
    elif len(rng) == 2:
        if rng[0] is None:
            isi_min = min(isis)
        else:
            try:
                isi_min = rng[0].rescale('s').magnitude
            except:
                raise ValueError('rng[0] must be a time Quantity')
        if rng[1] is None:
            isi_max = max(isis)
        else:
            try:
                isi_max = rng[1].rescale('s').magnitude
            except:
                raise ValueError('rng[1] must be a time Quantity')
    else:
        raise ValueError('Range can only be None or sequence of length two')

    # If bins is a Quantity, convert it to a dimensionless array
    # of bin edges (meant in seconds)
    if type(bins) == pq.quantity.Quantity:
        binsize = bins.simplified.magnitude
        # If bins has length 1, interpret it as binsize and create bin array.
        # Otherwise return error
        if binsize.ndim == 0:
            bins = numpy.arange(isi_min, isi_max + binsize / 2, binsize)
        else:
            raise ValueError(
                'bins can be either int or single-value Quantity. Quantity '
                'array of shape ' + bins.shape + ' given instead')

    vals, edges = numpy.histogram(isis, bins, density=density)

    # Add unit (inverse of time) to the histogram values if normalized
    if density is True:
        vals = (vals / pq.s).rescale(1. / spiketrain[0].units)
    else:
        vals = vals * pq.dimensionless

    # Rescale edges to the 1st spike train's unit and compute bin size
    edges = (edges * pq.s).rescale(spiketrain[0].units)
    w = edges[1] - edges[0]

    # Return histogram values and bins; the latter are r
    return neo.AnalogSignal(signal=vals, sampling_period=w, t_start=edges[0])


def isi(spiketrain, axis=-1):
    """
    This ISI function is a adjusted port from the elephant repository and will
    be removed from here once the stats module is merged into the elephant
    library!!!

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
    intervals : NumPy array or Quantities array.
    """
    if axis is None:
        axis = -1
    intervals = np.diff(spiketrain, axis=axis)
    if isinstance(spiketrain, neo.core.SpikeTrain):
        intervals = pq.Quantity(intervals.magnitude, units=spiketrain.units)
    return intervals


def cv(spiketrains):
    """
    Evaluate the empirical coefficient of variation (CV) of the inter-spike
    intervals (ISIs) collected from one or more spike trains.

    Given the vector v containing the observed ISIs of one spike train,
    the CV is defined as

                    CV := std(v)/mean(v).

    The CV of a list of spike trains is computed collecting the ISIs of all
    spike trains.

    The CV represents a measure of irregularity in the spiking activity. For
    For a time-stationary Poisson process, the theoretical CV=1.

    Parameters
    ---------
    spiketrains: SpikeTrain or list of SpikeTrains
        A `neo.SpikeTrain` object or a list of `neo.core.SpikeTrain` objects,
        for which to compute the CV.

    Returns
    -------
    CV : float
        The CV of all ISIs in the input SpikeTrain(s).  If no ISI can be
        calculated (less than 2 spikes in each SpikeTrains), then CV=0.
    """

    # Convert input to a list if it is a SpikeTrain object
    if isinstance(spiketrains, neo.core.SpikeTrain):
        spiketrains = [spiketrains]

    # Collect the ISIs of all trains in spiketrains, and return their CV
    isis = numpy.array([])
    for st in spiketrains:
        if len(st) > 1:
            isis = numpy.hstack([isis, numpy.diff(st.simplified.base)])

    # Compute CV of ISIs
    if len(isis) == 0:
        CV = 0.
    else:
       CV = isis.std() / isis.mean()
    return CV


def cv_timeresolved(spiketrain, win=None, start=None, stop=None, step=None):
    """
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
    spiketrain : SpikeTrain or list of SpikeTrains
        a neo.SpikeTrain object (or a list of), for which to compute the CV
    win : Quantity (optional)
        the length of the time windows over which to compute the CV.
        If None, the CV is computed over the largest window possible;
        otherwise, the window slides along time (see argument 'step')
        Default: None
    start : Quantity, optional
        initial time for the computation of the CV. If None, the largest
        t_start among those of the input spike trains in `spiketrain` is used
        Default: None
    stop : Quantity, optional
        last time for the computation of the CV.
        If None, the smallest t_stop among those of the input spike trains
        in `spiketrain` is used
        Default: None
    step : Quantity, optional
        Time shift between two consecutive sliding windows.
        If None, successive windows are adjacent
        Default: None

    Returns
    -------
    values : array
        Array of CV values computed over consecutive time windows
    windows : array
        Array of shape (n, 2) of time windows over which the CV has been
        computed
    """

    # Convert spiketrain to a list if it is a SpikeTrain
    if type(spiketrain) == neo.core.SpikeTrain:
        spiketrain = [spiketrain]

    max_tstart = min([t.t_start for t in spiketrain])
    min_tstop = max([t.t_stop for t in spiketrain])

    if not (all([max_tstart == t.t_start for t in spiketrain]) and
                all([min_tstop == t.t_stop for t in spiketrain])):
        warnings.warning('spike trains have different t_start or t_stop'
                         ' values. CV computed for inner values only')

    t_start = max_tstart if start is None else start
    t_stop = min_tstop if stop is None else stop
    wlen = t_stop - t_start if win is None else win
    wstep = wlen if step is None else step

    # Convert all time quantities in dimensionless (_dl) units (meant in s)
    start_dl = float(t_start.simplified.base)
    stop_dl = float(t_stop.simplified.base)
    wlen_dl = float(wlen.simplified.base)
    step_dl = float(wstep.simplified.base)

    # Define the centers of the sliding windows where the CV must be computed
    cv_times = numpy.arange(wlen_dl / 2. + start_dl,
                            stop_dl - wlen_dl / 2. + step_dl / 2, step_dl)

    # Define the nx2 array of time windows within which to compute the CV
    windows = pq.s * numpy.array([numpy.max([cv_times - wlen_dl / 2.,
                                             start_dl * numpy.ones(
                                                 len(cv_times))], axis=0),
                                  numpy.min([cv_times +
                                             wlen_dl / 2.,
                                             stop_dl * numpy.ones(
                                                 len(cv_times))], axis=0)]).T

    # Compute the CV in each window defined above
    cv_values = numpy.zeros(len(cv_times))  # Initialize CV values to 0
    for i, w in enumerate(windows):
        x_sliced = [t.time_slice(w[0], w[1]) for t in spiketrain]
        cv_values[i] = cv(x_sliced)

    return cv_values, windows


def peth(sts, w, t_start=None, t_stop=None, output='counts', clip=False):
    """
    Peri-Event Time Histogram (PETH) of a list of spike trains.

    Parameters
    ----------
    sts : list of neo.core.SpikeTrain objects
        Spiketrains with a common time axis (same t_start and t_stop)
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
    analogSignal : neo.core.AnalogSignal
        neo.core.AnalogSignal object containing the PETH values.
        AnalogSignal[j] is the PETH computed between
        t_start + j * w and t_start + (j + 1) * w.

    """

    # Find the internal range t_start, t_stop where all spike trains are
    # defined; cut all spike trains taking that time range only
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

    # Bin the spike trains and sum across columns
    bs = rep.binned_st(sts_cut, t_start=t_start, t_stop=t_stop, binsize=w)

    if clip is True:
        bin_hist = np.sum(bs.matrix_clipped(), axis=0)
    else:
        bin_hist = np.sum(bs.matrix_unclipped(), axis=0)

    # Renormalise the histogram
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


def complexity(
        sts, w, empty_bin=False, t_start=None, t_stop=None,
        output='normalized'):
    """
    Complexity distribution of a list of spike trains.

    Parameters
    ----------
    sts : list of SpikeTrain
        spike trains with a common time axis (same t_start and t_stop)
    w : Quantity
        width of the population histogram (peth)'s time bins.
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
        * 'normalized': probability of number of neuron firing together


    Returns
    -------
    analogSignal : neo.core.AnalogSignal
        neo.core.Analogsignal object containing the Complexity values.
        AnalogSignal[j] is the is the probability that j neurons fire together
        or the number of spikes involved in a synchrony event of size j,
        both case with a w time precision.
    """

    # computation of population histogram
    pophist = peth(sts, w, t_start=t_start, t_stop=t_stop)

    # computation of complexity with considering the empty bins
    if empty_bin:
        complexity_hist = numpy.histogram(
            pophist, bins=range(0, len(sts) + 2))[0]
        t_start = 0

    # computation of complexity without considering the empty bins
    else:
        complexity_hist = numpy.histogram(
            pophist, bins=range(1, len(sts) + 2))[0]
        t_start = 1
    # normalization of the count
    if output == 'normalized':
        complexity_hist /= float(numpy.sum(complexity_hist))
    return neo.AnalogSignal(
        signal=complexity_hist * pq.dimensionless,
        t_start=t_start * pq.dimensionless,
        sampling_period=1 * pq.dimensionless)


def complexity_histogram(
        sts, w, t_start=None, t_stop=None):
    """
    Complexity histogram of a list of `neo.core.SpikeTrain` objects.

    Parameters
    ----------
    sts : list of neo.core.SpikeTrain objects
        Spike trains with a common time axis (same t_start and t_stop)
    w : Quantity
        Width of the time bins of the complexity histogram.
    t_start, t_stop : Quantity (optional)
        Start and stop time of the histogram. Only events in the input
        spike trains falling between t_start and t_stop (both included) are
        considered in the histogram. If t_start and/or t_stop are not
        specified, the maximum t_start of all Spiketrains is used as t_start,
        and the minimum t_stop is used as t_stop.
        Default: t_start=t_stop=None

    Returns
    -------
    analogSignal : neo.core.AnalogSignal
        neo.core.AnalogSignal object containing the Complexity values.
        AnalogSignal[j] is the is the probability that j neurons fire together
        or the number of spikes involved in a synchrony event of size j,
        both case with a w time precision.
    """

    # Find the internal range t_start, t_stop where all spike trains are
    # defined; cut all spike trains taking that time range only
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

    # Bin the spike trains and take the lists of filled bins
    binned_sts = rep.binned_st(sts_cut, t_start=t_start, t_stop=t_stop,
                               binsize=w)
    filled_bins = numpy.sort(numpy.hstack(binned_sts.filled))

    # Compute the complexities from the filled bins
    filled_bins_ext = numpy.hstack([filled_bins, filled_bins[-1] + 1])
    filled_bins_diff_non0 = numpy.where(numpy.diff(filled_bins_ext) != 0)[0]
    complexities = numpy.hstack([filled_bins_diff_non0[0] + 1, numpy.diff(
        filled_bins_diff_non0)])

    # Compute the complexity histogram, from 1 to n=len(sts)
    complexity_hist, edges = numpy.histogram(
        complexities, bins=numpy.arange(len(sts) + 2))

    # Compute the histogram at complexity 0
    complexity_hist[0] = binned_sts.num_bins - numpy.sum(complexity_hist)

    return neo.AnalogSignal(signal=complexity_hist * pq.dimensionless,
                            t_start=0 * pq.dimensionless,
                            sampling_period=1 * pq.dimensionless)


def peth_old(spiketrains, w, start=None, stop=None, output='counts'):
    """
    Peri-Event Time Histogram of a list of spike trains.

    Parameters:
    -----------
    spiketrains : neo.core.SpikeTrain
        A list of neo.core.SpikeTrain objects
    w : Quantity
        Bin width for the histogram
    start, stop : Quantity (optional)
        Starting and stopping time of the histogram. Only events in the
        input spike trains falling between start and stop (both included)
        are considered in the histogram
    output : str
        type of values contained in histogram. Can be one of:
        * 'counts': spike counts at each bin (as integer numbers)
        * 'mean': mean spike counts per spike train
        * 'rate': mean rate per spike train. Like 'mean', but the counts
          are additionally normalized by the bin width.
    Returns:
    --------
    peth, bins : numpy.array
        Returns the pair of arrays (values, edges), where 'values' contains
        the histogram values and 'edges' the bin edges (half-open to the
        right, except the last bin which includes the right end.)

    """
    max_tstart = min([t.t_start for t in spiketrains])
    min_tstop = max([t.t_stop for t in spiketrains])

    if not (all([max_tstart == t.t_start for t in spiketrains]) and all(
            [min_tstop == t.t_stop for t in spiketrains])):
        warnings.warn(
            'spike trains have different t_start or t_stop values. '
            'PETH computed for inner values only')

    t_start = max_tstart if start == None else start
    t_stop = min_tstop if stop == None else stop

    # For each spike train in spiketrains, compute indices of the bins
    # where each spike fall
    x_mod_w = [numpy.array(((xx.view(pq.Quantity) - t_start) / w).rescale(
        pq.dimensionless).magnitude, dtype=int) for xx in spiketrains]

    # For each spike train in spiketrains, compute the PETH as the histogram
    # of the number of spikes per time bin
    Nbins = int(numpy.ceil(((t_stop - t_start) / w).rescale(pq.dimensionless)))
    bins = numpy.arange(0, Nbins + 1)

    bin_hist = numpy.zeros(Nbins)
    for xx in x_mod_w:
        bin_hist += numpy.histogram(xx, bins=bins)[0]

    if output == 'mean':
        # divide by number of input spike trains
        bin_hist *= 1. / len(spiketrains)

    elif output == 'rate':
        # divide by number of input spike trains
        bin_hist *= 1. / len(spiketrains)
        bin_hist *= 1. / w  # and by bin width

    # Return the PETH (normalized by the bin size)
    # and the bins used to compute it
    return bin_hist, t_start + bins * w


def ISIpdf(x, bins=10, range=None, density=False):
    '''
    Deprecated! Use isi_pdf() instead...

    Evaluate the empirical inter-spike-interval (ISI) probability density
    function (pdf) from a list of spike trains.

    Parameters:
    ----------
    x : list of neo.SpikeTrain
        a list of spike trains for which to compute the FF

    bins : int or time Quantity. Optional, default is 10
        If int, number of bins of the pdf histogram.
        If single value Quantity, length of each histogram time bin.
        If Quantity array, bin edges for the ISI histogram

    range : Quantity array or None. Optional, default is None
        range (in time unit) over which to compute the histogram

    density : bool. Optional, default is False
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the pdf at the bin, normalized
        such that the *integral* over the range is 1.
        Note that the sum of the histogram values will not be equal to 1
        unless bins of unity width are chosen.

    Output:
    ------
    Returns the pair (values, windows), where:
    * values is the array of FF values computed over consecutive windows,
    * windows is the 2-dim array of such window (one row per window)

    '''

    # Convert x to a list if not such
    if type(x) == neo.core.SpikeTrain: x = [x]

    # Collect all ISIs from each spike train in x (as arrays, meant in s)
    ISIs = []
    for t in x: ISIs = numpy.hstack([ISIs, numpy.diff(t.simplified.magnitude)])

    # If bins is a Quantity, convert it to an array (meant in seconds)
    if type(bins) == pq.quantity.Quantity:
        bins = bins.simplified.base
        # If bins has 1 element, interpret it as bin size and create arrays of bins
        if bins.ndim == 0:
            bins = numpy.arange(min(ISIs), max(ISIs) + bins, bins)

    # Transform the range into a dimensionless list (values )
    r_dl = range if range == None else [r.simplified.magnitude for r in range]

    # Compute the histogram of ISIs
    vals, edges = numpy.histogram(ISIs, bins, range=r_dl, density=density)

    # Return histogram values and bins; the latter are rescaled to the unit
    # of the first spike train
    return vals, (edges * pq.s).rescale(x[0].units)
