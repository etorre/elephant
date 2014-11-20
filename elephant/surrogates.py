#--------------------------------------------------------------------------
# NAME         : spiketrains_utils.py:
# DESCRIPTION  : routines for generating and binning time series, extracting
#                information and generating surrogates
# AUTHOR       : Emiliano Torre
# CREATED      : September 12, 2012
#--------------------------------------------------------------------------

import numpy as np
import quantities as pq
import neo


def spike_dithering(x, dither, n=1, decimals=None, edges='['):
    """
    Generates surrogates of a spike train by spike dithering.

    The surrogates are obtained by uniformly dithering times around the
    original position. The dithering is performed independently for each
    surrogate.

    The surrogates retain the t_start and t_stop of the original spike train.
    Spikes moved beyond this range are lost or moved to the range's ends,
    depending on the parameter edge.


    Parameters
    ----------
    x :  SpikeTrain
        the spike train from which to generate the surrogates
    dither : Quantity
        amount of dithering. A spike at time t is placed randomly within
        ]t-dither, t+dither[.
    n : int (optional)
        number of surrogates to be generated.
        Default: 1
    decimals : int or None (optional)
        number of decimal points for every spike time in the surrogates
        If None, machine precision is used.
        Default: None
    edges : str (optional)
        For surrogate spikes falling outside the range [x.t_start, x.t_stop),
        whether to drop them out (for edges = '[' or 'cliff') or set
        that to the range's closest end (for edges = ']' or 'wall').
        Default: '['

    Returns
    -------
    list of SpikeTrain
      a list of spike trains, each obtained from x by randomly dithering
      its spikes. The range of the surrogate spike trains is the same as x.

    Example
    -------
    >>> import quantities as pq
    >>> import neo
    >>>
    >>> st = neo.SpikeTrain([100, 250, 600, 800]*pq.ms, t_stop=1*pq.s)
    >>> print spike_dithering(st, dither = 20*pq.ms)
    [<SpikeTrain(array([  96.53801903,  248.57047376,  601.48865767,
     815.67209811]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print spike_dithering(st, dither = 20*pq.ms, n=2)
    [<SpikeTrain(array([ 104.24942044,  246.0317873 ,  584.55938657,
        818.84446913]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 111.36693058,  235.15750163,  618.87388515,
        786.1807108 ]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print spike_dithering(st, dither = 20*pq.ms, decimals=0)
    [<SpikeTrain(array([  81.,  242.,  595.,  799.]) * ms,
        [0.0 ms, 1000.0 ms])>]
    """

    # Transform x into a Quantity object (needed for matrix algebra)
    data = x.view(pq.Quantity)

    # Main: generate the surrogates
    surr = data.reshape((1, len(data))) + 2 * dither * \
        np.random.random_sample((n, len(data))) - dither

    # Round the surrogate data to decimal position, if requested
    if decimals is not None:
        surr = surr.round(decimals)

    if edges in (']', 'wall'):
        # Move all spikes outside [x.t_start, x.t_stop] to the range's ends
        surr = np.minimum(np.maximum(surr.base,
            (x.t_start / x.units).base), (x.t_stop / x.units).base) * x.units
    elif edges in ('[', 'cliff'):
        # Leave out all spikes outside [x.t_start, x.t_stop]
        Tstart, Tstop = (x.t_start / x.units).base, (x.t_stop / x.units).base
        surr = [s[np.all([s >= Tstart, s < Tstop], axis=0)] * x.units
            for s in surr.base]

    # Return the surrogates as SpikeTrains
    return [neo.SpikeTrain(s, t_start=x.t_start, t_stop=x.t_stop).rescale(
        x.units) for s in surr]


def spike_time_rand(x, n=1, decimals=None):
    """
    Generates surrogates of a spike trains by spike time randomisation.

    The surrogates are obtained by keeping the spike count of the original
    spike train x, but placing them randomly into the interval
    [x.t_start, x.t_stop].
    This generates independent Poisson SpikeTrains (exponentially distributed
    inter-spike intervals) while keeping the spike count as in x.

    Parameters
    ----------
    x :  SpikeTrain
        the spike train from which to generate the surrogates
    n : int (optional)
        number of surrogates to be generated.
        Default: 1
    decimals : int or None (optional)
        number of decimal points for every spike time in the surrogates
        If None, machine precision is used.
        Default: None

    Returns
    -------
    list of SpikeTrain
      a list of spike trains, each obtained from x by randomly dithering
      its spikes. The range of the surrogate spike trains is the same as x.

    Example
    -------
    >>> import quantities as pq
    >>> import neo
    >>>
    >>> st = neo.SpikeTrain([100, 250, 600, 800]*pq.ms, t_stop=1*pq.s)
    >>> print spike_time_rand(st)
        [<SpikeTrain(array([ 131.23574603,  262.05062963,  549.84371387,
                            940.80503832]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print spike_time_rand(st, n=2)
        [<SpikeTrain(array([  84.53274955,  431.54011743,  733.09605806,
              852.32426583]) * ms, [0.0 ms, 1000.0 ms])>,
         <SpikeTrain(array([ 197.74596726,  528.93517359,  567.44599968,
              775.97843799]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print spike_time_rand(st, decimals=0)
        [<SpikeTrain(array([  29.,  667.,  720.,  774.]) * ms,
              [0.0 ms, 1000.0 ms])>]
    """

    # Create surrogate spike trains as rows of a Quantity array
    sts = ((x.t_stop - x.t_start) * np.random.random(size=(n, len(x))) + \
        x.t_start).rescale(x.units)

    # Round the surrogate data to decimal position, if requested
    if decimals is not None:
        sts = sts.round(decimals)

    # Convert the Quantity array to a list of SpikeTrains, and return them
    return [neo.SpikeTrain(np.sort(st), t_start=x.t_start, t_stop=x.t_stop)
        for st in sts]


def isi_shuffling(x, n=1, decimals=None):
    """
    Generates surrogates of a spike trains by inter-spike-interval (ISI)
    shuffling.

    The surrogates are obtained by keeping the randomly sorting the ISIs of
    the original spike train x.
    This generates independent SpikeTrains with same ISI distribution
    and spike count as in x, while destroying temporal dependencies and
    firing rate profile.

    Parameters
    ----------
    x :  SpikeTrain
        the spike train from which to generate the surrogates
    n : int (optional)
        number of surrogates to be generated.
        Default: 1
    decimals : int or None (optional)
        number of decimal points for every spike time in the surrogates
        If None, machine precision is used.
        Default: None

    Returns
    -------
    list of SpikeTrain
      a list of spike trains, each obtained from x by randomly ISI shuffling.
      The range of the surrogate spike trains is the same as x.

    Example
    -------
    >>> import quantities as pq
    >>> import neo
    >>>
    >>> st = neo.SpikeTrain([100, 250, 600, 800]*pq.ms, t_stop=1*pq.s)
    >>> print isi_shuffling(st)
        [<SpikeTrain(array([ 200.,  350.,  700.,  800.]) * ms,
                 [0.0 ms, 1000.0 ms])>]
    >>> print isi_shuffling(st, n=2)
        [<SpikeTrain(array([ 100.,  300.,  450.,  800.]) * ms,
              [0.0 ms, 1000.0 ms])>,
         <SpikeTrain(array([ 200.,  350.,  700.,  800.]) * ms,
              [0.0 ms, 1000.0 ms])>]

    """

    # Compute ISIs of x as a numpy array (meant in units of x)
    x_dl = x.magnitude

    if len(x) > 0:
        isi0 = x[0] - x.t_start
        ISIs = np.hstack([isi0.magnitude, np.diff(x_dl)])

        # Round the ISIs to decimal position, if requested
        if decimals is not None:
            ISIs = ISIs.round(decimals)

        # Create list of surrogate spike trains by random ISI permutation
        sts = []
        for i in xrange(n):
            surr_times = np.cumsum(np.random.permutation(ISIs)) * x.units + \
                x.t_start
            sts.append(neo.SpikeTrain(
                surr_times, t_start=x.t_start, t_stop=x.t_stop))

    else:
        sts = []
        empty_train = neo.SpikeTrain([]*x.units, t_start=x.t_start,
            t_stop=x.t_stop)
        for i in xrange(n):
            sts.append(empty_train)

    return sts


def train_shifting(x, shift, n=1, decimals=None, edges='['):
    """
    Generates surrogates of a spike trains by spike train shifting.

    The surrogates are obtained by shifting the whole spike train by a
    random amount (independent for each surrogate). Thus, ISIs and temporal
    correlations within the spike train are kept. For small shifts, the
    firing rate profile is also kept with reasonable accuracy.

    The surrogates retain the t_start and t_stop of the original spike train.
    Spikes moved beyond this range are lost or moved to the range's ends,
    depending on the parameter edge.

    Parameters
    ----------
    x :  SpikeTrain
        the spike train from which to generate the surrogates
    shift : Quantity
        amount of shift. x is shifted by a random amount uniformly drawn
        from the range ]-shift, +shift[.
    n : int (optional)
        number of surrogates to be generated.
        Default: 1
    decimals : int or None (optional)
        number of decimal points for every spike time in the surrogates
        If None, machine precision is used.
        Default: None
    edges : str (optional)
        For surrogate spikes falling outside the range [x.t_start, x.t_stop),
        whether to drop them out (for edges = '[' or 'cliff') or set
        that to the range's closest end (for edges = ']' or 'wall').
        Default: '['

    Returns
    -------
    list of SpikeTrain
      a list of spike trains, each obtained from x by randomly dithering
      its spikes. The range of the surrogate spike trains is the same as x.

    Example
    -------
    >>> import quantities as pq
    >>> import neo
    >>>
    >>> st = neo.SpikeTrain([100, 250, 600, 800]*pq.ms, t_stop=1*pq.s)
    >>>
    >>> print train_shifting(st, shift = 20*pq.ms)
    [<SpikeTrain(array([  96.53801903,  248.57047376,  601.48865767,
     815.67209811]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print train_shifting(st, shift = 20*pq.ms, n=2)
    [<SpikeTrain(array([  92.89084054,  242.89084054,  592.89084054,
        792.89084054]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([  84.61079043,  234.61079043,  584.61079043,
        784.61079043]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print train_shifting(st, shift = 20*pq.ms, decimals=0)
    [<SpikeTrain(array([  82.,  232.,  582.,  782.]) * ms,
        [0.0 ms, 1000.0 ms])>]
    """

    # Transform x into a Quantity object (needed for matrix algebra)
    data = x.view(pq.Quantity)

    # Main: generate the surrogates by spike train shifting
    surr = data.reshape((1, len(data))) + 2 * shift * \
        np.random.random_sample((n, 1)) - shift

    # Round the surrogate data to decimal position, if requested
    if decimals is not None:
        surr = surr.round(decimals)

    if edges in (']', 'wall'):
        # Move all spikes outside [x.t_start, x.t_stop] to the range's ends
        surr = np.minimum(np.maximum(surr.base,
            (x.t_start / x.units).base), (x.t_stop / x.units).base) * x.units
    elif edges in ('[', 'cliff'):
        # Leave out all spikes outside [x.t_start, x.t_stop]
        Tstart, Tstop = (x.t_start / x.units).base, (x.t_stop / x.units).base
        surr = [s[np.all([s >= Tstart, s < Tstop], axis=0)] * x.units
            for s in surr.base]

    # Return the surrogates as SpikeTrains
    return [neo.SpikeTrain(s, t_start=x.t_start, t_stop=x.t_stop).rescale(
        x.units) for s in surr]


def spike_jittering(x, binsize, n=1, decimals=None, edges='['):
    """
    Generates surrogates of a spike train by spike jittering.

    The surrogates are obtained by defining adjacent time bins spanning the
    spike train range, and random re-positioning (independently for each
    surrogate) each spike in the time bin it falls into.

    The surrogates retain the t_start and t_stop of the original spike train.
    Note that within each time bin the surrogate spike trains are locally
    Poissonian (the inter-spike-interval are exponentially distributed).

    Parameters
    ----------
    x :  SpikeTrain
        the spike train from which to generate the surrogates
    binsize : Quantity
        size of the time bins within which to randomise the spike times.
        Note: the last bin arrives until x.t_stop and might have width different
        than binsize.
    n : int (optional)
        number of surrogates to be generated.
        Default: 1
    decimals : int or None (optional)
        number of decimal points for every spike time in the surrogates
        If None, machine precision is used.
        Default: None

    Returns
    -------
    list of SpikeTrain
      a list of spike trains, each obtained from x by randomly replacing its
      spikes within bins of user-defined width.
      The range of the surrogate spike trains is the same as x.

    Example
    -------
    >>> import quantities as pq
    >>> import neo
    >>>
    >>> st = neo.SpikeTrain([80, 150, 320, 480]*pq.ms, t_stop=1*pq.s)
    >>> print spike_jittering(st, binsize=100*pq.ms)
    [<SpikeTrain(array([  98.82898293,  178.45805954,  346.93993867,
        461.34268507]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print spike_jittering(st, binsize=100*pq.ms, n=2)
    [<SpikeTrain(array([  97.15720041,  199.06945744,  397.51928207,
        402.40065162]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([  80.74513157,  173.69371317,  338.05860962,
        495.48869981]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print spike_jittering(st, binsize=100*pq.ms, decimals=0)
    [<SpikeTrain(array([  4.55064897e-01,   1.31927046e+02,   3.57846265e+02,
         4.69370604e+02]) * ms, [0.0 ms, 1000.0 ms])>]
    """
    # Define standard time unit; all time Quantities are converted to
    # scalars after being rescaled to this unit, to use the power of numpy
    std_unit = binsize.units

    # Compute bin edges for the jittering procedure
    # !: the last bin arrives until x.t_stop and might have size != binsize
    start_dl = x.t_start.rescale(std_unit).magnitude
    stop_dl = x.t_stop.rescale(std_unit).magnitude

    bin_edges = start_dl + np.arange(start_dl, stop_dl, binsize.magnitude)
    bin_edges = np.hstack([bin_edges, stop_dl])

    # Create n surrogates with spikes randomly placed in the interval (0,1)
    surr_poiss01 = np.random.random_sample((n, len(x)))

    # Compute the bin id of each spike
    bin_ids = np.array(
        (x.view(pq.Quantity) / binsize).rescale(pq.dimensionless).magnitude,
        dtype=int)

    # Compute the size of each time bin (as a numpy array)
    bin_sizes_dl = np.diff(bin_edges)

    # For each spike compute its offset (the left end of the bin it falls
    # into) and the size of the bin it falls into
    offsets = start_dl + np.array([bin_edges[bin_id] for bin_id in bin_ids])
    dilats = np.array([bin_sizes_dl[bin_id] for bin_id in bin_ids])

    # Compute each surrogate by dilatating and shifting each spike s in the
    # poisson 0-1 spike trains to dilat * s + offset. Attach time unit again
    surr = np.sort(surr_poiss01 * dilats + offsets, axis=1) * std_unit

    return [neo.SpikeTrain(s, t_start=x.t_start, t_stop=x.t_stop).rescale(
        x.units) for s in surr]