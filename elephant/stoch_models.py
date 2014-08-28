import numpy as np
import random
import warnings

import quantities as pq
import neo


def poisson(rate, t_stop, t_start=0 * pq.s, n=None, decimals=None):
    """
    Generates one or more independent Poisson spike trains.

    Parameters
    ----------
    rate : Quantity
        Expected firing rate (frequency) of each output SpikeTrain.
        Can be one of:
        *  a single Quantity value: expected firing rate of each output
           SpikeTrain
        *  a Quantity array: rate[i] is the expected firing rate of the i-th
           output SpikeTrain
    t_stop : Quantity
        Single common stop time of each output SpikeTrain. Must be > t_start.
    t_start : Quantity (optional)
        Single common start time of each output SpikeTrain. Must be < t_stop.
        Default: 0 s.
    decimals : int or None (optional)
        Precision, i.e., number of decimal places, for the spikes in the
        SpikeTrains. To create spike times as whole numbers, i.e., no decimal
        digits, use decimals = 0. If set to None, no rounding takes place and
        default computer precision will be used.
        Default: None
    n : int or None (optional)
        If rate is a single Quantity value, n specifies the number of
        SpikeTrains to be generated. If rate is an array, n is ignored and the
        number of SpikeTrains is equal to len(rate).
        Default: None


    Returns
    -------
    list of neo.SpikeTrain
        Each SpikeTrain contains one of the independent Poisson spike trains,
        either n SpikeTrains of the same rate, or len(rate) SpikeTrains with
        varying rates according to the rate parameter. The time unit of the
        SpikeTrains is given by t_stop.


    Example
    -------

    >>> import numpy as np
    >>> import quantities as pq
    >>> np.random.seed(1)
    >>> print stocmod.poisson(rate = 3*pq.Hz, t_stop=1*pq.s)

    [<SpikeTrain(array([ 0.14675589,  0.30233257]) * s, [0.0 s, 1.0 s])>]

    >>> print stocmod.poisson(rate = 3*pq.Hz, t_stop=1*pq.s, decimals=2)

    [<SpikeTrain(array([ 0.35]) * s, [0.0 s, 1.0 s])>]

    >>> print stocmod.poisson(rate = 3*pq.Hz, t_stop=1*pq.s, decimals=2, n=2)

    [<SpikeTrain(array([ 0.14,  0.42,  0.56,  0.67]) * s, [0.0 s, 1.0 s])>,
     <SpikeTrain(array([ 0.2]) * s, [0.0 s, 1.0 s])>]

    >>> # Return the spike counts of 3 generated spike trains
    >>> print [len(x) for x in stocmod.poisson(
            rate = [20,50,80]*pq.Hz, t_stop=1*pq.s)]

    [17, 38, 66]
    """

    # Check t_start < t_stop and create their strip dimensions
    if not t_start < t_stop:
        raise ValueError(
            't_start (=%s) must be < t_stop (=%s)' % (t_start, t_stop))
    stop_dl = t_stop.simplified.magnitude
    start_dl = t_start.simplified.magnitude

    # Set number N of output spike trains (specified or set to len(rate))
    if n is not None:
        if not (type(n) == int and n > 0):
            raise ValueError('n (=%s) must be a positive integer' % str(n))
        N = n
    else:
        if rate.ndim == 0:
            N = 1
        else:
            N = len(rate.flatten())
            if N == 0:
                raise ValueError('No rate specified.')

    rate_dl = rate.simplified.magnitude.flatten()

    # Check rate input parameter
    if len(rate_dl) == 1:
        if rate_dl < 0:
            raise ValueError('rate (=%s) must be non-negative.' % rate)
        rates = np.array([rate_dl] * N)
    else:
        rates = rate_dl.flatten()
        if any(rates < 0):
            raise ValueError('rate must have non-negative elements.')

    if N != len(rates):
        warnings.warn('rate given as Quantity array, n will be ignored.')

    # Generate the array of (random, Poisson) number of spikes per spiketrain
    num_spikes = np.random.poisson(rates * (stop_dl - start_dl))
    if isinstance(num_spikes, int):
        num_spikes = np.array([num_spikes])

    # Create the Poisson spike trains
    series = [neo.SpikeTrain(
        ((t_stop - t_start) * np.sort(np.random.random(ns)) + t_start),
        t_start=t_start, t_stop=t_stop)
        for ns in num_spikes]

    # Round to decimal position, if requested
    if decimals is not None:
        series = [neo.SpikeTrain(
            s.round(decimals=decimals), t_start=t_start, t_stop=t_stop)
            for s in series]

    return series


def sip_poisson(
        M, N, T, rate_b, rate_c, jitter=0 * pq.s, tot_coinc='det',
        start=0 * pq.s, min_delay=0 * pq.s, decimals=4,
        return_coinc=False, output_format='list'):
    """
    Generates a multidimensional Poisson SIP (single interaction process)
    plus independent Poisson processes

    A Poisson SIP consists of Poisson time series which are independent
    except for simultaneous events in all of them. This routine generates
    a SIP plus additional parallel independent Poisson processes.

    **Args**:
      M [int]
          number of Poisson processes with coincident events to be generated.
          These will be the first M processes returned in the output.
      N [int]
          number of independent Poisson processes to be generated.
          These will be the last N processes returned in the output.
      T [float. Quantity assignable, default to sec]
          total time of the simulated processes. The events are drawn between
          0 and T. A time unit from the 'quantities' package can be assigned
          to T (recommended)
      rate_b [float | iterable. Quantity assignable, default to Hz]
          overall mean rate of the time series to be generated (coincidence
          rate rate_c is subtracted to determine the background rate). Can be:
          * a float, representing the overall mean rate of each process. If
            so, it must be higher than rate_c.
          * an iterable of floats (one float per process), each float
            representing the overall mean rate of a process. If so, the first
            M entries must be larger than rate_c.
      rate_c [float. Quantity assignable, default to Hz]
          coincidence rate (rate of coincidences for the M-dimensional SIP).
          The SIP time series will have coincident events with rate rate_c
          plus independent 'background' events with rate rate_b-rate_c.
      jitter [float. Quantity assignable, default to sec]
          jitter for the coincident events. If jitter == 0, the events of all
          M correlated processes are exactly coincident. Otherwise, they are
          jittered around a common time randomly, up to +/- jitter.
      tot_coinc [string. Default to 'det']
          whether the total number of injected coincidences must be determin-
          istic (i.e. rate_c is the actual rate with which coincidences are
          generated) or stochastic (i.e. rate_c is the mean rate of coincid-
          ences):
          * 'det', 'd', or 'deterministic': deterministic rate
          * 'stoc', 's' or 'stochastic': stochastic rate
      start [float <T. Default to 0. Quantity assignable, default to sec]
          starting time of the series. If specified, it must be lower than T
      min_delay [float <T. Default to 0. Quantity assignable, default to sec]
          minimum delay between consecutive coincidence times.
      decimals [int| None. Default to 4]
          number of decimal points for the events in the time series. E.g.:
          decimals = 0 generates time series with integer elements,
          decimals = 4 generates time series with 4 decimals per element.
          If set to None, no rounding takes place and default computer
          precision will be used
      return_coinc [bool]
          whether to retutrn the coincidence times for the SIP process
      output_format [str. Default: 'list']
          the output_format used for the output data:
          * 'gdf' : the output is a np ndarray having shape (2,-1). The
                    first column contains the process ids, the second column
                    represents the corresponding event times.
          * 'list': the output is a list of M+N sublists. Each sublist repres-
                    ents one process among those generated. The first M lists
                    contain the injected coincidences, the last N ones are
                    independent Poisson processes.
          * 'dict': the output is a dictionary whose keys are process IDs and
                    whose values are np arrays representing process events.

    **OUTPUT**:
      realization of a SIP consisting of M Poisson processes characterized by
      synchronous events (with the given jitter), plus N independent Poisson
      time series. The output output_format can be either 'gdf', list or
      dictionary (see output_format argument). In the last two cases a time
      unit is assigned to the output times (same as T's. Default to sec).

      If return_coinc == True, the coincidence times are returned as a second
      output argument. They also have an associated time unit (same as T's.
      Default to sec).

    .. note::
        See also: poisson(), msip_poisson(), genproc_mip_poisson(),
                  genproc_mmip_poisson()

    *************************************************************************
    EXAMPLE:

    >>> import quantities as qt
    >>> import jelephant.core.stocmod as sm
    >>> sip, coinc = sm.sip_poisson(M=10, N=0, T=1*qt.sec, \
            rate_b=20*qt.Hz,  rate_c=4, return_coinc = True)

    *************************************************************************
    """

    # return empty objects if N=M=0:
    if N == 0 and M == 0:
        if output_format == 'list':
            return [] * T.units
        elif output_format == 'gdf':
            return np.array([[]] * 2).T
        elif output_format == 'dict':
            return {}

    # Assign time unit to jitter, or check that its existing unit is a time
    # unit
    jitter = abs(jitter)

    # Define the array of rates from input argument rate. Check that its length
    # matches with N
    if rate_b.ndim == 0:
        if rate_b < 0:
            raise ValueError(
                'rate_b (=%s) must be non-negative.' % str(rate_b))
        rates_b = np.array(
            [rate_b.magnitude for _ in xrange(N + M)]) * rate_b.units
    else:
        rates_b = np.array(rate_b).flatten() * rate_b.units
        if not all(rates_b >= 0):
            raise ValueError('*rate_b* must have non-negative elements')
        elif N + M != len(rates_b):
            raise ValueError(
                "*N* != len(*rate_b*). Either specify rate_b as"
                "a vector, or set the number of spike trains by *N*")

    # Check: rate_b>rate_c
    if np.any(rates_b < rate_c):
        raise ValueError('all elements of *rate_b* must be >= *rate_c*')

    # Check min_delay < 1./rate_c
    if not (rate_c == 0 or min_delay < 1. / rate_c):
        raise ValueError(
            "'*min_delay* (%s) must be lower than 1/*rate_c* (%s)." %
            (str(min_delay), str((1. / rate_c).rescale(min_delay.units))))

    # Check that the variable decimals is integer or None.
    if decimals is not None and type(decimals) != int:
        raise ValueError(
            'decimals type must be int or None. %s specified instead' %
            str(type(decimals)))

    # Generate the N independent Poisson processes
    if N == 0:
        independ_poisson_trains = [] * T.units
    else:
        independ_poisson_trains = poisson(
            rate=rates_b[M:], t_stop=T, t_start=start, decimals=decimals)
        # Convert the trains from neo SpikeTrain objects to  simpler Quantity
        # objects
        independ_poisson_trains = [
            pq.Quantity(ind.base) * ind.units
            for ind in independ_poisson_trains]

    # Generate the M Poisson processes there are the basis for the SIP
    # (coincidences still lacking)
    if M == 0:
        embedded_poisson_trains = [] * T.units
    else:
        embedded_poisson_trains = poisson(
            rate=rates_b[:M] - rate_c, t_stop=T, t_start=start, n=M,
            decimals=decimals)
        # Convert the trains from neo SpikeTrain objects to simpler Quantity
        # objects
        embedded_poisson_trains = [
            pq.Quantity(emb.base) * emb.units
            for emb in embedded_poisson_trains]

    # Generate the array of times for coincident events in SIP, not closer than
    # min_delay. The array is generated as a quantity from the Quantity class
    # in the quantities module
    if tot_coinc in ['det', 'd', 'deterministic']:
        Nr_coinc = int(((T - start) * rate_c).rescale(pq.dimensionless))
        while 1:
            coinc_times = start + \
                np.sort(np.random.random(Nr_coinc)) * (T - start)
            if len(coinc_times) < 2 or min(np.diff(coinc_times)) >= min_delay:
                break
    elif tot_coinc in ['s', 'stoc', 'stochastic']:
        while 1:
            coinc_times = poisson(rate=rate_c, t_stop=T, t_start=start, n=1)[0]
            if len(coinc_times) < 2 or min(np.diff(coinc_times)) >= min_delay:
                break
        # Convert coinc_times from a neo SpikeTrain object to a Quantity object
        # pq.Quantity(coinc_times.base)*coinc_times.units
        coinc_times = coinc_times.view(pq.Quantity)
        # Set the coincidence times to T-jitter if larger. This ensures that
        # the last jittered spike time is <T
        for i in coinc_times:
            if coinc_times[i] > T - jitter:
                coinc_times[i] = T - jitter

    # Replicate coinc_times M times, and jitter each event in each array by
    # +/- jitter (within (start, T))
    embedded_coinc = coinc_times + \
        np.random.random((M, len(coinc_times))) * 2 * jitter - jitter
    embedded_coinc = embedded_coinc + \
        (start - embedded_coinc) * (embedded_coinc < start) - \
        (T - embedded_coinc) * (embedded_coinc > T)

    # Inject coincident events into the M SIP processes generated above, and
    # merge with the N independent processes
    sip_process = [
        np.sort(np.concatenate((
            embedded_poisson_trains[m].rescale(T.units),
            embedded_coinc[m].rescale(T.units))) * T.units)
        for m in xrange(M)]

    # Append the independent spike train to the list of trains
    sip_process.extend(independ_poisson_trains)

    # Convert back sip_process and coinc_times from Quantity objects to
    # neo.SpikeTrain objects
    sip_process = [
        neo.SpikeTrain(t, t_start=start, t_stop=T).rescale(T.units)
        for t in sip_process]
    coinc_times = [
        neo.SpikeTrain(t, t_start=start, t_stop=T).rescale(T.units)
        for t in embedded_coinc]

    # Return the processes in the specified output_format
    if output_format == 'list':
        if not return_coinc:
            output = sip_process  # [np.sort(s) for s in sip_process]
        else:
            output = sip_process, coinc_times
    elif output_format == 'gdf':
        neuron_ids = np.concatenate(
            [np.ones(len(s)) * (i + 1) for i, s in enumerate(sip_process)])
        spike_times = np.concatenate(sip_process)
        ids_sortedtimes = np.argsort(spike_times)
        output = np.array(
            (neuron_ids[ids_sortedtimes], spike_times[ids_sortedtimes])).T
        if return_coinc:
            output = output, coinc_times
    elif output_format == 'dict':
        dict_sip = {}
        for i, s in enumerate(sip_process):
            dict_sip[i + 1] = s
        if not return_coinc:
            output = dict_sip
        else:
            output = dict_sip, coinc_times

    return output


def msip_poisson(
        M, N, T, rate_b, rate_c, jitter=0 * pq.s, tot_coinc='det',
        start=0 * pq.s, min_delay=0 * pq.s, decimals=4, return_coinc=False,
        output_format='gdf'):
    """
    Generates Poisson multiple single-interaction-processes (mSIP) plus
    independent Poisson processes.

    A Poisson SIP consists of Poisson time series which are independent
    except for events occurring simultaneously in all of them. This routine
    generates multiple, possibly overlapping SIP plus additional parallel
    independent Poisson processes.

    **Args**:
      M [iterable | iterable of iterables]
          The list of neuron tags composing SIPs that have to be generated.
          Can be:
          * an iterable of integers: each integer is a time series ID, the
            list represents one SIP. A single SIP is generated this way.
          * an iterable of iterables: each internal iterable must contain
            integers, and represents a SIP that has to be generated.
            Different SIPs can be overlapping
      N [int | iterable]
          Refers to the full list of time series to be generated. Can be:
          * an integer, representing the number of Poisson processes to be
            generated. If so, the time series IDs will be integers from 1 to
            N.
          * an iterable, representing the full list of time series IDs to be
            generated.
      T [float. Quantity assignable, default to sec]
          total time of the simulated processes. The events are drawn between
          0 and T. A time unit from the 'quantities' package can be assigned
          to T (recommended)
      rate_b [float | iterable. Quantity assignable, default to Hz]
          overall mean rate of the time series to be generated (coincidence
          rate rate_c is subtracted to determine the background rate). Can be:
          * a float, representing the overall mean rate of each process. If
            so, it must be higher than each entry in rate_c.
          * an iterable of floats (one float per process), each float
            representing the overall mean rate of a process. For time series
            embedded in a SIP, the corresponding entry in rate_b must be
            larger than that SIP's rate (see rate_c).
      rate_c [float. Quantity assignable, default to Hz]
          coincidence rate (rate of coincidences for the M-dimensional SIP).
          Each SIP time series will have coincident events with rate rate_c,
          plus independent background events with rate rate_b-rate_c.
      jitter [float. Quantity assignable, default to sec]
          jitter for the coincident events. If jitter == 0, the events of all
          M correlated processes are exactly coincident. Otherwise, they are
          jittered around a common time randomly, up to +/- jitter.
      tot_coinc [string. Default to 'det']
          whether the total number of injected coincidences must be determin-
          istic (i.e. rate_c is the actual rate with which coincidences are
          generated) or stochastic (i.e. rate_c is the mean rate of coincid-
          ences):
          * 'det', 'd', or 'deterministic': deterministic rate
          * 'stoc', 's' or 'stochastic': stochastic rate
      start [float <T. Default to 0. Quantity assignable, default to sec]
          starting time of the series. If specified, it must be lower than T
      min_delay [float <T. Default to 0. Quantity assignable, default to sec]
          minimum delay between consecutive coincidence times of a SIP.
          This does not affect coincidences from two different SIPs, which
          can fall arbitrarily closer to each other.
      decimals [int| None. Default to 4]
          number of decimal points for the events in the time series. E.g.:
          decimals = 0 generates time series with integer elements,
          decimals = 4 generates time series with 4 decimals per element.
          If set to None, no rounding takes place and default computer
          precision will be used
      return_coinc [bool]
          whether to retutrn the coincidence times for the SIP process
      output_format   : [string. Default to 'gdf']
          the output_format used for the output data:
          * 'gdf' : the output is a np ndarray having shape (2,-1). The
                    first column contains the process ids, the second column
                    represents the corresponding event times.
          * 'list': the output is a list of M+N sublists. Each sublist repres-
                    ents one process among those generated. The first M lists
                    contain the injected coincidences, the last N ones are
                    independent Poisson processes.
          * 'dict': the output is a dictionary whose keys are process IDs and
                    whose values are np arrays representing process events.

    **OUTPUT**:
      Realization of mSIP plus independent Poisson time series. M and N
      determine the number of SIP assemblies and overall time series,
      respectively.
      The output output_format can be either 'gdf', list or dictionary
      (see output_format argument). In the last two cases a time unit is
      assigned to the output times (same as T's. Default to sec).

      If return_coinc == True, the mSIP coincidences are returned as an
      additional output variable. They are represented a list of lists, each
      sublist containing the coincidence times of a SIP. They also have an
      associated time unit (same as T's. Default to sec).

    **See also**:
      poisson(), sip_poisson(), genproc_mip_poisson(),
      genproc_mmip_poisson()

    *************************************************************************
    EXAMPLE:

    >>> import quantities as qt
    >>> import jelephant.core.stocmod as sm
    >>>
    >>> M = [1,2,3], [4,5]
    >>> N = 6
    >>> T = 1*qt.sec
    >>> rate_b, rate_c = 5 * qt.Hz, [2,3] *qt.Hz
    >>>
    >>> msip, coinc = sm.msip_poisson(M=M, N=N, T=T, rate_b=rate_b, \
            rate_c=rate_c, return_coinc = True, output_format='list')

    *************************************************************************
    """

    # Create from M the list all_units of all unit IDs to be generated, and
    # check N
    if hasattr(N, '__iter__'):
        all_units = N
    elif type(N) == int and N > 0:
        all_units = range(1, N + 1)
    else:
        raise ValueError(
            'N (=%s) must be a positive integer or an iterable' %
            str(N))

    # Create from M the list all_sip of all SIP assemblies to be generated, and
    # check M
    if hasattr(M, '__iter__'):
        if all([hasattr(m, '__iter__') for m in M]):
            all_sip = M
        elif all([type(m) == int for m in M]):
            all_sip = [M]
        else:
            raise ValueError(
                "M must be either a list of lists (one for every SIP) or "
                "a list of integers (a single SIP)")
    else:
        raise ValueError(
            "M must be either a list of lists (one for every SIP)"
            " or a list of integers (a single SIP)")

    # Check that the list of all units includes that of all sip-embedded units
    if not all([set(all_units).issuperset(sip) for sip in all_sip]):
        raise ValueError(
            "The set of all units (defined by N) must include each SIP"
            " (defined by M)")

    # Create the array of coincidence rates (one rate per SIP). Check the
    # number of elements and their non-negativity
    if rate_c.ndim == 0:
        rates_c = np.array([rate_c.magnitude for sip in all_sip]) * \
            rate_c.units
    else:
        rates_c = np.array(rate_c).flatten() * rate_c.units
        if not all(rates_c >= 0):
            raise ValueError('variable rate_c must have non-negative elements')
        elif len(all_sip) != len(rates_c):
            raise ValueError(
                "length of rate_c (=%d) and number of SIPs (=%d) mismatch" %
                (len(rate_c), len(all_sip)))

    # Define the array of rates from input argument rate. Check that its length
    # matches with N
    if rate_b.ndim == 0:
        if rate_b < 0:
            raise ValueError(
                "rate_b (=%s) must be non-negative." %
                str(rate_b))
        rates_b = np.array([rate_b.magnitude for _ in all_units]) * \
            rate_b.units
    else:
        rates_b = np.array(rate_b).flatten() * rate_b.units
        if not all(rates_b >= 0):
            raise ValueError("variable rate_b must have non-negative elements")
        elif len(all_units) != len(rates_b):
            raise ValueError(
                "the length of rate_b (=%d) must match the number "
                "of units (%d)" % (len(rates_b), len(all_units)))

    # Compute the background firing rate (total rate - coincidence rate) and
    # simulate background activity as a list...
    rates_bg = rates_b
    for sip_idx, sip in enumerate(all_sip):
        for n_id in sip:
            rates_bg[n_id - 1] -= rates_c[sip_idx]

    # Simulate the background activity and convert from neo SpikeTrain to
    # Quantity object
    background_activity = poisson(
        rate=rates_bg, t_stop=T, t_start=start, decimals=decimals)
    background_activity = [
        pq.Quantity(bkg.base) * bkg.units for bkg in background_activity]

    # Add SIP-like activity (coincidences only!) to background activity, and
    # list for each SIP its coincidences
    sip_coinc = []
    for sip, sip_rate in zip(all_sip, rates_c):
        sip_activity, coinc_times = sip_poisson(
            M=len(sip), N=0, T=T, rate_b=sip_rate, rate_c=sip_rate,
            jitter=jitter, tot_coinc=tot_coinc, start=start,
            min_delay=min_delay, decimals=decimals,
            return_coinc=True, output_format='list')
        sip_coinc.append(coinc_times)
        for i, n_id in enumerate(sip):
            background_activity[n_id - 1] = np.sort(
                np.concatenate(
                    [background_activity[n_id - 1], sip_activity[i]]) *
                T.units)

    # Convert background_activity from a Quantity object back to a neo
    # SpikeTrain object
    background_activity = [
        neo.SpikeTrain(bkg, t_start=start, t_stop=T).rescale(T.units)
        for bkg in background_activity]

    # Return the processes in the specified output_format
    if output_format == 'list':
        if not return_coinc:
            return background_activity
        else:
            return background_activity, sip_coinc
    elif output_format == 'gdf':
        neuron_ids = np.concatenate([
            np.ones(len(s)) * (i + 1)
            for i, s in enumerate(background_activity)])
        spike_times = np.concatenate(background_activity)
        ids_sortedtimes = np.argsort(spike_times)
        if not return_coinc:
            return np.array((
                neuron_ids[ids_sortedtimes], spike_times[ids_sortedtimes])).T
        else:
            return (
                np.array((
                    neuron_ids[ids_sortedtimes],
                    spike_times[ids_sortedtimes])).T,
                sip_coinc)
    elif output_format == 'dict':
        dict_sip = {}
        for i, s in enumerate(background_activity):
            dict_sip[i + 1] = s
        if not return_coinc:
            return dict_sip
        else:
            return dict_sip, sip_coinc


def poisson_cos(t_stop, a, b, f, phi=0, t_start=0 * pq.s):
    '''
    Generate a non-stationary Poisson spike train with cosine rate profile r(t)
    given as:
    $$r(t)= a * cos(f*2*\pi*t + phi) + b$$

    Parameters
    ----------
    t_stop : Quantity
        Stop time of the output spike trains
    a : Quantity
        Amplitude of the cosine rate profile. The unit should be of frequency
        type.
    b : Quantity
        Baseline amplitude of the cosine rate modulation. The rate oscillates
        between b+a and b-a. The unit should be of frequency type.
    f : Quantity
        Frequency of the cosine oscillation.
    phi : float (optional)
        Phase offset of the cosine oscillation.
        Default: 0
    t_start : Quantity (optional)
        Start time of each output SpikeTrain.
        Default: 0 s

    Returns
    -------
    SpikeTrain
        Poisson spike train with the expected cosine firing rate profile.
    '''

    # Generate Poisson spike train at maximum rate
    max_rate = a + b
    poiss = poisson(rate=max_rate, t_stop=t_stop, t_start=t_start)[0]

    # Calculate rate profile at each spike time
    cos_arg = (2 * f * np.pi * poiss).simplified.magnitude
    rate_profile = b + a * np.cos(cos_arg + phi)

    # Accept each spike at time t with probability r(t)/max_rate
    u = np.random.uniform(size=len(poiss)) * max_rate
    spike_train = poiss[u < rate_profile]

    return spike_train


def _sample_int_from_pdf(a, n):
    '''
    Draw n independent samples from the set {0,1,...,L}, where L=len(a)-1,
    according to the probability distribution a.
    a[j] is the probability to sample j, for each j from 0 to L.


    Parameters
    -----
    a [array|list]
        Probability vector (i..e array of sum 1) that at each entry j carries
        the probability to sample j (j=0,1,...,len(a)-1).

    n [int]
        Number of samples generated with the function

    Output
    -------
    array of n samples taking values between 0 and n=len(a)-1.
    '''

    # a = np.array(a)
    A = np.cumsum(a)  # cumulative distribution of a
    u = np.random.uniform(0, 1, size=n)
    U = np.array([u for i in a]).T  # copy u (as column vector) len(a) times
    return (A < U).sum(axis=1)


def _pool_two_spiketrains(a, b, range='outer'):
    '''
    Pool two spike trains a and b into a unique spike train containing
    all spikes from a and b.
    The new spike train starts at the minimum t-start between a and b, and
    ends at the maximum t-stop between a and b.


    Parameters
    ----------
    a, b : neo.SpikeTrains
        Spike trains to be pooled

    Output
    ------
    neo.SpikeTrain containing all spikes from a and b.
    '''

    unit = a.units
    times_a_dimless = list(a.view(pq.Quantity).magnitude)
    times_b_dimless = list(b.rescale(unit).view(pq.Quantity).magnitude)
    times = (times_a_dimless + times_b_dimless) * unit

    if range == 'inner':
        start = min(a.t_start, b.t_start)
        stop = max(a.t_stop, b.t_stop)
        times = times[times > start]
        times = times[times < stop]
    elif range == 'outer':
        start = max(a.t_start, b.t_start)
        stop = min(a.t_stop, b.t_stop)
    else:
        raise ValueError('range (%s) can only be "inner" or "outer"' % range)

    pooled_train = neo.SpikeTrain(
        times=times, units=unit, t_start=start, t_stop=stop)
    return pooled_train


def _mother_proc_cpp_stat(A, T, r, start=0 * pq.ms):
    '''
    Generate the hidden ("mother") Poisson process for a Compound Poisson
    Process (CPP).


    Parameters
    ----------
    r : Quantity, Hz
        Homogeneous rate of the n spike trains that will be genereted by the
        CPP function
    a : array
        Amplitude distribution. A[j] represents the probability of a
        synchronous event of size j.
        The sum over all entries of a must be equal to one.
    T : Quantity (time)
        The stopping time of the mother process
    start : Quantity (time). Optional, default is 0 ms
        The starting time of the mother process


    Output
    ------
    Poisson spike train representing the mother process generating the CPP
    '''

    N = len(A) - 1
    exp_A = np.dot(A, range(N + 1))  # expected value of a
    exp_mother = (N * r) / float(exp_A)  # rate of the mother process
    return poisson(rate=exp_mother, t_stop=T, t_start=start)[0]


def _mother_proc_cpp_cos(A, T, a, b, w, phi, start=0 * pq.ms):
    '''
    Generate the hidden ("mother") Poisson process for a non-stationary
    Compound Poisson Process (CPP) with oscillatory rates
                    r(t)=a * cos(w*2*\pi*t + phi) + b

    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    T : Quantity (time)
        stop time of the output spike trains
    a : Quantity (1/time)
        amplitude of the cosine rate profile
    b : Quantity (1/time)
        baseline of the cosine rate modulation. In a full period, the
        rate oscillates between b+a and b-a.
    w : Quantity (1/time)
        frequency of the cosine oscillation
    phi : float
        phase of the cosine oscillation
    start : Quantity (time). Optional, default to 0 s
        start time of each output spike trains

    Output
    ------
    Poisson spike train representing the mother process generating the CPP
    '''

    N = len(A) - 1  # Number of spike train in the CPP
    exp_A = float(np.dot(A, xrange(N + 1)))  # Expectation of A
    spike_train = poisson_cos(
        t_stop=T, a=N * a / exp_A, b=N * b / exp_A, f=w, phi=phi,
        t_start=start)

    return spike_train


def _cpp_hom_stat(A, T, r, start=0 * pq.s):
    '''
    Generate a Compound Poisson Process (CPP) with amplitude distribution
    A and heterogeneous firing rates r=r[0], r[1], ..., r[-1].

    Parameters
    ----------
    A : array
        Amplitude distribution. A[j] represents the probability of a
        synchronous event of size j.
        The sum over all entries of A must be equal to one.
    T : Quantity (time)
        The end time of the output spike trains
    r : Quantity (1/time)
        Average rate of each spike train generated
    start : Quantity (time). Optional, default to 0 s
        The start time of the output spike trains

    Output
    ------
    List of n neo.SpikeTrains, having average firing rate r and correlated
    such to form a CPP with amplitude distribution a
    '''

    # Generate mother process and associated spike labels
    mother = _mother_proc_cpp_stat(A=A, T=T, r=r, start=start)
    labels = _sample_int_from_pdf(A, len(mother))

    N = len(A) - 1  # Number of trains in output

    try:  # Faster but more memory-consuming approach
        M = len(mother)  # number of spikes in the mother process
        spike_matrix = np.zeros((N, M), dtype=bool)
        # for each spike, take its label l
        for spike_id, l in enumerate(labels):
            # choose l random trains
            train_ids = random.sample(xrange(N), l)
            # and set the spike matrix for that train
            for train_id in train_ids:
                spike_matrix[train_id, spike_id] = True  # and spike to True

        trains = [mother[row] for row in spike_matrix]

    except MemoryError:  # Slower (~2x) but less memory-consuming approach
        times = [[] for i in range(N)]
        for t, l in zip(mother, labels):
            train_ids = random.sample(xrange(N), l)
            for train_id in train_ids:
                times[train_id].append(t)

        trains = [neo.SpikeTrain(
            times=t, units=T.units, t_start=start, t_stop=T) for t in times]

    return trains


def _cpp_het_stat(A, T, r, start=0.*pq.s):
    '''
    Generate a Compound Poisson Process (CPP) with amplitude distribution
    A and heterogeneous firing rates r=r[0], r[1], ..., r[-1].

    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    T : Quantity (time)
        The end time of the output spike trains
    r : Quantity (1/time)
        Average rate of each spike train generated
    start : Quantity (time). Optional, default to 0 s
        The start time of the output spike trains

    Output
    ------
    List of neo.SpikeTrains with different firing rates, forming
    a CPP with amplitude distribution A
    '''

    # Computation of Parameters of the two CPPs that will be merged
    # (uncorrelated with heterog. rates + correlated with homog. rates)
    N = len(r)  # number of output spike trains
    A_exp = np.dot(A, xrange(N + 1))  # expectation of A
    r_sum = np.sum(r)  # sum of all output firing rates
    r_min = np.min(r)  # minimum of the firing rates
    r1 = r_sum - N * r_min  # rate of the uncorrelated CPP
    r2 = r_sum / float(A_exp) - r1  # rate of the correlated CPP
    r_mother = r1 + r2  # rate of the hidden mother process

    # Check the analytical constraint for the amplitude distribution
    if A[1] < (r1 / r_mother).rescale(pq.dimensionless).magnitude:
        raise ValueError('A[1] too small / A[i], i>1 too high')

    # Compute the amplitude distrib of the correlated CPP, and generate it
    a = [(r_mother * i) / float(r2) for i in A]
    a[1] = a[1] - r1 / float(r2)
    CPP = _cpp_hom_stat(a, T, r_min, start)

    # Generate the independent heterogeneous Poisson processes
    POISS = [poisson(i - r_min, T, start)[0] for i in r]

    # Pool the correlated CPP and the corresponding Poisson processes
    out = [_pool_two_spiketrains(CPP[i], POISS[i]) for i in range(N)]
    return out


def cpp(A, t_stop, rate, t_start=0 * pq.s):
    '''
    Generate a Compound Poisson Process (CPP) with a given amplitude
    distribution A and stationary marginal rates r.

    The CPP process is a model for parallel, correlated processes with Poisson
    spiking statistics at pre-defined firing rates. It is composed of len(A)-1
    spike trains with a correlation structure determined by the amplitude
    distribution A: A[j] is the probability that a spike occurs synchronously
    in any j spike trains.

    The CPP is generated by creating a hidden mother Poisson process, and then
    copying spikes of the mother process to j of the output spike trains with
    probability A[j].

    Note that this function decorrelates the firing rate of each SpikeTrain
    from the probability for that SpikeTrain to participate in a synchronous
    event (which is uniform across SpikeTrains).

    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    t_stop : Quantity (time)
        The end time of the output spike trains
    rate : Quantity (1/time)
        Average rate of each spike train generated. Can be:
        * single-valued: if so, all spike trains will have same rate rate
        * a sequence of values (of length len(A)-1), each indicating the
          firing rate of one process in output
    t_start : Quantity (time). Optional, default to 0 s
        The t_start time of the output spike trains

    Returns
    -------
    List of SpikeTrain
        SpikeTrains with specified firing rates forming the CPP with amplitude
        distribution A.
    '''

    if rate.ndim == 0:
        return _cpp_hom_stat(A=A, T=t_stop, r=rate, start=t_start)
    else:
        return _cpp_het_stat(A=A, T=t_stop, r=rate, start=t_start)


def cpp_cos(A, T, a, b, w, phi, start=0 * pq.s):
    '''
    Generate a Compound Poisson Process (CPP) with amplitude distribution
    A and non-stationary firing rates
                    r(t)=a * cos(w*2*\pi*t + phi) + b
    for all output spike trains.

    The CPP process is composed of n=length(A)-1 parallel Poisson spike
    trains with correlation structure determined by A: A[j] is the
    probability that a spike is occurring synchronously in j spike trains


    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    T : Quantity (time)
        stop time of the output spike trains
    a : Quantity (1/time)
        amplitude of the cosine rate profile
    b : Quantity (1/time)
        baseline of the cosine rate modulation. The rate oscillates
        between b+a and b-a.
    w : Quantity (1/time)
        frequency of the cosine oscillation
    phi : float
        phase of the cosine oscillation
    start : Quantity (time). Optional, default to 0 s
        start time of each output spike trains

    Output
    ------
    list of spike trains with same sinusoidal rates profile, and forming
    a CPP with specified amplitude distribution.
    '''

    N = len(A) - 1  # number of trains in output
    mother = _mother_proc_cpp_cos(A, T, a, b, w, phi, start=0 * pq.ms)
    labels = _sample_int_from_pdf(A, len(mother))

    try:  # faster but more memory-consuming approach
        M = len(mother)  # number of spikes in the mother process
        spike_matrix = np.zeros((N, M), dtype=bool)

        for spike_id, l in enumerate(labels):  # for each spike label l,
            train_ids = random.sample(xrange(N), l)  # choose l random trains
            for train_id in train_ids:  # and for each of them
                spike_matrix[train_id, spike_id] = True  # set copy to True

        trains = [mother[row] for row in spike_matrix]

    except MemoryError:  # slower (~2x) but less memory-consuming approach
        times = [[] for i in range(N)]
        for t, l in zip(mother, labels):
            train_ids = random.sample(xrange(N), l)
            for train_id in train_ids:
                times[train_id].append(t)

        trains = [neo.SpikeTrain(
            times=t, units=T.units, t_start=start, t_stop=T) for t in times]

    return trains


def poisson_nonstat_rate(rate_signal, cont_sign_method='step'):
    '''
    Generate a non-stationary poisson process with rate profile sampled from
    the analog-signal rate_signal


    Parameters
    -----
    T : Quantity (time)
        The stopping time of the output spike train


    rate_signal : neo.core.AnalogSignal, units=Hz
        The analog signal containing the discretization on the time axis of the
        rate profile function of the spike trains to generate

    cont_sign_method : string
        The approximation method used to make continuous the analog signal:
        *'linear': linear interplation is used
        *'step': the signal is approximed in each nterval of rate_signal.times
        with the value of the signal at the left extrem of the interval
        Default: 'linear'

    Output
    -----
    Poisson spike train representing the hidden process generating a CPP model
    with prfile rate lambda(t)= rate_signal
    '''
    # Dic of the interpolation methods
    methods_dic = {
        'linear': _analog_signal_linear_interp,
        'step': _analog_signal_step_interp}
    if cont_sign_method not in methods_dic:
        raise ValueError("Unknown method selected.")
    method = methods_dic[cont_sign_method]

    # Adding to the rate signal the value at t_stop coping the last one
    rate_signal = neo.AnalogSignal(
        np.append(rate_signal.magnitude, rate_signal.magnitude[-1]),
        units=rate_signal.units, sampling_period=rate_signal.sampling_period,
        t_start=rate_signal.t_start)

    # Generaton of the hidden Poisson process
    lambda_star = max(rate_signal)
    poiss = poisson(
        lambda_star, rate_signal.times[-1], t_start=rate_signal.t_start)[0]

    # Calculate rate profile at each spike time
    lamb = method(signal=rate_signal, times=poiss)

    # Accept each spike at time t with probability r(t)/max_rate
    u = np.random.uniform(size=len(poiss)) * lambda_star
    spiketrain = poiss[u < lamb]
    return spiketrain


def _analog_signal_linear_interp(signal, times):
    '''
    Compute the linear interpolation in all the point in the vector times with
    the values in signal


    Parameters
    -----
    times : Quantity vector(time)
        The points in whic is comuted the interpolation

    signal : neo.core.AnalogSignal
        The analog signal containing the discretization of the funtion to
        interpolate


    Output
    -----
    Quantity vector (units=signal.units) of the values of the interpolation of
    time with the signal
    '''
    out = np.array([])
    unit = signal.units
    times_rescaled = times.rescale(signal.times.units)

    # Linear interpolation of the signals in all the points t in times
    for i, t in enumerate(signal.times[1:]):
        m = (signal[i + 1] - signal[i]) / signal.sampling_period
        r = signal[i] - m * t
        h = times_rescaled[np.where(times_rescaled <= t)]
        times_rescaled = times_rescaled[len(h):]
        out = np.hstack([out, (r + h * m).magnitude])

    out = out * unit
    return out


def _analog_signal_step_interp(signal, times):
    '''
    Compute, in all the point t in the vector times,
    the values in signal at the first signal.times at left of t. In order to
    generate a step function approximation of the signal


    Parameters
    -----
    times : Quantity vector(time)
        The points in whic is comuted the interpolation

    signal : neo.core.AnalogSignal
        The analog signal containing the discretization of the funtion to
        interpolate


    Output
    -----
    Quantity vector (units=signal.units) of the values of the interpolation of
    time with the signal
    '''
    out = np.array([])
    unit = signal.units
    times_rescaled = times.rescale(signal.times.units)
    # Step constant interpolation of the signals in all the points t in times
    for i, t in enumerate(signal.times[1:]):
        h = times_rescaled[np.where(times_rescaled <= t)]
        times_rescaled = times_rescaled[len(h):]
        out = np.hstack([out, [signal[i]] * len(h)])

    out = out * unit
    return out


def _mother_proc_cpp_nonstat(A, rate_signal, cont_sign_method='step'):
    '''
    Generate the "mother" poisson process for a non-stationary
    Compound Poisson Process (CPP) with rate profile described by the
    analog-signal rate_signal.


    Parameters
    -----
    A : array
        Amplitude distribution, representing at each j-th entry the probability
        of a synchronous event of size j.
        The sum over all entries of a must be equal to one.

    rate_signal : neo.core.AnalogSignal, units=Hz
        The analog signal containing the discretization on the time axis of the
        rate profile function of the spike trains to generate

    cont_sign_method : string
        The approximation method used to make continuous the analog signal:
        *'linear': linear interplation is used
        *'step': the signal is approximed in each nterval of rate_signal.times
        with the value of the signal at the left extrem of the interval
        Default: 'linear'


    Output
    -----
    Poisson spike train representing the hidden process generating a CPP model
    with prfile rate lambda(t)=a*cos(w*2*greekpi*t+phi)+b
    '''

    N = len(A) - 1
    exp_A = np.dot(A, range(N + 1))  # expected value of a
    return poisson_nonstat_rate(
        rate_signal=rate_signal * N / float(exp_A),
        cont_sign_method=cont_sign_method)


def cpp_nonstat(A, rate_signal, cont_sign_method='step'):
    '''
    Generation a compound poisson process (CPP) with amplitude distribution A,
    homogeneus non-stationary profile rate described by the analog-signal
    rate_signal.

    The CPP process is composed of n=length(A)-1 different parallel poissonian
    spike trains with a correlation structure determined by the amplitude
    distribution A.


    Parameters
    -----
    A : array
        Amplitude distribution, representing at each j-th entry the probability
        of a synchronous event of size j.
        The sum over all entries of a must be equal to one.

    rate_signal [neo.core.AnalogSignal, units=Hz]
        The analog signal containing the discretization on the time axis of the
        rate profile function of the spike trains to generate

    cont_sign_method : string
        The approximation method used to make continuous the analog signal:
        *'linear': linear interplation is used
        *'step': the signal is approximed in each nterval of rate_signal.times
        with the value of the signal at the left extrem of the interval
        Default: 'linear'


    Output
    -----
    trains [list f spike trains]
        list of n spike trains all with same rates profile and distribuited as
        a CPP with amplitude given by A
    '''

    N = len(A) - 1  # number of trains in output
    # generation of mother process
    mother = _mother_proc_cpp_nonstat(
        A, rate_signal, cont_sign_method=cont_sign_method)
    # generation of labels from the amplitude
    labels = _sample_int_from_pdf(A, len(mother))
    N = len(A) - 1  # number of trains in output
    M = len(mother)  # number of spikes in the mother process

    spike_matrix = np.zeros((N, M), dtype=bool)

    for spike_id, l in enumerate(labels):  # for each spike, take its label l,
        train_ids = random.sample(xrange(N), l)  # choose l random trains
        for train_id in train_ids:  # and set the spike matrix for that train
            spike_matrix[train_id, spike_id] = True  # and spike to True

    trains = [mother[row] for row in spike_matrix]
    return trains


def cpp_corrcoeff(ro, xi, t_stop, rate, N, t_start=0 * pq.s):
    '''
    Generation a compound poisson process (CPP) with a prescribed pairwise
    correlation coefficient ro.

    The CPP process is composed of N different parallel poissonian
    spike trains with a correlation structure determined by the correlation
    coefficient ro and maximum order of correlation xi.


    Parameters
    ----------
    ro : float
        Pairwise correlation coefficient of the population, $0 <= ro <= 1$.
    xi : int
        Maximum order of correlation of the neuron population, $1 <= xi <= N$.
    t_stop : Quantity
        The stopping time of the output spike trains.
    rate : Quantity
        Average rate of each spike train generated expressed in units of
        frequency.
    N : int
        Number of parallel spike trains to create.
    t_start : Quantity (optional)
        The starting time of the output spike trains.

    Returns
    -------
    list of SpikeTrain
        list of N spike trains all with same rate and distributed as a CPP with
        correlation cefficient ro and maximum order of crrelation xi.
    '''

    if xi > N or xi < 1:
        raise ValueError('xi must be an integer such as 1 <= xi <= N.')
    if xi ** 2 - xi - (N - 1) * ro * xi < 0:
        raise ValueError(
            'Analytical check failed: ro= %f too big with xi= %d' % (ro, xi))
    # Computetion of the pick amplitude for xi=1
    nu = (xi ** 2 - xi - (N - 1) * ro * xi) / float(
        xi ** 2 - xi - (N - 1) * ro * xi + (N - 1) * ro)

    # Amplitude vector in the form A=[0,nu,0...0,nu,0...0]
    A = [0] + [nu] + [0] * (xi - 2) + [1 - nu] + [0] * (N - xi)
    return cpp(A=A, t_stop=t_stop, rate=rate, t_start=t_start)
