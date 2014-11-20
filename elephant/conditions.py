import elephant.neo_tools as nt
import numpy as np
import neo


def at_least_n_trains(container, n):
    """
    Given input is checked if it has at least `n` **neo.core.SpikeTrain**
    objects.

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo container
        The container for the neo objects.
    n: int
        Number of SpikeTrain objects given input must have at least.

    Returns
    -------
    list : list of `neo.core.SpikeTrain` objects.

    """
    if n < 1:
        raise ValueError(
            "Please provide a number greater than %d, when setting "
            "the condition for a minimal number of SpikeTrains." % n)
    sts = nt.get_all_spiketrains(container)
    return sts if len(sts) >= n else []


def exact_n_trains(container, n):
    """
    Given input is checked if it has exactly `n` **neo.core.SpikeTrain**
    objects.

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo container
        The container for the neo objects.
    n: int
        Number of SpikeTrain objects given input must have.

    Returns
    -------
    list : list of `neo.core.SpikeTrain` objects.

    """
    if n < 1:
        raise ValueError(
            "Please provide a number greater than %d, when setting "
            "the condition for exact number of SpikeTrains." % n)
    sts = nt.get_all_spiketrains(container)
    return sts if len(sts) == n else []


def each_train_has_n_spikes(container, n):
    """
    Each SpikeTrain of the trial must have `n` or more spikes.

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo container
        The container for the neo objects.
    n: int
        Number of spikes each SpikeTrain object must have at least.

    Returns
    -------

    """
    if n < 1:
        raise ValueError("Please provide a number greater than %d, "
                         "when setting the condition for a minimal number "
                         "of Spikes in each SpikeTrain." % n)
    sts = nt.get_all_spiketrains(container)
    return [st for st in filter(lambda x: np.size(x) >= n, sts)]


def each_train_exact_n_spikes(container, n):
    """
    Each SpikeTrain of the trial must have `n` spikes.

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo container
        The container for the neo objects.
    n: int
        Number of spikes each SpikeTrain object must have.

    Returns
    -------

    """
    if n < 1:
        raise ValueError("Please provide a number greater than %d, "
                         "when setting the condition for an exact number "
                         "of Spikes in each SpikeTrain." % n)
    sts = nt.get_all_spiketrains(container)
    return [st for st in filter(lambda st: np.size(st) == n, sts)]


def at_least_n_analogsignals(container, n):
    """
    Given input is checked if it has at least `n` **neo.core.AnalogSignal**
    objects.

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo container
        The container for the neo objects.
    n: int
        Number of AnalogSignal objects given input must have at least.

    Returns
    -------
    list : list of `neo.core.AnalogSignal` objects.

    """
    if n < 1:
        raise ValueError(
            "Please provide a number greater than %d, when setting "
            "the condition for a minimal number of AnalogSignals." % n)
    signals = nt.get_all_analogsignals(container)
    return signals if len(signals) >= n else []


def exact_n_analogsignals(container, n):
    """
    Given input is checked if it has exactly `n` **neo.core.AnalogSignal**
    objects.

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo container
        The container for the neo objects.
    n: int
        Number of AnalogSignal objects given input must have.

    Returns
    -------
    list : list of `neo.core.AnalogSignal` objects.

    """
    if n < 1:
        raise ValueError(
            "Please provide a number greater than %d, when setting "
            "the condition for exact number of AnalogSignals." % n)
    signals = nt.get_all_analogsignals(container)
    return signals if len(signals) == n else []


def at_least_n_units(container, n):
    """
    Given input is checked if it has at least `n` **neo.core.Unit**
    objects.

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo container
        The container for the neo objects.
    n: int
        Number of Unit objects given input must have at least.

    Returns
    -------
    list : list of `neo.core.Unit` objects.

    """
    if n < 1:
        raise ValueError(
            "Please provide a number greater than %d, when setting "
            "the condition for a minimal number of Units." % n)
    units = nt.get_all_units(container)
    return units if len(units) >= n else []


def exact_n_units(container, n):
    """
    Given input is checked if it has exactly`n` **neo.core.Unit**
    objects.

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo container
        The container for the neo objects.
    n: int
        Number of Unit objects given input must have.

    Returns
    -------
    list : list of `neo.core.Unit` objects.

    """
    if n < 1:
        raise ValueError(
            "Please provide a number greater than %d, when setting "
            "the condition for exact number of Units." % n)
    units = nt.get_all_units(container)
    return units if len(units) == n else []


def at_least_n_recordingchannels(container, n):
    """
    Given input is checked if it has at least `n` **neo.core.RecordingChannel**
    objects.

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo container
        The container for the neo objects.
    n: int
        Number of RecordingChannel objects given input must have at least.

    Returns
    -------
    list : list of `neo.core.RecordingChannel` objects.

    """
    if n < 1:
        raise ValueError(
            "Please provide a number greater than %d, when setting "
            "the condition for a minimal number of RecordingChannels." % n)
    rcs = nt.get_all_recordingchannels(container)
    return rcs if len(rcs) >= n else []


def exact_n_recordingchannels(container, n):
    """
    Given input is checked if it has exactly `n` **neo.core.RecordingChannel**
    objects.

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo container
        The container for the neo objects.
    n: int
        Number of RecordingChannel objects given input must have.

    Returns
    -------
    list : list of `neo.core.RecordingChannel` objects.

    """
    if n < 1:
        raise ValueError(
            "Please provide a number greater than %d, when setting "
            "the condition for exact number of RecordingChannels." % n)
    rcs = nt.get_all_recordingchannels(container)
    return rcs if len(rcs) == n else []


def contains_each_recordingchannel(container):
    """
    Each `neo.core.AnalogSignal` object of given input  has a link to each
    `neo.core.RecodingChannel`.

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo container
        The container for the neo objects.

    Returns
    -------
    sigs : list of neo.core.AnalogSignals
        List of `neo.core.AnalogSignal` objects with a link to each
        `neo.core.RecordingChannel`

    """
    rcs = nt.get_all_recordingchannels(container)
    sigs = nt.get_all_analogsignals(container)
    return [sig for sig in
            filter(lambda signal: signal.recordingchannel in rcs, sigs)]


def contains_each_unit(container):
    """
    Each `neo.core.SpikeTrain` object of given input has a link to each
    `neo.core.Unit`.

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo container
        The container for the neo objects.

    Returns
    -------
    sts : list of neo.core.SpikeTrain objects
        List of `neo.core.SpikeTrain` objects with a link to each neo.core.Unit
    """
    units = nt.get_all_units(container)
    sts = nt.get_all_spiketrains(container)
    return [spk for spk in
            filter(lambda st: st.unit in units, sts)]


def data_aligned(container):
    """
    Check if all signals are aligned at the same time line, regarding
    (start, stop).

    Parameters
    ----------
    container : list, tuple, iterable, dict, neo container
        The container for the neo objects.

    tuple: list of AnalogSignal objects and list of SpikeTrain objects
        A tuple consisting of two list with AnalogSignal and SpikeTrain
        objects.

    Returns
    -------
    sigs, sts: `neo.core.AnalogSignal` and `neo.core.SpikeTrain` objects
        All time aligned objects regarding the same start and stop time
        are returned.
    """
    as_time_list = []
    st_time_list = []
    sts = []
    sigs = []
    try:
        sts = nt.get_all_spiketrains(container)
        sigs = nt.get_all_analogsignals(container)
    except TypeError:
        if sts:
            sigs = []
            sts = nt.get_all_spiketrains(container)
        else:
            sts = []
            sigs = nt.get_all_analogsignals(container)
    # Empty signals
    if not (sts or sigs):
        return []
    for st in sts:
        st_time_list.append((st.t_start, st.t_stop))
    for sig in sigs:
        as_time_list.append((sig.t_start, sig.t_stop))
    # Check if start and stop times are equal
    if len(as_time_list) == 0:
        if st_time_list.count(st_time_list[0]) != len(
                st_time_list):
            return []
    elif len(st_time_list) == 0:
        if as_time_list.count(as_time_list[0]) != len(
                as_time_list):
            return []
    else:
        if np.equal(st_time_list, as_time_list).all() is False:
            return []
    if sigs:
        sigs.extend(sts)
        return sigs
    else:
        return sts


def signals_no_overlap(container, take_first=False):
    """
        Checks if trial overlap with other trials. If it has overlap the trial
        ID won't be considered as valid trial.

        Parameters
        ----------
        trial_list: list of int
            List of trial IDs, which will be iterated trough.
        take_first: 0 or 1
            True: Even if a overlap appears the first trial within all the
            overlapping trials will be taken.
            False: None of the overlapping trials will be taken.

        Returns
        -------
        list: list of int
            Trial IDs according to trial condition (trial has no overlap).

        Raises
        ------
        ValueError
            If no minimal start and maximal stop times are found.
        """

    lst_valid_trials = []
    trials_time = {}
    # Functions to define the min/max of a list of tuples
    # in each trial;
    # first unzips the list, then finds the min/max for each tuple position
    x_min = lambda x: list(map(min, zip(*x)))[0]
    y_max = lambda y: list(map(max, zip(*y)))[1]
    if isinstance(container, neo.core.Block):
        for tr_id in range(len(container.segments)):
            seg = container.segments[tr_id]
            st_start_stop = [(st.t_start, st.t_stop) for st in
                             seg.spiketrains]
            as_start_stop = [(asig.t_start, asig.t_stop) for asig in
                             seg.analogsignals]
            if st_start_stop and as_start_stop:
                min_start = min(x_min(st_start_stop), x_min(as_start_stop))
                max_stop = max(y_max(st_start_stop), y_max(as_start_stop))
            elif st_start_stop:
                min_start = x_min(st_start_stop)
                max_stop = y_max(st_start_stop)
            elif as_start_stop:
                min_start = x_min(as_start_stop)
                max_stop = y_max(as_start_stop)
            else:
                raise ValueError("No min start, max stop times found.")
            trials_time[tr_id] = (min_start, max_stop)
        # Store in list
        tr_time_lst = list(trials_time.items())
        # Order items according to smallest start time
        # Result is a sorted list of following form: (int, (int, int))
        # sorting by first item in tuple
        tr_time_lst.sort(key=lambda x: x[1][0])
        # Create binary mask for indexing,
        # in order not to visit invalid trials again
        bin_mask = np.ones(len(tr_time_lst), dtype=int)
        # Iterate over trial times list
        for idx, tpl in enumerate(tr_time_lst):
            # Get trial id
            ids = tpl[0]
            # Boolean to indicate if trial is valid or not
            valid = True
            # Check if in binary mask
            if not bin_mask[idx] == 0:
                valid = __overlap(bin_mask, tr_time_lst, idx, valid)
                if valid and bool(take_first) is False:
                    lst_valid_trials.append(ids)
                elif take_first:
                    lst_valid_trials.append(ids)
    # elif isinstance(container, neo.core.Segment):
    #         lst_valid_trials.append(tr_id)
    # elif isinstance(container, neo.core.SpikeTrain):
    #     lst_valid_trials.append(0)
    # elif isinstance(container, neo.core.AnalogSignal):
    #     lst_valid_trials.append(0)
    return lst_valid_trials


def __overlap(bin_mask, lst, i, b):
    """
    Recursive, helper method, to compare the start and stop time points of
    two neighbouring trials
    (actual and next element in list).

    Parameters
    ----------
    bin_mask: numpy.ndarray
        A mask with 1's or 0's, to see which trial is overlapping. 1 means
        no overlap and 0 means overlap.
        Corresponds to trial ID.
    lst: list of tuples of int
        A list with the trials.
    i: int
        Actual position in `lst`.
    b: bool
        A boolean variable which will be returned. Indicates if a trial is
        valid or not.

    Returns
    -------
    valid: bool
        A boolean variable to indicate if a trial is valid or not.

    Notes
    -----
    Algorithm compares actual element i (trial_1) with next element i+1
    (trial_2) from the list. List contains
    tuple of trial IDs and tuple of start and stop points per trial.
    The list is ordered according to the smallest
    start point.
    Each step in the calling function (__check_trial_has_no_overlap()) an
    element of the list will be picked out.
    If it is not flagged as an overlapping trial.
    """
    valid = b
    if i + 1 >= len(lst):
        return valid
    else:
        # Get the actual and next element of list
        trial_1 = lst[i]
        trial_2 = lst[i + 1]
        # If start time of second trial is smaller
        # than stop time of actual trial
        if trial_2[1][0] < trial_1[1][1]:
            bin_mask[i] = 0
            bin_mask[i + 1] = 0
            valid = False
            __overlap(bin_mask, lst, i + 1, valid)
    return valid


def signals_same_tstart(signals):
    '''
    Check whether a list of signals (AnalogSignals or SpikeTrains) have same
    attribute t_start. If so return that value. Otherwise raise a ValueError.

    Parameters:
    ----------
    signals : list
        a list of signals (e.g. AnalogSignals or SpikeTrains) having
        attribute t_start

    Returns:
    -------
    t_start : Quantity
        The common attribute t_start of the list of signals.
        If the signals have a different t_start, a ValueError is raised.
    '''

    t_start = signals[0].t_start

    if len(signals) == 1:
        return t_start
    else:
        for sig in signals[1:]:
            if sig.t_start != t_start:
                raise ValueError('signals have different t_start values')
        return t_start


def signals_same_tstop(signals):
    '''
    Check whether a list of signals (AnalogSignals or SpikeTrains) have same
    attribute t_stop. If so return that value. Otherwise raise a ValueError.

    Parameters:
    ----------
    signals : list
        a list of signals (e.g. AnalogSignals or SpikeTrains) having
        attribute t_stop

    Returns:
    -------
    t_stop : Quantity
        The common attribute t_stop of the list of signals.
        If the signals have a different t_stop, a ValueError is raised.
    '''

    t_stop = signals[0].t_stop

    if len(signals) == 1:
        return t_stop
    else:
        for sig in signals[1:]:
            if sig.t_stop != t_stop:
                raise ValueError('signals have different t_stop values')
        return t_stop