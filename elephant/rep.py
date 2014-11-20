import neo.core
import quantities as pq
import numpy as np
import elephant.conditions as conditions


def time_axis(tstart, tstop, h):
    """ Create a time axis
    """

    return np.arange(tstart, tstop, h)


def bin_spiketrain(spiketrain, h):
    """ Create a binned spike train from a neo object.

        Documentation yet missing.
    """

    # create binned spike train
    st_binned = np.zeros((spiketrain.duration))

    timestamps = time_axis(spiketrain.tstart, spiketrain.tstop)

    for t in spiketrain:
        st_binned[np.argmin(np.abs(t - timestamps))] = st_binned[np.argmin(
            np.abs(t - timestamps))] + 1

    return st_binned


def gdf(neo_spiketrains, ids=[]):

    """
    Converts a list of spike trains to gdf format.

    Gdf is a 2-column data structure containing neuron ids on the first
    column and spike times (sorted in increasing order) on the second
    column. Information about the time unit, not preserved in the float-
    like gdf, is returned as a second output

    *Args*
    ------
    sts [list]:
        a list of neo spike trains.

    ids [list. Default to []]:
        List of neuron IDs. Id[i] is the id associated to spike train
        sts[i]. If empty list provided (default), dds are assigned as
        integers from 0 to n_spiketrains-1.

    *Returns*
    ---------
    gdf : ndarray of floats with shape (n_spikes, 2)
        ndarray of unit ids (first column) and spike times (second column)
    time_unit : Quantity
        the time unit of the spike times (second column of the array)
    """

    # Find smallest time unit
    time_unit = neo_spiketrains[0].units
    for st in neo_spiketrains[1:]:
        if st.units < time_unit:
            time_unit = st.units

    # By default assign integers 0,1,... as ids of sts[0],sts[1],...
    if len(ids) == 0:
        ids = range(len(neo_spiketrains))

    gdf = np.zeros((1, 2))
    # Rescale all spike trains to that time unit, extract the magnitude
    # and add to the gdf
    for st_idx, st in zip(ids, neo_spiketrains):
        to_be_added = np.array([[st_idx] * len(st),
                                st.view(pq.Quantity).rescale(
                                    time_unit).magnitude]).T
        gdf = np.vstack([gdf, to_be_added])

    # Eliminate first row in gdf and sort the others by increasing spike
    # times
    gdf = gdf[1:]
    gdf = gdf[np.argsort(gdf[:, 1])]

    # Return gdf and time unit corresponding to second column
    return gdf, time_unit


###############################################################################
#
# Methods to calculate parameters, t_start, t_stop, bin size, number of bins
#
###############################################################################
def calc_tstart(num_bins, binsize, t_stop):
    """
    Calculates the start point from given parameter.

    Calculates the start point :attr:`t_start` from the three parameter
    :attr:`t_stop`, :attr:`num_bins` and :attr`binsize`.

    Parameters
    ----------
    num_bins: int
        Number of bins
    binsize: quantities.Quantity
        Size of Bins
    t_stop: quantities.Quantity
        Stop time

    Returns
    -------
    t_start : quantities.Quantity
        Starting point calculated from given parameter.
    """
    if num_bins is not None and binsize is not None and t_stop is not None:
        return t_stop.rescale(binsize.units) - num_bins * binsize


def calc_tstop(num_bins, binsize, t_start):
    """
    Calculates the stop point from given parameter.

    Calculates the stop point :attr:`t_stop` from the three parameter
    :attr:`t_start`, :attr:`num_bins` and :attr`binsize`.

    Parameters
    ----------
    num_bins: int
        Number of bins
    binsize: quantities.Quantity
        Size of bins
    t_start: quantities.Quantity
        Start time

    Returns
    -------
    t_stop : quantities.Quantity
        Stoping point calculated from given parameter.
    """
    if num_bins is not None and binsize is not None and t_start is not None:
        return t_start.rescale(binsize.units) + num_bins * binsize


def calc_num_bins(binsize, t_start, t_stop):
    """
    Calculates the number of bins from given parameter.

    Calculates the number of bins :attr:`num_bins` from the three parameter
    :attr:`t_start`, :attr:`t_stop` and :attr`binsize`.

    Parameters
    ----------
    binsize: quantities.Quantity
        Size of Bins
    t_start : quantities.Quantity
        Start time
    t_stop: quantities.Quantity
        Stop time

    Returns
    -------
    num_bins : int
       Number of bins  calculated from given parameter.

    Raises
    ------
    ValueError :
        Raised when :attr:`t_stop` is smaller than :attr:`t_start`".

    """
    if binsize is not None and t_start is not None and t_stop is not None:
        if t_stop < t_start:
            raise ValueError("t_stop (%s) is smaller than t_start (%s)"
                             % (t_stop, t_start))
        return int(((t_stop - t_start).rescale(
            binsize.units) / binsize).magnitude)


def calc_binsize(num_bins, t_start, t_stop):
    """
    Calculates the stop point from given parameter.

    Calculates the size of bins :attr:`binsize` from the three parameter
    :attr:`num_bins`, :attr:`t_start` and :attr`t_stop`.

    Parameters
    ----------
    num_bins: int
        Number of bins
    t_start: quantities.Quantity
        Start time
    t_stop
       Stop time

    Returns
    -------
    binsize : quantities.Quantity
        Size of bins calculated from given parameter.

    Raises
    ------
    ValueError :
        Raised when :attr:`t_stop` is smaller than :attr:`t_start`".
    """

    if num_bins is not None and t_start is not None and t_stop is not None:
        if t_stop < t_start:
            raise ValueError("t_stop (%s) is smaller than t_start (%s)"
                             % (t_stop, t_start))
        return (t_stop - t_start) / num_bins


def set_start_stop_from_input(spiketrains):
    """
    Sets the start :attr:`t_start`and stop :attr:`t_stop` point
    from given input.

    If one nep.SpikeTrain objects is given the start :attr:`t_stop `and stop
    :attr:`t_stop` of the spike train is returned.
    Otherwise the aligned times are returned, which are the maximal start point
    and minimal stop point.

    Parameters
    ----------
    spiketrains: neo.SpikeTrain object, list or array of neo.core.SpikeTrain
                 objects
        List of neo.core SpikeTrain objects to extract `t_start` and
        `t_stop` from.

    Returns
    -------
    start : quantities.Quantity
        Start point extracted from input :attr:`spiketrains`
    stop : quantities.Quantity
        Stop point extracted from input :attr:`spiketrains`
    """
    if isinstance(spiketrains, neo.SpikeTrain):
        return spiketrains.t_start, spiketrains.t_stop
    else:
        start = max([elem.t_start for elem in spiketrains])
        stop = min([elem.t_stop for elem in spiketrains])
    return start, stop


class binned_st:
    """
    Class which calculates a binned spike train and provides methods to
    transform the binned spike train to clipped or unclipped matrix.

    A binned spike train represents the occurrence of spikes in a certain time
    frame.
    I.e., a time series like [0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] is
    represented as [0, 0, 1, 3, 4, 5, 6]. The outcome is dependent on given
    parameter such as size of bins, number of bins, start and stop point.

    A clipped matrix represents the binned spike train in a binary manner.
    It's rows represent the number of spike trains
    and the columns represent the binned index position of a spike in a
    spike train.
    The calculated matrix columns contain only ones, which indicate
    a spike.

    An unclipped matrix is calculated the same way, but its columns
    contain the number of spikes that occurred in the spike train(s). It counts
    the occurrence of the timing of a spike in its respective spike train.

    Parameters
    ----------
    spiketrains : List of `neo.SpikeTrain` or a `neo.SpikeTrain` object
        Object or list of `neo.core.SpikeTrain` objects to be binned.
    binsize : quantities.Quantity
        Width of each time bin.
        Default is `None`
    num_bins : int
        Number of bins of the binned spike train.
    t_start : quantities.Quantity
        Time of the first bin (left extreme; included).
        Default is `None`
    t_stop : quantities.Quantity
        Stopping time of the last bin (right extreme; excluded).
        Default is `None`
    store_mat : bool
        If set to **True** matrix will be stored in memory.
        If set to **False** matrix will always be calculated on demand.
        This boolean indicates that both the clipped and unclipped matrix will
        be stored in memory. It is also possible to select which matrix to be
        stored, see Methods `matrix_clipped()` and `matrix_unclipped()`.
        Default is False.

    See also
    --------
    * from_neo()
    * matrix_clipped()
    * matrix_unclipped()

    Notes
    -----
    There are four cases the given parameters must fulfill.
    Each parameter must be a combination of following order or it will raise
    a value error:
    * t_start, num_bins, binsize
    * t_start, num_bins, t_stop
    * t_start, bin_size, t_stop
    * t_stop, num_bins, binsize

    It is possible to give the SpikeTrain objects and one parameter
    (:attr:`num_bins` or :attr:`binsize`). The start and stop time will be
    calculated from given SpikeTrain objects (max start and min stop point).
    Missing parameter will also be calculated automatically.

    """

    def __init__(self, spiketrains, binsize=None, num_bins=None, t_start=None,
                 t_stop=None, store_mat=False):
        """
        Defines a binned spike train class

        """
        # Converting spiketrains to a list, if spiketrains is one
        # SpikeTrain object
        if type(spiketrains) == neo.core.SpikeTrain:
            spiketrains = [spiketrains]

        # Check that spiketrains is a list of neo Spike trains.
        if not all([type(elem) == neo.core.SpikeTrain for elem in spiketrains]):
            raise TypeError(
                "All elements of the input list must be neo.core.SpikeTrain "
                "objects ")
        # Link to input
        self.lst_input = spiketrains
        # Set given parameter
        self.t_start = t_start
        self.t_stop = t_stop
        self.num_bins = num_bins
        self.binsize = binsize
        self.matrix_columns = num_bins
        self.matrix_rows = len(spiketrains)
        self.store_mat_c = store_mat
        self.store_mat_u = store_mat
        # Empty matrix for storage
        self.mat_c = None
        self.mat_u = None
        # Check all parameter, set also missing values
        self.__calc_start_stop(spiketrains)
        self.__check_init_params(binsize, num_bins, self.t_start, self.t_stop)
        self.__check_consistency(spiketrains, self.binsize, self.num_bins,
                                 self.t_start, self.t_stop)
        self.filled = []  # contains the index of the bins
        # Now create filled
        self.__convert_to_binned(spiketrains)

    # =========================================================================
    # There are four cases the given parameters must fulfill
    # Each parameter must be a combination of following order or it will raise
    # a value error:
    # t_start, num_bins, binsize
    # t_start, num_bins, t_stop
    # t_start, bin_size, t_stop
    # t_stop, num_bins, binsize
    # ==========================================================================

    def __check_init_params(self, binsize, num_bins, t_start, t_stop):
        """
        Checks given parameter.
        Calculates also missing parameter.

        Parameters
        ----------
        binsize : quantity.Quantity
            Size of Bins
        num_bins : int
            Number of Bins
        t_start: quantity.Quantity
            Start time of the spike
        t_stop: quantity.Quantity
            Stop time of the spike

        Raises
        ------
        ValueError :
            If the all parameter are `None`, a ValueError is raised.

        TypeError:
            If type of :attr:`num_bins` is not an Integer.
        """
        # Raise error if no argument is given
        if binsize is None and t_start is None and t_stop is None \
                and num_bins is None:
            raise ValueError(
                "No arguments given. Please enter at least three arguments")
        # Check if num_bins is an integer (special case)
        if num_bins is not None:
            if type(num_bins) is not int:
                raise TypeError("num_bins is not an integer!")
        # Check if all parameters can be calculated, otherwise raise ValueError
        if t_start is None:
            self.t_start = calc_tstart(num_bins, binsize, t_stop)
        elif t_stop is None:
            self.t_stop = calc_tstop(num_bins, binsize, t_start)
        elif num_bins is None:
            self.num_bins = calc_num_bins(binsize, t_start, t_stop)
            if self.matrix_columns is None:
                self.matrix_columns = self.num_bins
        elif binsize is None:
            self.binsize = calc_binsize(num_bins, t_start, t_stop)

    def __calc_start_stop(self, spiketrains):
        """
        Calculates start stop from given spike trains.

         The start and stop point are calculated from given spike trains, only
         if they are not calculable from given parameter or the number of
         parameter is less than three.

        """
        if self.__count_params() is False:
            if self.t_stop is None:
                self.t_start = set_start_stop_from_input(spiketrains)[0]
            if self.t_stop is None:
                self.t_stop = set_start_stop_from_input(spiketrains)[1]

    def __count_params(self):
        """
        Counts the parameter and returns **True** if the count is greater
        or equal to `3`.

        The calculation of the binned matrix is only possible if there are at
        least three parameter (fourth parameter will be calculated out of
        them).
        This method counts the necessary parameter and returns **True** if the
        count is greater or equal to `3`.

        Returns
        -------
        bool :
            True, if the count is greater or equal to `3`.
            False, otherwise.

        """
        param_count = 0
        if self.t_start:
            param_count += 1
        if self.t_stop:
            param_count += 1
        if self.binsize:
            param_count += 1
        if self.num_bins:
            param_count += 1
        return False if param_count < 3 else True

    def __check_consistency(self, spiketrains, binsize, num_bins, t_start,
                            t_stop):
        """
        Checks the given parameters for consistency

        Raises
        ------
        ValueError :
            A ValueError is raised if an inconsistency regarding the parameter
            appears.
        AttributeError :
            An AttributeError is raised if there is an insufficient number of
            parameters.

        """
        if self.__count_params() is False:
            raise AttributeError("Too less parameter given. Please provide "
                                 "at least one of the parameter which are "
                                 "None.\n"
                                 "t_start: %s, t_stop: %s, binsize: %s, "
                                 "numb_bins: %s" % (
                                     self.t_start,
                                     self.t_stop,
                                     self.binsize,
                                     self.num_bins))
        t_starts = [elem.t_start for elem in spiketrains]
        t_stops = [elem.t_stop for elem in spiketrains]
        max_tstart = max(t_starts)
        min_tstop = min(t_stops)
        if max_tstart >= min_tstop:
            raise ValueError(
                "Starting time of each spike train must be smaller than each "
                "stopping time")
        elif t_start < max_tstart or t_start > min_tstop:
            raise ValueError(
                'some spike trains are not defined in the time given '
                'by t_start')
        elif num_bins != int((
            (t_stop - t_start).rescale(binsize.units) / binsize).magnitude):
            raise ValueError(
                "Inconsistent arguments t_start (%s), " % t_start +
                "t_stop (%s), binsize (%d) " % (t_stop, binsize) +
                "and num_bins (%d)" % num_bins)
        elif not (t_start < t_stop <= min_tstop):
            raise ValueError(
                'too many / too large time bins. Some spike trains are '
                'not defined in the ending time')
        elif num_bins - int(num_bins) != 0 or num_bins < 0:
            raise TypeError(
                "Number of bins (num_bins) is not an integer: " + str(
                    num_bins))
        elif t_stop > min_tstop or t_stop < max_tstart:
            raise ValueError(
                'some spike trains are not defined in the time given '
                'by t_stop')
        elif self.matrix_columns < 1 or self.num_bins < 1:
            raise ValueError(
                "Calculated matrix columns and/or num_bins are smaller "
                "than 1: (matrix_columns: %s, num_bins: %s). "
                "Please check your input parameter." % (
                    self.matrix_columns, self.num_bins))

    @property
    def filled(self):
        """
        Returns the binned spike train.
        Is a property.

        Returns
        -------
        filled : numpy.array
            Numpy array containing binned spike train.
        """
        return self.filled

    @filled.setter
    def filled(self, f):
        """
        Setter for a new binned spike train.

        Sets the binned spike train array `filled` to another binned array.

        Parameters
        ----------
        f : numpy array or list
            Array or list which is going to replace the actual binned spike
            train.
        """
        self.filled = f

    @property
    def edges(self):
        """
        Returns all time edges with :attr:`num_bins` bins as a quantity array.

        The borders of all time steps between start and stop [start, stop]
        with:attr:`num_bins` bins are regarded as edges.
        The border of the last bin is included.

        Returns
        -------
        bin_edges : quantities.Quantity array
            All edges in interval [start, stop] with :attr:`num_bins` bins
            are returned as a quantity array.
        """
        return np.linspace(self.t_start, self.t_stop,
                           self.num_bins + 1, endpoint=True)

    @property
    def left_edges(self):
        """
        Returns an quantity array containing all left edges and
        :attr:`num_bins` bins.

        The left borders of all time steps between start and excluding stop
        [start, stop) with number of bins from given input are regarded as
        edges.
        The border of the last bin is excluded.

        Returns
        -------
        bin_edges : quantities.Quantity array
            All edges in interval [start, stop) with :attr:`num_bins` bins
            are returned as a quantity array.
        """
        return np.linspace(self.t_start, self.t_stop, self.num_bins,
                           endpoint=False)

    @property
    def right_edges(self):
        """
        Returns the right edges with :attr:`num_bins` bins as a
        quantities array.

        The right borders of all time steps between excluding start and
        including stop (start, stop] with :attr:`num_bins` from given input
        are regarded as edges.
        The border of the first bin is excluded, but the last border is
        included.

        Returns
        -------
        bin_edges : quantities.Quantity array
            All edges in interval [start, stop) with :attr:`num_bins` bins
            are returned as a quantity array.
        """
        return self.left_edges + self.binsize

    @property
    def center_edges(self):
        """
        Returns each center time point of all bins between start and stop
        points.

        The center of each bin of all time steps between start and stop
        (start, stop).

        Returns
        -------
        bin_edges : quantities.Quantity array
            All center edges in interval (start, stop) are returned as
            a quantity array.
        """
        return self.left_edges + self.binsize/2

    def matrix_clipped(self, **kwargs):
        """
        Calculates a matrix, which rows represent the number of spike trains
        and the columns represent the binned
        index position of a spike in a spike train.
        The calculated matrix columns contain only ones, which indicate
        a spike.
        If **bool** `store_mat` is set to **True** last calculated `clipped`
        matrix will be returned.

        Parameters
        ----------
        kwargs:
            store_mat : boolean
                If set to **True** calculated matrix will be stored in
                memory. If the method is called again, the stored (clipped)
                matrix will be returned.
                If set to **False** matrix will always be calculated on demand.

        Raises
        ------
        AssertionError:
            If :attr:`store_mat` is not a Boolean an Assertion error is raised.
        IndexError:
            If the cols and and rows of the matrix are inconsistent, an Index
            error is raised.

        Returns
        -------
        clipped matrix : numpy.ndarray
            Matrix with ones indicating a spike and zeros for non spike.
            The ones in the columns represent the index
            position of the spike in the spike train and rows represent the
            number of spike trains.

        Examples
        --------
        >>> import jelephant.core.rep as rep
        >>> import neo as n
        >>> import quantities as pq
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> x = rep.binned_st(a, num_bins=10, binsize=1 * pq.s, t_start=0 * pq.s)
        >>> print x.matrix_clipped()
            [[ 1.  1.  0.  1.  1.  1.  1.  0.  0.  0.]]
        """
        if 'store_mat' in kwargs:
            if not isinstance(kwargs['store_mat'], bool):
                raise AssertionError('store_mat is not a boolean')
            self.store_mat_c = kwargs['store_mat']
        if self.mat_c is not None:
            return self.mat_c
        # Matrix shall be stored
        if self.store_mat_c:
            self.mat_c = np.zeros((self.matrix_rows, self.matrix_columns))
            for elem_idx, elem in enumerate(self.filled):
                if len(elem) != 0:
                    try:
                        self.mat_c[elem_idx, elem] = 1
                    except IndexError as ie:
                        raise IndexError(str(
                            ie) + "\n You are trying to build a matrix which "
                                  "is inconsistent in size. "
                                  "Please check your input parameter.")
                return self.mat_c
        # Matrix on demand
        else:
            tmp_mat = np.zeros(
                (self.matrix_rows, self.matrix_columns))  # temporary matrix
            for elem_idx, elem in enumerate(self.filled):
                if len(elem) != 0:
                    try:
                        tmp_mat[elem_idx, elem] = 1
                    except IndexError as ie:
                        raise IndexError(str(
                            ie) + "\n You are trying to build a matrix which "
                                  "is inconsistent in size. "
                                  "Please check your input parameter.")
            return tmp_mat

    def matrix_unclipped(self, **kwargs):
        """

        Calculates a matrix, which rows represents the number of spike trains
        and the columns represents the binned index position of a spike in a
        spike train.
        The calculated matrix columns contain the number of spikes that
        occurred in the spike train(s).
        If **bool** `store_mat` is set to **True** last calculated `unclipped`
        matrix will be returned.

        Parameters
        ----------
        kwargs:
            store_mat : boolean
                If set to **True** last calculated matrix will be stored in
                memory. If the method is called again, the stored (unclipped)
                matrix will be returned.
                If set to **False** matrix will always be calculated on demand.

        Returns
        -------
        unclipped matrix : numpy.ndarray
            Matrix with spike times. Columns represent the index position of
            the binned spike and rows represent the number of spike trains.

        Raises
        ------
        AssertionError:
            If :attr:`store_mat` is not a Boolean an Assertion error is raised.
        IndexError:
            If the cols and and rows of the matrix are inconsistent, an Index
            error is raised.

        Examples
        --------
        >>> import jelephant.core.rep as rep
        >>> import neo as n
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> x = rep.binned_st(a, num_bins=10, binsize=1 * pq.s, t_start=0 * pq.s)
        >>> print x.matrix_unclipped()
            [[ 2.  1.  0.  1.  1.  1.  1.  0.  0.  0.]]
        """
        if 'store_mat' in kwargs:
            if not isinstance(kwargs['store_mat'], bool):
                raise AssertionError('store_mat is not a boolean')
            self.store_mat_u = kwargs['store_mat']
        if self.mat_u is not None:
            return self.mat_u
        if self.store_mat_u:
            self.mat_u = np.zeros((self.matrix_rows, self.matrix_columns))
            for elem_idx, elem in enumerate(self.filled):
                if len(elem) != 0:
                    try:
                        if len(elem) >= 1:
                            for inner_elem in elem:
                                self.mat_u[elem_idx, inner_elem] += 1
                    except IndexError as ie:
                        raise IndexError(str(ie) + "\n You are trying to "
                                                   "build a matrix which is "
                                                   "inconsistent in size. "
                                                   "Please check your input "
                                                   "parameter.")
            return self.mat_u
        # Matrix on demand
        else:
            tmp_mat = np.zeros((self.matrix_rows, self.matrix_columns))
            for elem_idx, elem in enumerate(self.filled):
                if len(elem) != 0:
                    try:
                        if len(elem) > 1:
                            for inner_elem in elem:
                                tmp_mat[elem_idx, inner_elem] += 1
                        else:
                            tmp_mat[elem_idx, elem[0]] += 1
                    except IndexError as ie:
                        raise IndexError(str(
                            ie) + "\n You are trying to build a matrix which "
                                  "is inconsistent in size. "
                                  "Please check your input parameter.")
            return tmp_mat

    def __convert_to_binned(self, spiketrains):
        """

        Converts neo.core.SpikeTrain objects to a list of numpy.ndarray's
        called **filled**, which contains the binned times.

        Parameters
        ----------
        spiketrains : neo.SpikeTrain object or list of SpikeTrain objects
           The binned time array :attr:filled is calculated from a SpikeTrain
           object or from a list of SpikeTrain objects.
        binsize : quantities.Quantity
            Size of bins

        Examples
        --------
        >>> import jelephant.core.rep as rep
        >>> import neo as n
        >>> import quantities as pq
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> x = rep.binned_st(a, num_bins=10, binsize=1 * pq.s, t_start=0 * pq.s)
        >>> print x.filled
            [array([0, 0, 1, 3, 4, 5, 6])]
        """
        for elem in spiketrains:
            idx_filled = np.array(
                ((elem.view(pq.Quantity) - self.t_start).rescale(
                    self.binsize.units) / self.binsize).magnitude, dtype=int)
            self.filled.append(idx_filled[idx_filled < self.num_bins])

    def prune(self):
        """
        Prunes the :attr:`filled` list, so that each element contains no
        duplicated values any more

        Returns
        -------
        self : jelephant.rep.binned_st object
              Returns a new class with a pruned `filled` list.
        """
        if len(self.filled) > 1:
            self.filled = [np.unique(elems) for elems in self.filled]
        else:
            self.filled = [np.unique(np.asarray(self.filled))]
        return self

    def __eq__(self, other):
        """
        Overloads the `==` operator.

        Parameters
        ----------
        other: jelephant.core.rep.binned_st
            Another class of binned_st

        Returns
        -------
        bool :
            True, if :attr:`other` is equal to :attr:`self`
            False, otherwise.

        Examples
        --------
        >>> import jelephant.core.rep as rep
        >>> import neo as n
        >>> import quantities as pq
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> b = n.SpikeTrain([0.1, 0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s)
        >>> x = rep.binned_st(a, binsize=pq.s, t_start=0 * pq.s, t_stop=10. * pq.s)
        >>> y = rep.binned_st(b, binsize=pq.s, t_start=0 * pq.s, t_stop=10. * pq.s)
        >>> print (x == y)
            False
        """
        return np.array_equal(self.filled, other.filled)

    def __add__(self, other):
        """
        Overloads `+` operator

        Parameters
        ----------
        other: jelephant.core.rep.binned_st
            Another class of binned_st

        Returns
        -------
        obj : jelephant.core.rep.binned_st object
            Summed joint object of `self` and `other`

        Examples
        --------
        >>> import jelephant.core.rep as rep
        >>> import neo as n
        >>> import quantities as pq
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> b = n.SpikeTrain([0.1, 0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s)
        >>> x = rep.binned_st(a, binsize=pq.s, t_start=0 * pq.s, t_stop=10. * pq.s)
        >>> y = rep.binned_st(b, binsize=pq.s, t_start=0 * pq.s, t_stop=10. * pq.s)
        >>> z = x + y
        >>> print z.filled
            [[0 0 1 3 4 5 6 0 0 1 2 4 5 8 0 0 1 2 4 5 8]]

        Notes
        -----
        A new object is created! That means parameter of Object A of
        (A+B) are copied.

        """
        new_class = self.create_class(self.t_start, self.t_stop, self.binsize,
                                      self.matrix_rows, self.matrix_columns)
        # filled is the important structure to change
        # For merged spiketrains, or when filled has more than one binned
        # spiketrain
        if len(self.filled) > 1 or len(other.filled) > 1:
            new_class.filled = [self.filled, other.filled]
        else:
            new_class.filled = np.hstack((self.filled, other.filled))
        return new_class

    def __iadd__(self, other):
        """
        Overloads `+=` operator

        Returns
        -------
        obj : jelephant.core.rep.binned_st object
            Summed joint object of `self` and `other`


        Examples
        --------
        >>> import jelephant.core.rep as rep
        >>> import neo as n
        >>> import quantities as pq
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> b = n.SpikeTrain([0.1, 0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s)
        >>> x = rep.binned_st(a, binsize=pq.s, t_start=0 * pq.s, t_stop=10. * pq.s)
        >>> y = rep.binned_st(b, binsize=pq.s, t_start=0 * pq.s, t_stop=10. * pq.s)
        >>> x += y
        >>> print x.filled
            [[0, 0, 1, 3, 4, 5, 6, 0, 0, 1, 2, 4, 5, 8]]

        Notes
        -----
        A new object is created! That means parameter of Object A of
        (A+=B) are copied.
        The input SpikeTrain is altered!

        """
        # Create new object; if object is not necessary,
        # only __add__ could be returned
        new_self = self.__add__(other)
        # Set missing parameter
        new_self.binsize = self.binsize
        new_self.t_start = self.t_start
        new_self.t_stop = self.t_stop
        new_self.num_bins = self.num_bins
        return new_self

    def __sub__(self, other):
        """
        Overloads the `-` operator.

        Returns
        -------
        obj : jelephant.core.rep.binned_st object
           Subtracted joint object of `self` and `other`

        Examples
        --------
        >>> import jelephant.core.rep as rep
        >>> import neo as n
        >>> import quantities as pq
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> b = n.SpikeTrain([0.1, 0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s)
        >>> x = rep.binned_st(a, binsize=pq.s, t_start=0 * pq.s, t_stop=10. * pq.s)
        >>> y = rep.binned_st(b, binsize=pq.s, t_start=0 * pq.s, t_stop=10. * pq.s)
        >>> z = x - y
        >>> print z.filled
            [[2, 3, 6, 8]]

        Notes
        -----
        A new object is created! That means parameter of Object A of
        (A-B) are copied.
        The input SpikeTrain is altered!

        """
        import itertools

        new_class = self.new_class = self.create_class(self.t_start,
                                                       self.t_stop,
                                                       self.binsize,
                                                       self.matrix_rows,
                                                       self.matrix_columns)
        # The cols and rows have to be equal to the rows and cols of self
        # and other.
        new_class.matrix_columns = self.matrix_columns
        new_class.matrix_rows = self.matrix_rows
        if len(self.filled) > 1 or len(other.filled) > 1:
            for s, o in itertools.izip(self.filled, other.filled):
                new_class.filled.append(np.array(list(set(s) ^ set(o))))
        else:
            new_class.filled.append(
                np.setxor1d(self.filled[0], other.filled[0]))
            if not len(new_class.filled[0] > 0):
                new_class.filled[0] = np.zeros(len(self.filled[0]))
        return new_class

    def __isub__(self, other):
        """
        Overloads the `-` operator.

        Returns
        -------
        obj : jelephant.core.rep.binned_st object
            Subtracted joint object of `self` and `other`


        Examples
        --------
        >>> import jelephant.core.rep as rep
        >>> import neo as n
        >>> import quantities as pq
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> b = n.SpikeTrain([0.1, 0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s)
        >>> x = rep.binned_st(a, binsize=pq.s, t_start=0 * pq.s, t_stop=10. * pq.s)
        >>> y = rep.binned_st(b, binsize=pq.s, t_start=0 * pq.s, t_stop=10. * pq.s)
        >>> x -= y
        >>> print x.filled
            [[2, 3, 6, 8]]

        Notes
        -----
        A new object is created! That means parameter of Object A of
        (A-=B) are copied.
        The input SpikeTrain is altered!

        """
        new_self = self.__sub__(other)
        # Set missing parameter
        new_self.binsize = self.binsize
        new_self.t_start = self.t_start
        new_self.t_stop = self.t_stop
        new_self.num_bins = self.num_bins
        return new_self

    @classmethod
    def create_class(cls, start, stop, binsize, mat_row, mat_col):
        # Dummy SpikeTrain is created to pass the checks in the constructor
        spk = neo.core.SpikeTrain([] * pq.s, t_stop=stop)
        # Create a new dummy class to return
        new_class = cls(spk, t_start=start, t_stop=stop, binsize=binsize)
        # Clear the filed list, which is created when creating an instance
        del new_class.filled[:]
        # The cols and rows has to be equal to the rows and cols of self
        # and other.
        new_class.matrix_rows = mat_row
        new_class.matrix_col = mat_col
        return new_class


def binned2train(x, width, align='left', **kwargs):
    """
    Transforms a 0-1 sequence x into a sequence of times, each time
    corresponding to the position of a non-zero element into the x array,
    scaled by binsize.

    **Args**:
      x [list | array]
          an array of zeros and non-zeros; elements with non-zero values
          represent time bins where an event (spike) takes place
      width [quantity.Quantity, with time unit]
          the width of the bins corresponding to spike train x
      align [list. Default to 'left']
          where to align the times corresponding to spikes (non-zero entries
          of x):
          * 'left': to the left border of the bin;
          * 'center': at the center of the bin;
          * 'right': to the right of the bin

    **Kwargs**:
      start [quantity.Quantity. Default to 0 sec]
          starting time of the spike train. If not specified, it is set to
          0 sec.
      stop [quantity.Quantity. Default to 0]
          ending time of the spike train. If not specified, it is set to
          start+width*len(x)

    **OUTPUT**:
      returns a time series as a neo.SpikeTrain object. The spike times in
      the train are computed from x by multiplying the position of its
      non-zero elements with the variable "width", shifting by "start" and
      then alignign the times either to the bin center or to one border.

    *************************************************************************
    EXAMPLE:

    >>> import quantities as pq
    >>> binned = [0,0,1,0,1,0,0]
    >>> train2 = binned2train(binned, width=5*pq.ms)
    >>> print train2

    *************************************************************************

    .. Seealso:: train2binned()


    See also:
      train2binned()

    """
    # Define the shift (as a fraction of the binsize)
    shift = 0.5 * (align == 'center') + 0 * (align == 'left') + 1 * (
    align == 'right')

    # Define t_start and t_stop for output spike train
    start = 0 * width.units if 'start' not in kwargs else kwargs['start']
    stop = start + len(x) * width if 'stop' not in kwargs else kwargs['stop']

    # Convert the input to a neo.SpikeTrain if not, and return it
    return neo.SpikeTrain(
        np.unique((np.where(np.array(x) > 0)[0] + shift) * width),
        t_start=start, t_stop=stop)


def train2binned(x, bins, clip=True):
    """
    Discretize a spike train (time series) into equally spaced intervals
    (time bins), assigning to each bin a value corresponding to the number of
    events of the train falling into it.

    **Args**:
      x [neo.SpikeTrain]
          a neo.SpikeTrain object
      bins [int|quantity.Quantity]
          defines the time bins for binning. Can be one of:
          * int: number of bins.  x is binned into n=bins equally-spaced time
            bins from x.t_start to x.t_stop
          * float with time unit: x is discretized into bins of width w=bins,
            apart for the last bin which ends at x.t_stop
          * array with time unit: bins to be used. x is discretized into
            these bins.

    **OUTPUT**:
      a sequence of integer numbers; zero correspond to no element of x falling
      into the corresponding time window. The bins are half-open on the right:
      [a, b).


    *************************************************************************
    Examples:

    >>> import quantities as pq
    >>> import neo
    >>> train = neo.SpikeTrain([1, 7, 9, 16, 31]*pq.ms, t_stop=38*pq.ms)

    # Defining the bin width may lead to the last bin being shorter:
    >>> print train2binned(train, bins=5*pq.ms)

    # Setting the number of bins creates equally-spaced bins
    >>> print train2binned(train, bins=8)

    # Clipping yields a 0-1 sequence
    >>> print train2binned(train, bins=8, clip=True)

    *************************************************************************

    .. note::

        See also: binned2train()

    """

    # Define the time bin edges
    edges = None
    if type(bins) == int:
        edges = np.linspace(x.t_start, x.t_stop, bins + 1)
    elif type(bins) == pq.Quantity:
        # Check that bins has time unit
        if bins.simplified.units != pq.sec:
            raise ValueError(
                'bins must have time unit. %s assigned instead' % (bins.units))
        # Define bin edges
        if bins.shape == ():
            edges = np.arange(x.t_start.simplified.base,
                              x.t_stop.simplified.base, bins.simplified.base)
            edges = (
            np.hstack([edges, x.t_stop.simplified.base]) * pq.s).rescale(
                x.units)
        else:
            edges = bins

    # Compute the non-clipped binned spike train (#spikes/bin)
    binned_train = np.histogram(x, bins=edges)[0]

    # Clip spikes if falling into the same time bin
    if clip is True:
        binned_train = 1 * (binned_train > (0. * x.units))

    # Return the binned time series and the edges
    return binned_train, edges


def transactions(spiketrains, binsize, t_start=None, t_stop=None, ids=[]):
    """
    Transform parallel spike trains a into list of sublists, called
    transactions, each corresponding to a time bin and containing the list
    of spikes in spiketrains falling into that bin.

    To compute each transaction, the spike trains are binned (with adjacent
    exclusive binning) and clipped (i.e. spikes from the same train falling
    in the same bin are counted as one event). The list of spike ids within
    each bin form the corresponding transaction.

    NOTE: the fpgrowth function in the fim module by Christian Borgelt
    requires int or string type for the SpikeTrain ids.

    Parameters:
    -----------
    spiketrains [list]
        list of neo.core.SpikeTrain objects, or list of pairs
        (Train_ID, SpikeTrain), where Train_ID can be any hashable object
    binsize [quantity.Quantity]
        width of each time bin; time is binned to determine synchrony
        t_start [quantity.Quantity. Default: None]
        starting time; only spikes occurring at times t >= t_start are
        considered; the first transaction contains spike falling into the
        time segment [t_start, t_start+binsize[.
        If None, takes the t_start value of the spike trains in spiketrains
        if the same for all of them, or returns an error.
    t_stop [quantity.Quantity. Default: None]
        ending time; only spikes occurring at times t < t_stop are
        considered.
        If None, takes the t_stop value of the spike trains in spiketrains
        if the same for all of them, or returns an error

    Returns:
    --------
    trans : list of lists
        a list of transactions; each transaction corresponds to a time bin
        and represents the list of spike trains ids having a spike in that
        time bin.


    """
    # Define the spike trains and their IDs depending on the input arguments
    if all([hasattr(elem, '__iter__') and len(elem) == 2 and
        type(elem[1]) == neo.SpikeTrain for elem in spiketrains]):
        ids = [elem[0] for elem in spiketrains]
        trains = [elem[1] for elem in spiketrains]
    elif all([type(st) == neo.SpikeTrain for st in spiketrains]):
        trains = spiketrains
        ids = range(len(spiketrains)) if ids == [] else ids
    else:
        raise TypeError('spiketrains must be either a list of ' + \
            'SpikeTrains or a list of (id, SpikeTrain) pairs')

    # Take the minimum and maximum t_start and t_stop of all spike trains
    tstarts = [xx.t_start for xx in trains]
    tstops = [xx.t_stop for xx in trains]
    max_tstart, min_tstart = max(tstarts), min(tstarts)
    max_tstop, min_tstop = max(tstops), min(tstops)

    # Set starting time of binning
    if t_start == None:
        start = conditions.signals_same_tstart(trains)
    elif t_start < max_tstart:
        raise ValueError('Some SpikeTrains have a larger t_start ' + \
            'than the specified t_start value')
    else:
        start = t_start

    # Set stopping time of binning
    if t_stop == None:
        stop = conditions.signals_same_tstop(trains)
    elif t_stop > min_tstop:
        raise ValueError('Some SpikeTrains have a smaller t_stop ' + \
            'than the specified t_stop value')
    else:
        stop = t_stop

    # Bin the spike trains and take for each of them the ids of filled bins
    binned = binned_st(
        trains, binsize=binsize, t_start=start, t_stop=stop)

    # Compute and return the transaction list. Note that each spike train
    # is implicitly binned!
    return [[train_id for train_id, b in zip(ids, binned.filled) if bin_id
        in b] for bin_id in xrange(binned.num_bins)]


def st_to_operation_time(st, rate):
    '''
    Trasposition of the spike times to operational time through the following
    equation:
                $t'_k = integral[0, t_k](rate(t))$
    Where t_k is the k-th spike and rate(t) is the rate profile of the
    spiketrain.

    Parameters
    ----------
    st : neo.SpikeTrain
        spiketrain to transpose in operational time
    rate : neo.AnalogSignal
        rate profile of the spiketrain
    '''
    #rescling to seconds
    dt = rate.sampling_period.simplified.magnitude
    st_magn = st.simplified.magnitude
    t_op = []
    #cumulative rate function
    crf_sum = np.cumsum(rate.simplified.magnitude)
    index = np.where(np.floor(st_magn/dt) < len(crf_sum))
    #computation of the new spike times
    for i in range(len(st_magn[index]) - 1):
        t = crf_sum[st_magn[i]/dt] * dt
        t_op.append(t)
    st_op = neo.SpikeTrain(
        t_op * pq.s, t_stop=crf_sum[-1] * dt * pq.s, t_start=st.t_start)
    return st_op
