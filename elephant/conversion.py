# -*- coding: utf-8 -*-
"""
docstring goes here.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: CeCILL, see LICENSE.txt for details.
"""

from __future__ import division, print_function

import numpy as np
import quantities as pq
import neo.core
from warnings import warn
import logging


def binarize(spiketrain, sampling_rate=None, t_start=None, t_stop=None,
             return_times=None):
    """
    Return an array indicating if spikes occured at individual time points.

    The array contains boolean values identifying whether one or more spikes
    happened in the corresponding time bin.  Time bins start at `t_start`
    and end at `t_stop`, spaced in `1/sampling_rate` intervals.

    Accepts either a Neo SpikeTrain, a Quantity array, or a plain NumPy array.
    Returns a boolean array with each element being the presence or absence of
    a spike in that time bin.  The number of spikes in a time bin is not
    considered.

    Optionally also returns an array of time points corresponding to the
    elements of the boolean array.  The units of this array will be the same as
    the units of the SpikeTrain, if any.

    Parameters
    ----------

    spiketrain : Neo SpikeTrain or Quantity array or NumPy array
                 The spike times.  Does not have to be sorted.
    sampling_rate : float or Quantity scalar, optional
                    The sampling rate to use for the time points.
                    If not specified, retrieved from the `sampling_rate`
                    attribute of `spiketrain`.
    t_start : float or Quantity scalar, optional
              The start time to use for the time points.
              If not specified, retrieved from the `t_start`
              attribute of `spiketrain`.  If that is not present, default to
              `0`.  Any value from `spiketrain` below this value is
              ignored.
    t_stop : float or Quantity scalar, optional
             The start time to use for the time points.
             If not specified, retrieved from the `t_stop`
             attribute of `spiketrain`.  If that is not present, default to
             the maximum value of `sspiketrain`.  Any value from
             `spiketrain` above this value is ignored.
    return_times : bool
                   If True, also return the corresponding time points.

    Returns
    -------

    values : NumPy array of bools
             A `True``value at a particular index indicates the presence of
             one or more spikes at the corresponding time point.
    times : NumPy array or Quantity array, optional
            The time points.  This will have the same units as `spiketrain`.
            If `spiketrain` has no units, this will be an NumPy array.

    Notes
    -----
    Spike times are placed in the bin of the closest time point, going to the
    higher bin if exactly between two bins.

    So in the case where the bins are `5.5` and `6.5`, with the spike time
    being `6.0`, the spike will be placed in the `6.5` bin.

    The upper edge of the last bin, equal to `t_stop`, is inclusive.  That is,
    a spike time exactly equal to `t_stop` will be included.

    If `spiketrain` is a Quantity or Neo SpikeTrain and
    `t_start`, `t_stop` or `sampling_rate` is not, then the arguments that
    are not quantities will be assumed to have the same units as`spiketrain`.

    Raises
    ------

    TypeError
        If `spiketrain` is a NumPy array and `t_start`, `t_stop`, or
        `sampling_rate` is a Quantity..

    ValueError
        `t_start` and `t_stop` can be inferred from `spiketrain` if
        not explicitly defined and not an attribute of `spiketrain`.
        `sampling_rate` cannot, so an exception is raised if it is not
        explicitly defined and not present as an attribute of `spiketrain`.
    """
    # get the values from spiketrain if they are not specified.
    if sampling_rate is None:
        sampling_rate = getattr(spiketrain, 'sampling_rate', None)
        if sampling_rate is None:
            raise ValueError('sampling_rate must either be explicitly defined '
                             'or must be an attribute of spiketrain')
    if t_start is None:
        t_start = getattr(spiketrain, 't_start', 0)
    if t_stop is None:
        t_stop = getattr(spiketrain, 't_stop', np.max(spiketrain))

    # we don't actually want the sampling rate, we want the sampling period
    sampling_period = 1./sampling_rate

    # figure out what units, if any, we are dealing with
    if hasattr(spiketrain, 'units'):
        units = spiketrain.units
        spiketrain = spiketrain.magnitude
    else:
        units = None

    # convert everything to the same units, then get the magnitude
    if hasattr(sampling_period, 'units'):
        if units is None:
            raise TypeError('sampling_period cannot be a Quantity if '
                            'spiketrain is not a quantity')
        sampling_period = sampling_period.rescale(units).magnitude
    if hasattr(t_start, 'units'):
        if units is None:
            raise TypeError('t_start cannot be a Quantity if '
                            'spiketrain is not a quantity')
        t_start = t_start.rescale(units).magnitude
    if hasattr(t_stop, 'units'):
        if units is None:
            raise TypeError('t_stop cannot be a Quantity if '
                            'spiketrain is not a quantity')
        t_stop = t_stop.rescale(units).magnitude

    # figure out the bin edges
    edges = np.arange(t_start-sampling_period/2, t_stop+sampling_period*3/2,
                      sampling_period)
    # we don't want to count any spikes before t_start or after t_stop
    if edges[-2] > t_stop:
        edges = edges[:-1]
    if edges[1] < t_start:
        edges = edges[1:]
    edges[0] = t_start
    edges[-1] = t_stop

    # this is where we actually get the binarized spike train
    res = np.histogram(spiketrain, edges)[0].astype('bool')

    # figure out what to output
    if not return_times:
        return res
    elif units is None:
        return res, np.arange(t_start, t_stop+sampling_period, sampling_period)
    else:
        return res, pq.Quantity(np.arange(t_start, t_stop+sampling_period,
                                          sampling_period), units=units)


class binned_st:
    """
     Defining a binned spike train class
    """

    def __init__(self, x, binsize=None, num_bins=None, t_start=None, t_stop=None, store_mat=False):
        """
        Defines a binned spike train class

        Parameters
        ----------
        x : neo.core.SpikeTrain
            List of spike train objects to be binned.
        binsize : quantities.Quantity
            Width of each time bin.
        num_bins : int
            Number of bins of the binned spike train.
        t_start : quantities.Quantity
            Time of the first bin (left extreme; included).
        t_stop : quantities.Quantity
            Stopping time of the last bin (right extreme; excluded).
        store_mat : bool
            If set to True last calculated matrix will be stored in memory.
            If set to False matrix will always be calculated on demand.
        """
        # Converting x to a list if x is a spiketrain.
        if type(x) == neo.core.SpikeTrain:
            x = [x]

        # Check that x is a list of neo Spike trains.
        if not all([type(elem) == neo.core.SpikeTrain for elem in x]):
            raise TypeError("All elements of the input list must be neo.core.SpikeTrain objects ")

        # Set given parameter
        self.t_start = t_start
        self.t_stop = t_stop
        self.num_bins = num_bins
        self.binsize = binsize
        self.matrix_columns = num_bins
        self.matrix_rows = len(x)
        self.store_mat = store_mat
        #Empty matrix for storage
        self.mat = None
        #Bool for storing unclipped/clipped matrix
        self.__unclipped = False
        self.__clipped = False
        # Check all parameter, set also missing values
        if self.t_start is None or self.t_stop is None:
            self.__set_start_stop_from_input(x)
        self.__check_init_params(binsize, num_bins, t_start, t_stop)
        self.__check_consistency(x, self.binsize, self.num_bins, self.t_start, self.t_stop)
        self.filled = []  # contains the index of the bins
        #Now create filled
        self.from_neo(x, self.binsize)
        #self.__check_input_length()

    #========================================================================================================
    # There are four cases the given parameters must fulfill
    # Each parameter must be a combination of following order or it will raise a value error:
    # t_start, num_bins, binsize
    # t_start, num_bins, t_stop
    # t_start, bin_size, t_stop
    # t_stop, num_bins, binsize
    #=========================================================================================================

    def __check_init_params(self, binsize, num_bins, t_start, t_stop):
        """
        Checks for given parameter.
        Otherwise it raises a ValueError. Also calculates the missing
        parameter.

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
            If the check fails a ValueError is raised.
        """
        # Raise error if no argument is given
        if binsize is None and t_start is None and t_stop is None and num_bins is None:
            raise ValueError("No arguments given. Please enter at least three arguments")
        # Check if num_bins is an integer (special case)
        if num_bins is not None:
            if type(num_bins) is not int:
                raise TypeError("num_bins is not an integer!")
        # Check if all parameters can be calculated, otherwise raise ValueError
        if t_start is None:
            self.t_start = self.__calc_tstart(num_bins, binsize, t_stop)
        elif t_stop is None:
            self.t_stop = self.__calc_tstop(num_bins, binsize, t_start)
        elif num_bins is None:
            self.num_bins = self.__calc_num_bins(binsize, t_start, t_stop)
            if self.matrix_columns is None:
                self.matrix_columns = self.num_bins
        elif binsize is None:
            self.binsize = self.__calc_binsize(num_bins, t_start, t_stop)

    #Routine Methods for calculating missing params
    def __calc_tstart(self, num_bins, binsize, t_stop):
        if num_bins is not None and binsize is not None and t_stop is not None:
            return t_stop.rescale(binsize.units) - num_bins * binsize
        else:
            raise ValueError(
                "Insufficient input arguments. Please provide at least one of the following arguments: "
                "(num_bins, binsize, t_stop)")

    def __calc_tstop(self, num_bins, binsize, t_start):
        if num_bins is not None and binsize is not None and t_start is not None:
            return t_start.rescale(binsize.units) + num_bins * binsize
        else:
            raise ValueError(
                "Insufficient input arguments. Please provide at least one of the following arguments: "
                "(num_bins, binsize, t_start)")

    def __calc_num_bins(self, binsize, t_start, t_stop):
        if binsize is not None and t_start is not None and t_stop is not None:
            return int(((t_stop - t_start).rescale(binsize.units) / binsize).magnitude)
        else:
            raise ValueError(
                "Insufficient input arguments. Please provide at least one of the following arguments: "
                "(binsize, t_start, t_stop)")

    def __calc_binsize(self, num_bins, t_start, t_stop):
        if num_bins is not None and t_start is not None and t_stop is not None:
            return (t_stop - t_start) / num_bins
        else:
            raise ValueError(
                "Insufficient input arguments. Please provide at least one of the following arguments: "
                "(num_bins, t_start, t_stop)")

    def __check_consistency(self, x, binsize, num_bins, t_start, t_stop):
        """
        Checks the given parameters for consistency

        There are special requirements for creating a matrix, which represents the binned structure of a spike train.

        """
        t_starts = [elem.t_start for elem in x]
        t_stops = [elem.t_stop for elem in x]
        max_tstart = max(t_starts)
        min_tstop = min(t_stops)
        if max_tstart >= min_tstop:
            raise ValueError("Starting time of each spike train must be smaller than each stopping time")
        else:
            if t_start < max_tstart or t_start > min_tstop:
                raise ValueError('some spike trains are not defined in the time given by t_start')
            if num_bins != int(((t_stop-t_start)/binsize).rescale(pq.dimensionless).magnitude):
                raise ValueError("Inconsistent arguments t_start, t_stop, binsize and num_bins")
            if not (t_start < t_stop <= min_tstop):
                raise ValueError('too many / too large time bins. Some spike trains are not defined in the ending time')
            if num_bins - int(num_bins) != 0 or num_bins < 0:
                raise TypeError("Number of bins (num_bins) is not an integer: " + str(num_bins))
            if t_stop > min_tstop or t_stop < max_tstart:
                raise ValueError('some spike trains are not defined in the time given by t_stop')
            if self.matrix_columns < 1 or self.num_bins < 1:
                warn("Calculated matrix columns and/or num_bins are smaller than 1: (%s, %s). "
                     "Please check your input parameter." % (self.matrix_columns, self.num_bins))

    def __check_input_length(self, other=None):
        if other is None:
            if np.all([type(self.filled[i]).__name__ == list for i in xrange(len(self.filled))]):
                if not np.all([len(self.filled[0]) == len(self.filled[i]) for i in xrange(len(self.filled))]):
                    warn("Inconsistent length of input elements.")
                    logging.critical("Inconsistent length of input element. "
                                     "\n May cause an exception when using adding op or calculating the matrix.")
        else:
            pass

    def __set_start_stop_from_input(self, x):
        if self.t_stop is None:
            self.t_start = max([elem.t_start for elem in x])
        if self.t_stop is None:
            self.t_stop = min([elem.t_stop for elem in x])

    @property
    def filled(self):
        return self.filled

    @filled.setter
    def filled(self, f):
        self.filled = f

    def matrix_clipped(self):
        """
        Calculates a matrix, which rows represent the number of spike trains and the columns represent the binned
        index position of a spike in a spike train.
        The calculated matrix columns contain only ones, which indicate a spike.
        If **bool** `store_mat` is set to **True** last calculated `clipped` matrix will be returned.

        Returns
        -------
        clipped matrix : numpy.ndarray
            Matrix with ones indicating a spike and zeros for non spike. The ones in the columns represent the index
            position of the spike in the spike train and rows represent the number of spike trains.

        Example
        -------
        >>> import jelephant.core.rep as e
        >>> import neo as n
        >>> import quantities as pq
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> x = e.binned_st(a, num_bins=10, binsize=1 * pq.s, t_start=0 * pq.s)
        >>> print x.matrix_clipped()
        >>> [[ 1.  1.  0.  1.  1.  1.  1.  0.  0.  0.]]
        """
        if self.mat is not None and self.__clipped is True:
            #print "Clipped Matrix already calculated and will be returned"
            return self.mat
        #Matrix shall be stored
        if self.store_mat:
            self.__clipped = True
            self.__unclipped = False
            self.mat = np.zeros((self.matrix_rows, self.matrix_columns))
            for elem_idx, elem in enumerate(self.filled):
                # if the spike train had no spikes, pass
                if len(elem) == 0:
                    pass
                else:
                    try:
                        self.mat[elem_idx, elem] = 1
                    except IndexError as ie:
                        raise IndexError(str(ie) + "\n You are trying to build a matrix which is inconsistent in size. "
                                                   "Please check your input parameter.")
                return self.mat
        #Matrix on demand
        else:
            tmp_mat = np.zeros((self.matrix_rows, self.matrix_columns))  # temporary matrix
            for elem_idx, elem in enumerate(self.filled):
                # if the spike train had no spikes, pass
                if len(elem) == 0:
                    pass
                else:
                    try:
                        tmp_mat[elem_idx, elem] = 1
                    except IndexError as ie:
                        raise IndexError(str(ie) + "\n You are trying to build a matrix which is inconsistent in size. "
                                               "Please check your input parameter.")
            return tmp_mat

    def matrix_unclipped(self):
        """

        Calculates a matrix, which rows represents the number of spike trains and the columns represents the binned
        index position of a spike in a spike train.
        The calculated matrix columns contain the number of spikes that occurred in the spike train(s).
        If **bool** `store_mat` is set to **True** last calculated `unclipped` matrix will be returned.

        Returns
        -------
        unclipped matrix : numpy.ndarray
            Matrix with spike times. Columns represent the index position of the
            binned spike and rows represent the number of spike trains.

        Example
        -------
        >>> import jelephant.core.rep as e
        >>> import neo as n
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> x = e.binned_st(a, num_bins=10, binsize=1 * pq.s, t_start=0 * pq.s)
        >>> print x.matrix_unclipped()
        >>> [[ 2.  1.  0.  1.  1.  1.  1.  0.  0.  0.]]
        """
        if self.mat is not None and self.__unclipped is True:
            #print "Unclipped Matrix already calculated and will be returned"
            return self.mat
        if self.store_mat:
            self.__unclipped = True
            self.__clipped = False
            self.mat = np.zeros((self.matrix_rows, self.matrix_columns))
            for elem_idx, elem in enumerate(self.filled):
                if len(elem) == 0:
                    pass
                else:
                    try:
                        if len(elem) >= 1:
                            for inner_elem in elem:
                                self.mat[elem_idx, inner_elem] += 1
#                        else:
#                            self.mat[elem_idx, elem[0]] += 1
                    except IndexError as ie:
                        raise IndexError(str(ie) + "\n You are trying to build a matrix which is inconsistent in size. "
                                                   "Please check your input parameter.")
            return self.mat
        #Matrix on demand
        else:
            tmp_mat = np.zeros((self.matrix_rows, self.matrix_columns))
            for elem_idx, elem in enumerate(self.filled):
                if len(elem) == 0:
                    pass
                else:
                    try:
                        if len(elem) > 1:
                            for inner_elem in elem:
                                tmp_mat[elem_idx, inner_elem] += 1
                        else:
                            tmp_mat[elem_idx, elem[0]] += 1
                    except IndexError as ie:
                        raise IndexError(str(ie) + "\n You are trying to build a matrix which is inconsistent in size. "
                                                   "Please check your input parameter.")
            return tmp_mat

    def from_neo(self, x, binsize):
        """

        Converts Neo SpikeTrain Object to a list of numpy.ndarray's called **filled**, which contains the binned times

        Parameters
        ----------
        x : neo.SpikeTrain
            A Neo SpikeTrain Object.
        binsize : quantity.Quantity
            Size of bins

        Example
        -------
        >>> import jelephant as e
        >>> import neo as n
        >>> import quantities as pq
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> x = e.binned_st(a, num_bins=10, binsize=1 * pq.s, t_start=0 * pq.s)
        >>> print x.filled
        >>> [array([0, 0, 1, 3, 4, 5, 6])]
        """
        for elem in x:
            idx_filled = np.array(
                ((elem.view(pq.Quantity) - self.t_start).rescale(self.binsize.units) / binsize).magnitude, dtype=int)
            self.filled.append(idx_filled[idx_filled < self.num_bins])

    def __eq__(self, y):
        return np.array_equal(self.filled, y.filled)

    def __add__(self, other):
        """
        Overloads + operator
        Beware a new object is created! That means parameter of Object A of (A+B) are copied.
        But the spiketrain is lost.
        """
        new_class = self.__create_dummy_class()
        #filled is the important structure to change
        #For merged spiketrains, or when filled has more than one binned spiketrain
        if len(self.filled) > 1 or len(other.filled) > 1:
            new_class.filled = [self.filled, other.filled]
        else:
            new_class.filled = np.hstack((self.filled, other.filled))
        return new_class

    def prune(self):
        """
        Prunes the `filled` list, so that each element contains no duplicated values any more

        Returns
        -------
        self : class binnsed_st
              Returns a new class with a pruned `filled` list.
        """
        if len(self.filled) > 1:
            self.filled = [np.unique(elems) for elems in self.filled]
        else:
            self.filled = [np.unique(np.asarray(self.filled))]
        return self

    def __iadd__(self, other):
        """
        Overloads += operator
        """
        #Create new object; if object is not necessary, only __add__ could be returned
        new_self = self.__add__(other)
        #Set missing parameter
        new_self.binsize = self.binsize
        new_self.t_start = self.t_start
        new_self.t_stop = self.t_stop
        new_self.num_bins = self.num_bins
        return new_self

    def __sub__(self, other):
        import itertools
        new_class = self.__create_dummy_class()
        #The cols and rows has to be equal to the rows and cols of self and other
        new_class.matrix_columns = self.matrix_columns
        new_class.matrix_rows = self.matrix_rows
        #Clear the list
        del new_class.filled[:]
        if len(self.filled) > 1 or len(other.filled) > 1:
            index = 0
            for s, o in itertools.izip(self.filled, other.filled):
                new_class.filled[index] = np.array(set(s) ^ set(o))
                index += 1
        else:
            new_class.filled.append(np.setxor1d(self.filled[0], other.filled[0]))
            if not len(new_class.filled[0] > 0):
                new_class.filled[0] = np.zeros(len(self.filled[0]))
        return new_class

    def __isub__(self, other):
        new_self = self.__sub__(other)
        #Set missing parameter
        new_self.binsize = self.binsize
        new_self.t_start = self.t_start
        new_self.t_stop = self.t_stop
        new_self.num_bins = self.num_bins
        return new_self

    def __create_dummy_class(self):
        #This is super hackish, so that the method works somehow
        import neo as n
        #Dummy SpikeTrain is created to fool the constructor
        spk = n.SpikeTrain([0]*pq.s, t_stop=self.t_stop)
        #Create a new dummy class to return
        dummy_class = binned_st(spk, t_start=self.t_start, t_stop=self.t_stop, binsize=self.binsize)
        #The cols and rows has to be equal to the rows and cols of self and other
        dummy_class.matrix_columns = self.matrix_columns
        dummy_class.matrix_rows = self.matrix_rows
        return dummy_class
