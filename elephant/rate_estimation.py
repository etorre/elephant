'''
Module: jelephant.analysis.rate_estimation

Contains utility functions for instantaneous rate estimation of SpikeTrains.
'''

import quantities as pq
import numpy as np
import numpy
import scipy.signal
import neo

#def make_kernel(form, sigma, resolution, direction=1):
#
#
#    """Creates kernel functions for convolution.
#
#    Constructs a numeric linear convolution kernel of basic shape to be used
#    for data smoothing (linear low pass filtering) and firing rate estimation
#    from single trial or trial-averaged spike trains.
#
#    Exponential and alpha kernels may also be used to represent postynaptic
#    currents / potentials in a linear (current-based) model.
#
#    Adapted from original script written by Martin P. Nawrot for the
#    FIND MATLAB toolbox [1]_ [2]_.
#
#    Parameters
#    ----------
#    form : {'BOX', 'TRI', 'GAU', 'EPA', 'EXP', 'ALP'}
#        Kernel form. Currently implemented forms are BOX (boxcar),
#        TRI (triangle), GAU (gaussian), EPA (epanechnikov), EXP (exponential),
#        ALP (alpha function). EXP and ALP are aymmetric kernel forms and
#        assume optional parameter `direction`.
#    sigma : float
#        Standard deviation of the distribution associated with kernel shape.
#        This parameter defines the time resolution of the kernel estimate
#        and makes different kernels comparable (cf. [1] for symetric kernels).
#        This is used here as an alternative definition to the cut-off
#        frequency of the associated linear filter.
#    time_stamp_resolution : float
#        Temporal resolution of input and output.
#    direction : {-1, 1}
#        Asymmetric kernels have two possible directions.
#        The values are -1 or 1, default is 1. The
#        definition here is that for direction = 1 the
#        kernel represents the impulse response function
#        of the linear filter. Default value is 1.
#
#    Returns
#    -------
#    kernel : array_like
#        Array of kernel. The length of this array is always an odd
#        number to represent symmetric kernels such that the center bin
#        coincides with the median of the numeric array, i.e for a
#        triangle, the maximum will be at the center bin with equal
#        number of bins to the right and to the left.
#   norm : float
#        For rate estimates. The kernel vector is normalized such that
#        the sum of all entries equals unity sum(kernel)=1. When
#        estimating rate functions from discrete spike data (0/1) the
#        additional parameter `norm` allows for the normalization to
#        rate in spikes per second.
#
#        For example:
#        ``rate = norm * scipy.signal.lfilter(kernel, 1, spike_data)``
#    m_idx : int
#        Index of the numerically determined median (center of gravity)
#        of the kernel function.
#
#    Examples
#    --------
#    To obtain single trial rate function of trial one should use::
#
#        r = norm * scipy.signal.fftconvolve(sua, kernel)
#
#    To obtain trial-averaged spike train one should use::
#
#        r_avg = norm * scipy.signal.fftconvolve(sua, np.mean(X,1))
#
#    where `X` is an array of shape `(l,n)`, `n` is the number of trials and
#    `l` is the length of each trial.
#
#    See also
#    --------
#    SpikeTrain.instantaneous_rate
#    SpikeList.averaged_instantaneous_rate
#
#    .. [1] Meier R, Egert U, Aertsen A, Nawrot MP, "FIND - a unified framework
#       for neural data analysis"; Neural Netw. 2008 Oct; 21(8):1085-93.
#
#    .. [2] Nawrot M, Aertsen A, Rotter S, "Single-trial estimation of neuronal
#       firing rates - from single neuron spike trains to population activity";
#       J. Neurosci Meth 94: 81-92; 1999.
#
#    """
#    assert form.upper() in ('BOX','TRI','GAU','EPA','EXP','ALP'), "form must \
#    be one of either 'BOX','TRI','GAU','EPA','EXP' or 'ALP'!"
#
#    assert direction in (1,-1), "direction must be either 1 or -1"
#
#    # conversion to SI units (s)
#    SI_sigma = sigma.rescale('s').magnitude
#    SI_time_stamp_resolution = resolution.rescale('s').magnitude
#
#    norm = 1./SI_time_stamp_resolution
#
#    if form.upper() == 'BOX':
#        w = 2.0 * SI_sigma * np.sqrt(3)
#        width = 2 * np.floor(w / 2.0 / SI_time_stamp_resolution) + 1  # always odd number of bins
#        height = 1. / width
#        kernel = np.ones((1, width)) * height  # area = 1
#
#    elif form.upper() == 'TRI':
#        w = 2 * SI_sigma * np.sqrt(6)
#        halfwidth = np.floor(w / 2.0 / SI_time_stamp_resolution)
#        trileft = np.arange(1, halfwidth + 2)
#        triright = np.arange(halfwidth, 0, -1)  # odd number of bins
#        triangle = np.append(trileft, triright)
#        kernel = triangle / triangle.sum()  # area = 1
#
#    elif form.upper() == 'EPA':
#        w = 2.0 * SI_sigma * np.sqrt(5)
#        halfwidth = np.floor(w / 2.0 / SI_time_stamp_resolution)
#        base = np.arange(-halfwidth, halfwidth + 1)
#        parabula = base**2
#        epanech = parabula.max() - parabula  # inverse parabula
#        kernel = epanech / epanech.sum()  # area = 1
#
#    elif form.upper() == 'GAU':
#        w = 2.0 * SI_sigma * 2.7  # > 99% of distribution weight
#        halfwidth = np.floor(w / 2.0 / SI_time_stamp_resolution)  # always odd
#        base = np.arange(-halfwidth, halfwidth + 1) * SI_time_stamp_resolution
#        g = np.exp(-(base**2) / 2.0 / SI_sigma**2) / SI_sigma / np.sqrt(2.0 * np.pi)
#        kernel = g / g.sum()
#
#    elif form.upper() == 'ALP':
#        w = 5.0 * SI_sigma
#        alpha = np.arange(1, (2.0 * np.floor(w / SI_time_stamp_resolution
#                                             / 2.0) + 1) + 1) * SI_time_stamp_resolution
#        alpha = (2.0 / SI_sigma**2) * alpha * np.exp(-alpha * np.sqrt(2) / SI_sigma)
#        kernel = alpha / alpha.sum()  # normalization
#        if direction == -1:
#            kernel = np.flipud(kernel)
#
#    elif form.upper() == 'EXP':
#        w = 5.0 * SI_sigma
#        expo = np.arange(1, (2.0 * np.floor(w / SI_time_stamp_resolution
#                                            / 2.0) + 1) + 1) * SI_time_stamp_resolution
#        expo = np.exp(-expo / SI_sigma)
#        kernel = expo / expo.sum()
#        if direction == -1:
#            kernel = np.flipud(kernel)
#
#    kernel = kernel.ravel()
#    m_idx = np.nonzero(kernel.cumsum() >= 0.5)[0].min()
#
#    return kernel, norm, m_idx
#
#
#
#
#def instantaneous_rate(spiketrain, resolution, kernel, norm, m_idx=None,
#                       t_start=None, t_stop=None, acausal=True, trim=False):
#
#    """
#    Estimate instantaneous firing rate by kernel convolution.
#
#    Inputs:
#        resolution  - time stamp resolution of the spike times. the
#                      same resolution will be assumed for the kernel
#        kernel      - kernel function used to convolve with
#        norm        - normalization factor associated with kernel function
#                      (see analysis.make_kernel for details)
#        t_start     - start time of the interval used to compute the firing
#                      rate
#        t_stop      - end time of the interval used to compute the firing
#                      rate (included)
#        acausal     - if True, acausal filtering is used, i.e., the gravity
#                      center of the filter function is aligned with the
#                      spike to convolve
#        m_idx       - index of the value in the kernel function vector that
#                      corresponds to its gravity center. this parameter is
#                      not mandatory for symmetrical kernels but it is
#                      required when assymmetrical kernels are to be aligned
#                      at their gravity center with the event times
#        trim        - if True, only the 'valid' region of the convolved
#                      signal are returned, i.e., the points where there
#                      isn't complete overlap between kernel and spike train
#                      are discarded
#                      NOTE: if True and an assymetrical kernel is provided
#                      the output will not be aligned with [t_start, t_stop]
#
#    See also:
#        analysis.make_kernel
#    """
#
#    units = pq.CompoundUnit("%s*s"%str(resolution.rescale('s').magnitude))
#    print(units)
#    spiketrain = spiketrain.rescale(units)
#    if t_start is None:
#        t_start = spiketrain.t_start
#    else:
#        t_start = t_start.rescale(spiketrain.units)
#
#    if t_stop is None:
#        t_stop = spiketrain.t_stop
#    else:
#        t_stop = t_stop.rescale(spiketrain.units)
#
#    if m_idx is None:
#        m_idx = kernel.size / 2
#
#    print(t_stop)
#
#    time_vector = numpy.zeros(int((t_stop - t_start)) + 1)
#
#    spikes_slice = []
#    if len(spiketrain):
#        spikes_slice = spiketrain.time_slice(t_start,t_stop)
#
#    for spike in spikes_slice:
#        index = int((spike - t_start))
#        time_vector[index] = 1
#
#    r = norm * scipy.signal.fftconvolve(time_vector, kernel, 'full')
#
#    if acausal is True:
#        if trim is False:
#            r = r[m_idx:-(kernel.size - m_idx)]
#            t_axis = numpy.linspace(t_start, t_stop, r.size)
#            return t_axis, r
#
#        elif trim is True:
#            r = r[2 * m_idx:-2*(kernel.size - m_idx)]
#            t_start = t_start + m_idx * spiketrain.units
#            t_stop = t_stop - ((kernel.size) - m_idx) * spiketrain.units
#            t_axis = numpy.linspace(t_start, t_stop, r.size)
#            return t_axis, r
#
#    if acausal is False:
#        if trim is False:
#            r = r[m_idx:-(kernel.size - m_idx)]
#            t_axis = (numpy.linspace(t_start, t_stop, r.size) +
#                      m_idx * spiketrain.units)
#            return t_axis, r
#
#        elif trim is True:
#            r = r[2 * m_idx:-2*(kernel.size - m_idx)]
#            t_start = t_start + m_idx * spiketrain.units
#            t_stop = t_stop - ((kernel.size) - m_idx) * spiketrain.units
#            t_axis = (numpy.linspace(t_start, t_stop, r.size) +
#                      m_idx * spiketrain.units)
#            return t_axis, r

#adaptation to output neo.AnalogSignal and wapper make_kernel() in
#instantaneous_rate()
def make_kernel(form, sigma, resolution, direction=1):
    """Creates kernel functions for convolution.

    Constructs a numeric linear convolution kernel of basic shape to be used
    for data smoothing (linear low pass filtering) and firing rate estimation
    from single trial or trial-averaged spike trains.

    Exponential and alpha kernels may also be used to represent postynaptic
    currents / potentials in a linear (current-based) model.

    Adapted from original script written by Martin P. Nawrot for the
    FIND MATLAB toolbox [1]_ [2]_.

    Parameters
    ----------
    form : {'BOX', 'TRI', 'GAU', 'EPA', 'EXP', 'ALP'}
        Kernel form. Currently implemented forms are BOX (boxcar),
        TRI (triangle), GAU (gaussian), EPA (epanechnikov), EXP (exponential),
        ALP (alpha function). EXP and ALP are aymmetric kernel forms and
        assume optional parameter `direction`.
    sigma : float
        Standard deviation of the distribution associated with kernel shape.
        This parameter defines the time resolution of the kernel estimate
        and makes different kernels comparable (cf. [1] for symetric kernels).
        This is used here as an alternative definition to the cut-off
        frequency of the associated linear filter.
    time_stamp_resolution : float
        Temporal resolution of input and output.
    direction : {-1, 1}
        Asymmetric kernels have two possible directions.
        The values are -1 or 1, default is 1. The
        definition here is that for direction = 1 the
        kernel represents the impulse response function
        of the linear filter. Default value is 1.

    Returns
    -------
    kernel : array_like
        Array of kernel. The length of this array is always an odd
        number to represent symmetric kernels such that the center bin
        coincides with the median of the numeric array, i.e for a
        triangle, the maximum will be at the center bin with equal
        number of bins to the right and to the left.
   norm : float
        For rate estimates. The kernel vector is normalized such that
        the sum of all entries equals unity sum(kernel)=1. When
        estimating rate functions from discrete spike data (0/1) the
        additional parameter `norm` allows for the normalization to
        rate in spikes per second.

        For example:
        ``rate = norm * scipy.signal.lfilter(kernel, 1, spike_data)``
    m_idx : int
        Index of the numerically determined median (center of gravity)
        of the kernel function.

    Examples
    --------
    To obtain single trial rate function of trial one should use::

        r = norm * scipy.signal.fftconvolve(sua, kernel)

    To obtain trial-averaged spike train one should use::

        r_avg = norm * scipy.signal.fftconvolve(sua, np.mean(X,1))

    where `X` is an array of shape `(l,n)`, `n` is the number of trials and
    `l` is the length of each trial.

    See also
    --------
    SpikeTrain.instantaneous_rate
    SpikeList.averaged_instantaneous_rate

    .. [1] Meier R, Egert U, Aertsen A, Nawrot MP, "FIND - a unified framework
       for neural data analysis"; Neural Netw. 2008 Oct; 21(8):1085-93.

    .. [2] Nawrot M, Aertsen A, Rotter S, "Single-trial estimation of neuronal
       firing rates - from single neuron spike trains to population activity";
       J. Neurosci Meth 94: 81-92; 1999.

    """
    assert form.upper() in (
        'BOX', 'TRI', 'GAU', 'EPA', 'EXP', 'ALP'), "form must \
    be one of either 'BOX','TRI','GAU','EPA','EXP' or 'ALP'!"

    assert direction in (1, -1), "direction must be either 1 or -1"

    # conversion to SI units (s)
    SI_sigma = sigma.rescale('s').magnitude
    SI_time_stamp_resolution = resolution.rescale('s').magnitude

    norm = 1./SI_time_stamp_resolution

    if form.upper() == 'BOX':
        w = 2.0 * SI_sigma * np.sqrt(3)
        # always odd number of bins
        width = 2 * np.floor(w / 2.0 / SI_time_stamp_resolution) + 1
        height = 1. / width
        kernel = np.ones((1, width)) * height  # area = 1

    elif form.upper() == 'TRI':
        w = 2 * SI_sigma * np.sqrt(6)
        halfwidth = np.floor(w / 2.0 / SI_time_stamp_resolution)
        trileft = np.arange(1, halfwidth + 2)
        triright = np.arange(halfwidth, 0, -1)  # odd number of bins
        triangle = np.append(trileft, triright)
        kernel = triangle / triangle.sum()  # area = 1

    elif form.upper() == 'EPA':
        w = 2.0 * SI_sigma * np.sqrt(5)
        halfwidth = np.floor(w / 2.0 / SI_time_stamp_resolution)
        base = np.arange(-halfwidth, halfwidth + 1)
        parabula = base**2
        epanech = parabula.max() - parabula  # inverse parabula
        kernel = epanech / epanech.sum()  # area = 1

    elif form.upper() == 'GAU':
        w = 2.0 * SI_sigma * 2.7  # > 99% of distribution weight
        halfwidth = np.floor(w / 2.0 / SI_time_stamp_resolution)  # always odd
        base = np.arange(-halfwidth, halfwidth + 1) * SI_time_stamp_resolution
        g = np.exp(
            -(base**2) / 2.0 / SI_sigma**2) / SI_sigma / np.sqrt(2.0 * np.pi)
        kernel = g / g.sum()

    elif form.upper() == 'ALP':
        w = 5.0 * SI_sigma
        alpha = np.arange(
            1, (
                2.0 * np.floor(w / SI_time_stamp_resolution / 2.0) + 1) +
            1) * SI_time_stamp_resolution
        alpha = (2.0 / SI_sigma**2) * alpha * np.exp(
            -alpha * np.sqrt(2) / SI_sigma)
        kernel = alpha / alpha.sum()  # normalization
        if direction == -1:
            kernel = np.flipud(kernel)

    elif form.upper() == 'EXP':
        w = 5.0 * SI_sigma
        expo = np.arange(
            1, (
                2.0 * np.floor(w / SI_time_stamp_resolution / 2.0) + 1) +
            1) * SI_time_stamp_resolution
        expo = np.exp(-expo / SI_sigma)
        kernel = expo / expo.sum()
        if direction == -1:
            kernel = np.flipud(kernel)

    kernel = kernel.ravel()
    m_idx = np.nonzero(kernel.cumsum() >= 0.5)[0].min()

    return kernel, norm, m_idx


def instantaneous_rate(spiketrain, resolution, form, sigma, m_idx=None,
                       t_start=None, t_stop=None, acausal=True, trim=False):

    """
    Estimate instantaneous firing rate by kernel convolution.

    Parameters
    -----------
    resolution : Quantity
        time stamp resolution of the spike times. the same resolution will
        be assumed for the kernel
    form : {'BOX', 'TRI', 'GAU', 'EPA', 'EXP', 'ALP'}
        Kernel form. Currently implemented forms are BOX (boxcar),
        TRI (triangle), GAU (gaussian), EPA (epanechnikov), EXP (exponential),
        ALP (alpha function). EXP and ALP are aymmetric kernel forms and
        assume optional parameter `direction`.
    sigma : Quantity
        Standard deviation of the distribution associated with kernel shape.
        This parameter defines the time resolution of the kernel estimate
        and makes different kernels comparable (cf. [1] for symetric kernels).
        This is used here as an alternative definition to the cut-off
        frequency of the associated linear filter.
    t_start : Quantity (Optional)
        start time of the interval used to compute the firing rate, if None
        assumed equal to spiketrain.t_start
        Default:None
    t_stop : Qunatity
        End time of the interval used to compute the firing rate (included).
        If none assumed equal to spiketrain.t_stop
        Default:None
    acausal : bool
        if True, acausal filtering is used, i.e., the gravity center of the
        filter function is aligned with the spike to convolve
        Default:None
    m_idx : int
        index of the value in the kernel function vector that corresponds
        to its gravity center. this parameter is not mandatory for
        symmetrical kernels but it is required when assymmetrical kernels
        are to be aligned at their gravity center with the event times if None
        is assued to be the median value of the kernel support
        Default : None
    trim : bool
        if True, only the 'valid' region of the convolved
        signal are returned, i.e., the points where there
        isn't complete overlap between kernel and spike train
        are discarded
        NOTE: if True and an assymetrical kernel is provided
        the output will not be aligned with [t_start, t_stop]

    See also:
        analysis.make_kernel
    """
    kernel, norm, m_idx = make_kernel(
        form=form, sigma=sigma, resolution=resolution)
    units = pq.CompoundUnit("%s*s" % str(resolution.rescale('s').magnitude))
    spiketrain = spiketrain.rescale(units)
    if t_start is None:
        t_start = spiketrain.t_start
    else:
        t_start = t_start.rescale(spiketrain.units)

    if t_stop is None:
        t_stop = spiketrain.t_stop
    else:
        t_stop = t_stop.rescale(spiketrain.units)

    if m_idx is None:
        m_idx = kernel.size / 2

    time_vector = numpy.zeros(int((t_stop - t_start)) + 1)

    spikes_slice = []
    if len(spiketrain):
        spikes_slice = spiketrain.time_slice(t_start, t_stop)

    for spike in spikes_slice:
        index = int((spike - t_start))
        time_vector[index] = 1

    r = norm * scipy.signal.fftconvolve(time_vector, kernel, 'full')

    if acausal is True:
        if trim is False:
            r = r[m_idx:-(kernel.size - m_idx)]
            rate = neo.AnalogSignal(
                signal=r,  sampling_period=resolution, units=pq.Hz,
                t_start=t_start)
            return rate

        elif trim is True:
            r = r[2 * m_idx:-2*(kernel.size - m_idx)]
            t_start = t_start + m_idx * spiketrain.units
            t_stop = t_stop - ((kernel.size) - m_idx) * spiketrain.units
            rate = neo.AnalogSignal(
                signal=r,  sampling_period=resolution, units=pq.Hz,
                t_start=t_start)
            return rate

    if acausal is False:
        if trim is False:
            r = r[m_idx:-(kernel.size - m_idx)]
            rate = neo.AnalogSignal(
                signal=r,  sampling_period=resolution, units=pq.Hz,
                t_start=t_start)
            return rate

        elif trim is True:
            r = r[2 * m_idx:-2*(kernel.size - m_idx)]
            t_start = t_start + m_idx * spiketrain.units
            t_stop = t_stop - ((kernel.size) - m_idx) * spiketrain.units
            rate = neo.AnalogSignal(
                signal=r,  sampling_period=resolution, units=pq.Hz,
                t_start=t_start)
            return rate
