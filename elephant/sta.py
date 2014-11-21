# -*- coding: utf-8 -*-
'''
Module: jelephant.analysis.sta

Contains functions to calculate spike-triggered averages of AnalogSignals.
'''


import numpy as np
import scipy.signal
import quantities as pq
from neo.core import AnalogSignal, AnalogSignalArray



if __name__ == '__main__':
    pass


#===============================================================================
# Spike-triggered average main functions
#===============================================================================
def sta(lfps, spiketrains, window, method="correlation", crosstrials=False, single_data=None):
    """
    Calls the resective sta function specified by 'method'. 'method' can either be 'correlation' for
    correlation-based STA calculation or 'average' for average-based STA calculation.

    **Args**:
        lfps: AnalogSignal object or AnalogSignalArray object or list of AnalogSignals
        spikes: SpikeTrain or list of SpikeTrains objects, its time intervall needs to
            be completely covered by the lfp
        window: positive time interval to specify the cutout around spikes given as Quantity or
            number of bins to use
        method: default 'correlation'. Specifies method to calculate STA
        crosstrials: indicates if STA is averaged over all provided trials or calculated trial-wise
            default value 'False'
            True: STA is calculated for each pair of lfp and spiketrain an is averaged afterwards
            False: STAs are calculated for each pair of lfp and spiketrain and are returned as list
        single_data: (None,'train','lfp')
            specifies whether one (first) spiketrain is used for all STAs ('train'),
            each AnalogSignal comes with its own spiketrain (None, Default) or one (first)
            Analogsignal is used for all spiketrains ('lfp') Default value 'None'

    **Return**:
        Returns a tuple (STA,time,used_spikes), where STA is a list of one-dimensional arrays with
        the spike triggered average, and time is a list of the corresponding time bins.
        The length of the respective array is defined by 2*window + 1, where window is
        the number of bins around the spike times used.
        used_spikes contains the number of spikes used for the STA. If the spiketrain did
        not contain suitable spikes, the returned STA will be filled with zeros.

    **Example**:
        (result,time,used_spikes)=sta_corr(lfp,spiketrain,Quantity(10,"ms"))
        matplotlib.pyplot.plot(time,result)
    """

    if single_data == 'lfp':
        # ## 1 ### In case of single lfp provided

        # wrapping spiketrains
        if isinstance(spiketrains, np.ndarray):
            box = []
            box.append(spiketrains)
            spiketrains = box



        # if (lfps has usefull type)
        if (isinstance(lfps, np.ndarray) or isinstance(lfps, list)):
            # (itselfe is data, but also contains lists) or (contains only one list as first element))
            if  (len(lfps) > 1 and (isinstance(lfps[0], list) or isinstance(lfps[0], np.ndarray))):
                pass
            elif (len(lfps == 1) and not(isinstance(lfps[0], list) or isinstance(lfps[0], np.ndarray))):
                # unwrapping lfps
                lfps = lfps[0]
            else:
                raise ValueError("There is no single lfp signal present in the supplied lfp signal")
        else:
            raise ValueError("Supplied LFP does not have the correct data format but %s" % (str(type(lfps))))


        loops = len(spiketrains)



        result = []
        for i in range(loops):
            if method == "corr" or method == "correlation":
                result.append(sta_corr(lfps, spiketrains[i], window, crosstrials, single_data))
            elif method == "aver" or method == "average":
                result.append(sta_average(lfps, spiketrains[i], window, crosstrials, single_data))
            else:
                raise ValueError("Specified STA method is not available. Please use 'correlation' or 'average'")

        if single_data == 'lfp':
            return averaging_STAs([a[0] for a in result], [a[2] for a in result]), result[0][1], np.sum([a[2] for a in result])

        return result[0]


    # ## 2 ### normal calling of sta function in case of single_data != 'lfp'
    if method == "corr" or method == "correlation":
        return (sta_corr(lfps, spiketrains, window, crosstrials, single_data))
    elif method == "aver" or method == "average":
        return (sta_average(lfps, spiketrains, window, crosstrials, single_data))
    else:
        raise ValueError("Specified STA method is not available. Please use 'correlation' or 'average'")





def sta_corr(lfps, spiketrains, window, crosstrials=False, single_data=None):
    """
    Calculates the respective spike-triggered average of a analog signals of multiple trials
    by binning the spiketrain and correlation of lfp and respective spiketrain.

    Calculates the spike triggered average of a AnalogSignal or AnalogSignalArray object in a
    time window +-window around the spike times in a SpikeTrain object.

    **Args**:
        lfps: AnalogSignal object or AnalogSignalArray object or list of AnalogSignals
        spikes: SpikeTrain or list of SpikeTrains objects, its time intervall needs to
            be completely covered by the lfp
        window: positive time interval to specify the cutout around spikes given as Quantity or
            number of bins to use
        crosstrail: indicates if STA is averaged over all provided trials or calculated trial-wise

    **Return**:
        Returns a tuple (STA,time,used_spikes), where STA is a list of one-dimensional arrays with
        the spike triggered average, and time is a list of the corresponding time bins.
        The length of the respective array is defined by 2*window + 1, where window is
        the number of bins around the spike times used.
        used_spikes contains the number of spikes used for the STA. If the spiketrain did
        not contain suitable spikes, the returned STA will be filled with zeros.

    **Example**:
        (result,time,used_spikes)=sta_corr(lfp,spiketrain,Quantity(10,"ms"))
        matplotlib.pyplot.plot(time,result)
    """

    # checking compatibility of data, calculating parameters of trials
    (lfps, spiketrains, window_times, wrapped, num_trials, window_bins, st_lfp_offsetbins, spiketrainbins) = data_quality_check(lfps, spiketrains, window, crosstrials, single_data)


    # create binned spiketrains of spikes in suitable time window
    st_binned = []
    for trial in np.arange(num_trials):
        # binning spiketrain with respect to its starting time
        st_binned.append(np.zeros(spiketrainbins[trial], dtype=int))
        for t in spiketrains[trial]:
            # calculating spikebin from spiketime (respective to spiketrainstart)
            spikebin = int(np.round(float(t - spiketrains[trial].t_start) / (spiketrains[trial].t_stop - spiketrains[trial].t_start) * spiketrainbins[trial]))
             # checking if lfp signal around spiketime t is available
            if spikebin + st_lfp_offsetbins[trial] > window_bins[trial] and len(lfps[trial]) - (st_lfp_offsetbins[trial] + spikebin) > window_bins[trial]:
                # adds 1 to the bin corresponding to spiketime t
                st_binned[trial][spikebin] += 1




    # use the correlation function to calculate the STA
    result_sta = []
    result_time = []
    used_spikes = []
    for trial in np.arange(num_trials):

        if all(np.equal(st_binned[trial] , 0)):  # This is slow!
            print "No suitable spikes in trial detected. Reduce window size or supply more LFP data."
            output = np.zeros(2 * window_bins[trial] + 1) * lfps[trial].units
            result_sta.append(output)
            # used_spikes.append(0)
        else:
            # cutting correct segment of lfp with respect to additional information outside of spiketrain intervall
            lfp_start = st_lfp_offsetbins[trial] - window_bins[trial]
            pre = []
            post = []
            if lfp_start < 0:
                pre = np.zeros(-lfp_start)
                lfp_start = 0
            lfp_stop = st_lfp_offsetbins[trial] + spiketrainbins[trial] + window_bins[trial]
            if lfp_stop > len(lfps[trial]):
                post = np.zeros(lfp_stop - len(lfps[trial]))
                lfp_stop = len(lfps[trial])

            # appending pre and post for symetrie reasons of correlation
            lfp = lfps[trial][lfp_start:lfp_stop]
            if pre != []:
                lfp = np.append(pre, lfp)
            if post != []:
                lfp = np.append(lfp, post)

            # actual calculation of correlation and therefore STA of both signals
            output = scipy.signal.correlate(lfp, st_binned[trial], mode='same') / np.sum(st_binned[trial]) * lfps[trial].units

            bin_start = int(len(output) / 2) - window_bins[trial]
            bin_end = int(len(output) / 2) + window_bins[trial]

            # one additional bin to cut STA symmetrically around time = 0
            result_sta.append(output[bin_start: bin_end + 1])

        result_time.append(np.arange(-window_times[trial], (window_times[trial] + 1 / lfps[trial].sampling_rate).rescale(window_times[trial].units), (1 / lfps[trial].sampling_rate).rescale(window_times[trial].units))[0: 2 * window_bins[trial] + 1] * window_times[trial].units)
        used_spikes.append(int(np.sum(st_binned[trial])))

    # Averaging over all trials in case of crosstrialing
    if crosstrials:
        result_sta[0] = averaging_STAs(result_sta, used_spikes)


    # Returns array in case only single LFP and spiketrains was passed
    if wrapped or crosstrials:
        return result_sta[0], result_time[0], used_spikes[0]
    else:
        return result_sta, result_time, used_spikes


#-------------------------------------------------------------------------------

def sta_average(lfps, spiketrains, window, crosstrials=False, single_data=None):
    """
    Calculates the respective spike-triggered average of a analog signals of multiple trials
    by averaging the respective parts of the lfp signal.

    Calculates the spike triggered average of a neo AnalogSignal or AnalogSignal object in a
    time window +-window around the spike times in a SpikeTrain object. Acts the same as
    analysis.sta_corr(lfps, spiketrains, window)

    **Args**:
        lfps: AnalogSignal object or AnalogSignalArray object
        spikes: SpikeTrain or list of SpikeTrains objects
        window: positive time interval to specify the cutout around given as Quantity or
        number of bins to use
        crosstrail: indicates if STA is averaged with all given trial or calculated trial-wise

    **Return**:
        Returns a tuple (STA,time,used_spikes), where STA is a list of one-dimensional arrays with
        the spike triggered average, and time is a list of the corresponding time bins.
        The length of the respective array is defined by 2*window + 1, where window is
        the number of bins around the spike times used.
        used_spikes contains the number of spikes used for the STA. If the spiketrain did
        not contain suitable spikes, the returned STA will be filled with zeros.

    **Example**:
        (result,time,used_spikes)=sta_average([lfp1,lfp2], [spiketrain1,spiketrain2], Quantity(10,"ms"), crosstrials)
        matplotlib.pyplot.plot(time,result)
    """

    # checking compatibility of data, calculating parameters of trials
    (lfps, spiketrains, window_times, wrapped, num_trials, window_bins, st_lfp_offsetbins, spiketrainbins) = data_quality_check(lfps, spiketrains, window, crosstrials, single_data)


    # calculate the spike-triggered-average by averaging the respective intervals of the lfp
    result_sta = []
    result_time = []
    used_spikes = np.zeros(num_trials, dtype=int)
    for trial in range(num_trials):
        # summing over all respective lfp intervals around spiketimes
        lfp_sum = np.zeros(2 * window_bins[trial] + 1) * lfps[trial].units

        for spiketime in spiketrains[trial]:
            # converting spiketime to respective bin in binned spiketrain (which starts at t_start of spiketrain)
            spikebin = int(np.round(float(spiketime - spiketrains[trial].t_start) / (spiketrains[trial].t_stop - spiketrains[trial].t_start) * spiketrainbins[trial]))

           # checks for sufficient lfp data around spikebin
            if spikebin + st_lfp_offsetbins[trial] > window_bins[trial]  and len(lfps[trial]) - (spikebin + st_lfp_offsetbins[trial]) > window_bins[trial]:

                # determines lfp interval to cut with respect to spiketrain timing
                bin_start = spikebin - window_bins[trial]
                bin_end = spikebin + window_bins[trial] + 1
                # actual copying of lfp interval
                lfp_cutout = lfps[trial][st_lfp_offsetbins[trial] + bin_start:st_lfp_offsetbins[trial] + bin_end]

                # conversion of lfp AnalogSignal to quantity numpy array and summing up
                # TODO: This step is slow due to copying the whole array -> Faster version?
                lfp_sum = lfp_sum + np.array(lfp_cutout) * lfp_cutout.units

                used_spikes[trial] += 1

        if used_spikes[trial] == 0:
            print "No suitable spikes in trial detected. Reduce window size or supply more LFP data."
            result_sta.append(lfp_sum)
        else:
            # normalizing STA
            result_sta.append(lfp_sum / used_spikes[trial])
        # generating timesteps for STA
        result_time.append(np.arange(-window_times[trial], (window_times[trial] + 1 / lfps[trial].sampling_rate).rescale(window_times[trial].units), (1 / lfps[trial].sampling_rate).rescale(window_times[trial].units))[0:len(result_sta[trial])] * window_times[trial].units)


    # Averaging over all trials in case of crosstrialing
    if crosstrials:
        result_sta[0] = averaging_STAs(result_sta, used_spikes)


    # Returns array in case only single LFP and spiketrains was passed or averaging over trials was done
    if wrapped or crosstrials:
        return result_sta[0], result_time[0], used_spikes[0]
    else:
        return result_sta, result_time, used_spikes








#===============================================================================
# Supplementary functions
#===============================================================================

def data_quality_check(lfps, spiketrains, window, crosstrials, single_data):
    """
    Supplementary function
    Checks the properties of the given data and transforms them into a defined format for STA analysis.

    **Args**:
        lfps: AnalogSignal object or AnalogSignalArray object or list of AnalogSignal objects
        spikes: SpikeTrain or list of SpikeTrains objects
        window: positive time interval to specify the cutout around given as time Quantity or
        number of bins to use
        crosstrials: indicates if STA will be calculated trial-wise or across all given trials

    **Return**:
        Returns a tuple (lfps, spiketrains, window_times, wrapped, num_trials, window_bins)
        lfps, spiketrains, and window_times are of type list covering single trails
        wrapped indicates whether the data needed to be wrapped or not
        num_trial and window_bins are lists containing the respective values for each trial
        st_lfp_offsetbins: array with number of bins between lfp start and spiketrain start
        spiketrainbins: length of spiketrain in number of bins

    **Example**
        TODO
    """

    if window <= 0:
        raise ValueError("Argument 'window' must be positive.")

    wrapped = False
    # wrapping lfps
    if type(lfps) != list and lfps.ndim == 1:
        box = []
        box.append(lfps)
        lfps = box
        wrapped = True

    # wrapping spiketrains
    if isinstance(spiketrains, np.ndarray):
        box = []
        box.append(spiketrains)
        spiketrains = box

    # multipling spiketrain in case of single_train option
    if single_data == 'train':
        template = spiketrains[0]
        spiketrains = []
        for trial in range(len(lfps)):
            spiketrains.append(template)

    # Checking for matching numbers of LFPs and spiketrains
    # This makes trouble for single_data = 'lfp' option due to variable length of lfp intervals
    if len(lfps) != len(spiketrains):
        raise ValueError("Number of LFPs and spiketrains has to be the same")


    # Checking trial-wise for matching times of lfp and spiketrain
    num_trials = len(lfps)
    st_lfp_offsetbins = np.zeros(num_trials, dtype=int)
    spiketrainbins = np.zeros(num_trials, dtype=int)

    for trial in range(num_trials):
       # bin distance between start of lfp and spiketrain signal
        st_lfp_offsetbins[trial] = int (((spiketrains[trial].t_start - lfps[trial].t_start) * lfps[trial].sampling_rate).rescale('dimensionless'))
        spiketrainbins[trial] = int (((spiketrains[trial].t_stop - spiketrains[trial].t_start) * lfps[trial].sampling_rate).rescale('dimensionless'))

        # checking time length in bins of lfps and spiketrains
        if len(lfps[trial]) < spiketrainbins[trial]:
            raise ValueError("LFP signal covers less bins than spiketrain. (LFP length: %i bins, spiketrain: %i bins)" % (len(lfps[trial]), len(spiketrainbins[trial])))
        if st_lfp_offsetbins[trial] < 0  or len(lfps[trial]) < st_lfp_offsetbins[trial] + spiketrainbins[trial]:
            raise ValueError("LFP does not cover the whole time of the spiketrain")


    # checking if STA across trials is possible to calculate due to sampling rates
    if crosstrials == True and any(lfp.sampling_rate != lfps[0].sampling_rate for lfp in lfps):
        print "Warning: Trials to cross do not have the same sampling rate"
        raise ValueError("For calculating STA of multiple trials all need the same sampling rate")


    # determine correct window size for each trial and calculating the missing variable window_bins or window_times
    window_times = np.zeros(num_trials) * pq.s
    window_bins = []
    if type(window) == pq.quantity.Quantity:
        # for loop is necessary in the following lines, otherwise units will be disregarded
        for trial in range(num_trials):
            window_times[trial] = window
            window_bins.append(int((window_times[trial] * lfps[trial].sampling_rate).rescale("dimensionless")))
    # check if windowsize gives number of bins which has to be converted into time interval
    elif type(window) == int:
        for trial in np.arange(num_trials):
            window_times[trial] = window / lfps[trial].sampling_rate
            window_bins.append(window)
    else:
        raise ValueError("window needs to be either a time quantity or an integer")


    return (lfps, spiketrains, window_times, wrapped, num_trials, window_bins, st_lfp_offsetbins, spiketrainbins)


#-------------------------------

def averaging_STAs(stas, used_spikes):
    """
    Supplementary function
    Calculates the average of multiple sta taking into account that they are based on
    different numbers of spikes

     **Args**:
        stas: list of STAs to average. STAs need to be quantities with np.arrays
        used_spikes: list of number of spikes used for calculating stas

    **Return**:
        Returns an averaged STA
    """


    cross_sta = np.zeros(len(stas[0]))
    for trial in np.arange(len(stas)):
        cross_sta[:] += stas[trial] * used_spikes[trial]
    if np.sum(used_spikes) != 0:
        return cross_sta / np.sum(used_spikes)
    else: return cross_sta
