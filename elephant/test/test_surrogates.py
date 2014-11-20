'''
Created on Feb 22nd, 2013

@author: torre
'''

import unittest
import numpy as np
import quantities as pq
import neo
import elephant.surrogates as surr


# TESTS to implement:
#* spike_dithering:
#  - abs(spike difference) not larger than dither
#  - empty train
#* spike_time_rand:
#  - empty train
#  - very same nr. spikes
#* isi_shuffling:
#  - very same ISIs
#  - very same nr. spikes
#  - empty train
#* train_shifting:
#  - same ISIs
#  - empty train

class SurrogatesTestCase(unittest.TestCase):

    def test_spike_dithering_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        dither = 10 * pq.ms
        surrs = surr.spike_dithering(st, dither=dither, n=nr_surr)

        # Bug encountered when the output has wrong format
        self.assertEqual(type(surrs), list)
        self.assertEqual(len(surrs), nr_surr)

        for surrog in surrs:
            self.assertEqual(type(surrs[0]), neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

        pass

    def test_spike_dithering_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        dither = 10 * pq.ms
        surrog = surr.spike_dithering(st, dither=dither, n=1)[0]
        self.assertEqual(len(surrog), 0)
        pass

    def test_spike_time_rand_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        surrs = surr.spike_time_rand(st, n=nr_surr)

        # Bug encountered when the output has wrong format
        self.assertEqual(type(surrs), list)
        self.assertEqual(len(surrs), nr_surr)

        # Bug encountered when the list's elements are not all SpikeTrains
        # with same time unit and nr. of spikes as the original train.
        for surrog in surrs:
            self.assertEqual(type(surrs[0]), neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

        pass

    def test_spike_time_rand_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        surrog = surr.spike_time_rand(st, n=1)[0]
        self.assertEqual(len(surrog), 0)
        pass

    def test_isi_shuffling_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        surrs = surr.isi_shuffling(st, n=nr_surr)

        # Bug encountered when the output has wrong format
        self.assertEqual(type(surrs), list)
        self.assertEqual(len(surrs), nr_surr)

        # Bug encountered when the list's elements are not all SpikeTrains
        # with same time unit and nr. of spikes as the original train.
        for surrog in surrs:
            self.assertEqual(type(surrs[0]), neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

        pass

    def test_isi_shuffling_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        surrog = surr.isi_shuffling(st, n=1)[0]
        self.assertEqual(len(surrog), 0)
        pass

    def test_isi_shuffling_same_isis(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        surrog = surr.isi_shuffling(st, n=1)[0]

        st_pq = st.view(pq.Quantity)
        surr_pq = surrog.view(pq.Quantity)

        # Bug encountered if the set of ISIs is not the same for the original
        # and surrogate spike train
        isi0_orig = st[0] - st.t_start
        ISIs_orig = np.sort([isi0_orig] + [isi for isi in np.diff(st_pq)])

        isi0_surr = surrog[0] - surrog.t_start
        ISIs_surr = np.sort([isi0_surr] + [isi for isi in np.diff(surr_pq)])

        self.assertTrue(np.all(ISIs_orig == ISIs_surr))

    def test_train_shifting_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        shift = 10 * pq.ms
        surrs = surr.train_shifting(st, shift=shift, n=nr_surr)

        # Bug encountered when the output has wrong format
        self.assertEqual(type(surrs), list)
        self.assertEqual(len(surrs), nr_surr)

        # Bug encountered when the list's elements are not all SpikeTrains
        # with same time unit and nr. of spikes as the original train.
        for surrog in surrs:
            self.assertEqual(type(surrs[0]), neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

        pass

    def test_train_shifting_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        shift = 10 * pq.ms
        surrog = surr.train_shifting(st, shift=shift, n=1)[0]
        self.assertEqual(len(surrog), 0)
        pass

    def test_spike_jittering_output_format(self):

        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        nr_surr = 2
        binsize = 100 * pq.ms
        surrs = surr.spike_jittering(st, binsize=binsize, n=nr_surr)

        # Bug encountered when the output is not a list of nr_surr elements
        self.assertEqual(type(surrs), list)
        self.assertEqual(len(surrs), nr_surr)

        # Bug encountered when the list's elements are not all SpikeTrains
        # with same time unit and nr. of spikes as the original train.
        for surrog in surrs:
            self.assertEqual(type(surrs[0]), neo.SpikeTrain)
            self.assertEqual(surrog.units, st.units)
            self.assertEqual(surrog.t_start, st.t_start)
            self.assertEqual(surrog.t_stop, st.t_stop)
            self.assertEqual(len(surrog), len(st))

        pass

    def test_spike_jittering_empty_train(self):

        st = neo.SpikeTrain([] * pq.ms, t_stop=500 * pq.ms)

        binsize = 75 * pq.ms
        surrog = surr.spike_jittering(st, binsize=binsize, n=1)[0]
        self.assertEqual(len(surrog), 0)
        pass

    def test_spike_jittering_same_bins(self):

        # Construct a spike train
        st = neo.SpikeTrain([90, 150, 180, 350] * pq.ms, t_stop=500 * pq.ms)

        # Create one surrogate by jittering spikes within 100ms time bins
        binsize = 100 * pq.ms
        surrog = surr.spike_jittering(st, binsize=binsize, n=1)[0]

        # Bug encountered when corresponding spikes from original and
        # surrogate trains fall in different time bins
        bin_ids_orig = np.array((st.view(pq.Quantity) / binsize).rescale(
            pq.dimensionless).magnitude, dtype=int)
        bin_ids_surr = np.array((surrog.view(pq.Quantity) / binsize).rescale(
            pq.dimensionless).magnitude, dtype=int)
        self.assertTrue(np.all(bin_ids_orig == bin_ids_surr))

        # Bug encountered when the original and surrogate trains have
        # different number of spikes
        self.assertEqual(len(st), len(surrog))

        pass

    def test_spike_jittering_unequal_binsize(self):

        st = neo.SpikeTrain([90, 150, 180, 480] * pq.ms, t_stop=500 * pq.ms)

        # Create one surrogate by jittering spikes within 100ms time bins
        binsize = 75 * pq.ms
        surrog = surr.spike_jittering(st, binsize=binsize, n=1)[0]

        # Check that spikes from orig and surr trains fall in the same bins
        bin_ids_orig = np.array((st.view(pq.Quantity) / binsize).rescale(
            pq.dimensionless).magnitude, dtype=int)
        bin_ids_surr = np.array((surrog.view(pq.Quantity) / binsize).rescale(
            pq.dimensionless).magnitude, dtype=int)

        self.assertTrue(np.all(bin_ids_orig == bin_ids_surr))

        pass


def suite():
    suite = unittest.makeSuite(SurrogatesTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


# if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Tesm.testName']
#    unittesm.main()
