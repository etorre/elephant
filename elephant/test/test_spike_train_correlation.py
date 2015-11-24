# -*- coding: utf-8 -*-
"""
Unit tests for the spike_train_correlation module.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import numpy as np
from numpy.testing.utils import assert_array_equal, assert_array_almost_equal
import quantities as pq
import neo
import elephant.conversion as conv
import elephant.spike_train_correlation as sc


class corrcoeff_TestCase(unittest.TestCase):

    def setUp(self):
        # These two arrays must be such that they do not have coincidences
        # spanning across two neighbor bins assuming ms bins [0,1),[1,2),...
        self.test_array_1d_0 = [
            1.3, 7.56, 15.87, 28.23, 30.9, 34.2, 38.2, 43.2]
        self.test_array_1d_1 = [
            1.02, 2.71, 18.82, 28.46, 28.79, 43.6]

        # Build spike trains
        self.st_0 = neo.SpikeTrain(
            self.test_array_1d_0, units='ms', t_stop=50.)
        self.st_1 = neo.SpikeTrain(
            self.test_array_1d_1, units='ms', t_stop=50.)

        # And binned counterparts
        self.binned_st = conv.BinnedSpikeTrain(
            [self.st_0, self.st_1], t_start=0 * pq.ms, t_stop=50. * pq.ms,
            binsize=1 * pq.ms)

    def test_corrcoef_binned(self):
        '''
        Test result of a correlation coefficient between two binned spike
        trains.
        '''

        # Calculate clipped and unclipped
        res_clipped = sc.corrcoef(
            self.binned_st, binary=True)
        res_unclipped = sc.corrcoef(
            self.binned_st, binary=False)

        # Check dimensions
        self.assertEqual(len(res_clipped), 2)
        self.assertEqual(len(res_unclipped), 2)

        # Check result unclipped against result calculated from scratch for
        # the off-diagonal element
        mat = self.binned_st.to_array()
        mean_0 = np.mean(mat[0])
        mean_1 = np.mean(mat[1])
        target_from_scratch = \
            np.dot(mat[0] - mean_0, mat[1] - mean_1) / \
            np.sqrt(
                np.dot(mat[0] - mean_0, mat[0] - mean_0) *
                np.dot(mat[1] - mean_1, mat[1] - mean_1))

        # Check result unclipped against result calculated by numpy.corrcoef
        target_numpy = np.corrcoef(mat)

        self.assertAlmostEqual(target_from_scratch, target_numpy[0][1])
        self.assertAlmostEqual(res_unclipped[0][1], target_from_scratch)
        self.assertAlmostEqual(res_unclipped[1][0], target_from_scratch)

        # Check result clipped against result calculated from scratch for
        # the off-diagonal elemant
        mat = self.binned_st.to_bool_array()
        mean_0 = np.mean(mat[0])
        mean_1 = np.mean(mat[1])
        target_from_scratch = \
            np.dot(mat[0] - mean_0, mat[1] - mean_1) / \
            np.sqrt(
                np.dot(mat[0] - mean_0, mat[0] - mean_0) *
                np.dot(mat[1] - mean_1, mat[1] - mean_1))

        # Check result unclipped against result calculated by numpy.corrcoef
        target_numpy = np.corrcoef(mat)

        self.assertAlmostEqual(target_from_scratch, target_numpy[0][1])
        self.assertAlmostEqual(res_clipped[0][1], target_from_scratch)
        self.assertAlmostEqual(res_clipped[1][0], target_from_scratch)

    def test_corrcoef_binned_same_spiketrains(self):
        '''
        Test if the correlation coefficient between two identical binned spike
        trains evaluates to a 2x2 matrix of ones.
        '''
        # Calculate correlation
        binned_st = conv.BinnedSpikeTrain(
            [self.st_0, self.st_0], t_start=0 * pq.ms, t_stop=50. * pq.ms,
            binsize=1 * pq.ms)
        target = sc.corrcoef(binned_st)

        # Check dimensions
        self.assertEqual(len(target), 2)
        # Check result
        assert_array_equal(target, 1.)

    def test_corrcoef_binned_short_input(self):
        '''
        Test if input list of one binned spike train yields 1.0.
        '''
        # Calculate correlation
        binned_st = conv.BinnedSpikeTrain(
            self.st_0, t_start=0 * pq.ms, t_stop=50. * pq.ms,
            binsize=1 * pq.ms)
        target = sc.corrcoef(binned_st)

        # Check result
        self.assertEqual(target, 1.)


class cross_correlation_histogram_TestCase(unittest.TestCase):

    def setUp(self):
        # These two arrays must be such that they do not have coincidences
        # spanning across two neighbor bins assuming ms bins [0,1),[1,2),...
        self.test_array_1d_0 = [
            1.3, 7.56, 15.87, 28.23, 30.9, 34.2, 38.2, 43.2]
        self.test_array_1d_1 = [
            1.02, 2.71, 18.82, 28.46, 28.79, 43.6]

        # Build spike trains
        self.st_0 = neo.SpikeTrain(
            self.test_array_1d_0, units='ms', t_stop=45.)
        self.st_1 = neo.SpikeTrain(
            self.test_array_1d_1, units='ms', t_stop=50.)

        # And binned counterparts
        self.binned_st1 = conv.BinnedSpikeTrain(
            [self.st_0], t_start=0 * pq.ms, t_stop=45. * pq.ms,
            binsize=1 * pq.ms)
        self.binned_st2 = conv.BinnedSpikeTrain(
            [self.st_1], t_start=0 * pq.ms, t_stop=50. * pq.ms,
            binsize=1 * pq.ms)
        self.binned_st3 = conv.BinnedSpikeTrain(
            [self.st_1], t_start=0 * pq.ms, t_stop=50. * pq.ms,
            binsize=5 * pq.ms)

    def test_cross_correlation_histogram(self):
        '''
        Test generic result of a cross-correlation histogram between two binned
        spike trains.
        '''
        # Calculate CCH using Elephant (normal and binary version) with
        # mode equal to 'full' (whole spike trains are correlated)
        cch_clipped, bin_ids_clipped = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, mode='full', window=None,
            binary=True)
        cch_unclipped, bin_ids_unclipped = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, mode='full', window=None,
            binary=False)

        cch_clipped_mem, bin_ids_clipped_mem = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, mode='full', window=None,
            binary=True, method='memory')
        cch_unclipped_mem, bin_ids_unclipped_mem = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, mode='full', window=None,
            binary=False, method='memory')
        # Check consistency two methods
        assert_array_equal(
            np.squeeze(cch_clipped.magnitude), np.squeeze(
                cch_clipped_mem.magnitude))
        assert_array_equal(
            np.squeeze(cch_clipped.times), np.squeeze(
                cch_clipped_mem.times))
        assert_array_equal(
            np.squeeze(cch_unclipped.magnitude), np.squeeze(
                cch_unclipped_mem.magnitude))
        assert_array_equal(
            np.squeeze(cch_unclipped.times), np.squeeze(
                cch_unclipped_mem.times))
        assert_array_almost_equal(bin_ids_clipped, bin_ids_clipped_mem)
        assert_array_almost_equal(bin_ids_unclipped, bin_ids_unclipped_mem)

        # Check normal correlation Note: Use numpy correlate to verify result.
        # Note: numpy conventions for input array 1 and input array 2 are
        # swapped compared to Elephant!
        mat1 = self.binned_st1.to_array()[0]
        mat2 = self.binned_st2.to_array()[0]
        target_numpy = np.correlate(mat2, mat1, mode='full')
        assert_array_equal(
            target_numpy, np.squeeze(cch_unclipped.magnitude))

        # Check correlation using binary spike trains
        mat1 = np.array(self.binned_st1.to_bool_array()[0], dtype=int)
        mat2 = np.array(self.binned_st2.to_bool_array()[0], dtype=int)
        target_numpy = np.correlate(mat2, mat1, mode='full')
        assert_array_equal(
            target_numpy, np.squeeze(cch_clipped.magnitude))

        # Check the time axis and bin IDs of the resulting AnalogSignalArray
        assert_array_almost_equal(
            (bin_ids_clipped - 0.5) * self.binned_st1.binsize,
            cch_unclipped.times)
        assert_array_almost_equal(
            (bin_ids_clipped - 0.5) * self.binned_st1.binsize,
            cch_clipped.times)

        # Calculate CCH using Elephant (normal and binary version) with
        # mode equal to 'valid' (only completely overlapping intervals of the
        # spike trains are correlated)
        cch_clipped, bin_ids_clipped = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, mode='valid', window=None,
            binary=True)
        cch_unclipped, bin_ids_unclipped = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, mode='valid', window=None,
            binary=False)
        cch_clipped_mem, bin_ids_clipped_mem = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, mode='valid', window=None,
            binary=True, method='memory')
        cch_unclipped_mem, bin_ids_unclipped_mem = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, mode='valid', window=None,
            binary=False, method='memory')

        # Check consistency two methods
        assert_array_equal(
            np.squeeze(cch_clipped.magnitude), np.squeeze(
                cch_clipped_mem.magnitude))
        assert_array_equal(
            np.squeeze(cch_clipped.times), np.squeeze(
                cch_clipped_mem.times))
        assert_array_equal(
            np.squeeze(cch_unclipped.magnitude), np.squeeze(
                cch_unclipped_mem.magnitude))
        assert_array_equal(
            np.squeeze(cch_unclipped.times), np.squeeze(
                cch_unclipped_mem.times))
        assert_array_almost_equal(bin_ids_clipped, bin_ids_clipped_mem)
        assert_array_almost_equal(bin_ids_unclipped, bin_ids_unclipped_mem)

        # Check normal correlation Note: Use numpy correlate to verify result.
        # Note: numpy conventions for input array 1 and input array 2 are
        # swapped compared to Elephant!
        mat1 = self.binned_st1.to_array()[0]
        mat2 = self.binned_st2.to_array()[0]
        target_numpy = np.correlate(mat2, mat1, mode='valid')
        assert_array_equal(
            target_numpy, np.squeeze(cch_unclipped.magnitude))

        # Check correlation using binary spike trains
        mat1 = np.array(self.binned_st1.to_bool_array()[0], dtype=int)
        mat2 = np.array(self.binned_st2.to_bool_array()[0], dtype=int)
        target_numpy = np.correlate(mat2, mat1, mode='valid')
        assert_array_equal(
            target_numpy, np.squeeze(cch_clipped.magnitude))

        # Check the time axis and bin IDs of the resulting AnalogSignalArray
        assert_array_almost_equal(
            (bin_ids_clipped - 0.5) * self.binned_st1.binsize,
            cch_unclipped.times)
        assert_array_almost_equal(
            (bin_ids_clipped - 0.5) * self.binned_st1.binsize,
            cch_clipped.times)

        # Check for wrong mode parameter setting
        self.assertRaises(
            KeyError, sc.cross_correlation_histogram, self.binned_st1,
            self.binned_st2, mode='dsaij')

    def test_normalize_option(self):
        '''
        Test result of a CCH between two binned spike trains with the
        normalization turned on.
        '''
        # Calculate normalized and raw cch
        cch_norm, bins = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, window=None, binary=False,
            normalize=True)
        cch_norm_mem, bins_mem = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, window=None, binary=False,
            normalize=True, method='memory')

        # Check that central bin is 1
        center_bin = np.where(bins == 0)
        target_time = cch_norm.times.magnitude[center_bin]
        target_value = cch_norm.magnitude[center_bin]

        center_bin_mem = np.where(bins_mem == 0)
        target_time_mem = cch_norm_mem.times.magnitude[center_bin_mem]
        target_value_mem = cch_norm.magnitude[center_bin_mem]

        self.assertEqual(target_time, -0.5*self.binned_st1.binsize)
        self.assertEqual(target_value, 1)

        self.assertEqual(target_time_mem, -0.5*self.binned_st1.binsize)
        self.assertEqual(target_value_mem, 1)

    def test_binsize(self):
        '''Check that an exception is thrown if the two spike trains are not
        binned with the same bin size.'''
        self.assertRaises(
            AssertionError,
            sc.cross_correlation_histogram, self.binned_st1, self.binned_st3)

        self.assertRaises(
            AssertionError,
            sc.cross_correlation_histogram, self.binned_st1, self.binned_st3,
            method='memory')

    def test_window(self):
        '''Test if the window parameter is correctly interpreted.'''
        cch_win, bin_ids = sc.cch(
            self.binned_st1, self.binned_st2, window=[-30, 30])
        cch_win_mem, bin_ids_mem = sc.cch(
            self.binned_st1, self.binned_st2, window=[-30, 30])

        assert_array_equal(bin_ids, np.arange(-30, 31, 1))
        assert_array_almost_equal(
            (bin_ids - 0.5) * self.binned_st1.binsize, cch_win.times)

        assert_array_equal(bin_ids_mem, np.arange(-30, 31, 1))
        assert_array_almost_equal(
            (bin_ids_mem - 0.5) * self.binned_st1.binsize, cch_win.times)

        cch_win, bin_ids = sc.cch(
            self.binned_st1, self.binned_st2, window=[-25*pq.ms, 25*pq.ms])
        cch_win_mem, bin_ids_mem = sc.cch(
            self.binned_st1, self.binned_st2, window=[-25*pq.ms, 25*pq.ms],
            method='memory')

        assert_array_equal(bin_ids, np.arange(-25, 26, 1))
        assert_array_almost_equal(
            (bin_ids - 0.5) * self.binned_st1.binsize, cch_win.times)

        assert_array_equal(bin_ids_mem, np.arange(-25, 26, 1))
        assert_array_almost_equal(
            (bin_ids_mem - 0.5) * self.binned_st1.binsize, cch_win.times)

        _, bin_ids = sc.cch(
            self.binned_st1, self.binned_st2, normalize=True, window=[20, 30])
        _, bin_ids_mem = sc.cch(
            self.binned_st1, self.binned_st2, normalize=True, window=[20, 30],
            method='memory')

        assert_array_equal(bin_ids, np.arange(20, 31, 1))
        assert_array_equal(bin_ids_mem, np.arange(20, 31, 1))

        _, bin_ids = sc.cch(
            self.binned_st1, self.binned_st2, normalize=True, window=[-20, 30])

        _, bin_ids_mem = sc.cch(
            self.binned_st1, self.binned_st2, normalize=True, window=[-20, 30],
            method='memory')

        assert_array_equal(bin_ids, np.arange(-20, 31, 1))
        assert_array_equal(bin_ids_mem, np.arange(-20, 31, 1))

        # Cehck for wrong assignments to the window parameter
        self.assertRaises(
            ValueError, sc.cross_correlation_histogram, self.binned_st1,
            self.binned_st2, window=[-60, 50])
        self.assertRaises(
            ValueError, sc.cross_correlation_histogram, self.binned_st1,
            self.binned_st2, window=[-60, 50], method='memory')

        self.assertRaises(
            ValueError, sc.cross_correlation_histogram, self.binned_st1,
            self.binned_st2, window=[-50, 60])
        self.assertRaises(
            ValueError, sc.cross_correlation_histogram, self.binned_st1,
            self.binned_st2, window=[-50, 60], method='memory')

        self.assertRaises(
            ValueError, sc.cross_correlation_histogram, self.binned_st1,
            self.binned_st2, window=[-25.5*pq.ms, 25*pq.ms])
        self.assertRaises(
            ValueError, sc.cross_correlation_histogram, self.binned_st1,
            self.binned_st2, window=[-25.5*pq.ms, 25*pq.ms], method='memory')

        self.assertRaises(
            ValueError, sc.cross_correlation_histogram, self.binned_st1,
            self.binned_st2, window=[-25*pq.ms, 25.5*pq.ms])
        self.assertRaises(
            ValueError, sc.cross_correlation_histogram, self.binned_st1,
            self.binned_st2, window=[-25*pq.ms, 25.5*pq.ms], method='memory')

        self.assertRaises(
            ValueError, sc.cross_correlation_histogram, self.binned_st1,
            self.binned_st2, window=[-60*pq.ms, 50*pq.ms])
        self.assertRaises(
            ValueError, sc.cross_correlation_histogram, self.binned_st1,
            self.binned_st2, window=[-60*pq.ms, 50*pq.ms], method='memory')

        self.assertRaises(
            ValueError, sc.cross_correlation_histogram, self.binned_st1,
            self.binned_st2, window=[-50*pq.ms, 60*pq.ms])
        self.assertRaises(
            ValueError, sc.cross_correlation_histogram, self.binned_st1,
            self.binned_st2, window=[-50*pq.ms, 60*pq.ms], method='memory')

    def test_border_correction(self):
        '''Test if the border correction for bins at the edges is correctly
        performed'''
        cch_corrected, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, window=None, normalize=False,
            border_correction=True, binary=False, kernel=None)
        cch_corrected_mem, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, window=None, normalize=False,
            border_correction=True, binary=False, kernel=None, method='memory')
        cch, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, window=None, normalize=False,
            border_correction=False, binary=False, kernel=None)
        cch_mem, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, window=None, normalize=False,
            border_correction=False, binary=False, kernel=None,
            method='memory')

        self.assertNotEqual(cch.all(), cch_corrected.all())
        self.assertNotEqual(cch_mem.all(), cch_corrected_mem.all())

    def test_kernel(self):
        '''Test if the smoothing kernel is correctly defined, and wheter it is
        applied properly.'''
        smoothed_cch, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, kernel=np.ones(3))
        smoothed_cch_mem, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, kernel=np.ones(3),
            method='memory')

        cch, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, kernel=None)
        cch_mem, _ = sc.cross_correlation_histogram(
            self.binned_st1, self.binned_st2, kernel=None, method='memory')

        self.assertNotEqual(smoothed_cch.all, cch.all)
        self.assertNotEqual(smoothed_cch_mem.all, cch_mem.all)

        self.assertRaises(
            ValueError, sc.cch, self.binned_st1, self.binned_st2,
            kernel=np.ones(100))
        self.assertRaises(
            ValueError, sc.cch, self.binned_st1, self.binned_st2,
            kernel=np.ones(100), method='memory')

        self.assertRaises(
            ValueError, sc.cch, self.binned_st1, self.binned_st2, kernel='BOX')
        self.assertRaises(
            ValueError, sc.cch, self.binned_st1, self.binned_st2, kernel='BOX',
            method='memory')

    def test_exist_alias(self):
        '''
        Test if alias cch still exists.
        '''
        self.assertEqual(sc.cross_correlation_histogram, sc.cch)

if __name__ == '__main__':
    unittest.main()
