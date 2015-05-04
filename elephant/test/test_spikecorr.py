# -*- coding: utf-8 -*-
"""
Unit tests for the spikecorr module.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import numpy as np
from numpy.testing.utils import assert_array_almost_equal
from numpy.testing.utils import assert_array_equal
import scipy
import quantities as pq
import neo
import elephant.conversion as conversion
import elephant.spikecorr as spikecorr


class cch_TestCase(unittest.TestCase):

    def setUp(self):
        # These two arrays must be such that they do not have coincidences
        # spanning across two neighbor bins assuming ms bins [0,1),[1,2),...
        self.test_array_1d_0 = [
            1.3, 7.56, 15.87, 28.23, 30.9, 34.2, 38.2, 43.2]
        self.test_array_1d_1 = [1.02, 2.71, 18.82, 28.46, 28.79, 43.6]

        # Build spike trains
        self.st_0 = neo.SpikeTrain(
            self.test_array_1d_0, units='ms', t_stop=50.)
        self.st_1 = neo.SpikeTrain(
            self.test_array_1d_1, units='ms', t_stop=50.)

        # And binned counterparts
        self.binned_st1 = conversion.BinnedSpikeTrain(
            [self.st_0], t_start=0 * pq.ms, t_stop=50. * pq.ms,
            binsize=1 * pq.ms)
        self.binned_st2 = conversion.BinnedSpikeTrain(
            [self.st_1], t_start=0 * pq.ms, t_stop=50. * pq.ms,
            binsize=1 * pq.ms)

    def test_cch(self):
        '''
        Test result of a correlation coefficient between two binned spike
        trains.
        '''
        # Calculate clipped and unclipped
        res_clipped = spikecorr.cch(
            self.binned_st1, self.binned_st2, window=None, binary=True)
        res_unclipped = spikecorr.cch(
            self.binned_st1, self.binned_st2, window=None, binary=False)

        # Check unclipped correlation
        # Use numpy correlate to verify result. Note: numpy conventions for
        # input array 1 and input array 2 are swapped compared to Elephant!
        mat1 = self.binned_st1.to_array()[0]
        mat2 = self.binned_st2.to_array()[0]
        target_numpy = np.correlate(mat2, mat1, mode='full')
        assert_array_equal(target_numpy, np.hstack(
            res_unclipped[0].magnitude))

        # Check clipped correlation
        # Use numpy correlate to verify result. Note: numpy conventions for
        # input array 1 and input array 2 are swapped compared to Elephant!
        mat1 = np.array(self.binned_st1.to_bool_array()[0], dtype=int)
        mat2 = np.array(self.binned_st2.to_bool_array()[0], dtype=int)
        target_numpy = np.correlate(mat2, mat1, mode='full')
        assert_array_equal(target_numpy, np.hstack(
            res_clipped[0].magnitude))

        # Check the time axis of the AnalogSignalArray
        assert_array_almost_equal(
            res_clipped[1]*self.binned_st1.binsize + self.binned_st1.binsize /
            float(2), res_clipped[0].times)
        assert_array_almost_equal(
            res_clipped[1]*self.binned_st1.binsize + self.binned_st1.binsize /
            float(2), res_clipped[0].times)


if __name__ == '__main__':
    unittest.main()
