import unittest
import numpy as np
import quantities as pq
import neo.core as n
import jelephant.core.rep as rep


class RepTestCase(unittest.TestCase):

    def setUp(self):
        self.spiketrain_a = n.SpikeTrain(
            [0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        self.spiketrain_b = n.SpikeTrain(
            [0.1, 0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s, t_stop=10.0 * pq.s)
        self.binsize = 1 * pq.s

    def tearDown(self):
        self.spiketrain_a = None
        del self.spiketrain_a
        self.spiketrain_b = None
        del self.spiketrain_b

    def test_binned_st_filled(self):
        a = n.SpikeTrain([1.7, 1.8, 4.3] * pq.s, t_stop=10.0 * pq.s)
        b = n.SpikeTrain([1.7, 1.8, 4.3] * pq.s, t_stop=10.0 * pq.s)
        binsize = 1 * pq.s
        nbins = 10
        x = rep.binned_st([a, b], num_bins=nbins, binsize=binsize,
                          t_start=0 * pq.s)
        x_filled = [[1, 1, 4], [1, 1, 4]]
        self.assertTrue(np.array_equal(x.filled, x_filled))

    def test_binned_st_shape(self):
        a = self.spiketrain_a
        x_unclipped = rep.binned_st(a, num_bins=10,
                                    binsize=self.binsize,
                                    t_start=0 * pq.s)
        x_clipped = rep.binned_st(a, num_bins=10, binsize=self.binsize,
                                  t_start=0 * pq.s)
        self.assertTrue(x_unclipped.matrix_unclipped().shape == (1, 10))
        self.assertTrue(x_clipped.matrix_clipped().shape == (1, 10))

    # shape of the matrix for a list of spike trains
    def test_binned_st_shape_list(self):
        a = self.spiketrain_a
        b = self.spiketrain_b
        c = [a, b]
        nbins = 5
        x_unclipped = rep.binned_st(c, num_bins=nbins, t_start=0 * pq.s,
                                    t_stop=10.0 * pq.s)
        x_clipped = rep.binned_st(c, num_bins=nbins, t_start=0 * pq.s,
                                  t_stop=10.0 * pq.s)
        self.assertTrue(x_unclipped.matrix_unclipped().shape == (2, 5))
        self.assertTrue(x_clipped.matrix_clipped().shape == (2, 5))

    # Various tests on constructing binned_st representations
    # and overloaded operators
    def test_binned_st_eq(self):
        a = n.SpikeTrain([1, 2, 3, 4, 5, 6, 6.5] * pq.s, t_stop=10.0 * pq.s)
        b = n.SpikeTrain([1000, 2000, 3000, 4000, 5000, 6000, 6500] * pq.ms,
                         t_stop=10.0 * pq.s)
        binsize = 10 * pq.ms
        x = rep.binned_st(a, num_bins=int(
            (a.t_stop / binsize).rescale(pq.dimensionless)), t_start=0 * pq.s,
                          binsize=binsize)
        y = rep.binned_st(b, num_bins=int(
            (b.t_stop / binsize).rescale(pq.dimensionless)), t_start=0 * pq.s,
                          binsize=binsize)
        self.assertTrue(x == y)

    def test_binned_st_neg_times(self):
        a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7, -6.5] * pq.s,
                         t_start=-6.5 * pq.s, t_stop=10.0 * pq.s)
        binsize = self.binsize
        nbins = 16
        x = rep.binned_st(a, num_bins=nbins, binsize=binsize,
                          t_start=-6.5 * pq.s)
        y = [np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0])]
        self.assertTrue(np.array_equal(x.matrix_clipped(), y))

    def test_binned_st_neg_times_list(self):
        a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7, -6.5] * pq.s,
                         t_start=-6.5 * pq.s, t_stop=10.5 * pq.s)
        b = n.SpikeTrain([-0.1, 0.7, 1.2, 2.2, 4.3, 5.5, 8.0] * pq.s,
                         t_start=-5.5 * pq.s, t_stop=11.5 * pq.s)
        c = [a, b]
        binsize = self.binsize
        nbins = 16
        x_clipped = rep.binned_st(c, num_bins=nbins, binsize=binsize,
                                  t_start=-5.5 * pq.s)
        y_clipped = [
            [0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1.],
            [0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 0.]]
        self.assertTrue(np.array_equal(x_clipped.matrix_clipped(), y_clipped))

    # checking filled(f) and matrix(m) for 1 spiketrain with clip(c) and
    # without clip(u)
    def test_binned_st_fmcu(self):
        a = self.spiketrain_a
        binsize = self.binsize
        nbins = 10
        x_unclipped = rep.binned_st(a, num_bins=nbins, binsize=binsize,
                                    t_start=0 * pq.s)
        x_clipped = rep.binned_st(a, num_bins=nbins, binsize=binsize,
                                  t_start=0 * pq.s)
        y_filled = [np.array([0., 0., 1., 3., 4., 5., 6.])]
        y_matrix_unclipped = [
            np.array([2., 1., 0., 1., 1., 1., 1., 0., 0., 0.])]
        y_matrix_clipped = [np.array([1., 1., 0., 1., 1., 1., 1., 0., 0., 0.])]
        self.assertTrue(
            np.array_equal(x_unclipped.matrix_unclipped(), y_matrix_unclipped))
        self.assertTrue(
            np.array_equal(x_clipped.matrix_clipped(), y_matrix_clipped))
        self.assertTrue(
            np.array_equal(x_clipped.matrix_clipped(), y_matrix_clipped))
        self.assertTrue(np.array_equal(x_clipped.filled, y_filled))

    def test_binned_st_fmcu_list(self):
        a = self.spiketrain_a
        b = self.spiketrain_b

        binsize = self.binsize
        nbins = 10
        c = [a, b]
        x_unclipped = rep.binned_st(c, num_bins=nbins, binsize=binsize,
                                    t_start=0 * pq.s)
        x_clipped = rep.binned_st(c, num_bins=nbins, binsize=binsize,
                                  t_start=0 * pq.s)
        y_filled = [np.array([0, 0, 1, 3, 4, 5, 6]),
                    np.array([0, 0, 1, 2, 4, 5, 8])]
        y_matrix_unclipped = np.array(
            [[2, 1, 0, 1, 1, 1, 1, 0, 0, 0], [2, 1, 1, 0, 1, 1, 0, 0, 1, 0]])
        y_matrix_clipped = np.array(
            [[1, 1, 0, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 1, 1, 0, 0, 1, 0]])
        self.assertTrue(
            np.array_equal(x_unclipped.matrix_unclipped(), y_matrix_unclipped))
        self.assertTrue(
            np.array_equal(x_clipped.matrix_clipped(), y_matrix_clipped))
        self.assertTrue(all(
            [np.array_equal(x_clipped.filled[i], y_filled[i]) for i in
             xrange(2)]))

    # t_stop is None
    def test_binned_st_list_t_stop(self):
        a = self.spiketrain_a
        b = self.spiketrain_b
        c = [a, b]
        binsize = self.binsize
        nbins = 10
        x = rep.binned_st(c, num_bins=nbins, binsize=binsize, t_start=0 * pq.s,
                          t_stop=None)
        x_clipped = rep.binned_st(c, num_bins=nbins, binsize=binsize,
                                  t_start=0 * pq.s)
        self.assertTrue(x.t_stop == 10 * pq.s)
        self.assertTrue(x_clipped.t_stop == 10 * pq.s)

    # Test number of bins
    def test_binned_st_list_numbins(self):
        a = self.spiketrain_a
        b = self.spiketrain_b
        c = [a, b]
        binsize = 1 * pq.s
        x_unclipped = rep.binned_st(c, binsize=binsize, t_start=0 * pq.s,
                                    t_stop=10. * pq.s)
        x_clipped = rep.binned_st(c, binsize=binsize, t_start=0 * pq.s,
                                  t_stop=10. * pq.s)
        self.assertTrue(x_unclipped.num_bins == 10)
        self.assertTrue(x_clipped.num_bins == 10)

    def test_matrix(self):
        # Init
        a = self.spiketrain_a
        b = self.spiketrain_b
        x_clipped_a = rep.binned_st(a, binsize=pq.s, t_start=0 * pq.s,
                                    t_stop=10. * pq.s)
        x_clipped_b = rep.binned_st(b, binsize=pq.s, t_start=0 * pq.s,
                                    t_stop=10. * pq.s, store_mat=True)

        # Operations
        clip_add = x_clipped_a + x_clipped_b
        x_clipped_b.matrix_clipped()  # store matrix

        # Assumed results
        y_matrix_unclipped_a = [np.array([2, 1, 0, 1, 1, 1, 1, 0, 0, 0])]
        y_matrix_clipped_a = [np.array([1, 1, 0, 1, 1, 1, 1, 0, 0, 0])]
        y_matrix_clipped_b = [np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0])]
        y_clip_add = [np.array(
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 0])]  # matrix of the clipped addition
        y_uclip_add = [np.array([4, 2, 1, 1, 2, 2, 1, 0, 1, 0])]
        # Asserts
        self.assertTrue(
            np.array_equal(x_clipped_a.matrix_clipped(store_mat=True),
                           y_matrix_clipped_a))
        self.assertTrue(np.array_equal(x_clipped_b.matrix_clipped(),
                                       y_matrix_clipped_b))
        self.assertTrue(
            np.array_equal(x_clipped_a.matrix_unclipped(),
                           y_matrix_unclipped_a))
        self.assertTrue(np.array_equal(clip_add.matrix_clipped(), y_clip_add))
        self.assertTrue(
            np.array_equal(clip_add.matrix_unclipped(), y_uclip_add))

        # Test prune add
        prune_add = x_clipped_a.prune() + x_clipped_b.prune()
        y_prune_filled = [np.array([0, 1, 3, 4, 5, 6, 0, 1, 2, 4, 5, 8])]
        self.assertTrue(np.array_equal(prune_add.filled, y_prune_filled))

    def test_matrix_storing(self):
        a = self.spiketrain_a
        b = self.spiketrain_b

        x_clipped = rep.binned_st(a, binsize=pq.s, t_start=0 * pq.s,
                                  t_stop=10. * pq.s, store_mat=True)
        x_unclipped = rep.binned_st(b, binsize=pq.s, t_start=0 * pq.s,
                                    t_stop=10. * pq.s, store_mat=True)
        # Store Matrix in variable
        matrix_clipped = x_clipped.matrix_clipped()
        matrix_unclipped = x_unclipped.matrix_unclipped()

        # Check for boolean
        self.assertEqual(x_clipped.store_mat_c, True)
        self.assertEqual(x_unclipped.store_mat_u, True)
        # Check if same matrix
        self.assertTrue(np.array_equal(x_clipped.mat_c, matrix_clipped))
        self.assertTrue(np.array_equal(x_unclipped.mat_u, matrix_unclipped))
        # New class without calculating the matrix
        x_clipped = rep.binned_st(a, binsize=pq.s, t_start=0 * pq.s,
                                  t_stop=10. * pq.s, store_mat=True)
        x_unclipped = rep.binned_st(b, binsize=pq.s, t_start=0 * pq.s,
                                    t_stop=10. * pq.s, store_mat=True)
        # No matrix calculated, should be None
        self.assertEqual(x_clipped.mat_c, None)
        self.assertEqual(x_unclipped.mat_u, None)
        # Test with stored matrix
        self.assertFalse(np.array_equal(x_clipped.mat_c, matrix_clipped))
        self.assertFalse(np.array_equal(x_unclipped, matrix_unclipped))

    # Test if matrix raises error when giving a non boolean instead an
    # expected boolean for store_mat
    def test_matrix_assertion_error(self):
        a = self.spiketrain_a
        x = rep.binned_st(a, binsize=pq.s, t_start=0 * pq.s,
                          t_stop=10. * pq.s)
        self.assertRaises(AssertionError, x.matrix_clipped, store_mat=4.)
        self.assertRaises(AssertionError, x.matrix_unclipped, store_mat=4)

    # Test if t_start is calculated correctly
    def test_parameter_calc_tstart(self):
        a = self.spiketrain_a
        x = rep.binned_st(a, binsize=1 * pq.s, num_bins=10,
                          t_stop=10. * pq.s)
        self.assertEqual(x.t_start, 0. * pq.s)
        self.assertEqual(x.t_stop, 10. * pq.s)
        self.assertEqual(x.binsize, 1 * pq.s)
        self.assertEqual(x.num_bins, 10)

    # Test if error raises when type of num_bins is not an integer
    def test_numbins_type_error(self):
        a = self.spiketrain_a
        self.assertRaises(TypeError, rep.binned_st, a, binsize=pq.s,
                          num_bins=1.4, t_start=0 * pq.s, t_stop=10. * pq.s)

    # Test if error is raised when providing insufficient number of parameter
    def test_insufficient_arguments(self):
        a = self.spiketrain_a
        self.assertRaises(AttributeError, rep.binned_st, a)

    # Test edges
    def test_bin_edges(self):
        a = self.spiketrain_a
        x = rep.binned_st(a, binsize=1 * pq.s, num_bins=10,
                          t_stop=10. * pq.s)
        # Test all edges
        edges = [float(i) for i in range(11)]
        self.assertTrue(np.array_equal(x.edges, edges))

        # Test left edges
        edges = [float(i) for i in range(10)]
        self.assertTrue(np.array_equal(x.left_edges, edges))

        # Test right edges
        edges = [float(i) for i in range(1, 11)]
        self.assertTrue(np.array_equal(x.right_edges, edges))

        # Test center edges
        edges = np.arange(0, 10) + 0.5
        self.assertTrue(np.array_equal(x.center_edges, edges))

    # Test addition of two binned classes with one spike train
    def test_iadd_1d(self):
        a = self.spiketrain_a
        b = self.spiketrain_b
        x = rep.binned_st(a, binsize=pq.s, t_start=0 * pq.s,
                          t_stop=10. * pq.s)
        y = rep.binned_st(b, binsize=pq.s, t_start=0 * pq.s,
                          t_stop=10. * pq.s)
        x += y
        targ = [[0, 0, 1, 3, 4, 5, 6, 0, 0, 1, 2, 4, 5, 8]]
        self.assertTrue(np.array_equal(x.filled, targ))

    # Test addition with 2 spike trains
    def test_iadd_2d(self):
        c = [self.spiketrain_a, self.spiketrain_b]
        x = rep.binned_st(c, binsize=pq.s, t_start=0 * pq.s,
                          t_stop=10. * pq.s)
        xfilled = x.filled
        x += x
        targ = [xfilled] * 2
        self.assertTrue(np.array_equal(x.filled, targ))

    # Test subtraction
    def test_subtraction_1d(self):
        a = self.spiketrain_a

        b = self.spiketrain_b
        x = rep.binned_st(a, binsize=pq.s, t_start=0 * pq.s,
                          t_stop=10. * pq.s)
        y = rep.binned_st(b, binsize=pq.s, t_start=0 * pq.s,
                          t_stop=10. * pq.s)
        x = x - y
        targ = [[2, 3, 6, 8]]
        self.assertTrue(np.array_equal(x.filled, targ))

    def test_isub_1d(self):
        a = self.spiketrain_a
        b = self.spiketrain_b
        x = rep.binned_st(a, binsize=pq.s, t_start=0 * pq.s,
                          t_stop=10. * pq.s)
        y = rep.binned_st(b, binsize=pq.s, t_start=0 * pq.s,
                          t_stop=10. * pq.s)
        x -= y
        targ = [[2, 3, 6, 8]]
        self.assertTrue(np.array_equal(x.filled, targ))

    # Subtracting from itself
    def test_sub_2d(self):
        c = [self.spiketrain_a, self.spiketrain_b]
        x = rep.binned_st(c, binsize=pq.s, t_start=0 * pq.s,
                          t_stop=10. * pq.s)
        x -= x
        self.assertTrue(np.array_equal(x.filled, [[]] * 2))

        st1 = n.SpikeTrain([1.7, 1.8, 4.3, 5.1, 6.7, 7.2, 9.4] * pq.s,
                           t_stop=10.0 * pq.s)
        st2 = n.SpikeTrain([0.1, 1.9, 2.1, 3.4, 6.1, 8.1, 8.9] * pq.s,
                           t_stop=10.0 * pq.s)
        # Subtracting from other class
        d = [st1, st2]
        x = rep.binned_st(c, binsize=pq.s, t_start=0 * pq.s,
                          t_stop=10. * pq.s)
        y = rep.binned_st(d, binsize=pq.s, t_start=0 * pq.s,
                          t_stop=10. * pq.s)
        targ = [np.array(list(set(x.filled[0]) ^ set(y.filled[0]))),
                np.array(list(set(x.filled[1]) ^ set(y.filled[1])))]
        x -= y
        self.assertTrue(np.array_equal(x.filled, targ))

    # Test for different units but same times
    def test_different_units(self):
        a = self.spiketrain_a
        b = a.rescale(pq.ms)
        binsize = 1*pq.s
        xa = rep.binned_st(a, binsize=binsize)
        xb = rep.binned_st(b, binsize=binsize.rescale(pq.ms))
        self.assertTrue(
            np.array_equal(xa.matrix_clipped(), xb.matrix_clipped()))
        self.assertTrue(
            np.array_equal(xa.filled, xb.filled))
        self.assertTrue(
            np.array_equal(xa.left_edges,
                           xb.left_edges.rescale(binsize.units)))
        self.assertTrue(xa == xb)


def suite():
    suit = unittest.makeSuite(RepTestCase, 'test')
    return suit


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

