# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:35:49 2014

@author: quaglio
"""

import unittest
import quantities as pq
import neo
import elephant.cubic as cubic
import elephant.stocmod as sm
import elephant.statistics as stats


class CubicTestCase(unittest.TestCase):

    def test_cubic(self):
        #TODO: check cubic with pophist of length 1,2,3
        #(or strange pophist in general)
        A = [0, .9, .1]
        t_stop = 10 * pq.s
        t_start = 5 * pq.s
        rate = 3 * pq.Hz
        cpp_hom = sm.cpp(A, t_stop, rate, t_start=t_start)
        pop_hist = stats.peth(cpp_hom, 10*pq.ms)
        alpha = 0.05
        xi, p_vals, k = cubic.cubic(pop_hist, alpha=alpha)
        self.assertEqual(type(xi), int)
        self.assertEqual(type(p_vals), list)
        self.assertEqual(type(k), list)
        self.assertEqual(xi, len(p_vals))
        for p in p_vals[:-1]:
            self.assertGreater(alpha, p)
        self.assertGreater(p_vals[-1], alpha)
        self.assertEqual(3, len(k))
        self.assertGreater(k[2], k[1])
        self.assertGreater(k[1], k[0])
        self.assertRaises(ZeroDivisionError, cubic.cubic, neo.AnalogSignal(
            []*pq.Hz, sampling_period=10*pq.ms))
        pop_hist = neo.AnalogSignal(
            [1, 2, 1, 3, 4, 2, 1, 1, 1, 1, 3]*pq.Hz, sampling_period=10*pq.ms)
        pass



def suite():
    suite = unittest.makeSuite(CubicTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
