import unittest
import numpy as np
import quantities as pq
import neo.core
import jelephant.analysis.stats as stats


class StatisticsTestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_spiketrains = [
            neo.core.SpikeTrain([0.3, 0.56, 0.87, 1.23] * pq.ms,
                                t_start=0 * pq.ms, t_stop=1.5 * pq.ms),
            neo.core.SpikeTrain([0.02, 0.71, 1.82, 8.46] * pq.ms,
                                t_start=0 * pq.ms, t_stop=8.5 * pq.ms),
            neo.core.SpikeTrain([0.03, 0.14, 0.15, 0.92] * pq.ms,
                                t_start=0 * pq.ms, t_stop=1 * pq.ms)]

    def cv_test(self, spiketrains):
        if isinstance(spiketrains, neo.core.SpikeTrain):
            spiketrains = [spiketrains]
        isis = np.array([])
        for st in spiketrains:
            if len(st) > 1:
                isis = np.hstack([isis, np.diff(st.simplified.base)])
        cv = isis.std() / isis.mean()
        return cv

    def isi_test(self, spiketrains):
        isis = np.diff(spiketrains)
        if isinstance(spiketrains, neo.core.SpikeTrain):
            isis = pq.Quantity(isis.magnitude,
                               units=isis.units)
        return isis

    def test_isi(self):
        # Test with list of spiketrains
        isi_res = stats.isi(self.test_array_spiketrains)
        isi_targ = self.isi_test(self.test_array_spiketrains)
        self.assertTrue(np.array_equal(isi_res, isi_targ))

        # Test with empty list
        self.assertTrue(np.array_equal(stats.isi([]), []))

        # Test with one spiketrain
        st = neo.core.SpikeTrain([0.3, 0.56, 0.87, 1.23] * pq.ms,
                                 t_start=0 * pq.ms, t_stop=1.5 * pq.ms)
        isi_res = stats.isi(st)
        isi_targ = self.isi_test(st)
        self.assertTrue(np.array_equal(isi_res, isi_targ))

        # Test with empty spiketrain
        st = neo.core.SpikeTrain([] * pq.ms, t_start=0 * pq.ms,
                                 t_stop=1.5 * pq.ms)
        isi_res = stats.isi(st)
        isi_targ = self.isi_test(st)
        self.assertTrue(np.array_equal(isi_res, isi_targ))

    def test_cv(self):
        # Test with list of spiketrains
        cv_res = stats.cv(self.test_array_spiketrains)
        cv_targ = self.cv_test(self.test_array_spiketrains)
        self.assertTrue(cv_res, cv_targ)
        self.assertEqual(cv_targ, cv_res)

        # Test with empty array
        self.assertEqual(stats.cv([]), 0.0)

        # Test with one spiketrain
        st = neo.core.SpikeTrain([0.3, 0.56, 0.87, 1.23] * pq.ms,
                                 t_start=0 * pq.ms, t_stop=1.5 * pq.ms)
        self.assertEqual(stats.cv(st), self.cv_test(st))


class FanoFactorTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(100)
        num_st = 300
        self.spiketrains = []
        self.sp_counts = np.zeros(num_st)
        for i in range(num_st):
            st = neo.core.SpikeTrain(
                np.random.rand(np.random.randint(20) + 1) * pq.s,
                t_start=0 * pq.s,
                t_stop=10.0 * pq.s)
            self.spiketrains.append(st)
            # for cross-validation
            self.sp_counts[i] = len(st)

    def test_fanofactor(self):
        # Test with list of spiketrains
        self.assertEqual(
            np.var(self.sp_counts) / np.mean(self.sp_counts),
            stats.fanofactor(self.spiketrains))

        # Test with empty list
        self.assertEqual(stats.fanofactor([]), 0.0)

        # Test with same spiketrains
        sts = []
        for i in range(3):
            sts.append(neo.core.SpikeTrain([0.3, 0.56, 0.87, 1.23] * pq.ms,
                                           t_start=0 * pq.ms,
                                           t_stop=1.5 * pq.ms))
        self.assertEqual(stats.fanofactor(sts), 0.0)

        # Empty spiketrain
        st = neo.core.SpikeTrain([] * pq.ms, t_start=0 * pq.ms,
                                 t_stop=1.5 * pq.ms)
        self.assertEqual(stats.fanofactor(st), 0.0)

        # One spiketrain
        st = neo.core.SpikeTrain([0.3, 0.56, 0.87, 1.23] * pq.ms,
                                 t_start=0 * pq.ms, t_stop=1.5 * pq.ms)
        self.assertEqual(stats.fanofactor([st]), 0.0)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(StatisticsTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)