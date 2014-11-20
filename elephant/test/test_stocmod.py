'''
Created on Nov 27, 2012

@author: torre, gollan
'''

import unittest
import jelephant.core.stocmod as sm
import numpy as np
import quantities as pq
import neo


class StocModelsTestCase(unittest.TestCase):

    def test_genproc_poisson(self):

        #Bug encountered when the output has no unit or this unit is not time
        pp = sm.poisson(rate=10*pq.Hz, t_stop=1000*pq.ms)
        self.assertEqual(type(pp[0]), neo.core.spiketrain.SpikeTrain)
        self.assertEqual(pp[0].simplified.units, pq.sec)

        #Bug encountered when the output has wrong format
        pp = sm.poisson(rate=10*pq.Hz, t_stop=1*pq.sec)
        self.assertEqual(type(pp), list)

        pass

    def test_genproc_sip_poisson(self):

        # Generate an example SIP model
        T = 1*pq.sec
        rate_c = 1*pq.Hz
        M = 2
        sip, coinc = sm.sip_poisson(
            M=M, N=0, T=T, rate_b=10 * pq.Hz, rate_c=rate_c,
            return_coinc=True)

        # Bug encountered when the output is not a neo SpikeTrain object
        self.assertEqual(type(sip[0]), neo.core.spiketrain.SpikeTrain)
        self.assertEqual(type(coinc[0]), neo.core.spiketrain.SpikeTrain)

        # Bug encountered when the output unit is not time
        self.assertEqual(sip[0].simplified.units, pq.sec)
        self.assertEqual(coinc[0].simplified.units, pq.sec)

        # Bug encountered when the output has wrong format or length
        self.assertEqual(type(sip), list)
        self.assertEqual(len(sip), M)
        self.assertEqual(len(coinc[0]), (rate_c*T).rescale(pq.dimensionless))

        pass

    def test_genproc_msip_poisson(self):

        # Define parameters for mSIP simulation
        M = [1, 2, 3], [4, 5]
        N = 6
        T = 1 * pq.sec
        rate_b, rate_c = 5 * pq.Hz, [2, 3] * pq.Hz
        msip, coinc = sm.msip_poisson(
            M=M, N=N, T=T, rate_b=rate_b, rate_c=rate_c, return_coinc=True,
            output_format='list')

        # Bug encountered when the output has is not a neo SpikeTrain object
        #or this unit is not time
        self.assertTrue(
            np.all([type(train) == neo.SpikeTrain for train in msip]))
            #, tuple([neo.SpikeTrain]*len(msip)))
        self.assertTrue(
            np.all([train.simplified.units == pq.sec for train in msip]))
        # Bug encountered when N and the number of output spike trains mismatch
        self.assertEqual(len(msip), N)

        # Bug encountered when the number of SIP and coincidence arrays
        #mismatch, or the number of coincidences in any SIP and
        #its rate mismatch.
        self.assertEqual(len(coinc), len(M))
        # TODO: uncomment the test below
        ##self.assertEqual(
            ##[len(c) for c in coinc], list((rate_c*T).simplified.magnitude))

        pass

    def test_cpp_hom(self):
        #testing output with generic inputs
        A = [0, .9, .1]
        t_stop = 10 * pq.s
        t_start = 5 * pq.s
        rate = 3 * pq.Hz
        cpp_hom = sm.cpp(A, t_stop, rate, t_start=t_start)
        #testing the ouput formats
        self.assertEqual(
            [type(train) for train in cpp_hom], [neo.SpikeTrain]*len(cpp_hom))
        self.assertEqual(cpp_hom[0].simplified.units, pq.sec)
        self.assertEqual(type(cpp_hom), list)
        #testing quantities format of the output
        self.assertEqual(
            [train.simplified.units for train in cpp_hom], [pq.sec]*len(
                cpp_hom))
        #testing output t_start t_stop
        for st in cpp_hom:
            self.assertEqual(st.t_stop, t_stop)
            self.assertEqual(st.t_start, t_start)
        self.assertEqual(len(cpp_hom), len(A) - 1)

        #testing the units
        A = [0, 0.9, 0.1]
        t_stop = 10000*pq.ms
        t_start = 5 * pq.s
        rate = 3 * pq.Hz
        cpp_unit = sm.cpp(A, t_stop, rate, t_start=t_start)

        self.assertEqual(cpp_unit[0].units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_stop.units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_start.units, t_stop.units)

        #testing output without copy of spikes
        A = [1]
        t_stop = 10 * pq.s
        t_start = 5 * pq.s
        rate = 3 * pq.Hz
        cpp_hom_empty = sm.cpp(A, t_stop, rate, t_start=t_start)

        self.assertEqual(
            [len(train) for train in cpp_hom_empty], [0]*len(cpp_hom_empty))

        #testing output with rate equal to 0
        A = [0, .9, .1]
        t_stop = 10 * pq.s
        t_start = 5 * pq.s
        rate = 0 * pq.Hz
        cpp_hom_empty_r = sm.cpp(A, t_stop, rate, t_start=t_start)
        self.assertEqual(
            [len(train) for train in cpp_hom_empty_r], [0]*len(
                cpp_hom_empty_r))

        #testing output with same spike trains in output
        A = [0, 0, 1]
        t_stop = 10 * pq.s
        t_start = 5 * pq.s
        rate = 3 * pq.Hz
        cpp_hom_eq = sm.cpp(A, t_stop, rate, t_start=t_start)

        self.assertTrue(np.allclose(cpp_hom_eq[0], cpp_hom_eq[1]))

    def test_cpp_hom_errors(self):
        #testing raises of ValueError (wrong inputs)
        #testing empty amplitude
        self.assertRaises(
            ValueError, sm.cpp, A=[], t_stop=10*pq.s, rate=3*pq.Hz)

        #testing sum of amplitude>1
        self.assertRaises(
            ValueError, sm.cpp, A=[1, 1, 1], t_stop=10*pq.s, rate=3*pq.Hz)
        #testing negative value in the amplitude
        self.assertRaises(
            ValueError, sm.cpp, A=[-1, 1, 1], t_stop=10*pq.s, rate=3*pq.Hz)
        #testing t_start>t_stop
        self.assertRaises(
            ValueError, sm.cpp, A=[0, 1, 0], t_stop=10*pq.s, rate=3*pq.Hz,
            t_start=15*pq.s)
        #testing t_start=t_stop
        self.assertRaises(
            ValueError, sm.cpp, A=[0, 1, 0], t_stop=10*pq.s, rate=3*pq.Hz,
            t_start=10*pq.s)
        #test negative rate
        self.assertRaises(
            ValueError, sm.cpp, A=[0, 1, 0], t_stop=10*pq.s, rate=-3*pq.Hz)
        #test wrong unit for rate
        self.assertRaises(
            ValueError, sm.cpp, A=[0, 1, 0], t_stop=10*pq.s, rate=3*pq.s)

        #testing raises of AttributeError (missing input units)
        #Testing missing unit to t_stop
        self.assertRaises(
            AttributeError, sm.cpp, A=[0, 1, 0], t_stop=10, rate=3*pq.Hz)
        #Testing missing unit to t_start
        self.assertRaises(
            AttributeError, sm.cpp, A=[0, 1, 0], t_stop=10*pq.s, rate=3*pq.Hz,
            t_start=3)
        #testing rate missing unit
        self.assertRaises(
            AttributeError, sm.cpp, A=[0, 1, 0], t_stop=10*pq.s, rate=3)

    def test_cpp_het(self):
        #testing output with generic inputs
        A = [0, .9, .1]
        t_stop = 10 * pq.s
        t_start = 5 * pq.s
        rate = [3, 4] * pq.Hz
        cpp_het = sm.cpp(A, t_stop, rate, t_start=t_start)
        #testing the ouput formats
        self.assertEqual(
            [type(train) for train in cpp_het], [neo.SpikeTrain]*len(cpp_het))
        self.assertEqual(cpp_het[0].simplified.units, pq.sec)
        self.assertEqual(type(cpp_het), list)
        #testing units
        self.assertEqual(
            [train.simplified.units for train in cpp_het], [pq.sec]*len(
                cpp_het))
        #testing output t_start and t_stop
        for st in cpp_het:
            self.assertEqual(st.t_stop, t_stop)
            self.assertEqual(st.t_start, t_start)
        #testing the number of output spiketrains
        self.assertEqual(len(cpp_het), len(A) - 1)
        self.assertEqual(len(cpp_het), len(rate))

        #testing the units
        A = [0, 0.9, 0.1]
        t_stop = 10000*pq.ms
        t_start = 5 * pq.s
        rate = [3, 4] * pq.Hz
        cpp_unit = sm.cpp(A, t_stop, rate, t_start=t_start)

        self.assertEqual(cpp_unit[0].units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_stop.units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_start.units, t_stop.units)
        #testing without copying any spikes
        A = [1, 0, 0]
        t_stop = 10 * pq.s
        t_start = 5 * pq.s
        rate = [3, 4] * pq.Hz
        cpp_het_empty = sm.cpp(A, t_stop, rate, t_start=t_start)

        self.assertEqual(len(cpp_het_empty[0]), 0)

        #testing output with rate equal to 0
        A = [0, .9, .1]
        t_stop = 10 * pq.s
        t_start = 5 * pq.s
        rate = [0, 0] * pq.Hz
        cpp_het_empty_r = sm.cpp(A, t_stop, rate, t_start=t_start)
        self.assertEqual(
            [len(train) for train in cpp_het_empty_r], [0]*len(
                cpp_het_empty_r))

        #testing completely sync spiketrains
        A = [0, 0, 1]
        t_stop = 10 * pq.s
        t_start = 5 * pq.s
        rate = [3, 3] * pq.Hz
        cpp_het_eq = sm.cpp(A, t_stop, rate, t_start=t_start)

        self.assertTrue(np.allclose(cpp_het_eq[0], cpp_het_eq[1]))

    def test_cpp_het_err(self):
    #testing raises of ValueError (wrong inputs)
        #testing empty amplitude
        self.assertRaises(
            ValueError, sm.cpp, A=[], t_stop=10*pq.s, rate=[3, 4]*pq.Hz)
        #testing sum amplitude>1
        self.assertRaises(
            ValueError, sm.cpp, A=[1, 1, 1], t_stop=10*pq.s, rate=[3, 4]*pq.Hz)
        #testing amplitude negative value
        self.assertRaises(
            ValueError, sm.cpp, A=[-1, 1, 1], t_stop=10*pq.s,
            rate=[3, 4]*pq.Hz)
        #testing t_start>t_stop
        self.assertRaises(
            ValueError, sm.cpp, A=[0, 1, 0], t_stop=10*pq.s, rate=[3, 4]*pq.Hz,
            t_start=15*pq.s)
        #testing t_start=t_stop
        self.assertRaises(
            ValueError, sm.cpp, A=[0, 1, 0], t_stop=10*pq.s, rate=[3, 4]*pq.Hz,
            t_start=10*pq.s)
        #testing negative rate
        self.assertRaises(
            ValueError, sm.cpp, A=[0, 1, 0], t_stop=10*pq.s,
            rate=[-3, 4]*pq.Hz)
        #testing empty rate
        self.assertRaises(
            ValueError, sm.cpp, A=[0, 1, 0], t_stop=10*pq.s, rate=[]*pq.Hz)
        #testing empty amplitude
        self.assertRaises(
            ValueError, sm.cpp, A=[], t_stop=10*pq.s, rate=[3, 4]*pq.Hz)
        #testing different len(A)-1 and len(rate)
        self.assertRaises(
            ValueError, sm.cpp, A=[0, 1], t_stop=10*pq.s, rate=[3, 4]*pq.Hz)
        #testing rate with different unit from Hz
        self.assertRaises(
            ValueError, sm.cpp, A=[0, 1], t_stop=10*pq.s, rate=[3, 4]*pq.s)
        #Testing analytical constrain between amplitude and rate
        self.assertRaises(
            ValueError, sm.cpp, A=[0, 0, 1], t_stop=10*pq.s,
            rate=[3, 4]*pq.Hz, t_start=3)

        #testing raises of AttributeError (missing input units)
        #Testing missing unit to t_stop
        self.assertRaises(
            AttributeError, sm.cpp, A=[0, 1, 0], t_stop=10, rate=[3, 4]*pq.Hz)
        #Testing missing unit to t_start
        self.assertRaises(
            AttributeError, sm.cpp, A=[0, 1, 0], t_stop=10*pq.s,
            rate=[3, 4]*pq.Hz, t_start=3)
        #Testing missing unit to rate
        self.assertRaises(
            AttributeError, sm.cpp, A=[0, 1, 0], t_stop=10*pq.s,
            rate=[3, 4])

    def test_cpp_nonstat_hom(self):
        #testing output with generic inputs
        A = [0, .9, .1]
        rate = neo.AnalogSignal(
            [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)
        cpp_hom = sm.cpp_nonstat(A, rate)
        #testing the ouput formats
        self.assertEqual(
            [type(train) for train in cpp_hom], [neo.SpikeTrain]*len(cpp_hom))
        self.assertEqual(cpp_hom[0].simplified.units, pq.sec)
        self.assertEqual(type(cpp_hom), list)
        #testing quantities format of the output
        self.assertEqual(
            [train.simplified.units for train in cpp_hom], [pq.sec]*len(
                cpp_hom))
        #testing output t_start t_stop
        for st in cpp_hom:
            self.assertEqual(st.t_stop, rate.t_stop)
            self.assertEqual(st.t_start, rate.t_start)
        self.assertEqual(len(cpp_hom), len(A) - 1)

        #testing output without copy of spikes
        A = [1]
        rate = neo.AnalogSignal(
            [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)
        cpp_hom_empty = sm.cpp_nonstat(A, rate)

        self.assertEqual(
            [len(train) for train in cpp_hom_empty], [0]*len(cpp_hom_empty))

        #testing output with rate equal to 0
        A = [0, .9, .1]
        rate = neo.AnalogSignal(
            [0] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)
        cpp_hom_empty_r = sm.cpp_nonstat(A, rate)
        self.assertEqual(
            [len(train) for train in cpp_hom_empty_r], [0]*len(
                cpp_hom_empty_r))

        #testing completely sync spiketrains
        A = [0, 0, 1]
        rate = neo.AnalogSignal(
            [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)
        cpp_hom_eq = sm.cpp_nonstat(A, rate)
        self.assertTrue(np.allclose(cpp_hom_eq[0], cpp_hom_eq[1]))

    def test_cpp_nonstat_hom_errors(self):
        #testing raises of ValueError (wrong inputs)
        #testing empty amplitude
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[], rate=neo.AnalogSignal(
                [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s))

        #testing sum of amplitude>1
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[1, 1, 1], rate=neo.AnalogSignal(
                [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s))
        #testing negative value in the amplitude
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[-1, 1, 1], rate=neo.AnalogSignal(
                [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s))

        #test negative rate
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=neo.AnalogSignal(
                [-3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s))
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=neo.AnalogSignal(
                [3] * 10000 + [-3], units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s))
        #testing wrong unit rate
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=neo.AnalogSignal(
                [3] * 10000, units=pq.s, sampling_period=0.001*pq.s,
                t_start=5*pq.s))

        #test empty rate
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=neo.AnalogSignal(
                [], units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s))

        #testing raises of AttributeError (missing input units)
        #testing list instead neo.AnalogSignal
        self.assertRaises(
            AttributeError, sm.cpp_nonstat, A=[0, 1, 0], rate=[3] * 10000)
        #testing quantities array instead neo.AnalogSignal
        self.assertRaises(
            AttributeError, sm.cpp_nonstat, A=[0, 1, 0],
            rate=[3] * 10000*pq.Hz)

    def test_cpp_nonstat_het(self):
        #testing output with generic inputs

        A = [0, .9, .1]
        rate = [neo.AnalogSignal(
            [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s), neo.AnalogSignal(
            [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)]
        cpp_het = sm.cpp_nonstat(A, rate)
        #testing the ouput formats
        self.assertEqual(
            [type(train) for train in cpp_het], [neo.SpikeTrain]*len(cpp_het))
        self.assertEqual(cpp_het[0].simplified.units, pq.sec)
        self.assertEqual(type(cpp_het), list)
        #testing quantities format of the output
        self.assertEqual(
            [train.simplified.units for train in cpp_het], [pq.sec]*len(
                cpp_het))
        #testing output t_start t_stop
        for st in cpp_het:
            self.assertEqual(st.t_stop, rate[0].t_stop)
            self.assertEqual(st.t_start, rate[0].t_start)
        self.assertEqual(len(cpp_het), len(A) - 1)

        #testing without copying any spikes
        A = [1, 0, 0]
        rate = [neo.AnalogSignal(
            [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s), neo.AnalogSignal(
            [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)]
        cpp_het_empty = sm.cpp_nonstat(A, rate)

        self.assertEqual(len(cpp_het_empty[0]), 0)

        #testing output with rate equal to 0
        A = [0, .9, .1]
        rate = [neo.AnalogSignal(
            [0] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s), neo.AnalogSignal(
            [0] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)]
        cpp_het_empty_r = sm.cpp_nonstat(A, rate)
        self.assertEqual(
            [len(train) for train in cpp_het_empty_r], [0]*len(
                cpp_het_empty_r))

        #testing output with same spike trains in output
        A = [0, 0, 1]
        rate = [neo.AnalogSignal(
            [0] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s), neo.AnalogSignal(
            [0] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)]
        cpp_het_eq = sm.cpp_nonstat(A, rate)

        self.assertTrue(np.allclose(cpp_het_eq[0], cpp_het_eq[1]))

    def test_cpp_nonstat_het_err(self):
    #testing raises of ValueError (wrong inputs)
        #testing empty amplitude
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[], rate=[
                neo.AnalogSignal(
                    [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s), neo.AnalogSignal(
                    [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s)])
        #testing sum amplitude>1
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[1, 1, 1],  rate=[
                neo.AnalogSignal(
                    [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s), neo.AnalogSignal(
                    [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s)])
        #testing amplitude negative value
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[-1, 1, 1], rate=[
                neo.AnalogSignal(
                    [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s), neo.AnalogSignal(
                    [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s)])
        #testing negative rate
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=[
                neo.AnalogSignal(
                    [-3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s), neo.AnalogSignal(
                    [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s)])
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=[
                neo.AnalogSignal(
                    [-3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s), neo.AnalogSignal(
                    [-4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s)])
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=[
                neo.AnalogSignal(
                    [3] * 10000 + [-3], units=pq.Hz,
                    sampling_period=0.001*pq.s, t_start=5*pq.s),
                neo.AnalogSignal(
                    [4] * 10000 + [-4], units=pq.Hz,
                    sampling_period=0.001*pq.s, t_start=5*pq.s)])

        #test empty rate
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=[neo.AnalogSignal(
                [], units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s), neo.AnalogSignal(
                [], units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s)])

        #testing different len(A)-1 and len(rate)
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1], rate=[
                neo.AnalogSignal(
                    [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s), neo.AnalogSignal(
                    [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s)])

        #testing raises of AttributeError (missing input units)
        #Testing missing unit to t_stop
        self.assertRaises(
            AttributeError, sm.cpp_nonstat, A=[0, 1, 0], rate=[
                [3] * 10000, [4] * 10000])
        #Testing missing unit to t_start
        self.assertRaises(
            AttributeError, sm.cpp_nonstat, A=[0, 1, 0], rate=[
                [3] * 10000 * pq.Hz, [4] * 10000 * pq.Hz])

    #Test cpp_nonstat with method='thinning'
    def test_cpp_nonstat_hom_thinning(self):
        #testing output with generic inputs
        A = [0, .9, .1]
        rate = neo.AnalogSignal(
            [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)
        cpp_hom = sm.cpp_nonstat(A, rate, method='thinning')
        #testing the ouput formats
        self.assertEqual(
            [type(train) for train in cpp_hom], [neo.SpikeTrain]*len(cpp_hom))
        self.assertEqual(cpp_hom[0].simplified.units, pq.sec)
        self.assertEqual(type(cpp_hom), list)
        #testing quantities format of the output
        self.assertEqual(
            [train.simplified.units for train in cpp_hom], [pq.sec]*len(
                cpp_hom))
        #testing output t_start t_stop
        for st in cpp_hom:
            self.assertEqual(st.t_stop, rate.t_stop)
            self.assertEqual(st.t_start, rate.t_start)
        self.assertEqual(len(cpp_hom), len(A) - 1)

        #testing output without copy of spikes
        A = [1]
        rate = neo.AnalogSignal(
            [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)
        cpp_hom_empty = sm.cpp_nonstat(A, rate, method='thinning')

        self.assertEqual(
            [len(train) for train in cpp_hom_empty], [0]*len(cpp_hom_empty))

        #testing output with rate equal to 0
        A = [0, .9, .1]
        rate = neo.AnalogSignal(
            [0] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)
        cpp_hom_empty_r = sm.cpp_nonstat(A, rate, method='thinning')
        self.assertEqual(
            [len(train) for train in cpp_hom_empty_r], [0]*len(
                cpp_hom_empty_r))

        #testing completely sync spiketrains
        A = [0, 0, 1]
        rate = neo.AnalogSignal(
            [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)
        cpp_hom_eq = sm.cpp_nonstat(A, rate, method='thinning')
        self.assertTrue(np.allclose(cpp_hom_eq[0], cpp_hom_eq[1]))

    def test_cpp_nonstat_hom_errors_thinning(self):
        #testing raises of ValueError (wrong inputs)
        #testing empty amplitude
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[], rate=neo.AnalogSignal(
                [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s, method='thinning'))

        #testing sum of amplitude>1
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[1, 1, 1], rate=neo.AnalogSignal(
                [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s, method='thinning'))
        #testing negative value in the amplitude
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[-1, 1, 1], rate=neo.AnalogSignal(
                [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s, method='thinning'))

        #test negative rate
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=neo.AnalogSignal(
                [-3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s, method='thinning'))
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=neo.AnalogSignal(
                [3] * 10000 + [-3], units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s, method='thinning'))
        #testing wrong unit rate
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=neo.AnalogSignal(
                [3] * 10000, units=pq.s, sampling_period=0.001*pq.s,
                t_start=5*pq.s, method='thinning'))

        #test empty rate
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=neo.AnalogSignal(
                [], units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s, method='thinning'))

        #testing raises of AttributeError (missing input units)
        #testing list instead neo.AnalogSignal
        self.assertRaises(
            AttributeError, sm.cpp_nonstat, A=[0, 1, 0], rate=[3] * 10000,
            method='thinning')
        #testing quantities array instead neo.AnalogSignal
        self.assertRaises(
            AttributeError, sm.cpp_nonstat, A=[0, 1, 0],
            rate=[3] * 10000*pq.Hz, method='thinning')

    def test_cpp_nonstat_het_thinning(self):
        #testing output with generic inputs

        A = [0, .9, .1]
        rate = [neo.AnalogSignal(
            [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s), neo.AnalogSignal(
            [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)]
        cpp_het = sm.cpp_nonstat(A, rate, method='thinning')
        #testing the ouput formats
        self.assertEqual(
            [type(train) for train in cpp_het], [neo.SpikeTrain]*len(cpp_het))
        self.assertEqual(cpp_het[0].simplified.units, pq.sec)
        self.assertEqual(type(cpp_het), list)
        #testing quantities format of the output
        self.assertEqual(
            [train.simplified.units for train in cpp_het], [pq.sec]*len(
                cpp_het))
        #testing output t_start t_stop
        for st in cpp_het:
            self.assertEqual(st.t_stop, rate[0].t_stop)
            self.assertEqual(st.t_start, rate[0].t_start)
        self.assertEqual(len(cpp_het), len(A) - 1)

        #testing without copying any spikes
        A = [1, 0, 0]
        rate = [neo.AnalogSignal(
            [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s), neo.AnalogSignal(
            [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)]
        cpp_het_empty = sm.cpp_nonstat(A, rate, method='thinning')

        self.assertEqual(len(cpp_het_empty[0]), 0)

        #testing output with rate equal to 0
        A = [0, .9, .1]
        rate = [neo.AnalogSignal(
            [0] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s), neo.AnalogSignal(
            [0] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)]
        cpp_het_empty_r = sm.cpp_nonstat(A, rate, method='thinning')
        self.assertEqual(
            [len(train) for train in cpp_het_empty_r], [0]*len(
                cpp_het_empty_r))

        #testing output with same spike trains in output
        A = [0, 0, 1]
        rate = [neo.AnalogSignal(
            [0] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s), neo.AnalogSignal(
            [0] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)]
        cpp_het_eq = sm.cpp_nonstat(A, rate, method='thinning')

        self.assertTrue(np.allclose(cpp_het_eq[0], cpp_het_eq[1]))

    def test_cpp_nonstat_het_errors_thinning(self):
    #testing raises of ValueError (wrong inputs)
        #testing empty amplitude
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[], rate=[
                neo.AnalogSignal(
                    [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s), neo.AnalogSignal(
                    [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s)], method='thinning')
        #testing sum amplitude>1
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[1, 1, 1],  rate=[
                neo.AnalogSignal(
                    [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s), neo.AnalogSignal(
                    [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s)], method='thinning')
        #testing amplitude negative value
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[-1, 1, 1], rate=[
                neo.AnalogSignal(
                    [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s), neo.AnalogSignal(
                    [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s)], method='thinning')
        #testing negative rate
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=[
                neo.AnalogSignal(
                    [-3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s), neo.AnalogSignal(
                    [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s)], method='thinning')
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=[
                neo.AnalogSignal(
                    [-3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s), neo.AnalogSignal(
                    [-4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s)], method='thinning')
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=[
                neo.AnalogSignal(
                    [3] * 10000 + [-3], units=pq.Hz,
                    sampling_period=0.001*pq.s, t_start=5*pq.s),
                neo.AnalogSignal(
                    [4] * 10000 + [-4], units=pq.Hz,
                    sampling_period=0.001*pq.s, t_start=5*pq.s)],
            method='thinning')

        #test empty rate
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1, 0], rate=[neo.AnalogSignal(
                [], units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s), neo.AnalogSignal(
                [], units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s)], method='thinning')

        #testing different len(A)-1 and len(rate)
        self.assertRaises(
            ValueError, sm.cpp_nonstat, A=[0, 1], rate=[
                neo.AnalogSignal(
                    [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s), neo.AnalogSignal(
                    [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                    t_start=5*pq.s)], method='thinning')

        #testing raises of AttributeError (missing input units)
        #Testing missing unit to t_stop
        self.assertRaises(
            AttributeError, sm.cpp_nonstat, A=[0, 1, 0],
            rate=[[3] * 10000, [4] * 10000], method='thinning')
        #Testing missing unit to t_start
        self.assertRaises(
            AttributeError, sm.cpp_nonstat, A=[0, 1, 0], rate=[
                [3] * 10000 * pq.Hz, [4] * 10000 * pq.Hz], method='thinning')

    def test_gamma_thinning(self):
        t_start = 0 * pq.s
        t_stop = 10000 * pq.ms
        shape = 3
        N = 3
        gp = sm.gamma_thinning(
            t_stop=t_stop, shape=shape, rate=10*pq.Hz, N=N, t_start=t_start)
        #testing output format
        self.assertEqual(type(gp), list)
        self.assertEqual(len(gp), N)
        #testing parameter of every single spike train
        for g in gp:
            self.assertEqual(type(g), neo.core.spiketrain.SpikeTrain)
            self.assertEqual(g.t_start, t_start)
            self.assertEqual(g.t_stop, t_stop)
            self.assertEqual(g.units, t_stop.units)
        #testin rate=0Hz
        gp_emptyrate = sm.gamma_thinning(
            t_stop=t_stop, shape=shape, rate=0*pq.Hz, t_start=t_start)
        self.assertEqual(len(gp_emptyrate[0]), 0)

    def test_gamma_thinning_errors(self):
        #testing t_stop<t_start
        self.assertRaises(
            ValueError, sm.gamma_thinning, t_stop=5*pq.s, shape=3,
            rate=3*pq.Hz, t_start=10*pq.s)
        #testing t_stop=t_start
        self.assertRaises(
            ValueError, sm.gamma_thinning, t_stop=10*pq.s, shape=3,
            rate=3*pq.Hz, t_start=10*pq.s)
        #testing rate<0
        self.assertRaises(
            ValueError, sm.gamma_thinning, t_stop=5*pq.s, shape=3,
            rate=-3*pq.Hz, t_start=0*pq.s)
        #testing N<0
        self.assertRaises(
            ValueError, sm.gamma_thinning, t_stop=5*pq.s, shape=3,
            rate=3*pq.Hz, N=-2, t_start=0*pq.s)
        #testing type(N)!=integer
        self.assertRaises(
            ValueError, sm.gamma_thinning, t_stop=5*pq.s, shape=3,
            rate=3*pq.Hz, N=2.5, t_start=0*pq.s)
        #testing shape<0
        self.assertRaises(
            ValueError, sm.gamma_thinning, t_stop=5*pq.s, shape=-3,
            rate=3*pq.Hz, N=2, t_start=0*pq.s)
        #testing type(shape)!=integer
        self.assertRaises(
            ValueError, sm.gamma_thinning, t_stop=5*pq.s, shape=2.4,
            rate=3*pq.Hz, N=2, t_start=0*pq.s)
        #testing t_stop not quantity object
        self.assertRaises(
            ValueError, sm.gamma_thinning, t_stop=5*pq.s, shape=3,
            rate=3*pq.Hz, N=2, t_start=0)
        #testing t_start not quantity object
        self.assertRaises(
            AttributeError, sm.gamma_thinning, t_stop=5, shape=3,
            rate=3*pq.Hz, N=2, t_start=0*pq.s)
        #testing rate not quantity object
        self.assertRaises(
            ValueError, sm.gamma_thinning, t_stop=5*pq.s, shape=3,
            rate=3, N=-2, t_start=0*pq.s)

    def test_gamma_nonstat_rate(self):
        shape = 3
        N = 3
        rate_signal = neo.AnalogSignal(
            [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)
        gp = sm.gamma_nonstat_rate(
            shape=shape, rate_signal=rate_signal, N=N)
        #testing output format
        self.assertEqual(type(gp), list)
        self.assertEqual(len(gp), N)
        #testing parameter of every single spike train
        for g in gp:
            self.assertEqual(type(g), neo.core.spiketrain.SpikeTrain)
            self.assertEqual(g.t_start, rate_signal.t_start)
            self.assertEqual(g.t_stop, rate_signal.t_stop)
            self.assertEqual(g.units, rate_signal.t_start.units)
        #testin rate_signal=0Hz
        gp_emptyrate_signal = sm.gamma_nonstat_rate(
            shape=shape, rate_signal=neo.AnalogSignal(
                [0] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s))
        self.assertEqual(len(gp_emptyrate_signal[0]), 0)

    def test_gamma_nonstat_rate_errors(self):
        #testing rate_signal<0
        self.assertRaises(
            ValueError, sm.gamma_nonstat_rate, shape=3,
            rate_signal=neo.AnalogSignal(
                [4] * 10000 + [-3], units=pq.Hz, sampling_period=0.001*pq.s))
        #testing N<0
        self.assertRaises(
            ValueError, sm.gamma_nonstat_rate, shape=3,
            rate_signal=neo.AnalogSignal(
                [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s),
            N=-2)
        #testing type(N)!=integer
        self.assertRaises(
            ValueError, sm.gamma_nonstat_rate, shape=3,
            rate_signal=neo.AnalogSignal(
                [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s),
            N=2.5)
        #testing shape<0
        self.assertRaises(
            ValueError, sm.gamma_nonstat_rate, shape=-3,
            rate_signal=neo.AnalogSignal(
                [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s))
        #testing type(shape)!=integer
        self.assertRaises(
            ValueError, sm.gamma_nonstat_rate, shape=2.4,
            rate_signal=neo.AnalogSignal(
                [4] * 10000, units=pq.Hz, sampling_period=0.001*pq.s))
        #testing rate_signal not quantity object
        self.assertRaises(
            AttributeError, sm.gamma_nonstat_rate, shape=3,
            rate_signal=[3]*1000, N=-2)
        self.assertRaises(
            ValueError, sm.gamma_nonstat_rate, shape=3,
            rate_signal=neo.AnalogSignal(
                [4] * 10000, units=pq.s, sampling_period=0.001*pq.s), N=2)

    def test_cgp(self):
        #testing output with generic inputs
        A = [0, .9, .1]
        t_stop = 10 * pq.s
        t_start = 5 * pq.s
        rate = 3 * pq.Hz
        shape = 3
        cgp = sm.cgp(A, t_stop, shape, rate, t_start=t_start)
        #testing the ouput formats
        self.assertEqual(
            [type(train) for train in cgp], [neo.SpikeTrain]*len(cgp))
        self.assertEqual(cgp[0].simplified.units, pq.sec)
        self.assertEqual(type(cgp), list)
        #testing quantities format of the output
        self.assertEqual(
            [train.simplified.units for train in cgp], [pq.sec]*len(
                cgp))
        #testing output t_start t_stop
        for st in cgp:
            self.assertEqual(st.t_stop, t_stop)
            self.assertEqual(st.t_start, t_start)
        self.assertEqual(len(cgp), len(A) - 1)

        #testing the units
        A = [0, 0.9, 0.1]
        t_stop = 10000*pq.ms
        t_start = 5 * pq.s
        rate = 3 * pq.Hz
        cpp_unit = sm.cgp(A, t_stop, shape, rate, t_start=t_start)

        self.assertEqual(cpp_unit[0].units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_stop.units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_start.units, t_stop.units)

        #testing output with rate equal to 0
        A = [0, .9, .1]
        t_stop = 10 * pq.s
        t_start = 5 * pq.s
        rate = 0 * pq.Hz
        cgp_empty_r = sm.cgp(A, t_stop, shape, rate, t_start=t_start)
        self.assertEqual(
            [len(train) for train in cgp_empty_r], [0]*len(
                cgp_empty_r))

    def test_cgp_errors(self):
        #testing raises of ValueError (wrong inputs)
        #testing empty amplitude
        self.assertRaises(
            ValueError, sm.cgp, A=[], t_stop=10*pq.s, shape=3, rate=3*pq.Hz)

        #testing sum of amplitude>1
        self.assertRaises(
            ValueError, sm.cgp, A=[1, 1, 1], t_stop=10*pq.s, shape=3,
            rate=3*pq.Hz)
        #testing negative value in the amplitude
        self.assertRaises(
            ValueError, sm.cgp, A=[-1, 1, 1], t_stop=10*pq.s, shape=3,
            rate=3*pq.Hz)
        #testing t_start>t_stop
        self.assertRaises(
            ValueError, sm.cgp, A=[0, 1, 0], t_stop=10*pq.s, shape=3,
            rate=3*pq.Hz, t_start=15*pq.s)
        #testing t_start=t_stop
        self.assertRaises(
            ValueError, sm.cgp, A=[0, .9, .1], t_stop=10*pq.s, shape=3,
            rate=3*pq.Hz, t_start=10*pq.s)
        #test negative rate
        self.assertRaises(
            ValueError, sm.cgp, A=[0, 1, 0], t_stop=10*pq.s, shape=3,
            rate=-3*pq.Hz)
        #test wrong unit for rate
        self.assertRaises(
            ValueError, sm.cgp, A=[0, 1, 0], t_stop=10*pq.s, shape=3,
            rate=3*pq.s)

        #testing raises of AttributeError (missing input units)
        #Testing missing unit to t_stop
        self.assertRaises(
            AttributeError, sm.cgp, A=[0, 1, 0], t_stop=10, shape=3,
            rate=3*pq.Hz)
        #Testing missing unit to t_start
        self.assertRaises(
            ValueError, sm.cgp, A=[0, 1, 0], t_stop=10*pq.s, shape=3,
            rate=3*pq.Hz, t_start=3)
        #testing rate missing unit
        self.assertRaises(
            ValueError, sm.cgp, A=[0, 1, 0], t_stop=10*pq.s, shape=3,
            rate=3)
        #testing negative value of shape
        self.assertRaises(
            ValueError, sm.cgp, A=[0, 1, 0], t_stop=10*pq.s, shape=3,
            rate=3*pq.s)
        #testing negative non-integer value of shape
        self.assertRaises(
            ValueError, sm.cgp, A=[0, 1, 0], t_stop=10*pq.s, shape=2.5,
            rate=3*pq.s)

    def test_cgp_nonstat(self):
        #testing output with generic inputs
        A = [0, .9, .1]
        rate = neo.AnalogSignal(
            [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)
        shape = 3
        cgp = sm.cgp_nonstat(A, shape, rate)
        #testing the ouput formats
        self.assertEqual(
            [type(train) for train in cgp], [neo.SpikeTrain]*len(cgp))
        self.assertEqual(cgp[0].simplified.units, pq.sec)
        self.assertEqual(type(cgp), list)
        #testing quantities format of the output
        self.assertEqual(
            [train.simplified.units for train in cgp], [pq.sec]*len(
                cgp))
        #testing output t_start t_stop
        for st in cgp:
            self.assertEqual(st.t_stop, rate.t_stop)
            self.assertEqual(st.t_start, rate.t_start)
        self.assertEqual(len(cgp), len(A) - 1)

        #testing output with rate equal to 0
        A = [0, .9, .1]
        rate = neo.AnalogSignal(
            [0] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
            t_start=5*pq.s)
        cgp_empty_r = sm.cgp_nonstat(A, shape, rate)
        self.assertEqual(
            [len(train) for train in cgp_empty_r], [0]*len(
                cgp_empty_r))


    def test_cgp_nonstat_errors(self):
        #testing raises of ValueError (wrong inputs)
        #testing empty amplitude
        self.assertRaises(
            ValueError, sm.cgp_nonstat, A=[], shape=3, rate=neo.AnalogSignal(
                [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s))

        #testing sum of amplitude>1
        self.assertRaises(
            ValueError, sm.cgp_nonstat, A=[1, 1, 1], shape=3,
            rate=neo.AnalogSignal(
                [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s))
        #testing negative value in the amplitude
        self.assertRaises(
            ValueError, sm.cgp_nonstat, A=[-1, 1, 1], shape=3,
            rate=neo.AnalogSignal(
                [3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s))

        #test negative rate
        self.assertRaises(
            ValueError, sm.cgp_nonstat, A=[0, 1, 0], shape=3,
            rate=neo.AnalogSignal(
                [-3] * 10000, units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s))
        self.assertRaises(
            ValueError, sm.cgp_nonstat, A=[0, 1, 0], shape=3,
            rate=neo.AnalogSignal(
                [3] * 10000 + [-3], units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s))
        #testing wrong unit rate
        self.assertRaises(
            ValueError, sm.cgp_nonstat, A=[0, 1, 0], shape=3,
            rate=neo.AnalogSignal(
                [3] * 10000, units=pq.s, sampling_period=0.001*pq.s,
                t_start=5*pq.s))

        #test empty rate
        self.assertRaises(
            IndexError, sm.cgp_nonstat, A=[0, 1, 0], shape=3,
            rate=neo.AnalogSignal(
                [], units=pq.Hz, sampling_period=0.001*pq.s,
                t_start=5*pq.s))

        #testing raises of AttributeError (missing input units)
        #testing list instead neo.AnalogSignal
        self.assertRaises(
            AttributeError, sm.cgp_nonstat, A=[0, 1, 0], shape=3,
            rate=[3] * 10000)
        #testing quantities array instead neo.AnalogSignal
        self.assertRaises(
            AttributeError, sm.cgp_nonstat, A=[0, 1, 0], shape=3,
            rate=[3] * 10000*pq.Hz)


def suite():
    suite = unittest.makeSuite(StocModelsTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
