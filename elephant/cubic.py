# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 13:48:18 2014

@author: quaglio
"""
import scipy.stats
import math
import warnings

# TODO: decide if we want to set the input as a population histogram (current
#version) or a list of SpikeTrains


def cubic(data, ximax=100, alpha=0.05, errorval=4., rate_distr='stat'):
    '''
    Performs the CuBIC analysis on a population histogram, calculated from
    a population of spiking neurons.

    The null hypothesis $H0: k_3(data)<=k*_3xi$ is iteratively tested with
    increasing correlation order xi until it's possible accept, with a
    significance level alpha, that xi_hat is the minimum order of correlation
    necessary to explain the third cumulant k_3(data).

    K*_3xi is the maximized third cumulant, supposed a CPP model with
    maximum order of correlation equal to xi.

    Parameters
    ----------
    data : neo.core.AnalogSignal
        The population histogram of the entire population of neurons.

    ximax : int
         The max number of iteration of the hypothesis test:
         if it's not possible compute the xi_hat before ximax
         iteration the CuBIC procedure is aborted.

    alpha : float
         The significance level of the hypothesis tests perfomed.

    errorval : float (optional)
         The value assigned to the p-value in the case that the test is
         aborted.
         Default: 4

    rate_distr: string (optional)
        The supposed distribution of the rate of the population used to compute
        CuBIC:
        * 'stat':  statonary constant rate
        * 'cos': spike trains with a cosine profile rate
        * 'gamma': instantaneous firing rates are distribuited as gamma
        * 'unif': instantaneous firing rates are distribuited as uniforms
        Default: 'stat'.

    Returns
    -------
    xi_hat : int
        The minimum correlation order stimated by CuBIC, necessary to
        explain the value of the third cumulant calculated from the population.

    p : list
        The ordred list of all the p-values of the hypothesis tests that have
        been performed.

    kappa : list
        The list of the first three cumulants of the data.

    References
    ----------
    Staude, Rotter, Gruen, (2009) J. Comp. Neurosci
    Staude, Gruen, Rotter, (2010) Frontiers Comp. Neurosci
    '''

    # dict of all possible rate functions
    rate_distr_funcs = {
        'stat':  _H03xi, 'cos': _H03xi_cosin,
        'gamma': _H03xi_gamma, 'unif': _H03xi_unif}
    if rate_distr not in rate_distr_funcs:
        raise ValueError("Unknown rate distribution selected.")

    # select appropriate function for hypothesis test
    H_func = rate_distr_funcs[rate_distr]

    L = len(data)
    # compute first three cumulants
    kappa = _kstat(data.magnitude)
    xi_hat = 1
    xi = 1
    pval = 0
    p = []
    # compute xi_hat iteratively
    while pval < alpha:
        xi_hat = xi
        if xi > ximax:
            warnings.warn('Test aborted, xihat > ximax')
            p.append(-4)
            break
        #compute p-value
        pval = H_func(
            kappa, xi, L, errorval, data.sampling_period)
        p.append(pval)
        xi = xi + 1

    return xi_hat, p, kappa


# TODO: remove default parameter from binsize=None --> (PQ:not done yet bacause
#not sure how to manage an input not ncessary in all the functions from the
#dict H_func)
# used to not break original code when adding Cubic_alternate
def _H03xi(kappa, xi, L, errorval, binsize=None):
    '''
    Computes the p_value for testing  the $H0: kappa[2]<=k*_3xi$ hypothesis of
    CuBIC in the stationary rate version

    Parameters
    -----
    kappa : list
        The first three cumulants of the populaton of spike trains

    xi : int
        The the maximum order of correlation xi xi supposed in the hypothesis
        for which is computed the p_value of H_0

    L : float
        The length of the orginal population histogram on which is performed
        the CuBIC analysis

    errorval : float
         The value assigned to the p-value in the case that the test is aborted


    Returns
    -----
    p : float
        The p-value of the hypothesis tests
    '''

    #Check the order condition of the cumulants necessary to perform CuBIC
    if kappa[1] < kappa[0]:
        p = errorval
        kstar = [0]
        warnings.warn('H_0 can not be tested: kappa(2)<kappa(1)!!! "\
            "p-value is set to p=4!!!')
    else:
        # computation of the maximized cumulants
        kstar = [_kappamstar(kappa[:2], i, xi) for i in range(2, 7)]
        k3star = kstar[1]

        # variance of third cumulant (from Stuart & Ord)
        sigmak3star = math.sqrt(
            kstar[4] / L + 9 * (kstar[2] * kstar[0] + kstar[1] ** 2) /
            (L - 1) + 6 * L * kstar[0] ** 3 / ((L - 1) * (L - 2)))
        #computation of the p-value (the third cumulant is supposed to
        #be gaussian istribuited)
        p = 1 - scipy.stats.norm(k3star, sigmak3star).cdf(kappa[2])
    return p


def _kappamstar(kappa, m, xi):
    '''
    Computes maximized cumulant of order m

    Parameters
    -----
    kappa : list
        The first two cumulants of the data

    xi : int
        The xi for which is computed the p_value of H_0

    m : float
        The order of the cumulant


    Returns
    -----
    k_out : list
        The maximized cumulant of order m
    '''

    if xi == 1:
        kappa_out = kappa[1]
    else:
        kappa_out = \
            (kappa[1] * (xi ** (m - 1) - 1) -
                kappa[0] * (xi ** (m - 1) - xi)) / (xi - 1)
    return kappa_out


#TODO: decide if itroduce the units for all the object that should need but are
# only internal computational features
def _H03xi_gamma(kappa, xi, L, errorval, binsize):
    '''
    Compute the p_value for testing  the H_0 hypothesis of CuBIC in the
    gamma distributed rate version

    Parameters
    -----
    kappa : list
        The first three cumulants of the data

    xi : int
        The the maximum order of correlation xi xi supposed in the hypothesis
        for which is computed the p_value of H_0

    L : float
        The length of the orginal population histogram on which is performed
        the CuBIC analysis

    errorval : float
        The value assigned to the p-value in the case that the test is aborted


    Returns
    -----
    p : float
        The p-value of the hypothesis tests
    '''

    #Check the order condition of the cumulants necessary to perform CuBIC
    if kappa[1] < kappa[0]:
        p = errorval
        warnings.warn(
            "H_0 can not be tested: kappa(2)<kappa(1)!!! "
            " p-value is set to p=4!!!")
    else:
        nu, beta, k3star = _maxk3_gamma(kappa, xi)
        # first cumulant of rate distribution
        knu = sum(nu)
        # amplitude distribution
        A = [n / float(sum(nu)) for n in nu]
        #first six moments of the amplitude distribution
        if xi == 1:
            mu = [1 for i in range(6)]
        else:
            mu = [A[0] + A[1] * xi ** k for k in range(1, 7)]
        if beta < 0:
                warnings.warn("beta_2<0! Set to 0!!!")
                beta = 0
        StatTol = 1e-10
        #check if beta is too small the maximization is computed as like as
        #in the stationary version
        if beta < StatTol:
            warnings.warn(
                '(step %d) beta too small, used stationary version' % xi)
            kstar = [knu * m for m in mu]
        else:
            #parameters of the rate distribution
            par = []
            par.append(1 / float(beta))
            par.append(beta * knu)
            # raw moments of the rate distribution
            munu = _HigherMoments_gamma(par, 6)
            binsize = binsize.rescale('sec')
            binsize = binsize.magnitude
            # convert to units of Hz^m
            munuHz = [
                munu[i - 1] / float(binsize ** i)
                for i in range(1, len(munu) + 1)]
            # maximization of cumulant in non-stationary case
            kstar = _KappaZNonStat(mu, munuHz, binsize)
        # check for numerical robustness
        err = abs(k3star - kstar[2])
        if err > 1e-9:
            warnings.warn(
                "_H03xi: Results not reliable!!! k3star - kappa(3)=", err)
        # variance of third k-statistics from Stuart & Ord
        sigmak3star = math.sqrt(
            kstar[5] / L + 9 * (kstar[3] * kstar[1] + kstar[2] ** 2) /
            float(L - 1) + 6 * L * kstar[1] ** 3 / float((L - 1) * (L - 2)))
        #bimodal distribution of the third cumulant case (sigma=0)
        if sigmak3star == 0:
                p = float(kappa[2] < k3star)
        #gaussian distribuited third cumulant case
        else:
                p = 1 - scipy.stats.norm(k3star, sigmak3star).cdf(kappa[2])
    return p


def _maxk3_gamma(_kstat, xi):
    '''
    Computes the maximized k3star, and the resulting model parameters nu and
    beta assuming a gamma distribution of the rate

    Parameters
    -----
    _kstat : list
       The first two cumulants of the data

    xi : int
        The xi for which is computed the p_value of H_0


    Returns
    -----
    nu : list
        The parameters of the model nu=[nu_1, nu_xi] for nu_1=A[1]*k_1[R] and
        nu_xi=A[xi]*k_1[R], where A is the amplitude of the CPP and R the rate
        distribution

    beta : float
        The parameter beta_2=k_2[R]/k_1[R]**2 where R is the gamma rate
        distribution

    k3star : float
        The the maximized third cumulant of the CPP model
    '''

    betafac = 2
    beta = (
        3 * _kstat[1] - (xi + 1) * _kstat[0]) / float((
        betafac * _kstat[0] ** 2))
    if beta < 0:
        beta = 0
    nu = []
    nu2 = (
        _kstat[1] - _kstat[0] - beta * _kstat[0] ** 2) / float((xi - 1) * xi)
    nu1 = _kstat[0] - xi * nu2
    nu.append(nu1)
    nu.append(nu2)
    if nu[0] < 0:
        nu[0] = 0
        nu[1] = _kstat[0] / xi
        beta = (_kstat[1] - xi * _kstat[0]) / float(_kstat[0] ** 2)
    elif nu[1] < 0:
        nu[1] = 0
        nu[0] = _kstat[0] - xi * nu[1]
        beta = (_kstat[1] - _kstat[0]) / float(_kstat[0] ** 2)
    beta3 = 2 * beta ** 2
    k3star = (
        nu[0] + nu[1] * xi ** 3 + beta3 * _kstat[0] ** 3 +
        3 * _kstat[0] * _kstat[1] * beta - 3 * _kstat[0] ** 3 * beta ** 2)
    return nu, beta, k3star


def _HigherMoments_gamma(par, n):
    '''
    Returnes the first n moments of a gamma distribution with
    the parameters in par

    Parameters
    -----
    par : list
        The parameters of the distribution

    n : int
        The order until which the moments are computed


    Returns
    -----
    moments : list
        The first n moments of a gamma distribution of parameters par
    '''

    moments = [
        par[1] ** k * math.gamma(k) / float(scipy.special.beta(par[0], k))
        for k in range(1, 7)]
    # beta(par(1),k)=gamma(par(1)+k)/gamma(par(1))
    return moments


def _H03xi_cosin(kappa, xi, L, errorval, binsize):
    '''
    Computes the p_value for testing  the H_0 hypothesis of CuBIC in the
    cosine distribuited rate version

    Parameters
    -----
    kappa : list
        The first three cumulants of the data

    xi : int
        The the maximum order of correlation xi xi supposed in the hypothesis
        for which is computed the p_value of H_0

    L : float
        The length of the orginal population histogram on which is performed
        the CuBIC analysis

    errorval : float
        The value assigned to the p-value in the case that the test is aborted


    Returns
    -----
    p : float
        The p-value of the hypothesis tests
    '''

    #Check the order condition of the cumulants necessary to perform CuBIC
    if kappa[1] < kappa[0]:
        p = errorval
        warnings.warn(
            'H_0 can not be tested: kappa(2)<kappa(1)!!! p-value is'
            ' set to p=4!!!')
    else:

        nu, beta, k3star, flag = _maxk3_cosin(kappa, xi)
        #in the case of failed maximization the p-value is equal to 1 if the
        #third cumulant of the population is ngative otherwise is set to 0
        if flag == 0:
            p = float(kappa[2] < k3star)
            warnings.warn(
                '_H03xi (step %d): Maximization of third cumulant failed.' %
                xi)
        else:
            # first cumulant of rate distribution
            knu = sum(nu)
            # amplitude distribution and its moments
            A = [n / float(sum(nu)) for n in nu]
            #first six moments of the amplitude distribution
            if xi == 1:
                mu = [1 for i in range(6)]
            else:
                mu = [A[0] + A[1] * xi ** k for k in range(1, 7)]
            if beta < 0:
                warnings.warn('beta_2<0! Set to 0!!!')
                beta = 0
            StatTol = 1e-10
            #check if beta is too small the maximization is computed as like as
            #in the stationary version
            if beta < StatTol:
                warnings.warn(
                    '(step %d) beta too small, used stationary version' % xi)
                kstar = [knu * m for m in mu]
            else:
                #parameter of the rate distribution
                par = []
                par.append(knu)
                par.append(math.sqrt(2 * beta) * knu)
                if par[0] < par[1]:
                    warnings.warn('maximization produced negative rate values')
                # raw moments of the rate distribution
                munu = _HigherMoments_cosin(par, 6)
                binsize = binsize.rescale('sec')
                binsize = binsize.magnitude
                # convert to units of Hz^m
                munuHz = [
                    munu[i - 1] / float(binsize ** i) for i in range(
                        1, len(munu) + 1)]
                # maximization of cumulant in non-stationary case
                kstar = _KappaZNonStat(mu, munuHz, binsize)
            # check for numerical robustness
            err = abs(k3star - kstar[2])
            if err > 1e-9:
                warnings.warn(
                    '_H03xi: Results not reliable!!! k3star - kappa(3)=', err)
            # variance of third k-statistics from Stuart & Ord
            sigmak3star = math.sqrt(
                kstar[5] / L + 9 * (kstar[3] * kstar[1] + kstar[2] ** 2) /
                float(L - 1) + 6 * L * kstar[1] ** 3 / float((L - 1) * (
                    L - 2)))
            #bimodal distribution of the third cumulant case (sigma=0)
            if sigmak3star == 0:
                p = float(kappa[2] < k3star)
            #gaussian distribuited third cumulant case
            else:
                p = 1 - scipy.stats.norm(k3star, sigmak3star).cdf(kappa[2])
        return p


def _maxk3_cosin(_kstat, xi):
    '''
    Computes the maximized k3star, and the resulting model parameters nu and
    beta, and the flag of the optimization in the cosin rate version

    Parameters
    -----
    _kstat : list
        The first two cumulants of the data

    xi : int
        The xi for which is computed the p_value of H_0


    Returns
    -----
    nu : list
        The parameters of the model nu=[nu_1, nu_xi] for nu_1=A[1]*k_1[R] and
        nu_xi=A[xi]*k_1[R], where A is the amplitude of the CPP and R the rate
        distribution given by a cosine rate profile

    beta : float
        The parameter beta_2=k_2[R]/k_1[R]**2 where R is the rate
        distribution

    k3star : float
        The the maximized third cumulant of the CPP model

    fl : int
        Binary result the check that the constrain are solveable, only in that
        case is equal to 1, else is 0
    '''

    betafac = 6
    betaMax = 1 / float(2)
    if _kstat[1] < _kstat[0] * (xi + _kstat[0] * betaMax):
    #only then are the constraints solveable!!!
        beta = (
            3 * _kstat[1] - (xi + 1) * _kstat[0]) / float((
            betafac * _kstat[0] ** 2))
        if beta > betaMax:
            beta = betaMax
        elif beta < 0:
            beta = 0
        nu = []
        nu2 = (
            _kstat[1] - _kstat[0] -
            beta * _kstat[0] ** 2) / float((xi - 1) * xi)
        nu1 = _kstat[0] - xi * nu2
        nu.append(nu1)
        nu.append(nu2)
        if nu[0] < 0:
            nu[0] = 0
            nu[1] = _kstat[0] / xi
            beta = (_kstat[1] - xi * _kstat[0]) / float(_kstat[0] ** 2)
        elif nu[1] < 0:
            nu[1] = 0
            nu[0] = _kstat[0] - xi * nu[1]
            beta = (_kstat[1] - _kstat[0]) / float(_kstat[0] ** 2)
        beta3 = 0
        k3star = (
            nu[0] + nu[1] * xi ** 3 + beta3 * _kstat[0] ** 3 +
            3 * _kstat[0] * _kstat[1] * beta - 3 * _kstat[0] ** 3 * beta ** 2)
        fl = 1
    #if the constrains are not solveable all the parameters are set to 0
    else:
        fl = 0
        nu = 0
        beta = 0
        k3star = 0
    return nu, beta, k3star, fl


def _HigherMoments_cosin(par, n):
    '''
    Returnes the first n moments of a cosin distribution with
    the parameters in par

    Parameters
    -----
    par : list
        The parameters of the distribution

    n : int
        The order until which the moments are computed


    Returns
    -----
    mom : list
        The first n moments of a cosin distribution of parameters par
    '''

    mom = []
    mom.append(par[0])
    mom.append(par[0] ** 2 + par[1] ** 2 / 2.)
    mom.append(par[0] ** 3 + (3 * par[0] * par[1] ** 2) / 2.)
    mom.append(
        par[0] ** 4 + 3 * par[0] ** 2 * par[1] ** 2 +
        (3 * par[1] ** 4) / 8.)
    mom.append(
        par[0] ** 5 + 5 * par[0] ** 3 * par[1] ** 2 + (
            15 * par[0] * par[1] ** 4) / 8.)
    mom.append(
        par[0] ** 6 + (15 * par[0] ** 4 * par[1] ** 2) / 2. +
        (45 * par[0] ** 2 * par[1] ** 4) / 8. + (5 * par[1] ** 6) / 16.)
    return mom


def _H03xi_unif(kappa, xi, L, errorval, binsize):
    '''
    Compute the p_value for testing  the H_0 hypothesis of CuBIC in the
    uniformly distribuited rate version

    Parameters
    -----
    kappa : list
        The first three cumulants of the data

    xi : int
        The the maximum order of correlation xi xi supposed in the hypothesis
        for which is computed the p_value of H_0

    L : float
        The length of the orginal population histogram on which is performed
        the CuBIC analysis

    errorval : float
        The value assigned to the p-value in the case that the test is aborted


    Returns
    -----
    p : float
        The p-value of the hypothesis tests
    '''

    #Check the order condition of the cumulants necessary to perform CuBIC
    if kappa[1] < kappa[0]:
        p = errorval
        warnings.warn(
            'H_0 can not be tested: kappa(2)<kappa(1)!!! p-value is set'
            ' to p=4!!!')
    else:

        nu, beta, k3star, flag = _maxk3_unif(kappa, xi)
        #in the case of failed maximization the p-value is equal to 1 if the
        #third cumulant of the population is ngative otherwise is set to 0
        if flag == 0:
            p = float(kappa[2] < k3star)
            warnings.warn(
                '_H03xi (step %d): Maximization of third cumulant failed.'
                % xi)
        else:
            # first cumulant of rate distribution
            knu = sum(nu)
            # amplitude distribution and its moments
            A = [n / float(sum(nu)) for n in nu]
            #first six moments of the amplitude distribution
            if xi == 1:
                mu = [1 for i in range(6)]
            else:
                mu = [A[0] + A[1] * xi ** k for k in range(1, 7)]

            if beta < 0:
                warnings.warn('beta_2<0! Set to 0!!!')
                beta = 0
            StatTol = 1e-10
            #check if beta is too small the maximization is computed as like as
            #in the stationary version
            if beta < StatTol:
                warnings.warn(
                    '(step %d) beta too small, used stationary version' % xi)
                kstar = [knu * m for m in mu]
            else:
                #parameter of the rate distribution
                par = []
                par.append(knu - math.sqrt(3 * beta * knu ** 2))
                par.append(knu + math.sqrt(3 * beta * knu ** 2))
                if par[0] < 0:
                    warnings.warn('maximization produced negative rate values')
                # raw moments of the rate distribution
                munu = _HigherMoments_unif(par, 6)

                binsize = binsize.rescale('sec')
                binsize = binsize.magnitude
                # convert to units of Hz^m
                munuHz = [
                    munu[i - 1] / float(binsize ** i) for i in range(
                        1, len(munu) + 1)]
                # maximization of cumulant in non-stationary case
                kstar = _KappaZNonStat(mu, munuHz, binsize)
            # check for numerical robustness
            err = abs(k3star - kstar[2])
            if err > 1e-9:
                warnings.warn(
                    '_H03xi: Results not reliable!!! k3star - kappa(3)=', err)
            # variance of third cumulant from Stuart & Ord
            sigmak3star = math.sqrt(
                kstar[5] / L + 9 * (kstar[3] * kstar[1] + kstar[2] ** 2) /
                float(L - 1) + 6 * L * kstar[1] ** 3 /
                float((L - 1) * (L - 2)))
            #bimodal distribution of the third cumulant case (sigma=0)
            if sigmak3star == 0:
                p = float(kappa[2] < k3star)
            #gaussian distribuited third cumulant case
            else:
                p = 1 - scipy.stats.norm(k3star, sigmak3star).cdf(kappa[2])
    return p


def _maxk3_unif(_kstat, xi):
    '''
    Computes the maximized third cumulant k3star, and the resulting model
    parameters nu, beta and the flag in the uniform rate version

     Parameters
    -----
    _kstat : list
        The first two cumulants of the data

    xi : int
        The xi for which is computed the p_value of H_0


    Returns
    -----
    nu : list
        The parameters of the model nu=[nu_1, nu_xi] for nu_1=A[1]*k_1[R] and
        nu_xi=A[xi]*k_1[R], where A is the amplitude of the CPP and R the
        uniform rate distribution

    beta : float
        The parameter beta_2=k_2[R]/k_1[R]**2 where R is the rate
        distribution

    k3star : float
        The the maximized third cumulant of the CPP model

    fl : int
        Binary result the check that the constrain are solveable, only in that
        case is equal to 1, else is 0
    '''

    betafac = 6
    betaMax = 1 / float(3)
    if _kstat[1] < _kstat[0] * (xi + _kstat[0] * betaMax):
    #only then are the constraints solveable!!!
        beta = (
            (3 * _kstat[1] - (xi + 1) * _kstat[0]) /
            (float((betafac * _kstat[0] ** 2))))
        if beta > betaMax:
            beta = betaMax
        elif beta < 0:
            beta = 0
        nu = []
        nu2 = (_kstat[1] - _kstat[0] - beta * _kstat[0] ** 2) / float(
            (xi - 1) * xi)
        nu1 = _kstat[0] - xi * nu2
        nu.append(nu1)
        nu.append(nu2)
        if nu[0] < 0:
            nu[0] = 0
            nu[1] = _kstat[0] / xi
            beta = (_kstat[1] - xi * _kstat[0]) / float(_kstat[0] ** 2)
        elif nu[1] < 0:
            nu[1] = 0
            nu[0] = _kstat[0] - xi * nu[1]
            beta = (_kstat[1] - _kstat[0]) / float(_kstat[0] ** 2)
        beta3 = 0
        k3star = (
            nu[0] + nu[1] * xi ** 3 + beta3 * _kstat[0] ** 3 + 3 *
            _kstat[0] * _kstat[1] * beta - 3 * _kstat[0] ** 3 * beta ** 2)
        flag = 1
    #if the constrains are not solveable all the parameters are set to 0
    else:
        flag = 0
        nu = 0
        beta = 0
        k3star = 0
    return nu, beta, k3star, flag


def _HigherMoments_unif(par, n):
    '''
    Returnes the first n moments of a uniform distribution with
    the parameters in par

    Parameters
    -----
    par : list
        The parameters of the distribution

    n : int
        The order until which the moments are computed


    Returns
    -----
    moments : list
        The first n moments of a uniform distribution of parameters par
    '''

    moments = []
    for i in range(1, n + 1):
        temp = [par[0] ** k * par[1] ** (i - k) for k in range(i + 1)]
        moments.append((1 / float(k + 1)) * sum(temp))
    return moments


def _KappaZNonStat(a, nu, h):
    '''
    Compute the first len(a) cumulants kappa of a non-stationary
    CPP with (raw) amplitude  moments a and rate moments nu.
    The bin size h should be in the correspondent time units of the rate nu

    method: copy results from Mathemtica script LawOfTotalCumulance.nb

    Parameters
    -----
    a : list
        The first len(a) moments of theamplitude distribution of the CPP model

    nu : list
         The corresondent moments of the rate distribution

    h : float
        The width of the bins


    Returns
    -----
    kappa : list
        The first length(a) cmulants of of the population
    '''

    order = len(a)
    kappa = []
    for i in range(order):
        kappa.append(_kthCumulant(a, nu, h, i + 1))
    return kappa


def _kthCumulant(a, nu, h, k):
    '''
    Compute the k-th cumulant  of a non-stationary CPP with (raw) amplitude
    moments a and rate moments nu
    The bin size h should be in the correspondent time units of the rate nu

    method: copy results from Mathemtica script LawOfTotalCumulance.nb

    Parameters
    -----
    a : list
        The first len(a) moments of theamplitude distribution of the CPP model

    nu : list
        The moments of the rate distribution

    h : float
        The width of the bins

    k : int
        The order of the cumulant to be computed


    Returns
    -----
    kappa : float
        The k-th cmulant of the population
    '''

    if k == 1:
        kappa = h * a[0] * nu[0]
    elif k == 2:
        kappa = h * (a[1] * nu[0] + (h * a[0] ** 2) * (nu[1] - nu[0] ** 2))
    elif k == 3:
        kappa = h * (
            a[2] * nu[0] + h * a[0] * (3 * a[1] * (nu[1] - nu[0] ** 2) + h * (
                a[0] ** 2) * (
                2 * nu[0] ** 3 - 3 * nu[0] * nu[1] + nu[2])))
    elif k == 4:
        kappa = h * (
            a[3] * nu[0] + h * (
                3 * a[1] ** 2 * (nu[1] - nu[0] ** 2) +
                6 * h * a[0] ** 2 * a[1] * (
                    2 * nu[0] ** 3 - 3 * nu[0] * nu[1] + nu[2]) + a[0] * (
                    4 * a[2] * (nu[1] - nu[0] ** 2) + h ** 2 * a[0] ** 3 * (
                        -6 * nu[0] ** 4 +
                        12 * nu[0] ** 2 * nu[1] -
                        3 * nu[1] ** 2 - 4 * nu[0] * nu[2] + nu[3]))))
    elif k == 5:
        kappa = h * (
            a[4] * nu[0] + h * (
                15 * h * a[0] * a[1] ** 2 * (
                    2 * nu[0] ** 3 - 3 * nu[0] * nu[1] + nu[2]) -
                10 * a[1] * (
                    a[2] * (nu[0] ** 2 - nu[1]) +
                    h ** 2 * a[0] ** 3 * (
                        6 * nu[0] ** 4 - 12 * nu[0] ** 2 * nu[1] +
                        3 * nu[1] ** 2 + 4 * nu[0] * nu[2] - nu[3])) +
                a[0] * (-5 * a[3] * (nu[0] ** 2 - nu[1]) + h * a[0] * (
                    10 * a[2] * (
                        2 * nu[0] ** 3 - 3 * nu[0] * nu[1] + nu[2]) +
                    h ** 2 * a[0] ** 3 * (
                        24 * nu[0] ** 5 - 60 * nu[0] ** 3 * nu[1] +
                        20 * nu[0] ** 2 * nu[2] -
                        10 * nu[1] * nu[2] + 5 * nu[0] * (
                            6 * nu[1] ** 2 - nu[3]) + nu[4])))))

        kappa = h * (
            a[4] * nu[0] + h * (
                15 * h * a[0] * a[1] ** 2 * (
                    2 * nu[0] ** 3 - 3 * nu[0] * nu[1] + nu[2]) -
                10 * a[1] * (
                    a[2] * (nu[0] ** 2 - nu[1]) + h ** 2 * a[0] ** 3 * (
                        6 * nu[0] ** 4 - 12 * nu[0] ** 2 * nu[1] +
                        3 * nu[1] ** 2 + 4 * nu[0] * nu[2] - nu[3])) + a[0] * (
                    -5 * a[3] * (nu[0] ** 2 - nu[1]) + h * a[0] * (
                        10 * a[2] * (
                            2 * nu[0] ** 3 - 3 * nu[0] * nu[1] + nu[2]) +
                        h ** 2 * a[0] ** 3 * (
                            24 * nu[0] ** 5 - 60 * nu[0] ** 3 * nu[1] +
                            20 * nu[0] ** 2 * nu[2] - 10 * nu[1] * nu[2] +
                            5 * nu[0] * (6 * nu[1] ** 2 - nu[3]) + nu[4])))))
    elif k == 6:
        kappa = h * (
            a[5] * nu[0] + h * (
                -10 * a[2] ** 2 * (nu[0] ** 2 - nu[1]) + 15 * h * a[1] ** 3 * (
                    2 * nu[0] ** 3 - 3 * nu[0] * nu[1] + nu[2]) -
                20 * h * a[0] * a[2] * (
                    -3 * a[1] * (2 * nu[0] ** 3 - 3 * nu[0] * nu[1] + nu[2]) +
                    h * a[0] ** 2 * (
                        6 * nu[0] ** 4 - 12 * nu[0] ** 2 * nu[1] +
                        3 * nu[1] ** 2 + 4 * nu[0] * nu[2] - nu[3])) -
                45 * h ** 2 * a[0] ** 2 * a[1] ** 2 * (
                    6 * nu[0] ** 4 - 12 * nu[0] ** 2 * nu[1] +
                    3 * nu[1] ** 2 + 4 * nu[0] * nu[2] - nu[3]) -
                15 * a[1] * (
                    a[3] * (nu[0] ** 2 - nu[1]) - h ** 3 * a[0] ** 4 * (
                        24 * nu[0] ** 5 - 60 * nu[0] ** 3 * nu[1] +
                        20 * nu[0] ** 2 * nu[2] - 10 * nu[1] * nu[2] +
                        5 * nu[0] * (6 * nu[1] ** 2 - nu[3]) +
                        nu[4])) + a[0] * (
                    -6 * a[4] * (
                        nu[0] ** 2 - nu[1]) + h * a[0] * (15 * a[3] * (
                        2 * nu[0] ** 3 - 3 * nu[0] * nu[1] + nu[2]) +
                        h ** 3 * a[0] ** 4 * (
                            -120 * nu[0] ** 6 + 360 * nu[0] ** 4 * nu[1] +
                            30 * nu[1] ** 3 - 120 * nu[0] ** 3 * nu[2] -
                            10 * nu[2] ** 2 - 30 * nu[0] ** 2 * (
                                9 * nu[1] ** 2 - nu[3]) -
                            15 * nu[1] * nu[3] +
                            6 * nu[0] * (
                                20 * nu[1] * nu[2] - nu[4]) + nu[5])))))
    return kappa


def _kstat(data):
    '''
    Compute first three cumulants of a population count of a population of
    spiking
    See http://mathworld.wolfram.com/k-Statistic.html or Stuart & Ord

    Parameters
    -----
    data : numpy.aray
        The population histogram of the population on which are computed
        the cumulants


    Returns
    -----
    kappa : list
        The first three cumulants of the population count
    '''
    L = len(data)
    S = [sum(data ** r) for r in range(1, 4)]
    kappa = []
    kappa.append(S[0] / float(L))
    kappa.append((L * S[1] - S[0] ** 2) / float(L * (L - 1)))
    kappa.append(
        (2 * S[0] ** 3 - 3 * L * S[0] * S[1] + L ** 2 * S[2]) / float(
            L * (L - 1) * (L - 2)))
    return kappa
