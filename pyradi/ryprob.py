################################################################
# The contents of this file are subject to the BSD 3Clause (New) License
# you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://directory.fsf.org/wiki/License:BSD_3Clause

# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
# License for the specific language governing rights and limitations
# under the License.

# The Original Code is part of the PyRadi toolkit.

# The Initial Developer of the Original Code is M Konnik and CJ Willers,
# Portions created by CJ Willers are Copyright (C) 2006-2015
# All Rights Reserved.

# Contributor(s): ______________________________________.
################################################################
"""
This module provides a few probability utility functions, most of which are used in  the 
high level  CCD and CMOS staring array model pyradi.rystare.

One of the utility functions provide a packaged version of scikit learn's 
kernel density estimation tool to provide a better estimate than does a 
simple histogram.

"""


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= ""
__author__='M Konnik and CJ Willers'
__all__=['distribution_exp','distribution_lognormal','distribution_inversegauss','distribution_logistic',
'distribution_wald','distributions_generator','validateParam','checkParamsNum','unifomOnSphere']

import sys
import numpy as np
import re


######################################################################################
def KDEbounded(x_d,x,bandwidth=np.nan,lowbnd=np.nan,uppbnd=np.nan,kernel = 'gaussian'):
    """Estimate the probability by Kernel Density Estimation

    If bandwidth is np.nan, calculate the optimal kernel width, aka bandwidth.  
    Be careful, this can take a while.

    Mirrors the data at either or both of the edges if the domain is bounded.

    Args:
        | x_d (np.array[N,]): domain over which values must be returned
        | x (np.array[N,]): input ample data set
        | bandwidth (float): the bandwidth width to be used, np.nan if to be calculated
        | lowbnd (float): lower mirror fold boundary, np.nan means no lower bound and mirror
        | uppbnd (float): upper mirror fold boundary, np.nan means no upper bound and mirror
        | kernel (str): kernel to be used ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']


    Returns:
        | x_d (np.array[N,]): input vector used as domain for the calculations
        | x (np.array[N,]): probability density over x_d, the range of the PDF
        | bandwidth (float): bandwidth used in the KDE
        | kernel (str): kernel used 

    Raises:
        | No exception is raised.

    See here for more detail and examples:
    https://github.com/NelisW/PythonNotesToSelf/tree/master/KernelDensityEstimation

    Other references:
    https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html#Motivating-KDE:-Histograms
    https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html
    https://stats.stackexchange.com/questions/405357/how-to-choose-the-bandwidth-of-a-kde-in-python
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    https://stats.stackexchange.com/questions/405357/how-to-choose-the-bandwidth-of-a-kde-in-python
    https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
    https://towardsdatascience.com/how-to-find-probability-from-probability-density-plots-7c392b218bab
    https://github.com/admond1994/calculate-probability-from-probability-density-plots/blob/master/cal_probability.ipynb
   
    """

    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import LeaveOneOut
   
    # find optimal bandwidth if not supplied
    if np.isnan(bandwidth):
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        kd = KernelDensity(kernel=kernel)
        grid = GridSearchCV(kd,param_grid={'bandwidth': bandwidths},
                            cv=LeaveOneOut())
        grid.fit(x[:, None]); # create additional axes of length one
        bandwidth = grid.best_params_['bandwidth']
 
    X = []
    Xo = []
    
    # base data, and if required lower flipped, upper flipped
    X.append(x)
    if not np.isnan(lowbnd):
        X.append(-x + 2 * lowbnd)
    if not np.isnan(uppbnd):
        X.append(-x + 2 * uppbnd)

    # do for base, and if present lower and upper flipped
    for i,x in enumerate(X):
        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(x[:, None]) # create additional axes of length one
        # score_samples returns the log of the probability density
        prob = np.exp(kde.score_samples(x_d[:, None])) # create additional axes of length one
        Xo.append(prob)

    # add base and flipped together
    x = np.zeros_like(x_d)
    for xi in Xo:
        x += xi

    # only cut out the base domain
    if not np.isnan(lowbnd):
        x = np.where(x_d<=lowbnd,0,x)
    if not np.isnan(uppbnd):
        x = np.where(x_d>=uppbnd,0,x)
    
    return x_d,x,bandwidth, kernel


######################################################################################
def distribution_exp(distribParams, out, funcName):
    r"""Exponential Distribution

    This function is meant to be called via the `distributions_generator` function.

    :math:`\textrm{pdf} = \lambda * \exp( -\lambda * y )`

    :math:`\textrm{cdf} = 1 - \exp(-\lambda * y)`

    - Mean = 1/lambda
    - Variance = 1/lambda^2
    - Mode = lambda
    - Median = log(2)/lambda
    - Skewness = 2
    - Kurtosis = 6

    GENERATING FUNCTION:   :math:`T=-\log_e(U)/\lambda`

    PARAMETERS: distribParams[0] is lambda - inverse scale or rate (lambda>0)

    SUPPORT: y,  y>= 0

    CLASS: Continuous skewed distributions

    NOTES: The discrete version of the Exponential distribution is 
    the Geometric distribution.

    USAGE: 

    - y = randraw('exp', lambda, sampleSize) - generate sampleSize number of variates from the Exponential distribution with parameter 'lambda';
 
    EXAMPLES:

    1.   y = randraw('exp', 1, [1 1e5]);
    2.   y = randraw('exp', 1.5, 1, 1e5);
    3.   y = randraw('exp', 2, 1e5 );
    4.   y = randraw('exp', 3, [1e5 1] );

    SEE ALSO:
    GEOMETRIC, GAMMA, POISSON, WEIBULL distributions
    http://en.wikipedia.org/wiki/Exponential_distribution
    """
    if distribParams is None or len(distribParams)==0:
        distribParams = [1.]
    if checkParamsNum(funcName,'Exponential','exp',distribParams,[1]):
        _lambda = distribParams[0]
        if validateParam(funcName,'Exponential','exp','lambda','lambda',_lambda,[str('> 0')]):
            out = - np.log(np.random.rand(*out.shape)) / _lambda

    return out



######################################################################################
def distribution_lognormal(distribParams, out, funcName):
    """THe Log-normal Distribution (sometimes: Cobb-Douglas or antilognormal distribution)

    This function is meant to be called via the `distributions_generator` function.

    pdf = 1/(y*sigma*sqrt(2*pi)) * exp(-1/2*((log(y)-mu)/sigma)^2)
    cdf = 1/2*(1 + erf((log(y)-mu)/(sigma*sqrt(2))));

    - Mean = exp( mu + sigma^2/2 );
    - Variance = exp(2*mu+sigma^2)*( exp(sigma^2)-1 );
    - Skewness = (exp(1)+2)*sqrt(exp(1)-1), for mu=0 and sigma=1;
    - Kurtosis = exp(4) + 2*exp(3) + 3*exp(2) - 6; for mu=0 and sigma=1;
    - Mode = exp(mu-sigma^2);

    PARAMETERS: mu - location, sigma - scale (sigma>0)

    SUPPORT:  y,  y>0

    CLASS: Continuous skewed distribution                      

    NOTES:

    1. The LogNormal distribution is always right-skewed
    2. Parameters mu and sigma are the mean and standard deviation of y in (natural) log space.
    3. mu = log(mean(y)) - 1/2*log(1 + var(y)/(mean(y))^2)
    4. sigma = sqrt( log( 1 + var(y)/(mean(y))^2) )

    USAGE:

    - randraw('lognorm', [], sampleSize) - generate sampleSize number
      of variates from the standard Lognormal distribution with 
      location parameter mu=0 and scale parameter sigma=1 
    - randraw('lognorm', [mu, sigma], sampleSize) - generate sampleSize number
      of variates from the Lognormal distribution with 
      location parameter 'mu' and scale parameter 'sigma'

    EXAMPLES:

    1.   y = randraw('lognorm', [], [1 1e5]);
    2.   y = randraw('lognorm', [0, 4], 1, 1e5);
    3.   y = randraw('lognorm', [-1, 10.2], 1e5 );
    4.   y = randraw('lognorm', [3.2, 0.3], [1e5 1] );
    """

    if distribParams is None or len(distribParams)==0:
        distribParams = [0., 1.]

    if checkParamsNum(funcName,'Lognormal','lognorm',distribParams,[0,2]):
        mu = distribParams[0]
        sigma = distribParams[1]
        if validateParam(funcName,'Lognormal','lognorm','[mu, sigma]','sigma',sigma,[str('> 0')]):
            out = np.exp(mu + sigma * np.random.randn(*out.shape)) 
    
    return out
  

######################################################################################
def distribution_inversegauss(distribParams, out, funcName):
    """The Inverse Gaussian Distribution

    This function is meant to be called via the `distributions_generator` function.

    The Inverse Gaussian distribution is left skewed distribution whose
    location is set by the mean with the profile determined by the
    scale factor.  The random variable can take a value between zero and
    infinity.  The skewness increases rapidly with decreasing values of
    the scale parameter.

    pdf(y) = sqrt(_lambda/(2*pi*y^3)) * exp(-_lambda./(2*y).*(y/mu-1).^2)

    cdf(y) = normcdf(sqrt(_lambda./y).*(y/mu-1)) + exp(2*_lambda/mu)*normcdf(sqrt(_lambda./y).*(-y/mu-1))

    where  normcdf(x) = 0.5*(1+erf(y/sqrt(2))); is the standard normal CDF
         
    - Mean     = mu
    - Variance = mu^3/_lambda
    - Skewness = sqrt(9*mu/_lambda)
    - Kurtosis = 15*mean/scale
    - Mode = mu/(2*_lambda)*(sqrt(9*mu^2+4*_lambda^2)-3*mu)

    PARAMETERS: mu - location; (mu>0), _lambda - scale; (_lambda>0)

    SUPPORT: y,  y>0

    CLASS: Continuous skewed distribution

    NOTES:

    1. There are several alternate forms for the PDF, some of which have more than two parameters
    2. The Inverse Gaussian distribution is often called the Inverse Normal
    3. Wald distribution is a special case of The Inverse Gaussian distribution where the mean is a 
       constant with the value one.
    4. The Inverse Gaussian distribution is a special case of The Generalized Hyperbolic Distribution

    USAGE:

    - randraw('ig', [mu, _lambda], sampleSize) - generate sampleSize number of variates 
      from the Inverse Gaussian distribution with parameters mu and _lambda;

    EXAMPLES:

    1.   y = randraw('ig', [0.1, 1], [1 1e5]);
    2.   y = randraw('ig', [3.2, 10], 1, 1e5);
    3.   y = randraw('ig', [100.2, 6], 1e5 );
    4.   y = randraw('ig', [10, 10.5], [1e5 1] );
 
    SEE ALSO:     WALD distribution

    Method:

    There is an efficient procedure that utilizes a transformation yielding two roots.
    If Y is Inverse Gauss random variable, then following [1] we can write:
    V = _lambda*(Y-mu)^2/(Y*mu^2) ~ Chi-Square(1)

    i.e. V is distributed as a _lambda-square random variable with one degree of freedom.
    So it can be simply generated by taking a square of a standard normal random number.
    Solving this equation for Y yields two roots:

    y1 = mu + 0.5*mu/_lambda * ( mu*V - sqrt(4*mu*_lambda*V + mu^2*V.^2) );
    and
    y2 = mu^2/y1;

    In [2] showed that  Y can be simulated by choosing y1 with probability
    mu/(mu+y1) and y2 with probability 1-mu/(mu+y1)

    References:
    [1] Shuster, J. (1968). On the Inverse Gaussian Distribution Function, Journal of the American 
    Statistical Association 63: 1514-1516.

    [2] Michael, J.R., Schucany, W.R. and Haas, R.W. (1976). Generating Random Variates Using 
    Transformations with Multiple Roots, The American Statistician 30: 88-90.

    http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
    """

    if distribParams is None or len(distribParams)==0:
        distribParams = [0., 1.]

    if checkParamsNum(funcName,'Inverse Gaussian','ig',distribParams,[2]):
        mu = distribParams[0]
        _lambda = distribParams[1]
        if validateParam(funcName,'Inverse Gaussian','ig','[mu, _lambda]','mu',mu,[str('> 0')]) & \
            validateParam(funcName,'Inverse Gaussian','ig','[mu, _lambda]','_lambda',_lambda,[str('> 0')]):
            chisq1 = np.random.randn(*out.shape) ** 2
            out = mu + 0.5 * mu / _lambda * (mu * chisq1 - np.sqrt(4 * mu * _lambda * chisq1 + mu ** 2 * chisq1 ** 2))
            l = np.random.rand(*out.shape) >= mu / (mu + out)
            out[l] = mu ** 2.0 / out[l]

    return out


######################################################################################
def distribution_logistic(distribParams, out, funcName):
    """The Logistic Distribution

    This function is meant to be called via the `distributions_generator` function.

    The logistic distribution is a symmetrical bell shaped distribution.
    One of its applications is an alternative to the Normal distribution
    when a higher proportion of the population being modeled is
    distributed in the tails.

    pdf(y) = exp((y-a)/k)./(k*(1+exp((y-a)/k)).^2)

    cdf(y) = 1 ./ (1+exp(-(y-a)/k))

    - Mean = a
    - Variance = k^2*pi^2/3
    - Skewness = 0
    - Kurtosis = 1.2

    PARAMETERS: a - location,  k - scale (k>0);

    SUPPORT: y,  -Inf < y < Inf

    CLASS: Continuous symmetric distribution                      

    USAGE:

    - randraw('logistic', [], sampleSize) - generate sampleSize number of variates from the 
      standard Logistic distribution with location parameter a=0 and scale parameter k=1;                    
    - Logistic distribution with location parameter 'a' and scale parameter 'k';

    EXAMPLES:

    1.   y = randraw('logistic', [], [1 1e5]);
    2.   y = randraw('logistic', [0, 4], 1, 1e5);
    3.   y = randraw('logistic', [-1, 10.2], 1e5 );
    4.   y = randraw('logistic', [3.2, 0.3], [1e5 1] );

    Method:

    Inverse CDF transformation method.

    http://en.wikipedia.org/wiki/Logistic_distribution
    """

    if distribParams is None or len(distribParams)==0:
        distribParams = [0., 1.]
    if checkParamsNum(funcName,'Logistic','logistic',distribParams,[0,2]):
        a = distribParams[0]
        k = distribParams[1]
        if validateParam(funcName,'Laplace','laplace','[a, k]','k',k,[str('> 0')]):
            u1 = np.random.rand(*out.shape)
            out = a - k * np.log(1.0 / u1 - 1)

    return out


######################################################################################
def distribution_wald(distribParams, out, funcName):
    """The Wald Distribution

    This function is meant to be called via the `distributions_generator` function.

    The Wald distribution is as special case of the Inverse Gaussian Distribution
    where the mean is a constant with the value one.

    pdf = sqrt(chi/(2*pi*y^3)) * exp(-chi./(2*y).*(y-1).^2);

    - Mean     = 1
    - Variance = 1/chi
    - Skewness = sqrt(9/chi)
    - Kurtosis = 3+ 15/scale

    PARAMETERS: chi - scale parameter; (chi>0)

    SUPPORT: y,  y>0

    CLASS: Continuous skewed distributions

    USAGE:

    - randraw('wald', chi, sampleSize) - generate sampleSize number of variates from the 
      Wald distribution with scale parameter 'chi';

    EXAMPLES:

    1.   y = randraw('wald', 0.5, [1 1e5]);
    2.   y = randraw('wald', 1, 1, 1e5);
    3.   y = randraw('wald', 1.5, 1e5 );
    4.   y = randraw('wald', 2, [1e5 1] );
    """                      

    if distribParams is None or len(distribParams)==0:
        distribParams = [0.]
    if checkParamsNum(funcName,'Wald','wald',distribParams,[1]):
        chi = distribParams[0]
        if validateParam(funcName,'Wald','wald','chi','chi',chi,[str('> 0')]):
            # out = feval_(funcName,'ig',[1,chi],*out.shape)
            out = distributions_generator('ig', [1, chi], sampleSize=out.shape)

    return out


######################################################################################
def distributions_generator(distribName=None, distribParams=None, sampleSize=None): 
    """The routine contains various models for simulation of FPN (DSNU or PRNU).

    This function allows the user to select the distribution by name and pass requisite
    parameters in a list (which differs for different distrubutions).  The size of the 
    distribution is defined by a scalar or list.

    sampleSize follows Matlab conventions:

    - if None then return a single scalar value
    - if scalar int N then return NxN array
    - if tuple then return tuple-sized array


    Possible values for distribName:
        | 'exp','exponential'
        | 'lognorm','lognormal','cobbdouglas','antilognormal'
        | 'ig', 'inversegauss', 'invgauss'
        | 'logistic'
        | 'wald'

    Args:
        | distribName (string): required distribution name
        | distribParams ([float]): list of distribution parameters (see below)
        | sampleSize (None,int,[int,int]): Size of the returned random set

    Returns:
        | out (float, np.array[N,M]): set of random variables for selected distribution.

    Raises:
        | No exception is raised.

    The routine set generates various types of random distributions, and is based on the 
    code randraw  by Alex Bar Guy  &  Alexander Podgaetsky
    These programs are distributed in the hope that they will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    Author: Alex Bar Guy, comments to alex@wavion.co.il
    """

    funcName = distributions_generator.__name__

    #remove spaces from distribution name
    # print(distribName)
    pattern = re.compile(r'\s+')
    distribNameInner = re.sub(pattern, '', distribName).lower()    

    if type(sampleSize) is int:
        out = np.zeros((sampleSize, sampleSize))
    elif type(sampleSize) is list or type(sampleSize) is tuple:
        out = np.zeros(sampleSize)
    else:
        out = np.zeros(1)

    if distribName is not None:

        if distribNameInner in ['exp','exponential']:
            out = distribution_exp(distribParams=None, out=out, funcName=funcName)

        elif distribNameInner in ['lognorm','lognormal','cobbdouglas','antilognormal']:
            out = distribution_lognormal(distribParams=None, out=out, funcName=funcName)

        elif distribNameInner in ['ig', 'inversegauss', 'invgauss']:
            out = distribution_inversegauss(distribParams=None, out=out, funcName=funcName)

        elif distribNameInner in ['logistic']:
            out = distribution_logistic(distribParams=None, out=out, funcName=funcName)

        elif distribNameInner in ['wald']:
            out = distribution_wald(distribParams=None, out=out, funcName=funcName)

        else:
            print('\n distributions_generator: Unknown distribution name: {}/{} \n'.format(distribName, distribNameInner))

    if out.shape == (1,):
        out = out[0]
    return out  



##########################################################################################3
def validateParam(funcName=None, distribName=None, runDistribName=None, distribParamsName=None, paramName=None, param=None, conditionStr=None):
    """Validate the range and number of parameters

    Args:
        | funcName (string):  distribution name
        | distribName (string):  distribution name
        | runDistribName (string):  run distribution name
        | distribParamsName
        | paramName 
        | param
        | conditionStr

    Returns:
        | True if the requirements are matched

    Raises:
        | No exception is raised.    
    """
    import math

    condLogical = True

    eqCondStr = ''
    for i,strn in enumerate(conditionStr):
        if i == 0:
            eqCondStr =  eqCondStr + conditionStr[i]
        else:
            eqCondStr =  eqCondStr + ' and ' + conditionStr[i]

        eqCond = conditionStr[i][0:2]
        # print('{} {} '.format(conditionStr[i], eqCond), end='')
        #remove spaces 
        pattern = re.compile(r'\s+')
        eqCond = re.sub(pattern, '', eqCond)   
        # print('{}'.format(eqCond))

        # print(eqCond)
        # funcName=funcName, distribName='Wald', runDistribName='wald', distribParamsName='chi', paramName='chi', param=chi, conditionStr[i]='> 0')

        if eqCond in ['<']: 
            condLogical &= param < float(conditionStr[i][2:])

        elif eqCond in ['<=']: 
            condLogical &= param <= float(conditionStr[i][2:])

        elif eqCond in ['>']: 
            condLogical &= param > float(conditionStr[i][2:])

        elif eqCond in ['>=']:
            condLogical &= param >= float(conditionStr[i][2:])

        elif eqCond in ['!=']: 
            condLogical &= param != float(conditionStr[i][2:])

        elif eqCond in ['==']:

            if 'integer' in conditionStr[i][2:]:
                condLogical &= param == math.floor_(param)
            else:
                condLogical &= param == math.float(conditionStr[i][2:])

    if not condLogical:
        print('{} Variates Generation: {}({}, {});\n Parameter {} should be {}\n (run {} ({}) for help)'.format(distribName, 
            funcName,runDistribName,distribParamsName,paramName,eqCondStr,funcName,runDistribName))

    return condLogical

##########################################################################################3
def checkParamsNum(funcName,distribName,runDistribName,distribParams,correctNum):
    """See if the correct number of parameters was supplied.  More than one number may apply

    Args:
        | funcName (string):  distribution name
        | distribName (string):  distribution name
        | distribParams ([float]): list of distribution parameters (see below)
        | correctNum ([int]): list with the possible numbers of parameters

    Returns:
        | True if the requirements are matched

    Raises:
        | No exception is raised.
    """

    proceed = True
    if not len(distribParams) in correctNum:
        print('{} Variates Generation:\n {} {} {} {} {}'.format(distribName, 
            'Wrong number of parameters (run ', funcName, "('", runDistribName, "') for help) "))
        proceed = False

    # print(distribParams, correctNum, proceed)

    return proceed


##########################################################################################3
def unifomOnSphere(number,dimension=3):
    """Create a set of n-dimensional vectors normally distributed on a sphere

    https://math.stackexchange.com/questions/444700/uniform-distribution-on-the-surface-of-unit-sphere
    https://stackoverflow.com/questions/59954810/generate-random-points-on-10-dimensional-unit-sphere


    Args:
        | number (int):  Number of points on sphere
        | dimension (int):  dimensionality of the normalised vector

    Returns:
        | vec (np.array): array of 3-D points 

    Raises:
        | No exception is raised.
    """

    vec = np.random.normal(0,1,(number,dimension))
    vec = vec/ np.linalg.norm(vec,axis=1).reshape(-1,1)
    return vec


################################################################
################################################################
##
## confirm the correctness of the functions

if __name__ == '__main__':

    import datetime as dt
    import ryutils
    rit = ryutils.intify_tuple

    doAll = False

    # numpoints = 10
    # vec = unifomOnSphere(numpoints)
    # import pyradi.ryplot as ryplot
    # p = ryplot.Plotter(1,1,1)
    # p.plot3d(1, vec[:,0],vec[:,1],vec[:,2], markers='.',linewidths=[0], linestyle='-')
    # p.saveFig('randomonunitsphere.png')

    #no tests at this time

