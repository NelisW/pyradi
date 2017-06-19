# -*- coding: utf-8 -*-


################################################################list[M],
# The contents of this file are subject to the BSD 3Clause (New) License
# you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://directory.fsf.org/wiki/License:BSD_3Clause

# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
# License for the specific language governing rights and limitations
# under the License.

# The Original Code is part of the PyRadi toolkit.

# The Initial Developer of the Original Code is CJ Willers,
# Portions created by CJ Willers are Copyright (C) 2006-2012
# All Rights Reserved.

# Contributor(s): ______________________________________.
################################################################
"""
This module provides functions for Planck law exitance calculations, as well as
temperature derivative calculations.
The functions provide spectral exitance in [W/(m^2.*)] or [q/(s.m^2.*)], given
the temperature and a vector of one of wavelength, wavenumbers or frequency
(six combinations each for exitance and temperature derivative). The total
exitance can also be calculated by using the Stefan-Boltzmann equation, in
[W/m^2] or [q/(s.m^2)].  'Exitance' is the CIE/ISO term for the older term 'emittance'.

The Planck and temperature-derivative Planck functions take the spectral variable
(wavelength, wavenumber or frequency) and/or temperature as either a scalar, a 
single element list, a multi-element list or a numpy array.

Spectral values must be strictly scalar or shape (N,) or (N,1).  
Shape (1,N) will not work.

Temperature values must be strictly scalar, list[M], shape (M,), (M,1), or (1,M).
Shape (Q,M) will not work.

If the spectral variable and temperature are both single numbers (scalars or lists
with one element), the return value is a scalar.  If either the temperature or the
spectral variable are single-valued, the return value is a rank-1 vector. If both
the temperature and spectral variable are multi-valued, the return value is a 
rank-2 array, with the spectral variable along axis=0.

This module uses the CODATA physical constants. For more details see
http://physics.nist.gov/cuu/pdf/RevModPhysCODATA2010.pdf

See the __main__ function for testing and examples of use.

This package was partly developed to provide additional material in support of students 
and readers of the book Electro-Optical System Analysis and Design: A Radiometry 
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""





__version__= "$Revision$"
__author__='pyradi team'
__all__=['planck','dplanck','stefanboltzman','planckef',  'planckel', 'plancken',
'planckqf', 'planckql', 'planckqn', 'dplnckef', 'dplnckel', 'dplncken', 'dplnckqf',
'dplnckql', 'dplnckqn','an','printConstants','planckInt']

import sys
import numpy as np
import scipy.constants as const
from functools import wraps

#np.exp() has upper limit in IEEE double range, catch this in Planck calcs
explimit = 709.7


import pyradi.ryutils as ryutils

class PlanckConstants:
    """Precalculate the Planck function constants using the values in
       scipy.constants.  Presumbly these constants are up to date and
       will be kept up to date.

       This module uses the CODATA physical constants. For more details see
       http://physics.nist.gov/cuu/pdf/RevModPhysCODATA2010.pdf
 
       
       
        Reference: http://docs.scipy.org/doc/scipy/reference/constants.html
    """

    def __init__(self):
        """ Precalculate the Planck function constants.

        Reference: http://www.spectralcalc.com/blackbody/appendixC.html
        """

        import scipy.optimize

        self.c1em = 2 * np.pi * const.h * const.c * const.c
        self.c1el = self.c1em * (1.0e6)**(5-1) # 5 for lambda power and -1 for density
        self.c1en = self.c1em * (100)**3 * 100 # 3 for wavenumber, 1 for density
        self.c1ef = 2 * np.pi * const.h / (const.c * const.c)

        self.c1qm = 2 * np.pi * const.c
        self.c1ql = self.c1qm * (1.0e6)**(4-1) # 5 for lambda power and -1 for density
        self.c1qn = self.c1qm * (100)**2 * 100 # 2 for wavenumber, 1 for density
        self.c1nf = 2 * np.pi  / (const.c * const.c)

        self.c2m = const.h * const.c / const.k
        self.c2l = self.c2m * 1.0e6 # 1 for wavelength density
        self.c2n = self.c2m * 1.0e2 # 1 for cm-1 density
        self.c2f = const.h / const.k

        self.sigmae = const.sigma
        self.zeta3 = 1.2020569031595942853
        self.sigmaq = 4 * np.pi * self.zeta3 * const.k ** 3 \
               / (const.h ** 3 * const.c ** 2)

        self.a2 = scipy.optimize.brentq(self.an, 1, 2, (2) )
        self.a3 = scipy.optimize.brentq(self.an, 2, 3, (3) )
        self.a4 = scipy.optimize.brentq(self.an, 3.5, 4, (4) )
        self.a5 = scipy.optimize.brentq(self.an, 4.5, 5, (5) )

        self.wel = 1e6 * const.h * const.c /(const.k * self.a5)
        self.wql = 1e6 * const.h * const.c /(const.k * self.a4)
        self.wen = self.a3 * const.k /(100 * const.h * const.c )
        self.wqn = self.a2 * const.k /(100 * const.h * const.c )
        self.wef = self.a3 * const.k /(const.h )
        self.wqf = self.a2 * const.k /(const.h )


    def an(self,x,n):
        return n * (1-np.exp(-x)) - x

    def printConstants(self):
        """Print Planck function constants.

        Args:
            | None

        Returns:
            | Print to stdout

        Raises:
            | No exception is raised.
        """
        print('h = {:.14e} Js'.format(const.h))
        print('c = {:.14e} m/s'.format(const.c))
        print('k = {:.14e} J/K'.format(const.k))
        print('q = {:.14e} C'.format(const.e))

        print(' ')
        print('pi = {:.14e}'.format(const.pi))
        print('e = {:.14e}'.format(np.exp(1)))
        print('zeta(3) = {:.14e}'.format(self.zeta3 ))
        print('a2 = {:.14e}, root of 2(1-exp(-x))-x'.format(self.a2 ))
        print('a3 = {:.14e}, root of 3(1-exp(-x))-x'.format(self.a3 ))
        print('a4 = {:.14e}, root of 4(1-exp(-x))-x'.format(self.a4 ))
        print('a5 = {:.14e}, root of 5(1-exp(-x))-x'.format(self.a5 ))

        print(' ')
        print('sigmae = {:.14e} W/(m^2 K^4)'.format(self.sigmae))
        print('sigmaq = {:.14e} q/(s m^2 K^3)'.format(self.sigmaq))
        print(' ')
        print('c1em = {:.14e} with wavelenth in m'.format(self.c1em))
        print('c1qm = {:.14e} with wavelenth in m'.format(self.c1qm))
        print('c2m = {:.14e} with wavelenth in m'.format(self.c2m))
        print(' ')
        print('c1el = {:.14e} with wavelenth in $\mu$m'.format(self.c1el))
        print('c1ql = {:.14e} with wavelenth in $\mu$m'.format(self.c1ql))
        print('c2l = {:.14e} with wavelenth in $\mu$m'.format(self.c2l))
        print(' ')
        print('c1en = {:.14e} with wavenumber in cm$^{{-1}}$'.format(self.c1en))
        print('c1qn = {:.14e} with wavenumber in cm$^{{-1}}$'.format(self.c1qn))
        print('c2n = {:.14e} with wavenumber in cm$^{{-1}}$'.format(self.c2n))
        print(' ')
        print('c1ef = {:.14e} with frequency in Hz'.format(self.c1ef))
        print('c1nf = {:.14e} with frequency in Hz'.format(self.c1nf))
        print('c2f = {:.14e} with frequency in Hz'.format(self.c2f))
        print(' ')
        print('wel = {:.14e} um.K  Wien for radiant and wavelength'.format(self.wel))
        print('wql = {:.14e} um.K  Wien for photon rate and wavelength'.format(self.wql))
        print('wen = {:.14e} cm-1/K  Wien for radiant and wavenumber'.format(self.wen))
        print('wqn = {:.14e} cm-1/K  Wien for photon rate and wavenumber'.format(self.wqn))
        print('wef = {:.14e} Hz/K  Wien for radiant and frequency'.format(self.wef))
        print('wqf = {:.14e} Hz/K  Wien for photon rate and frequency'.format(self.wqf))
        print(' ')

pconst = PlanckConstants()


################################################################
##
def fixDimensions(planckFun):
  """Decorator function to prepare the spectral and temperature array 
  dimensions and order before and after the actual Planck function.
  The Planck functions process elementwise and therefore require 
  flattened arrays.  This decorator flattens, executes the planck function
  and reshape afterwards the correct shape, according to input.
  """
  @wraps(planckFun)
  def inner(spectral, temperature):

    #confirm that only vector is used, break with warning if so.
    if isinstance(temperature, np.ndarray):
        if len(temperature.flat) != max(temperature.shape):
            print('ryplanck: temperature must be of shape (M,), (M,1) or (1,M)')
            return None
    #confirm that no row vector is used, break with warning if so.
    if isinstance(spectral, np.ndarray):
        if len(spectral.flat) != spectral.shape[0]:
            print('ryplanck: spectral must be of shape (N,) or (N,1)')
            return None
    tempIn = np.array(temperature, copy=True,  ndmin=1).astype(float)
    specIn = np.array(spectral, copy=True,  ndmin=1).astype(float)
    tempIn = tempIn.reshape(-1,1)
    specIn = specIn.reshape(-1,1)

    #create flattened version of the input dataset
    specgrid, tempgrid = np.meshgrid(specIn,tempIn)
    spec = np.ravel(specgrid)
    temp = np.ravel(tempgrid)

    #test for zero temperature
    temp = np.where(temp!=0.0, temp, 1e-300);

    #this is the actual planck calculation
    planckA = planckFun(spec,temp) 

    #now unflatten to proper structure again, spectral along axis=0
    if tempIn.shape[0] == 1 and specIn.shape[0] == 1:
        rtnVal = planckA[0]
    elif tempIn.shape[0] == 1 and specIn.shape[0] != 1:
        rtnVal = planckA.reshape(specIn.shape[0],)
    elif tempIn.shape[0] != 1 and specIn.shape[0] == 1:
        rtnVal = planckA.reshape(specIn.shape[0],-1)
    else:
        rtnVal = planckA.reshape(tempIn.shape[0],-1).T

    return rtnVal
  return inner


################################################################
##
@fixDimensions
def planckel(spectral, temperature):
    """ Planck function in wavelength for radiant exitance.

    Args:
        | spectral (scalar, np.array (N,) or (N,1)):  wavelength vector in  [um]
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]

    Returns:
        | (scalar, np.array[N,M]):  spectral radiant exitance in W/(m^2.um)

    Raises:
        | No exception is raised, returns None on error.
    """

    # planckA = pconst.c1el / (spec ** 5 * ( np.exp(pconst.c2l / (spec * temp))-1));

    #test value of exponent to prevent infinity, force to exponent to zero
    #this happens for low temperatures and short wavelengths
    exP =  pconst.c2l / (spectral * temperature)
    exP2 = np.where(exP<explimit, exP, 1);
    p = (pconst.c1el / ( np.exp(exP2)-1)) / (spectral ** 5)
    #if exponent is exP>=explimit, force Planck to zero
    planckA = np.where(exP<explimit, p, 0);

    return planckA


################################################################
##
@fixDimensions
def planckef(spectral, temperature):
    """ Planck function in frequency for radiant exitance.

    Args:
        | spectral (scalar, np.array (N,) or (N,1)):  frequency vector in  [Hz]
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]

    Returns:
        | (scalar, np.array[N,M]): spectral radiant exitance in W/(m^2.Hz)

    Raises:
        | No exception is raised, returns None on error.
    """

    # planckA = pconst.c1ef * spec**3 / (np.exp(pconst.c2f * spec / temp)-1);

    #test value of exponent to prevent infinity, force to exponent to zero
    #this happens for low temperatures and short wavelengths
    exP =  pconst.c2f * spectral / temperature
    exP2 = np.where(exP<explimit, exP, 1);
    p = pconst.c1ef * spectral**3 / (np.exp(exP2)-1);
    #if exponent is exP>=explimit, force Planck to zero
    planckA = np.where(exP<explimit, p, 0);

    return planckA



################################################################
##
@fixDimensions
def plancken(spectral, temperature):
    """ Planck function in wavenumber for radiant exitance.

    Args:
        | spectral (scalar, np.array (N,) or (N,1)):  wavenumber vector in   [cm^-1]
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]

    Returns:
        | (scalar, np.array[N,M]):  spectral radiant exitance in  W/(m^2.cm^-1)

    Raises:
        | No exception is raised, returns None on error.
    """

    # planckA = pconst.c1en * spec**3 / (np.exp(pconst.c2n * (spec / temp))-1)

    #test value of exponent to prevent infinity, force to exponent to zero
    #this happens for low temperatures and short wavelengths
    exP =  pconst.c2n * (spectral / temperature)
    exP2 = np.where(exP<explimit, exP, 1);
    p = ( pconst.c1en  / (np.exp(exP)-1) ) * spectral**3
    #if exponent is exP>=explimit, force Planck to zero
    planckA = np.where(exP<explimit, p, 0);

    return planckA


################################################################
##
@fixDimensions
def planckqf(spectral, temperature):
    """ Planck function in frequency domain for photon rate exitance.

    Args:
        | spectral (scalar, np.array (N,) or (N,1)): frequency vector in  [Hz]
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]

    Returns:
        | (scalar, np.array[N,M]):  spectral radiant exitance in q/(s.m^2.Hz)

    Raises:
        | No exception is raised, returns None on error.
    """

    #test value of exponent to prevent infinity, force to exponent to zero
    #this happens for low temperatures and short wavelengths
    exP =  pconst.c2f * spectral / temperature
    exP2 = np.where(exP<explimit, exP, 1);
    p = pconst.c1nf * spectral**2 / (np.exp(exP2)-1)
    #if exponent is exP>=explimit, force Planck to zero
    planckA = np.where(exP<explimit, p, 0);

    return planckA


################################################################
##
@fixDimensions
def planckql(spectral, temperature):
    """ Planck function in wavelength domain for photon rate exitance.

    Args:
        | spectral (scalar, np.array (N,) or (N,1)):  wavelength vector in  [um]
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]

    Returns:
        | (scalar, np.array[N,M]):  spectral radiant exitance in  q/(s.m^2.um)

    Raises:
        | No exception is raised, returns None on error.
    """
    #test value of exponent to prevent infinity, force to exponent to zero
    #this happens for low temperatures and short wavelengths
    exP = pconst.c2l / (spectral * temperature)
    exP2 = np.where(exP<explimit, exP, 1);
    # print(np.max(exP), np.max(exP2))
    p = (pconst.c1ql /( np.exp(exP2)-1) )  / (spectral**4 )
    #if exponent is exP>=explimit, force Planck to zero
    planckA = np.where(exP<explimit, p, 0);

    return planckA


################################################################
##
@fixDimensions
def planckqn(spectral, temperature):
    """ Planck function in wavenumber domain for photon rate exitance.

    Args:
        | spectral (scalar, np.array (N,) or (N,1)):  wavenumber vector in   [cm^-1]
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]

    Returns:
        | (scalar, np.array[N,M]):  spectral radiant exitance in  q/(s.m^2.cm^-1)

    Raises:
        | No exception is raised, returns None on error.
    """

    #test value of exponent to prevent infinity, force to exponent to zero
    #this happens for low temperatures and short wavelengths
    exP =  pconst.c2n * spectral / temperature
    exP2 = np.where(exP<explimit, exP, 1);
    p = pconst.c1qn * spectral**2 / (np.exp(exP2)-1);
    #if exponent is exP>=explimit, force Planck to zero
    planckA = np.where(exP<explimit, p, 0);

    return planckA


################################################################
##
@fixDimensions
def dplnckef(spectral, temperature):
    """Temperature derivative of Planck function in frequency domain
    for radiant exitance.

    Args:
        | spectral (scalar, np.array (N,) or (N,1)): frequency vector in  [Hz]
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]

    Returns:
        | (scalar, np.array[N,M]):  spectral radiant exitance/K in W/(K.m^2.Hz)

    Raises:
        | No exception is raised, returns None on error.
    """

    xx=(pconst.c2f * spectral /temperature);
    f=xx*np.exp(xx)/(temperature*(np.exp(xx)-1))
    y=pconst.c1ef * spectral**3 / (np.exp(pconst.c2f * spectral \
            / temperature)-1);
    dplanckA = f*y;

    return dplanckA


################################################################
##
@fixDimensions
def dplnckel(spectral, temperature):
    """Temperature derivative of Planck function in wavelength domain for
    radiant exitance.

    Args:
        | spectral (scalar, np.array (N,) or (N,1)):  wavelength vector in  [um]
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]

    Returns:
        | (scalar, np.array[N,M]):  spectral radiant exitance in W/(K.m^2.um)

    Raises:
        | No exception is raised, returns None on error.
    """

    # if xx > 350, then we get overflow
    xx = pconst.c2l /(spectral * temperature)
    # return (3.7418301e8 * xx * np.exp(xx) ) \
    #     / (temperature* spectral ** 5 * (np.exp(xx)-1) **2 )
    # refactor (np.exp(xx)-1)**2 to prevent overflow problem
    dplanckA = (pconst.c1el * xx * np.exp(xx) / (np.exp(xx)-1) ) \
        / (temperature* spectral ** 5 * (np.exp(xx)-1) )

    return dplanckA


################################################################
##
@fixDimensions
def dplncken(spectral, temperature):
    """Temperature derivative of Planck function in wavenumber domain for radiance exitance.

    Args:
        | spectral (scalar, np.array (N,) or (N,1)):  wavenumber vector in   [cm^-1]
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]

    Returns:
        | (scalar, np.array[N,M]):  spectral radiant exitance in  W/(K.m^2.cm^-1)

    Raises:
        | No exception is raised, returns None on error.
    """

    xx=(pconst.c2n * spectral /temperature)
    f=xx*np.exp(xx)/(temperature*(np.exp(xx)-1))
    y=(pconst.c1en* spectral **3 / (np.exp(pconst.c2n * spectral \
            / temperature)-1))
    dplanckA = f*y;

    return dplanckA


################################################################
##
@fixDimensions
def dplnckqf(spectral, temperature):
    """Temperature derivative of Planck function in frequency domain for photon rate.

    Args:
        | spectral (scalar, np.array (N,) or (N,1)): frequency vector in  [Hz]
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]

    Returns:
        | (scalar, np.array[N,M]):  spectral radiant exitance in q/(K.s.m^2.Hz)

    Raises:
        | No exception is raised, returns None on error.
    """

    xx=(pconst.c2f * spectral /temperature)
    f=xx*np.exp(xx)/(temperature*(np.exp(xx)-1))
    y=pconst.c1nf * spectral **2 / (np.exp(pconst.c2f * spectral \
            / temperature)-1)
    dplanckA = f*y;

    return dplanckA


################################################################
##
@fixDimensions
def dplnckql(spectral, temperature):
    """Temperature derivative of Planck function in wavenumber domain for radiance exitance.

    Args:
        | spectral (scalar, np.array (N,) or (N,1)):  wavelength vector in  [um]
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]

    Returns:
        | (scalar, np.array[N,M]):  spectral radiant exitance in  q/(K.s.m^2.um)

    Raises:
        | No exception is raised, returns None on error.
    """

    xx=(pconst.c2l /(spectral * temperature))
    f=xx*np.exp(xx)/(temperature*(np.exp(xx)-1))
    y=pconst.c1ql / (spectral ** 4 * ( np.exp(pconst.c2l \
            / (temperature * spectral))-1))
    dplanckA = f*y;

    return dplanckA


################################################################
##
@fixDimensions
def dplnckqn(spectral, temperature):
    """Temperature derivative of Planck function in wavenumber domain for photon rate.

    Args:
        | spectral (scalar, np.array (N,) or (N,1)):  wavenumber vector in   [cm^-1]
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]

    Returns:
        | (scalar, np.array[N,M]):  spectral radiant exitance in  q/(s.m^2.cm^-1)

    Raises:
        | No exception is raised, returns None on error.
    """

    xx=(pconst.c2n * spectral /temperature)
    f=xx*np.exp(xx)/(temperature*(np.exp(xx)-1))
    y=pconst.c1qn * spectral **2 / (np.exp(pconst.c2n * spectral \
            / temperature)-1)
    dplanckA = f*y;

    return dplanckA


################################################################
##
def stefanboltzman(temperature, type='e'):
    """Stefan-Boltzman wideband integrated exitance.

    Calculates the total Planck law exitance, integrated over all wavelengths,
    from a surface at the stated temperature. Exitance can be given in radiant or
    photon rate units, depending on user input in type.

    Args:
        | (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]
        | type (string):  'e' for radiant or 'q' for photon rate exitance.

    Returns:
        | (float): integrated radiant exitance in  [W/m^2] or [q/(s.m^2)].
        | Returns a -1 if the type is not 'e' or 'q'

    Raises:
        | No exception is raised.
    """

    #confirm that only vector is used, break with warning if so.
    if isinstance(temperature, np.ndarray):
        if len(temperature.flat) != max(temperature.shape):
            print('ryplanck.stefanboltzman: temperature must be of shape (M,), (M,1) or (1,M)')
            return -1

    tempr = np.asarray(temperature).astype(float)
    #use dictionary to switch between options, lambda fn to calculate, default -1
    rtnval = {
              'e': lambda temp: pconst.sigmae * np.power(temp, 4) ,
              'q': lambda temp: pconst.sigmaq * np.power(temp, 3)
              }.get(type, lambda temp: -1)(tempr)
    return rtnval



################################################################
##
# dictionaries to allow case-like statements in generic planck functions, below.
plancktype = {  'el' : planckel, 'ef' : planckef, 'en' : plancken, \
                      'ql' : planckql, 'qf' : planckqf, 'qn' : planckqn}
dplancktype = {'el' : dplnckel, 'ef' : dplnckef, 'en' : dplncken, \
                      'ql' : dplnckql, 'qf' : dplnckqf, 'qn' : dplnckqn}

################################################################
##
def planck(spectral, temperature, type='el'):
    """Planck law spectral exitance.

    Calculates the Planck law spectral exitance from a surface at the stated 
    temperature. Temperature can be a scalar, a list or an array. Exitance can 
    be given in radiant or photon rate units, depending on user input in type.

    Args:
        | spectral (scalar, np.array (N,) or (N,1)):  spectral vector.
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]
        | type (string):
        |  'e' signifies Radiant values in [W/m^2.*].
        |  'q' signifies photon rate values  [quanta/(s.m^2.*)].
        |  'l' signifies wavelength spectral vector  [micrometer].
        |  'n' signifies wavenumber spectral vector [cm-1].
        |  'f' signifies frequency spectral vecor [Hz].

    Returns:
        | (scalar, np.array[N,M]):  spectral radiant exitance (not radiance) in units selected.
        | For type = 'el' units will be [W/(m^2.um)].
        | For type = 'qf' units will be [q/(s.m^2.Hz)].
        | Other return types are similarly defined as above.
        | Returns None on error.

    Raises:
        | No exception is raised, returns None on error.
    """
    if type in list(plancktype.keys()):
        #select the appropriate fn as requested by user
        exitance = plancktype[type](spectral, temperature)
    else:
        # return all minus one if illegal type
        exitance = None

    return exitance


################################################################
##
def dplanck(spectral, temperature, type='el'):
    """Temperature derivative of Planck law exitance.

    Calculates the temperature derivative for Planck law spectral exitance
    from a surface at the stated temperature. dM/dT can be given in radiant or
    photon rate units, depending on user input in type. Temperature can be a 
    scalar, a list or an array. 

    Args:
        | spectral (scalar, np.array (N,) or (N,1)):  spectral vector in  [micrometer], [cm-1] or [Hz].
        | temperature (scalar, list[M], np.array (M,), (M,1) or (1,M)):  Temperature in [K]
        | type (string):
        |  'e' signifies Radiant values in [W/(m^2.K)].
        |  'q' signifies photon rate values  [quanta/(s.m^2.K)].
        |  'l' signifies wavelength spectral vector  [micrometer].
        |  'n' signifies wavenumber spectral vector [cm-1].
        |  'f' signifies frequency spectral vecor [Hz].

    Returns:
        | (scalar, np.array[N,M]):  spectral radiant exitance (not radiance) in units selected.
        | For type = 'el' units will be [W/(m2.um.K)]
        | For type = 'qf' units will be [q/(s.m2.Hz.K)]
        | Other return types are similarly defined as above.
        | Returns None on error.

    Raises:
        | No exception is raised, returns None on error.
    """

    if type in list(dplancktype.keys()):
        #select the appropriate fn as requested by user
        exitance = dplancktype[type](spectral, temperature)
    else:
        # return all zeros if illegal type
        exitance = - np.ones(spectral.shape)

    return exitance


################################################################
##
def planck_integral(wavenum, temperature, radiant,niter=512):
    """Integrated Planck law spectral exitance by summation from wavenum to infty.

    http://www.spectralcalc.com/blackbody/inband_radiance.html
    http://www.spectralcalc.com/blackbody/appendixA.html
    Integral of spectral radiance from wavenum (cm-1) to infinity. 
    follows Widger and Woodall, Bulletin of Am Met Soc, Vol. 57, No. 10, pp. 1217

    Args:
        | wavenum (scalar):  spectral limit.
        | temperature (scalar):  Temperature in [K]
        | radiant (bool): output units required, W/m2 or q/(s.m2)
 
    Returns:
        | (scalar):  exitance (not radiance) in units selected.
        | For radiant units will be [W/(m^2)].
        | For not radiant units will be [q/(s.m^2)].

    Raises:
        | No exception is raised, returns None on error.
    """
    # compute powers of x, the dimensionless spectral coordinate
    c1 = const.h * const.c / const.k
    x = c1 * 100. * wavenum / temperature 
    x2 = x * x  
    x3 = x * x2 

    # decide how many terms of sum are needed
    iter = int(2.0 + 20.0 / x )
    iter = iter if iter<niter else niter

    # add up terms of sum 
    sum = 0. 
    for n in range(1,iter):
        dn = 1.0 / n 
        if radiant:
            sum += np.exp(-n * x) * (x3 + (3.0 * x2 + 6.0 * (x + dn) * dn) * dn) * dn
        else:
            sum += np.exp(-n * x) * (x2 + 2.0 * (x + dn) * dn) * dn 

    if radiant:
        # in units of W/m2
        c2 = 2.0 * const.h * const.c * const.c
        rtnval = c2 * (temperature/c1)**4. * sum  
    else:
        # in units of photons/(s.m2)
        kTohc =  const.k * temperature / (const.h * const.c) 
        rtnval =  2.0 * const.c * kTohc**3.  * sum 

    # print('wn={} x={} T={} E={} n={}'.format(wavenum,x,temperature, rtnval/np.pi,iter))

    return rtnval * np.pi

################################################################
##
def planckInt(spectralLo, spectralHi, temperature, type='el'):
    """Integrated Planck law spectral exitance.

    Calculates the integral of the Planck law spectral exitance from a surface 
    at the stated temperature over the specified spectral range. 
    Temperature can be a scalar, a list or an array. 
    Exitance can be returned in radiant or photon rate units, 
    depending on user input in type.

    http://www.spectralcalc.com/blackbody/inband_radiance.html
    http://www.spectralcalc.com/blackbody/appendixA.html

    Args:
        | spectralLo (scalar):  spectral lower limit.
        | spectralHi (scalar):  spectral upper limit.
        | temperature (scalar):  Temperature in [K]
        | type (string):
        |  'e' signifies Radiant values in [W/m^2.*].
        |  'q' signifies photon rate values  [quanta/(s.m^2.*)].
        |  'l' signifies wavelength spectral vector  [micrometer].
        |  'n' signifies wavenumber spectral vector [cm-1].
        |  'f' signifies frequency spectral vecor [Hz].

    Returns:
        | (scalar):  spectral radiant exitance (not radiance) in units selected.
        | For type = 'e*' units will be [W/(m^2)].
        | For type = 'q*' units will be [q/(s.m^2)].
        | Returns None on error.

    Raises:
        | No exception is raised, returns None on error.
    """
    if type not in list(plancktype.keys()) or temperature==0.0:
        exitance = None
    else:
        if type[1]=='f':
            sLo = ryutils.convertSpectralDomain(spectralLo, type='fn') 
            sHi = ryutils.convertSpectralDomain(spectralHi, type='fn')  
        elif type[1]=='l':
            sLo = ryutils.convertSpectralDomain(spectralLo, type='ln') 
            sHi = ryutils.convertSpectralDomain(spectralHi, type='ln')  
        else:
            sLo = spectralLo
            sHi = spectralHi

        sLo = 1e100 if sLo>1e100 else sLo
        sHi = 1e100 if sHi>1e100 else sHi

        if sLo > sHi:
            sLo, sHi = sHi, sLo

        if sLo==0.:
            mLo = stefanboltzman(temperature, type=type[0])
        else:
            mLo = planck_integral(sLo, temperature, type[0]=='e')

        if sHi==0.:
            mHi = stefanboltzman(temperature, type=type[0])
        else:
            mHi = planck_integral(sHi, temperature, type[0]=='e')

        exitance = mLo - mHi

    return exitance



################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':
    
    import ryutils
    doAll = True

    rit = ryutils.intify_tuple

    if doAll:
        #--------------------------------------------------------------------------------------
        # print all available constants
        pconst.printConstants()

    if doAll:
        #--------------------------------------------------------------------------------------
        # test different input types for temperature but with spectral array
        wavelen = np.linspace(1.0, 2.0, 100)
        #test for scalar temperature
        m = planckel(wavelen, 300) 
        print('Array spectral {} & scalar temperature input, output shape is {}'.format(
            rit(wavelen.shape), rit(m.shape)))
        #test for list of temperature values
        temp =  [300]
        m = planckel(wavelen,temp) 
        print('Array spectral {} &  list with len()={} temperature input, output shape is {}'.format(
            rit(wavelen.shape), len(temp), rit(m.shape)))
        #test for list of temperature values
        temp =  [300, 350, 500]
        m = planckel(wavelen,temp) 
        print('Array spectral {} & list with len()={} temperature input, output shape is {}'.format(
            rit(wavelen.shape), len(temp), rit(m.shape)))
        #test for array of temperature values
        temp =  np.asarray([300, 350, 400, 450, 500])
        m = planckel(wavelen,temp) 
        print('Array spectral {} & array with shape={} temperature input, output shape is {}'.format(
            rit(wavelen.shape), rit(temp.shape), rit(m.shape)))
        #test for array of temperature values
        temp =  np.asarray([300, 350, 400, 450, 500]).reshape(1,-1)
        m = planckel(wavelen,temp.T) 
        print('Array spectral {} & array with shape={} temperature input, output shape is {}'.format(
            rit(wavelen.shape), rit(temp.shape), rit(m.shape)))

        # test different input types for temperature but with spectral as scalar
        #test for scalar temperature
        m = planckel(wavelen[0], 300) 
        print('Scalar spectral & scalar temperature input, output shape is {}'.format(rit(m.shape)))
        #test for list of temperature values
        temp =  [300]
        m = planckel(wavelen[0],temp) 
        print('Scalar spectral & list temperature with len()={} input, output shape is {}'.format(len(temp), rit(m.shape)))
        #test for list of temperature values
        temp =  [300, 350, 500]
        m = planckel(wavelen[0],temp) 
        print('Scalar spectral & list temperature with len()={} input, output shape is {}'.format(len(temp), rit(m.shape)))
        #test for list of temperature values
        temp =  np.asarray([300, 350, 400, 450, 500])
        m = planckel(wavelen[0],temp) 
        print('Scalar spectral & array temperature with shape={}  input, output shape is {}'.format(rit(temp.shape), rit(m.shape)))

        print(' ')

    if doAll:
        #--------------------------------------------------------------------------------------
        # test at fixed temperature
        # for each function we calculate a spectral exitance and then get the peak and where peak occurs.'
        # this is checked against a first principles equation calculation.

        tmprtr=1000  # arbitrary choice of temperature
        dTmprtr=0.01  # arbitrary choice of temperature

        print('Temperature for calculations             {0:f} [K]'.format(tmprtr))
        print('dTemperature for dM/dTcalculations       {0:f} [K]'.format(dTmprtr))

        numIntPts=10000  #number of integration points

        wl1=.05            # lower integration limit
        wl2= 1000         # upper integration limit
        wld=(wl2-wl1)/numIntPts  #integration increment
        wl=np.arange(wl1, wl2+wld, wld)

        print('\nplanckel WAVELENGTH DOMAIN, RADIANT EXITANCE')
        M =planckel(wl,tmprtr)
        peakM =  1.28665e-11*tmprtr**5
        spectralPeakM = 2897.9/tmprtr
        spectralPeak = wl[M.argmax()]
        I=np.trapz(M, wl)
        psum = planckInt(wl1, wl2, tmprtr, type='el')
        sblf = stefanboltzman(tmprtr, 'e')
        sbl=5.67033e-8*tmprtr**4
        dMe = ( planckel(spectralPeak, tmprtr+dTmprtr)  - planckel(spectralPeak,tmprtr))/dTmprtr
        dMf = dplnckel(spectralPeak,tmprtr)
        print('                            function       equations    (% error)')
        print('peak exitance             {0:e}   {1:e}   {2:+.4f}   [W/(m^2.um)]'.format(max(M),peakM, 100 * (max(M)-peakM)/peakM))
        print('peak exitance at          {0:e}   {1:e}   {2:+.4f}   [um]'.format(spectralPeak,spectralPeakM, 100 * (spectralPeak-spectralPeakM)/spectralPeakM))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(I, sbl, 100 * (I-sbl)/sbl))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(sblf, sbl, 100 * (sblf-sbl)/sbl))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(psum, sbl, 100 * (psum-sbl)/sbl))
        print('radiant exitance dM/dT    {0:e}   {1:e}   {2:+.4f}   [W/(m^2.um.K)]'.format(dMf,  dMe, 100 * (dMf-dMe)/dMe))


        print('\nplanckql WAVELENGTH DOMAIN, PHOTON EXITANCE');
        M =planckql(wl,tmprtr);
        peakM =  2.1010732e11*tmprtr*tmprtr*tmprtr*tmprtr
        spectralPeakM = 3669.7/tmprtr
        spectralPeak = wl[M.argmax()]
        I=np.trapz(M, wl)
        psum = planckInt(wl1, wl2, tmprtr, type='ql')
        sblf = stefanboltzman(tmprtr, 'q')
        sbl=1.5204e+15*tmprtr*tmprtr*tmprtr
        dMe = ( planckql(spectralPeak,tmprtr+dTmprtr)  - planckql(spectralPeak,tmprtr))/dTmprtr
        dMf = dplnckql(spectralPeak,tmprtr)
        print('                            function       equations    (% error)')
        print('peak exitance             {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2.um)]'.format(max(M),peakM, 100 * (max(M)-peakM)/peakM))
        print('peak exitance at          {0:e}   {1:e}   {2:+.4f}   [um]'.format(spectralPeak,spectralPeakM, 100 * (spectralPeak-spectralPeakM)/spectralPeakM))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2)]'.format(I, sbl, 100 * (I-sbl)/sbl))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2)]'.format(sblf, sbl, 100 * (sblf-sbl)/sbl))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(psum, sbl, 100 * (psum-sbl)/sbl))
        print('radiant exitance dM/dT    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2.um.K)]'.format(dMf,  dMe, 100 * (dMf-dMe)/dMe))


        f1=const.c/ (wl2*1e-6)
        f2=const.c/ (wl1*1e-6)
        fd=(f2-f1)/numIntPts  # integration increment
        f=np.arange(f1, f2+fd, fd)

        print('\nplanckef FREQUENCY DOMAIN, RADIANT EXITANCE');
        M =planckef(f,tmprtr);
        peakM =  5.95664593e-19*tmprtr*tmprtr*tmprtr
        spectralPeakM = 5.8788872e10*tmprtr
        spectralPeak = f[M.argmax()]
        I=np.trapz(M, f)
        psum = planckInt(f1, f2, tmprtr, type='ef')
        sblf = stefanboltzman(tmprtr, 'e')
        sbl=5.67033e-8*tmprtr*tmprtr*tmprtr*tmprtr
        dMe = ( planckef(spectralPeak,tmprtr+dTmprtr)  - planckef(spectralPeak,tmprtr))/dTmprtr
        dMf = dplnckef(spectralPeak,tmprtr)
        print('                            function       equations    (% error)')
        print('peak exitance             {0:e}   {1:e}   {2:+.4f}   [W/(m^2.Hz)]'.format(max(M),peakM, 100 * (max(M)-peakM)/peakM))
        print('peak exitance at          {0:e}   {1:e}   {2:+.4f}   [Hz]'.format(spectralPeak,spectralPeakM, 100 * (spectralPeak-spectralPeakM)/spectralPeakM))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(I, sbl, 100 * (I-sbl)/sbl))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(sblf, sbl, 100 * (sblf-sbl)/sbl))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(psum, sbl, 100 * (psum-sbl)/sbl))
        print('radiant exitance dM/dT    {0:e}   {1:e}   {2:+.4f}   [W/(m^2.Hz.K)]'.format(dMf,  dMe, 100 * (dMf-dMe)/dMe))


        print('\nplanckqf FREQUENCY DOMAIN, PHOTON EXITANCE');
        M =planckqf(f,tmprtr);
        peakM = 1.965658e4*tmprtr*tmprtr
        spectralPeakM = 3.32055239e10*tmprtr
        spectralPeak = f[M.argmax()]
        I=np.trapz(M, f)
        psum = planckInt(f1, f2, tmprtr, type='qf')
        sblf = stefanboltzman(tmprtr, 'q')
        sbl=1.5204e+15*tmprtr*tmprtr*tmprtr
        dMe = ( planckqf(spectralPeak,tmprtr+dTmprtr)  - planckqf(spectralPeak,tmprtr))/dTmprtr
        dMf = dplnckqf(spectralPeak,tmprtr)
        print('                            function       equations    (% error)')
        print('peak exitance             {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2.Hz)]'.format(max(M),peakM, 100 * (max(M)-peakM)/peakM))
        print('peak exitance at          {0:e}   {1:e}   {2:+.4f}   [Hz]'.format(spectralPeak,spectralPeakM,100 * (spectralPeak-spectralPeakM)/spectralPeakM))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2)]'.format(I, sbl, 100 * (I-sbl)/sbl))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2)]'.format(sblf, sbl, 100 * (sblf-sbl)/sbl))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(psum, sbl, 100 * (psum-sbl)/sbl))
        print('radiant exitance dM/dT    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2.Hz.K)]'.format(dMf,  dMe, 100 * (dMf-dMe)/dMe))


        n1=1e4 / wl2
        n2=1e4 / wl1
        nd=(n2-n1)/numIntPts  # integration increment
        n=np.arange(n1, n2+nd, nd)

        print('\nplancken WAVENUMBER DOMAIN, RADIANT EXITANCE');
        M =plancken(n,tmprtr);
        peakM =  1.78575759e-8*tmprtr*tmprtr*tmprtr
        spectralPeakM = 1.9609857086*tmprtr
        spectralPeak = n[M.argmax()]
        I=np.trapz(M, n)
        psum = planckInt(n1, n2, tmprtr, type='en')
        sblf = stefanboltzman(tmprtr, 'e')
        sbl=5.67033e-8*tmprtr*tmprtr*tmprtr*tmprtr
        dMe = ( plancken(spectralPeak,tmprtr+dTmprtr)  - plancken(spectralPeak,tmprtr))/dTmprtr
        dMf = dplncken(spectralPeak,tmprtr)
        print('                            function       equations    (% error)')
        print('peak exitance             {0:e}   {1:e}   {2:+.4f}   [W/(m^2.cm-1)]'.format(max(M),peakM, 100 * (max(M)-peakM)/peakM))
        print('peak exitance at          {0:e}   {1:e}   {2:+.4f}   [cm-1]'.format(spectralPeak,spectralPeakM,100 * (spectralPeak-spectralPeakM)/spectralPeakM))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(I, sbl, 100 * (I-sbl)/sbl))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(sblf, sbl, 100 * (sblf-sbl)/sbl))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(psum, sbl, 100 * (psum-sbl)/sbl))
        print('radiant exitance dM/dT    {0:e}   {1:e}   {2:+.4f}   [W/(m^2.cm-1.K)]'.format(dMf,  dMe, 100 * (dMf-dMe)/dMe))


        print('\nplanckqn WAVENUMBER DOMAIN, PHOTON EXITANCE');
        M =planckqn(n,tmprtr);
        peakM = 5.892639e14*tmprtr*tmprtr
        spectralPeakM = 1.1076170659*tmprtr
        spectralPeak =  n[M.argmax()]
        I=np.trapz(M, n)
        psum = planckInt(n1, n2, tmprtr, type='qn')
        sblf = stefanboltzman(tmprtr, 'q')
        sbl=1.5204e+15*tmprtr*tmprtr*tmprtr
        dMe = ( planckqn(spectralPeak,tmprtr+dTmprtr)  - planckqn(spectralPeak,tmprtr))/dTmprtr
        dMf = dplnckqn(spectralPeak,tmprtr)
        print('                            function       equations    (% error)')
        print('peak exitance             {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2.cm-1)]'.format(max(M),peakM, 100 * (max(M)-peakM)/peakM))
        print('peak exitance at          {0:e}   {1:e}   {2:+.4f}   [cm-1]'.format(spectralPeak,spectralPeakM, 100 * (spectralPeak-spectralPeakM)/spectralPeakM))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2)]'.format(I, sbl, 100 * (I-sbl)/sbl))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2)]'.format(sblf, sbl, 100 * (sblf-sbl)/sbl))
        print('radiant exitance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(psum, sbl, 100 * (psum-sbl)/sbl))
        print('radiant exitance dM/dT    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2.cm-1.K)]'.format(dMf,  dMe, 100 * (dMf-dMe)/dMe))
        print(' ')

        print('Test the functions by converting between different spectral domains.')
        wavelenRef = np.asarray([0.1,  1,  10 ,  100]) # in units of um
        wavenumRef = np.asarray([1.0e5,  1.0e4,  1.0e3,  1.0e2]) # in units of cm-1
        frequenRef = np.asarray([  2.99792458e+15,   2.99792458e+14,   2.99792458e+13, 2.99792458e+12])
        print('Input spectral vectors:')
        print('{0} micrometers'.format(wavelenRef))
        print('{0} wavenumber'.format(wavenumRef))
        print('{0} frequency'.format(frequenRef))

        # now test conversion of spectral density quantities
        #create planck spectral densities at the wavelength interval
        exitancewRef = planck(wavelenRef, 1000,'el')
        exitancefRef = planck(frequenRef, 1000,'ef')
        exitancenRef = planck(wavenumRef, 1000,'en')
        exitance = exitancewRef.copy()
        #convert to frequency density
        print('all following eight statements should print (close to) unity vectors:')
        (freq, exitance) = ryutils.convertSpectralDensity(wavelenRef, exitance, 'lf')
        print('exitance converted: wf against calculation')
        print(exitancefRef/exitance)
        #convert to wavenumber density
        (waven, exitance) = ryutils.convertSpectralDensity(freq, exitance, 'fn')
        print('exitance converted: wf->fn against calculation')
        print(exitancenRef/exitance)
        #convert to wavelength density
        (wavel, exitance) = ryutils.convertSpectralDensity(waven, exitance, 'nl')
        #now repeat in opposite sense
        print('exitance converted: wf->fn->nw against original')
        print(exitancewRef/exitance)
        print('spectral variable converted: wf->fn->nw against original')
        print(wavelenRef/wavel)
        #convert to wavenumber density
        exitance = exitancewRef.copy()
        (waven, exitance) = ryutils.convertSpectralDensity(wavelenRef, exitance, 'ln')
        print('exitance converted: wf against calculation')
        print(exitancenRef/exitance)
        #convert to frequency density
        (freq, exitance) = ryutils.convertSpectralDensity(waven, exitance, 'nf')
        print('exitance converted: wf->fn against calculation')
        print(exitancefRef/exitance)
        #convert to wavelength density
        (wavel, exitance) = ryutils.convertSpectralDensity(freq, exitance, 'fl')
        # if the spectral density conversions were correct, the following two should be unity vectors
        print('exitance converted: wn->nf->fw against original')
        print(exitancewRef/exitance)
        print('spectral variable converted: wn->nf->fw against original')
        print(wavelenRef/wavel)


    #--------------------------------------------------------------------------------------
    #now plot a number of graphs
    if doAll:
        import ryplot

        #plot a single planck curve on linear scale for 300K source
        wl=np.logspace(np.log10(0.2), np.log10(20), num=100).reshape(-1, 1)
        Mel = planck(wl, 300, type='el').reshape(-1, 1) # [W/(m$^2$.$\mu$m)]

        lp = ryplot.Plotter(1)
        lp.semilogX(1,wl,Mel,"Planck law exitance for 300 K source","Wavelength [$\mu$m]",
            "Exitance [W/(m$^2$.$\mu$m)]")
        lp.saveFig('M300k.eps')

        #plot all the planck functions.
        wl=np.logspace(np.log10(0.1), np.log10(100), num=100).reshape(-1, 1)
        n=np.logspace(np.log10(1e4/100),np. log10(1e4/0.1), num=100).reshape(-1, 1)
        f=np.logspace(np.log10(const.c/ (100*1e-6)),np. log10(const.c/ (0.1*1e-6)), num=100).reshape(-1, 1)
        temperature=[280,300,450,650,1000,1800,3000,6000]

        Mel = planck(wl, np.asarray(temperature).reshape(-1,1), type='el') # [W/(m$^2$.$\mu$m)]
        Mql = planck(wl, np.asarray(temperature).reshape(-1,1), type='ql') # [q/(s.m$^2$.$\mu$m)]
        Men = planck(n, np.asarray(temperature).reshape(-1,1), type='en')  # [W/(m$^2$.cm$^{-1}$)]
        Mqn = planck(n, np.asarray(temperature).reshape(-1,1), type='qn')  # [q/(s.m$^2$.cm$^{-1}$)]
        Mef = planck(f, np.asarray(temperature).reshape(-1,1), type='ef')  # [W/(m$^2$.Hz)]
        Mqf = planck(f, np.asarray(temperature).reshape(-1,1), type='qf')  # [q/(s.m$^2$.Hz)]

        legend = ["{0:.0f} K".format(temperature[0])]
        for temp in temperature[1:] :
            legend.append("{0:.0f} K".format(temp))

        fplanck = ryplot.Plotter(1, 2, 3,"Planck's Law", figsize=(18, 12))
        fplanck.logLog(1, wl, Mel, "Radiant, Wavelength Domain","Wavelength [$\mu$m]", \
            "Exitance [W/(m$^2$.$\mu$m)]",legendAlpha=0.5, label=legend, \
                        pltaxis=[0.1, 100, 1e-2, 1e9])
        fplanck.logLog(2, n, Men, "Radiant, Wavenumber Domain","Wavenumber [cm$^{-1}$]", \
            "Exitance [W/(m$^2$.cm$^{-1}$)]",legendAlpha=0.5, label=legend, \
                        pltaxis=[100, 100000, 1e-8, 1e+4])
        fplanck.logLog(3, f, Mef, "Radiant, Frequency Domain","Frequency [Hz]", \
            "Exitance [W/(m$^2$.Hz)]",legendAlpha=0.5, label=legend, \
                        pltaxis=[3e12, 3e15, 1e-20, 1e-6])

        fplanck.logLog(4, wl, Mql, "Photon Rate, Wavelength Domain","Wavelength [$\mu$m]", \
            "Exitance [q/(s.m$^2$.$\mu$m)]",legendAlpha=0.5, label=legend, \
                        pltaxis=[0.1, 100, 1e-0, 1e27])
        fplanck.logLog(5, n, Mqn, "Photon Rate, Wavenumber Domain","Wavenumber [cm$^{-1}$]", \
            "Exitance [q/(s.m$^2$.cm$^{-1}$)]",legendAlpha=0.5, label=legend, \
                        pltaxis=[100, 100000, 1e-8, 1e+23])
        fplanck.logLog(6, f, Mqf, "Photon Rate, Frequency Domain","Frequency [Hz]", \
            "Exitance [q/(s.m$^2$.Hz)]",legendAlpha=0.5, label=legend, \
                        pltaxis=[3e12, 3e15, 1e-20, 1e+13])

        #fplanck.GetPlot().show()
        fplanck.saveFig('planck.png')

        #now plot temperature derivatives
        Mel = dplanck(wl, np.asarray(temperature).reshape(-1,1), type='el') # [W/(m$^2$.$\mu$m.K)]
        Mql = dplanck(wl, np.asarray(temperature).reshape(-1,1), type='ql') # [q/(s.m$^2$.$\mu$m.K)]
        Men = dplanck(n , np.asarray(temperature).reshape(-1,1), type='en') # [W/(m$^2$.cm$^{-1}$.K)]
        Mqn = dplanck(n,  np.asarray(temperature).reshape(-1,1), type='qn') # [q/(s.m$^2$.cm$^{-1}$.K)]
        Mef = dplanck(f,  np.asarray(temperature).reshape(-1,1), type='ef') # [W/(m$^2$.Hz.K)]
        Mqf = dplanck(f,  np.asarray(temperature).reshape(-1,1), type='qf') # [q/(s.m$^2$.Hz.K)]

        fdplanck = ryplot.Plotter(2, 2, 3,"Planck's Law Temperature Derivative", figsize=(18, 12))
        fdplanck.logLog(1, wl, Mel, "Radiant, Wavelength Domain","Wavelength [$\mu$m]", \
            "dM/dT [W/(m$^2$.$\mu$m.K)]",legendAlpha=0.5, label=legend, \
                        pltaxis=[0.1, 100, 1e-5, 1e5])
        fdplanck.logLog(2, n, Men, "Radiant, Wavenumber Domain","Wavenumber [cm$^{-1}$]", \
            "dM/dT [W/(m$^2$.cm$^{-1}$.K)]",legendAlpha=0.5, label=legend, \
                        pltaxis=[100, 100000, 1e-10, 1e+1])
        fdplanck.logLog(3, f, Mef, "Radiant, Frequency Domain","Frequency [Hz]", \
            "dM/dT [W/(m$^2$.Hz.K)]",legendAlpha=0.5, label=legend, \
                        pltaxis=[3e12, 3e15, 1e-20, 1e-10])

        fdplanck.logLog(4, wl, Mql, "Photon Rate, Wavelength Domain","Wavelength [$\mu$m]", \
            "dM/dT [q/(s.m$^2$.$\mu$m.K)]",legendAlpha=0.5, label=legend, \
                        pltaxis=[0.1, 100, 1e-0, 1e24])
        fdplanck.logLog(5, n, Mqn, "Photon Rate, Wavenumber Domain","Wavenumber [cm$^{-1}$]", \
            "dM/dT [q/(s.m$^2$.cm$^{-1}$.K)]",legendAlpha=0.5, label=legend, \
                        pltaxis=[100, 100000, 1e-10, 1e+20])
        fdplanck.logLog(6, f, Mqf, "Photon Rate, Frequency Domain","Frequency [Hz]", \
            "dM/dT [q/(s.m$^2$.Hz.K)]",legendAlpha=0.5, label=legend, \
                        pltaxis=[3e12, 3e15, 1e-20, 1e+9])

        #fdplanck.GetPlot().show()
        fdplanck.saveFig('dplanck.png')
        print(' ')

    #--------------------------------------------------------------------------------------
    #a simple calculation to calculate the number of bits required to express colour ratio
    if doAll:
        print('Calculate the number of bits required to express the colour ratio')
        print('between an MTV flare and a relatively cold aircraft fuselage.')
        #calculate the radiance ratio of aircraft fuselage to MTV flare in 3-5 um band
        wl=np.arange(3.5, 5, 0.001)
        #MTV flare temperature is 2200 K. emissivity=0.15
        flareEmis = 0.15
        flareM = flareEmis * np.trapz(planckel(wl,2200).reshape(-1, 1),wl, axis=0)[0]
        #aircraft fuselage temperature is 250 K. emissivity=1,
        aircraftM =  np.trapz(planckel(wl,250).reshape(-1, 1),wl, axis=0)[0]
        print('Mflare={0:.2f} W/m2 \nMaircraft={1:.1f} W/m2'.format(flareM, aircraftM))
        print('Colour ratio: ratio={0:.3e} minimum number of bits required={1:.1f}'.\
            format(flareM/aircraftM,  np.log2(flareM/aircraftM)))

    #--------------------------------------------------------------------------------------
    print('\nmodule planck done!')
