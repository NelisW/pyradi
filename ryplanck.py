#  $Id$
#  $HeadURL$

################################################################
# The contents of this file are subject to the Mozilla Public License
# Version 1.1 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/

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
This module provides functions for Planck Law emittance calculations, as well as
temperature derivative calculations.
The functions provide spectral emittance in [W/(m^2.*)] or [q/(s.m^2.*)], given
the temperature and a vector of one of wavelength, wavenumbers or frequency
(six combinations each for emittance and temperature derivative). The total
emittance can also be calculated by using the Stefan-Boltzman equation, in
[W/m^2] or [q/(s.m^2)].

See the __main__ function for testing and examples of use.
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__='pyradi team'
__all__=['planck','dplanck','stefanboltzman','planckef',  'planckel', 'plancken',
'planckqf', 'planckql', 'planckqn', 'dplnckef', 'dplnckel', 'dplncken', 'dplnckqf',
'dplnckql', 'dplnckqn','an']

import numpy
import scipy.constants as const
import pyradi.ryutils as ryutils


class PlanckConstants:
    """Precalculate the Planck function constants using the values in
       scipy.constants.  Presumbly these constants are up to date and
       will be kept up to date.

        Reference: http://docs.scipy.org/doc/scipy/reference/constants.html
    """

    def __init__(self):
        """ Precalculate the Planck function constants.

        Reference: http://www.spectralcalc.com/blackbody/appendixC.html
        """

        import scipy.optimize

        self.c1em = 2 * numpy.pi * const.h * const.c * const.c
        self.c1el = self.c1em * (1.0e6)**(5-1) # 5 for lambda power and -1 for density
        self.c1en = self.c1em * (100)**3 * 100 # 3 for wavenumber, 1 for density
        self.c1ef = 2 * numpy.pi * const.h / (const.c * const.c)

        self.c1qm = 2 * numpy.pi * const.c
        self.c1ql = self.c1qm * (1.0e6)**(4-1) # 5 for lambda power and -1 for density
        self.c1qn = self.c1qm * (100)**2 * 100 # 2 for wavenumber, 1 for density
        self.c1nf = 2 * numpy.pi  / (const.c * const.c)

        self.c2m = const.h * const.c / const.k
        self.c2l = self.c2m * 1.0e6 # 1 for wavelength density
        self.c2n = self.c2m * 1.0e2 # 1 for cm-1 density
        self.c2f = const.h / const.k

        self.sigmae = const.sigma
        self.zeta3 = 1.2020569031595942853
        self.sigmaq = 4 * numpy.pi * self.zeta3 * const.k ** 3 \
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
        return n * (1-numpy.exp(-x)) - x

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
        print('e = {:.14e}'.format(numpy.exp(1)))
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
        print('wel = {:.14e} um.K'.format(self.wel))
        print('wql = {:.14e} um.K'.format(self.wql))
        print('wen = {:.14e} cm-1/K'.format(self.wen))
        print('wqn = {:.14e} cm-1/K'.format(self.wqn))
        print('wef = {:.14e} Hz/K'.format(self.wef))
        print('wqf = {:.14e} Hz/K'.format(self.wqf))
        print(' ')

pconst = PlanckConstants()

################################################################
##
def planckef(frequency, temperature):
    """ Planck function in frequency for radiant emittance.

    Args:
        | frequency (np.array[N,]):  frequency vector in  [Hz]
        | temperature (float):  Temperature scalar in [K]

    Returns:
        | (np.array[N,]): spectral radiant emittance in W/(m^2.Hz)

    Raises:
        | No exception is raised.
    """
    return pconst.c1ef * frequency**3 / (numpy.exp(pconst.c2f * \
        frequency / temperature)-1);

################################################################
##
def planckel(wavelength, temperature):
    """ Planck function in wavelength for radiant emittance.

    Args:
        | wavelength (np.array[N,]):  wavelength vector in  [um]
        | temperature (float):  Temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emittance in W/(m^2.um)

    Raises:
        | No exception is raised.
    """
    return pconst.c1el / (wavelength ** 5 * ( numpy.exp(pconst.c2l \
                / (wavelength * temperature))-1));

################################################################
##
def plancken(wavenumber, temperature):
    """ Planck function in wavenumber for radiant emittance.

    Args:
        | wavenumber (np.array[N,]):  wavenumber vector in   [cm^-1]
        | temperature (float):  Temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emittance in  W/(m^2.cm^-1)

    Raises:
        | No exception is raised.
    """
    return pconst.c1en * wavenumber**3 / (numpy.exp(pconst.c2n * wavenumber \
            / temperature)-1);

################################################################
##
def planckqf(frequency, temperature):
    """ Planck function in frequency domain for photon rate emittance.

    Args:
        | frequency (np.array[N,]): frequency vector in  [Hz]
        | temperature (float):  Temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emittance in q/(s.m^2.Hz)

    Raises:
        | No exception is raised.
    """
    return pconst.c1nf * frequency**2 / (numpy.exp(pconst.c2f * frequency \
            / temperature)-1);

################################################################
##
def planckql(wavelength, temperature):
    """ Planck function in wavelength domain for photon rate emittance.

    Args:
        | wavelength (np.array[N,]):  wavelength vector in  [um]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emittance in  q/(s.m^2.um)

    Raises:
        | No exception is raised.
    """
    return pconst.c1ql / (wavelength**4 * ( numpy.exp(pconst.c2l \
                / (wavelength * temperature))-1));

################################################################
##
def planckqn(wavenumber, temperature):
    """ Planck function in wavenumber domain for photon rate emittance.

    Args:
        | wavenumber (np.array[N,]):  wavenumber vector in   [cm^-1]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emittance in  q/(s.m^2.cm^-1)

    Raises:
        | No exception is raised.
    """
    return pconst.c1qn * wavenumber**2 / (numpy.exp(pconst.c2n * wavenumber \
            / temperature)-1);

################################################################
##
def dplnckef(frequency, temperature):
    """Temperature derivative of Planck function in frequency domain
    for radiant emittance.

    Args:
        | frequency (np.array[N,]): frequency vector in  [Hz]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emittance/K in W/(K.m^2.Hz)

    Raises:
        | No exception is raised.
    """
    xx=(pconst.c2f * frequency /temperature);
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=pconst.c1ef * frequency**3 / (numpy.exp(pconst.c2f * frequency \
            / temperature)-1);
    return f*y;


################################################################
##
def dplnckel(wavelength, temperature):
    """Temperature derivative of Planck function in wavelength domain for
    radiant emittance.

    Args:
        | wavelength (np.array[N,]):  wavelength vector in  [um]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emittance in W/(K.m^2.um)

    Raises:
        | No exception is raised.
    """
    # if xx > 350, then we get overflow
    xx = pconst.c2l /(wavelength * temperature)
    # return (3.7418301e8 * xx * numpy.exp(xx) ) \
    #     / (temperature* wavelength ** 5 * (numpy.exp(xx)-1) **2 )
    # refactor (numpy.exp(xx)-1)**2 to prevent overflow problem
    return (pconst.c1el * xx * numpy.exp(xx) / (numpy.exp(xx)-1) ) \
        / (temperature* wavelength ** 5 * (numpy.exp(xx)-1) )


################################################################
##
def dplncken(wavenumber, temperature):
    """Temperature derivative of Planck function in wavenumber domain for radiance emittance.

    Args:
        | wavenumber (np.array[N,]):  wavenumber vector in   [cm^-1]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emittance in  W/(K.m^2.cm^-1)

    Raises:
        | No exception is raised.
    """
    xx=(pconst.c2n * wavenumber /temperature)
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=(pconst.c1en* wavenumber **3 / (numpy.exp(pconst.c2n * wavenumber \
            / temperature)-1))
    return f*y

################################################################
##
def dplnckqf(frequency, temperature):
    """Temperature derivative of Planck function in frequency domain for photon rate.

    Args:
        | frequency (np.array[N,]): frequency vector in  [Hz]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emittance in q/(K.s.m^2.Hz)

    Raises:
        | No exception is raised.
    """
    xx=(pconst.c2f * frequency /temperature)
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=pconst.c1nf * frequency **2 / (numpy.exp(pconst.c2f * frequency \
            / temperature)-1)
    return f*y

################################################################
##
def dplnckql(wavelength, temperature):
    """Temperature derivative of Planck function in wavenumber domain for radiance emittance.

    Args:
        | wavelength (np.array[N,]):  wavelength vector in  [um]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emittance in  q/(K.s.m^2.um)

    Raises:
        | No exception is raised.
    """
    xx=(pconst.c2l /(wavelength * temperature))
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=pconst.c1ql / (wavelength ** 4 * ( numpy.exp(pconst.c2l \
            / (temperature * wavelength))-1))
    return f*y

################################################################
##
def dplnckqn(wavenumber, temperature):
    """Temperature derivative of Planck function in wavenumber domain for photon rate.

    Args:
        | wavenumber (np.array[N,]):  wavenumber vector in   [cm^-1]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emittance in  q/(s.m^2.cm^-1)

    Raises:
        | No exception is raised.
    """
    xx=(pconst.c2n * wavenumber /temperature)
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=pconst.c1qn * wavenumber **2 / (numpy.exp(pconst.c2n * wavenumber \
            / temperature)-1)
    return f*y



################################################################
##
def stefanboltzman(temperature, type='e'):
    """Stefan-Boltzman wideband integrated emittance.

    Calculates the total Planck Law emittance, integrated over all wavelengths,
    from a surface at the stated temperature. Emittance can be given in radiant or
    photon rate units, depending on user input in type.

    Args:
        | temperature (float):  temperature scalar in [K].
        | type (string):  'e' for radiant or 'q' for photon rate emittance.

    Returns:
        | (float): integrated radiant emittance in  [W/m^2] or [q/(s.m^2)].
        | Returns a -1 if the type is not 'e' or 'q'

    Raises:
        | No exception is raised.
    """

    #use dictionary to switch between options, lambda fn to calculate, default zero
    rtnval = {
              'e': lambda temperature: pconst.sigmae * temperature**4 ,
              'q': lambda temperature: pconst.sigmaq * temperature**3
              }.get(type, lambda temperature: -1)(temperature)
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
    """Planck Law spectral emittance.

    Calculates the Planck Law spectral emittance
    from a surface at the stated temperature. Emittance can be given in radiant or
    photon rate units, depending on user input in type.

    Args:
        | spectral (np.array[N,]):  spectral vector.
        | temperature (float):  Temperature scalar in [K].
        | type (string):
        |  'e' signifies Radiant values in [W/m^2.*].
        |  'q' signifies photon rate values  [quanta/(s.m^2.*)].
        |  'l' signifies wavelength spectral vector  [micrometer].
        |  'n' signifies wavenumber spectral vector [cm-1].
        |  'f' signifies frequency spectral vecor [Hz].

    Returns:
        | (np.array[N,]):  spectral radiant emittance (not radiance) in units selected.
        | For type = 'el' units will be [W/(m^2.um)].
        | For type = 'qf' units will be [q/(s.m^2.Hz)].
        | Other return types are similarly defined as above.
        | Returns vector of -1 values if illegal type is requested.

    Raises:
        | No exception is raised.
    """
    if type in plancktype.keys():
        #select the appropriate fn as requested by user
        emittance = plancktype[type](spectral, temperature)
    else:
        # return all zeros if illegal type
        emittance = - numpy.ones(spectral.shape)

    return emittance



################################################################
##
def dplanck(spectral, temperature, type='el'):
    """Temperature derivative of Planck Law emittance.

    Calculates the temperature derivative for Planck Law spectral emittance
    from a surface at the stated temperature. dM/dT can be given in radiant or
    photon rate units, depending on user input in type.

    Args:
        | spectral (np.array[N,]):  spectral vector in  [micrometer], [cm-1] or [Hz].
        | temperature (float):  Temperature scalar in [K].
        | type (string):
        |  'e' signifies Radiant values in [W/(m^2.K)].
        |  'q' signifies photon rate values  [quanta/(s.m^2.K)].
        |  'l' signifies wavelength spectral vector  [micrometer].
        |  'n' signifies wavenumber spectral vector [cm-1].
        |  'f' signifies frequency spectral vecor [Hz].

    Returns:
        | (np.array[N,]):  spectral radiant emittance (not radiance) in units selected.
        | For type = 'el' units will be [W/(m2.um.K)]
        | For type = 'qf' units will be [q/(s.m2.Hz.K)]
        | Other return types are similarly defined as above.
        | Returns vector of -1 values if illegal type is requested.

    Raises:
        | No exception is raised.
    """

    if type in dplancktype.keys():
        #select the appropriate fn as requested by user
        emittance = dplancktype[type](spectral, temperature)
    else:
        # return all zeros if illegal type
        emittance = - numpy.ones(spectral.shape)

    return emittance

################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    pconst.printConstants()

    #calculate the radiance ratio of aircraft fuselage to MTV flare in 3-5 um band
    wl=numpy.arange(3.5, 5, 0.001)
    #flare temperature is 2200 K. emissivity=0.15 at close range
    flareEmis = 0.2
    flareM=flareEmis * numpy.trapz(planckel(wl,2200).reshape(-1, 1),wl, axis=0)
    #aircraft fuselage temperature is 250 K. emissivity=1,
    flareA=  numpy.trapz(planckel(wl,250).reshape(-1, 1),wl, axis=0)
    print(wl)
    print(flareM)
    print(flareA)
    print('Flare/aircradt ratio: ratio={0} bits required={1}'.format(flareM/flareA,  numpy.log2(flareM/flareA)))


    #--------------------------------------------------------------------------------------
    #first test at fixed temperature
    # for each function we calculate a spectral emittance and then get the peak and where peak occurs.'
    # this is checked against a first principles equation calculation.

    tmprtr=1000  # arbitrary choice of temperature
    dTmprtr=0.01  # arbitrary choice of temperature

    print('Temperature for calculations             {0:f} [K]'.format(tmprtr))
    print('dTemperature for dM/dTcalculations       {0:f} [K]'.format(dTmprtr))

    numIntPts=10000  #number of integration points

    wl1=.05            # lower integration limit
    wl2= 1000         # upper integration limit
    wld=(wl2-wl1)/numIntPts  #integration increment
    wl=numpy.arange(wl1, wl2+wld, wld)

    print('\nplanckel WAVELENGTH DOMAIN, RADIANT EMITTANCE')
    M =planckel(wl,tmprtr)
    peakM =  1.28665e-11*tmprtr**5
    spectralPeakM = 2897.9/tmprtr
    spectralPeak = wl[M.argmax()]
    I=sum(M)*wld
    sblf = stefanboltzman(tmprtr, 'e')
    sbl=5.67033e-8*tmprtr**4
    dMe = ( planckel(spectralPeak, tmprtr+dTmprtr)  - planckel(spectralPeak,tmprtr))/dTmprtr
    dMf = dplnckel(spectralPeak,tmprtr)
    print('                            function       equations    (% error)')
    print('peak emittance             {0:e}   {1:e}   {2:+.4f}   [W/(m^2.um)]'.format(max(M),peakM, 100 * (max(M)-peakM)/peakM))
    print('peak emittance at          {0:e}   {1:e}   {2:+.4f}   [um]'.format(spectralPeak,spectralPeakM, 100 * (spectralPeak-spectralPeakM)/spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(I, sbl, 100 * (I-sbl)/sbl))
    print('radiant emittance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(sblf, sbl, 100 * (sblf-sbl)/sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}   {2:+.4f}   [W/(m^2.um.K)]'.format(dMf,  dMe, 100 * (dMf-dMe)/dMe))

    print('\nplanckql WAVELENGTH DOMAIN, PHOTON EMITTANCE');
    M =planckql(wl,tmprtr);
    peakM =  2.1010732e11*tmprtr*tmprtr*tmprtr*tmprtr
    spectralPeakM = 3669.7/tmprtr
    spectralPeak = wl[M.argmax()]
    I=sum(M)*wld
    sblf = stefanboltzman(tmprtr, 'q')
    sbl=1.5204e+15*tmprtr*tmprtr*tmprtr
    dMe = ( planckql(spectralPeak,tmprtr+dTmprtr)  - planckql(spectralPeak,tmprtr))/dTmprtr
    dMf = dplnckql(spectralPeak,tmprtr)
    print('                            function       equations    (% error)')
    print('peak emittance             {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2.um)]'.format(max(M),peakM, 100 * (max(M)-peakM)/peakM))
    print('peak emittance at          {0:e}   {1:e}   {2:+.4f}   [um]'.format(spectralPeak,spectralPeakM, 100 * (spectralPeak-spectralPeakM)/spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2)]'.format(I, sbl, 100 * (I-sbl)/sbl))
    print('radiant emittance (int)    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2)]'.format(sblf, sbl, 100 * (sblf-sbl)/sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2.um.K)]'.format(dMf,  dMe, 100 * (dMf-dMe)/dMe))


    f1=const.c/ (wl2*1e-6)
    f2=const.c/ (wl1*1e-6)
    fd=(f2-f1)/numIntPts  # integration increment
    f=numpy.arange(f1, f2+fd, fd)

    print('\nplanckef FREQUENCY DOMAIN, RADIANT EMITTANCE');
    M =planckef(f,tmprtr);
    peakM =  5.95664593e-19*tmprtr*tmprtr*tmprtr
    spectralPeakM = 5.8788872e10*tmprtr
    spectralPeak = f[M.argmax()]
    I=sum(M)*fd
    sblf = stefanboltzman(tmprtr, 'e')
    sbl=5.67033e-8*tmprtr*tmprtr*tmprtr*tmprtr
    dMe = ( planckef(spectralPeak,tmprtr+dTmprtr)  - planckef(spectralPeak,tmprtr))/dTmprtr
    dMf = dplnckef(spectralPeak,tmprtr)
    print('                            function       equations    (% error)')
    print('peak emittance             {0:e}   {1:e}   {2:+.4f}   [W/(m^2.Hz)]'.format(max(M),peakM, 100 * (max(M)-peakM)/peakM))
    print('peak emittance at          {0:e}   {1:e}   {2:+.4f}   [Hz]'.format(spectralPeak,spectralPeakM, 100 * (spectralPeak-spectralPeakM)/spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(I, sbl, 100 * (I-sbl)/sbl))
    print('radiant emittance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(sblf, sbl, 100 * (sblf-sbl)/sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}   {2:+.4f}   [W/(m^2.Hz.K)]'.format(dMf,  dMe, 100 * (dMf-dMe)/dMe))


    print('\nplanckqf FREQUENCY DOMAIN, PHOTON EMITTANCE');
    M =planckqf(f,tmprtr);
    peakM = 1.965658e4*tmprtr*tmprtr
    spectralPeakM = 3.32055239e10*tmprtr
    spectralPeak = f[M.argmax()]
    I=sum(M)*fd
    sblf = stefanboltzman(tmprtr, 'q')
    sbl=1.5204e+15*tmprtr*tmprtr*tmprtr
    dMe = ( planckqf(spectralPeak,tmprtr+dTmprtr)  - planckqf(spectralPeak,tmprtr))/dTmprtr
    dMf = dplnckqf(spectralPeak,tmprtr)
    print('                            function       equations    (% error)')
    print('peak emittance             {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2.Hz)]'.format(max(M),peakM, 100 * (max(M)-peakM)/peakM))
    print('peak emittance at          {0:e}   {1:e}   {2:+.4f}   [Hz]'.format(spectralPeak,spectralPeakM,100 * (spectralPeak-spectralPeakM)/spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2)]'.format(I, sbl, 100 * (I-sbl)/sbl))
    print('radiant emittance (int)    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2)]'.format(sblf, sbl, 100 * (sblf-sbl)/sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2.Hz.K)]'.format(dMf,  dMe, 100 * (dMf-dMe)/dMe))


    n1=1e4 / wl2
    n2=1e4 / wl1
    nd=(n2-n1)/numIntPts  # integration increment
    n=numpy.arange(n1, n2+nd, nd)

    print('\nplancken WAVENUMBER DOMAIN, RADIANT EMITTANCE');
    M =plancken(n,tmprtr);
    peakM =  1.78575759e-8*tmprtr*tmprtr*tmprtr
    spectralPeakM = 1.9609857086*tmprtr
    spectralPeak = n[M.argmax()]
    I=sum(M)*nd
    sblf = stefanboltzman(tmprtr, 'e')
    sbl=5.67033e-8*tmprtr*tmprtr*tmprtr*tmprtr
    dMe = ( plancken(spectralPeak,tmprtr+dTmprtr)  - plancken(spectralPeak,tmprtr))/dTmprtr
    dMf = dplncken(spectralPeak,tmprtr)
    print('                            function       equations    (% error)')
    print('peak emittance             {0:e}   {1:e}   {2:+.4f}   [W/(m^2.cm-1)]'.format(max(M),peakM, 100 * (max(M)-peakM)/peakM))
    print('peak emittance at          {0:e}   {1:e}   {2:+.4f}   [cm-1]'.format(spectralPeak,spectralPeakM,100 * (spectralPeak-spectralPeakM)/spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(I, sbl, 100 * (I-sbl)/sbl))
    print('radiant emittance (int)    {0:e}   {1:e}   {2:+.4f}   [W/m^2]'.format(sblf, sbl, 100 * (sblf-sbl)/sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}   {2:+.4f}   [W/(m^2.cm-1.K)]'.format(dMf,  dMe, 100 * (dMf-dMe)/dMe))


    print('\nplanckqn WAVENUMBER DOMAIN, PHOTON EMITTANCE');
    M =planckqn(n,tmprtr);
    peakM = 5.892639e14*tmprtr*tmprtr
    spectralPeakM = 1.1076170659*tmprtr
    spectralPeak =  n[M.argmax()]
    I=sum(M)*nd
    sblf = stefanboltzman(tmprtr, 'q')
    sbl=1.5204e+15*tmprtr*tmprtr*tmprtr
    dMe = ( planckqn(spectralPeak,tmprtr+dTmprtr)  - planckqn(spectralPeak,tmprtr))/dTmprtr
    dMf = dplnckqn(spectralPeak,tmprtr)
    print('                            function       equations    (% error)')
    print('peak emittance             {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2.cm-1)]'.format(max(M),peakM, 100 * (max(M)-peakM)/peakM))
    print('peak emittance at          {0:e}   {1:e}   {2:+.4f}   [cm-1]'.format(spectralPeak,spectralPeakM, 100 * (spectralPeak-spectralPeakM)/spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2)]'.format(I, sbl, 100 * (I-sbl)/sbl))
    print('radiant emittance (int)    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2)]'.format(sblf, sbl, 100 * (sblf-sbl)/sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}   {2:+.4f}   [q/(s.m^2.cm-1.K)]'.format(dMf,  dMe, 100 * (dMf-dMe)/dMe))
    print(' ')

    print('Test the functions by converting between different spectral domains.')
    wavelenRef = numpy.asarray([0.1,  1,  10 ,  100]) # in units of um
    wavenumRef = numpy.asarray([1.0e5,  1.0e4,  1.0e3,  1.0e2]) # in units of cm-1
    frequenRef = numpy.asarray([  2.99792458e+15,   2.99792458e+14,   2.99792458e+13, 2.99792458e+12])
    print('Input spectral vectors:')
    print('{0} micrometers'.format(wavelenRef))
    print('{0} wavenumber'.format(wavenumRef))
    print('{0} frequency'.format(frequenRef))

    # now test conversion of spectral density quantities
    #create planck spectral densities at the wavelength interval
    emittancewRef = planck(wavelenRef, 1000,'el')
    emittancefRef = planck(frequenRef, 1000,'ef')
    emittancenRef = planck(wavenumRef, 1000,'en')
    emittance = emittancewRef.copy()
    #convert to frequency density
    print('all following eight statements should print (close to) unity vectors:')
    (freq, emittance) = ryutils.convertSpectralDensity(wavelenRef, emittance, 'lf')
    print('emittance converted: wf against calculation')
    print(emittancefRef/emittance)
   #convert to wavenumber density
    (waven, emittance) = ryutils.convertSpectralDensity(freq, emittance, 'fn')
    print('emittance converted: wf->fn against calculation')
    print(emittancenRef/emittance)
    #convert to wavelength density
    (wavel, emittance) = ryutils.convertSpectralDensity(waven, emittance, 'nl')
    #now repeat in opposite sense
    print('emittance converted: wf->fn->nw against original')
    print(emittancewRef/emittance)
    print('spectral variable converted: wf->fn->nw against original')
    print(wavelenRef/wavel)
    #convert to wavenumber density
    emittance = emittancewRef.copy()
    (waven, emittance) = ryutils.convertSpectralDensity(wavelenRef, emittance, 'ln')
    print('emittance converted: wf against calculation')
    print(emittancenRef/emittance)
   #convert to frequency density
    (freq, emittance) = ryutils.convertSpectralDensity(waven, emittance, 'nf')
    print('emittance converted: wf->fn against calculation')
    print(emittancefRef/emittance)
    #convert to wavelength density
    (wavel, emittance) = ryutils.convertSpectralDensity(freq, emittance, 'fl')
    # if the spectral density conversions were correct, the following two should be unity vectors
    print('emittance converted: wn->nf->fw against original')
    print(emittancewRef/emittance)
    print('spectral variable converted: wn->nf->fw against original')
    print(wavelenRef/wavel)






    #--------------------------------------------------------------------------------------
    #now plot a number of graphs
    import ryplot

    wl=numpy.logspace(numpy.log10(0.1), numpy.log10(100), num=100).reshape(-1, 1)
    n=numpy.logspace(numpy.log10(1e4/100),numpy. log10(1e4/0.1), num=100).reshape(-1, 1)
    f=numpy.logspace(numpy.log10(const.c/ (100*1e-6)),numpy. log10(const.c/ (0.1*1e-6)), num=100).reshape(-1, 1)
    temperature=[280,300,450,650,1000,1800,3000,6000]

    Mel = planck(wl, temperature[0], type='el').reshape(-1, 1) # [W/(m$^2$.$\mu$m)]
    Mql = planck(wl, temperature[0], type='ql').reshape(-1, 1) # [q/(s.m$^2$.$\mu$m)]
    Men = planck(n, temperature[0], type='en').reshape(-1, 1)  # [W/(m$^2$.cm$^{-1}$)]
    Mqn = planck(n, temperature[0], type='qn').reshape(-1, 1)  # [q/(s.m$^2$.cm$^{-1}$)]
    Mef = planck(f, temperature[0], type='ef').reshape(-1, 1)  # [W/(m$^2$.Hz)]
    Mqf = planck(f, temperature[0], type='qf').reshape(-1, 1)  # [q/(s.m$^2$.Hz)]

    legend = ["{0:.0f} K".format(temperature[0])]

    for temp in temperature[1:] :
        Mel = numpy.hstack((Mel, planck(wl,temp, type='el').reshape(-1, 1)))
        Mql = numpy.hstack((Mql, planck(wl,temp, type='ql').reshape(-1, 1)))
        Men = numpy.hstack((Men, planck(n,temp, type='en').reshape(-1, 1)))
        Mqn = numpy.hstack((Mqn, planck(n,temp, type='qn').reshape(-1, 1)))
        Mef = numpy.hstack((Mef, planck(f,temp, type='ef').reshape(-1, 1)))
        Mqf = numpy.hstack((Mqf, planck(f,temp, type='qf').reshape(-1, 1)))
        legend.append("{0:.0f} K".format(temp))

    fplanck = ryplot.Plotter(1, 2, 3,"Planck's Law", figsize=(18, 12))
    fplanck.logLog(1, wl, Mel, "Radiant, Wavelength Domain","Wavelength [$\mu$m]", \
        "Emittance [W/(m$^2$.$\mu$m)]",legendAlpha=0.5, label=legend, \
                    pltaxis=[0.1, 100, 1e-2, 1e9])
    fplanck.logLog(2, n, Men, "Radiant, Wavenumber Domain","Wavenumber [cm$^{-1}$]", \
        "Emittance [W/(m$^2$.cm$^{-1}$)]",legendAlpha=0.5, label=legend, \
                    pltaxis=[100, 100000, 1e-8, 1e+4])
    fplanck.logLog(3, f, Mef, "Radiant, Frequency Domain","Frequency [Hz]", \
        "Emittance [W/(m$^2$.Hz)]",legendAlpha=0.5, label=legend, \
                    pltaxis=[3e12, 3e15, 1e-20, 1e-6])

    fplanck.logLog(4, wl, Mql, "Photon Rate, Wavelength Domain","Wavelength [$\mu$m]", \
        "Emittance [q/(s.m$^2$.$\mu$m)]",legendAlpha=0.5, label=legend, \
                    pltaxis=[0.1, 100, 1e-0, 1e27])
    fplanck.logLog(5, n, Mqn, "Photon Rate, Wavenumber Domain","Wavenumber [cm$^{-1}$]", \
        "Emittance [q/(s.m$^2$.cm$^{-1}$)]",legendAlpha=0.5, label=legend, \
                    pltaxis=[100, 100000, 1e-8, 1e+23])
    fplanck.logLog(6, f, Mqf, "Photon Rate, Frequency Domain","Frequency [Hz]", \
        "Emittance [q/(s.m$^2$.Hz)]",legendAlpha=0.5, label=legend, \
                    pltaxis=[3e12, 3e15, 1e-20, 1e+13])

    #fplanck.GetPlot().show()
    fplanck.saveFig('planck.png')


    #now plot temperature derivatives
    Mel = dplanck(wl, temperature[0], type='el').reshape(-1, 1) # [W/(m$^2$.$\mu$m.K)]
    Mql = dplanck(wl, temperature[0], type='ql').reshape(-1, 1) # [q/(s.m$^2$.$\mu$m.K)]
    Men = dplanck(n, temperature[0], type='en').reshape(-1, 1)  # [W/(m$^2$.cm$^{-1}$.K)]
    Mqn = dplanck(n, temperature[0], type='qn').reshape(-1, 1)  # [q/(s.m$^2$.cm$^{-1}$.K)]
    Mef = dplanck(f, temperature[0], type='ef').reshape(-1, 1)  # [W/(m$^2$.Hz.K)]
    Mqf = dplanck(f, temperature[0], type='qf').reshape(-1, 1)  # [q/(s.m$^2$.Hz.K)]

    for temp in temperature[1:] :
        Mel = numpy.hstack((Mel, dplanck(wl,temp, type='el').reshape(-1, 1)))
        Mql = numpy.hstack((Mql, dplanck(wl,temp, type='ql').reshape(-1, 1)))
        Men = numpy.hstack((Men, dplanck(n,temp, type='en').reshape(-1, 1)))
        Mqn = numpy.hstack((Mqn, dplanck(n,temp, type='qn').reshape(-1, 1)))
        Mef = numpy.hstack((Mef, dplanck(f,temp, type='ef').reshape(-1, 1)))
        Mqf = numpy.hstack((Mqf, dplanck(f,temp, type='qf').reshape(-1, 1)))

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


    print('\n\nmodule planck done!')
