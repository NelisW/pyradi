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

See the __main__ function for examples of use.
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__='pyradi team'
__all__=['planck','dplanck','stefanboltzman','planckef',  'planckel', 'plancken',
'planckqf', 'planckql', 'planckqn', 'dplnckef', 'dplnckel', 'dplncken', 'dplnckqf',
'dplnckql', 'dplnckqn']

import numpy

################################################################
##
def planckef(frequency, temperature):
    """ Planck function in frequency for radiant emittance.

    Args:
        | frequency (np.array[N,]):  frequency vector in  [Hz]
        | temperature (float):  Temperature scalar in [K]

    Returns:
        | (np.array[N,]): spectral radiant emitance in W/(m^2.Hz)

    Raises:
        | No exception is raised.
    """
    return 4.6323506e-50 * frequency**3 / (numpy.exp(4.79927e-11 * frequency/\
            temperature)-1);

################################################################
##
def planckel(wavelength, temperature):
    """ Planck function in wavelength for radiant emittance.

    Args:
        | wavelength (np.array[N,]):  wavelength vector in  [um]
        | temperature (float):  Temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emitance in W/(m^2.um)

    Raises:
        | No exception is raised.
    """
    return 3.7418301e8 / (wavelength ** 5 * ( numpy.exp(14387.86 /\
                (wavelength * temperature))-1));

################################################################
##
def plancken(wavenumber, temperature):
    """ Planck function in wavenumber for radiant emittance.

    Args:
        | wavenumber (np.array[N,]):  wavenumber vector in   [cm^-1]
        | temperature (float):  Temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emitance in  W/(m^2.cm^-1)

    Raises:
        | No exception is raised.
    """
    return 3.7418e-8 * wavenumber**3 / (numpy.exp(1.438786 * wavenumber /\
            temperature)-1);

################################################################
##
def planckqf(frequency, temperature):
    """ Planck function in frequency domain for photon rate emittance.

    Args:
        | frequency (np.array[N,]): frequency vector in  [Hz]
        | temperature (float):  Temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emitance in q/(s.m^2.Hz)

    Raises:
        | No exception is raised.
    """
    return 6.9911e-17 * frequency**2 / (numpy.exp(4.79927e-11 * frequency /\
            temperature)-1);

################################################################
##
def planckql(wavelength, temperature):
    """ Planck function in wavelength domain for photon rate emittance.

    Args:
        | wavelength (np.array[N,]):  wavelength vector in  [um]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emitance in  q/(s.m^2.um)

    Raises:
        | No exception is raised.
    """
    return 1.88365e27 / (wavelength**4 * ( numpy.exp(14387.86 /\
                (wavelength * temperature))-1));

################################################################
##
def planckqn(wavenumber, temperature):
    """ Planck function in wavenumber domain for photon rate emittance.

    Args:
        | wavenumber (np.array[N,]):  wavenumber vector in   [cm^-1]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emitance in  q/(s.m^2.cm^-1)

    Raises:
        | No exception is raised.
    """
    return 1.883635e15 * wavenumber**2 / (numpy.exp(1.438786 * wavenumber /\
            temperature)-1);

################################################################
##
def dplnckef(frequency, temperature):
    """Temperative derivative of Planck function in frequency domain for radiant emittance.

    Args:
        | frequency (np.array[N,]): frequency vector in  [Hz]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emitance/K in W/(K.m^2.Hz)

    Raises:
        | No exception is raised.
    """
    xx=(4.79927e-11 * frequency /temperature);
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=4.6323506e-50 * frequency**3 / (numpy.exp(4.79927e-11 * frequency /\
            temperature)-1);
    return f*y;


################################################################
##
def dplnckel(wavelength, temperature):
    """Temperative derivative of Planck function in wavelength domain for radiant emittance.

    Args:
        | wavelength (np.array[N,]):  wavelength vector in  [um]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emitance in W/(K.m^2.um)

    Raises:
        | No exception is raised.
    """
    # if xx > 350, then we get overflow
    xx=14387.86 /(wavelength * temperature)
    return (3.7418301e8 * xx * numpy.exp(xx) )/\
        (temperature* wavelength ** 5 * (numpy.exp(xx)-1) **2 )

################################################################
##
def dplncken(wavenumber, temperature):
    """Temperative derivative of Planck function in wavenumber domain for radiance emittance.

    Args:
        | wavenumber (np.array[N,]):  wavenumber vector in   [cm^-1]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emitance in  W/(K.m^2.cm^-1)

    Raises:
        | No exception is raised.
    """
    xx=(1.438786 * wavenumber /temperature)
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=(3.7418e-8 * wavenumber **3 / (numpy.exp(1.438786 * wavenumber /\
            temperature)-1))
    return f*y

################################################################
##
def dplnckqf(frequency, temperature):
    """Temperative derivative of Planck function in frequency domain for photon rate.

    Args:
        | frequency (np.array[N,]): frequency vector in  [Hz]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emitance in q/(K.s.m^2.Hz)

    Raises:
        | No exception is raised.
    """
    xx=(4.79927e-11 * frequency /temperature)
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=6.9911e-17 * frequency **2 / (numpy.exp(4.79927e-11 * frequency /\
            temperature)-1)
    return f*y

################################################################
##
def dplnckql(wavelength, temperature):
    """Temperative derivative of Planck function in wavenumber domain for radiance emittance.

    Args:
        | wavelength (np.array[N,]):  wavelength vector in  [um]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emitance in  q/(K.s.m^2.um)

    Raises:
        | No exception is raised.
    """
    xx=(14387.86 /(wavelength * temperature))
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=1.88365e27 / (wavelength ** 4 * ( numpy.exp(14387.86 /\
            (temperature * wavelength))-1))
    return f*y

################################################################
##
def dplnckqn(wavenumber, temperature):
    """Temperative derivative of Planck function in wavenumber domain for photon rate.

    Args:
        | wavenumber (np.array[N,]):  wavenumber vector in   [cm^-1]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emitance in  q/(s.m^2.cm^-1)

    Raises:
        | No exception is raised.
    """
    xx=(1.438786 * wavenumber /temperature)
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=1.883635e15 * wavenumber **2 / (numpy.exp(1.438786 * wavenumber /\
            temperature)-1)
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
        | (float): integrated radiant emitance in  [W/m^2] or [q/(s.m^2)].
        | Returns a -1 if the type is not 'e' or 'q'

    Raises:
        | No exception is raised.
    """

    #use dictionary to switch between options, lambda fn to calculate, default zero
    rtnval = {
              'e': lambda temperature: 5.67033e-8 * temperature**4 ,
              'q': lambda temperature: 1.5204e15 * temperature**3
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
        | (np.array[N,]):  spectral radiant emitance (not radiance) in units selected.
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
        | (np.array[N,]):  spectral radiant emitance (not radiance) in units selected.
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

    c=2.997924580e8 # speed of light

    numIntPts=10000  #number of integration points

    wl1=.1            # lower integration limit
    wl2= 100         # upper integration limit
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
    print('                            function       equations')
    print('peak emittance             {0:e}   {1:e}  [W/(m^2.um)]'.format(max(M),peakM))
    print('peak emittance at          {0:e}   {1:e}  [um]'.format(spectralPeak,spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}  [W/m^2]'.format(I, sbl))
    print('radiant emittance (int)    {0:e}   {1:e}  [W/m^2]'.format(sblf, sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}  [W/(m^2.um.K)]'.format(dMf,  dMe))

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
    print('                            function       equations')
    print('peak emittance             {0:e}   {1:e}  [q/(s.m^2.um)]'.format(max(M),peakM))
    print('peak emittance at          {0:e}   {1:e}  [um]'.format(spectralPeak,spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}  [q/(s.m^2)]'.format(I, sbl))
    print('radiant emittance (int)    {0:e}   {1:e}  [q/(s.m^2)]'.format(sblf, sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}  [q/(s.m^2.um.K)]'.format(dMf,  dMe))


    f1=c/ (wl2*1e-6)
    f2=c/ (wl1*1e-6)
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
    print('                            function       equations')
    print('peak emittance             {0:e}   {1:e}  [W/(m^2.Hz)]'.format(max(M),peakM))
    print('peak emittance at          {0:e}   {1:e}  [Hz]'.format(spectralPeak,spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}  [W/m^2]'.format(I, sbl))
    print('radiant emittance (int)    {0:e}   {1:e}  [W/m^2]'.format(sblf, sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}  [W/(m^2.Hz.K)]'.format(dMf,  dMe))


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
    print('                            function       equations')
    print('peak emittance             {0:e}   {1:e}  [q/(s.m^2.Hz)]'.format(max(M),peakM))
    print('peak emittance at          {0:e}   {1:e}  [Hz]'.format(spectralPeak,spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}  [q/(s.m^2)]'.format(I, sbl))
    print('radiant emittance (int)    {0:e}   {1:e}  [q/(s.m^2)]'.format(sblf, sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}  [q/(s.m^2.Hz.K)]'.format(dMf,  dMe))


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
    print('                            function       equations')
    print('peak emittance             {0:e}   {1:e}  [W/(m^2.cm-1)]'.format(max(M),peakM))
    print('peak emittance at          {0:e}   {1:e}  [cm-1]'.format(spectralPeak,spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}  [W/m^2]'.format(I, sbl))
    print('radiant emittance (int)    {0:e}   {1:e}  [W/m^2]'.format(sblf, sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}  [W/(m^2.cm-1.K)]'.format(dMf,  dMe))


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
    print('                            function       equations')
    print('peak emittance             {0:e}   {1:e}  [q/(s.m^2.cm-1)]'.format(max(M),peakM))
    print('peak emittance at          {0:e}   {1:e}  [cm-1]'.format(spectralPeak,spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}  [q/(s.m^2)]'.format(I, sbl))
    print('radiant emittance (int)    {0:e}   {1:e}  [q/(s.m^2)]'.format(sblf, sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}  [q/(s.m^2.cm-1.K)]'.format(dMf,  dMe))

    #--------------------------------------------------------------------------------------
    #now plot a number of graphs
    import ryplot

    wl=numpy.logspace(numpy.log10(0.1), numpy.log10(100), num=100).reshape(-1, 1)
    n=numpy.logspace(numpy.log10(1e4/100),numpy. log10(1e4/0.1), num=100).reshape(-1, 1)
    f=numpy.logspace(numpy.log10(c/ (100*1e-6)),numpy. log10(c/ (0.1*1e-6)), num=100).reshape(-1, 1)
    temperature=[280,300,450,650,1000,1800,3000,6000]

    Mel = planck(wl, temperature[0], type='el').reshape(-1, 1)
    Mql = planck(wl, temperature[0], type='ql').reshape(-1, 1)
    Men = planck(n, temperature[0], type='en').reshape(-1, 1)
    Mqn = planck(n, temperature[0], type='qn').reshape(-1, 1)
    Mef = planck(f, temperature[0], type='ef').reshape(-1, 1)
    Mqf = planck(f, temperature[0], type='qf').reshape(-1, 1)

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
    Mel = dplanck(wl, temperature[0], type='el').reshape(-1, 1)
    Mql = dplanck(wl, temperature[0], type='ql').reshape(-1, 1)
    Men = dplanck(n, temperature[0], type='en').reshape(-1, 1)
    Mqn = dplanck(n, temperature[0], type='qn').reshape(-1, 1)
    Mef = dplanck(f, temperature[0], type='ef').reshape(-1, 1)
    Mqf = dplanck(f, temperature[0], type='qf').reshape(-1, 1)

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


    print('module planck done!')
