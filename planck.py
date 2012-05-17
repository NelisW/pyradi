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




import numpy




################################################################
##
def planckef(frequency, temperature):
    """   Planck function in frequency domain.
    Parameters:
        frequency == frequency vector in  [Hz]
        temperature == Temperature scalar in [K]
    Return:
        returns spectral radiant emitance in W/(m^2.Hz)
    """
    return 4.6323506e-50 * frequency**3 / (numpy.exp(4.79927e-11 * frequency/\
            temperature)-1);

################################################################

##
def planckel(wavelength, temperature):
    """   Planck function in wavelength domain.
    Parameters:
        wavelength == wavelength vector in  [um]
        temperature == Temperature scalar in [K]
    Return:
        returns spectral radiant emitance in W/(m^2.um)
    """
    return 3.7418301e8 / (wavelength ** 5 * ( numpy.exp(14387.86 /\
                (wavelength * temperature))-1));

################################################################

##
def plancken(wavenumber, temperature):
    """   Planck function in wavenumber domain.
    Parameters:
        wavelength == wavenumber vector in   [cm^-1]
        temperature == Temperature scalar in [K]
    Return:
        returns spectral radiant emitance in  W/(m^2.cm^-1)
    """
    return 3.7418e-8 * wavenumber**3 / (numpy.exp(1.438786 * wavenumber /\
            temperature)-1);

################################################################

##
def planckqf(frequency, temperature):
    """   Planck function in frequency domain.
    Parameters:
        frequency == frequency vector in  [Hz]
        temperature == Temperature scalar in [K]
    Return:
        returns spectral radiant emitance in q/(s.m^2.Hz)
    """
    return 6.9911e-17 * frequency**2 / (numpy.exp(4.79927e-11 * frequency /\
            temperature)-1);

################################################################

##
def planckql(wavelength, temperature):
    """   Planck function in wavelength domain.
    Parameters:
        wavelength == wavelength vector in  [um]
        temperature == temperature scalar in [K]
    Return:
        returns spectral radiant emitance in  q/(s.m^2.um)
    """
    return 1.88365e27 / (wavelength**4 * ( numpy.exp(14387.86 /\
                (wavelength * temperature))-1));

################################################################

##
def planckqn(wavenumber, temperature):
    """   Planck function in wavenumber domain.
    Parameters:
        wavelength == wavenumber vector in   [cm^-1]
        temperature == temperature scalar in [K]
    Return:
        returns spectral radiant emitance in  q/(s.m^2.cm^-1)
    """
    return 1.883635e15 * wavenumber**2 / (numpy.exp(1.438786 * wavenumber /\
            temperature)-1);

################################################################

##
def dplnckef(frequency, temperature):
    """   Planck's law derivative wrt temperature in frequency domain, 
            in radiant emittance 
    Parameters:
        frequency == frequency vector in  [Hz]
        temperature == temperature scalar in [K]
    Return:
        returns spectral radiant emitance/K in W/(K.m^2.Hz)
    """
    xx=(4.79927e-11 * frequency /temperature);
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=4.6323506e-50 * frequency**3 / (numpy.exp(4.79927e-11 * frequency /\
            temperature)-1);
    return f*y;


################################################################

##
def dplnckel(wavelength, temperature):
    """   Planck's law derivative wrt temp in wavelength domain, 
            in radiant emittance 
    Parameters:
        wavelength == wavelength vector in  [um]
        temperature == temperature scalar in [K]
    Return:
        returns spectral radiant emitance in W/(K.m^2.um)
    """
    xx=14387.86 /(wavelength * temperature)
    return (3.7418301e8 * xx * numpy.exp(xx) )/\
        (temperature* wavelength ** 5 * (numpy.exp(xx)-1) **2 )

################################################################

##
def dplncken(wavenumber, temperature):
    """   Planck's law derivative wrt temperature in wavenumber domain, 
            in radiant emittance 
    Parameters:
        wavelength == wavenumber vector in   [cm^-1]
        temperature == temperature scalar in [K]
    Return:
        returns spectral radiant emitance in  W/(K.m^2.cm^-1)
    """
    xx=(1.438786 * wavenumber /temperature)
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=(3.7418e-8 * wavenumber **3 / (numpy.exp(1.438786 * wavenumber /\
            temperature)-1))
    return f*y

################################################################

##
def dplnckqf(frequency, temperature):
    """   Planck's law derivative wrt temperature in frequency domain,
            in photon emittance 
    Parameters:
        frequency == frequency vector in  [Hz]
        temperature == temperature scalar in [K]
    Return:
        returns spectral radiant emitance in q/(K.s.m^2.Hz)
    """
    xx=(4.79927e-11 * frequency /temperature)
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=6.9911e-17 * frequency **2 / (numpy.exp(4.79927e-11 * frequency /\
            temperature)-1)
    return f*y

################################################################

##
def dplnckql(wavelength, temperature):
    """   Planck's law derivative wrt temperature in wavelength domain, 
            in photon emittance 
    Parameters:
        wavelength == wavelength vector in  [um]
        temperature == temperature scalar in [K]
    Return:
        returns spectral radiant emitance in  q/(K.s.m^2.um)
    """
    xx=(14387.86 /(wavelength * temperature))
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=1.88365e27 / (wavelength ** 4 * ( numpy.exp(14387.86 /\
            (temperature * wavelength))-1))
    return f*y

################################################################

##
def dplnckqn(wavenumber, temperature):
    """   Planck function in wavenumber domain.
    Parameters:
        wavelength == wavenumber vector in   [cm^-1]
        temperature == temperature scalar in [K]
    Return:
        returns spectral radiant emitance in  q/(s.m^2.cm^-1)
    """
    xx=(1.438786 * wavenumber /temperature)
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=1.883635e15 * wavenumber **2 / (numpy.exp(1.438786 * wavenumber /\
            temperature)-1)
    return f*y


plancktype = {  'el' : planckel, 'ef' : planckef, 'en' : plancken, \
                      'ql' : planckql, 'qf' : planckqf, 'qn' : planckqn}
dplancktype = {'el' : dplnckel, 'ef' : dplnckef, 'en' : dplncken, \
                      'ql' : dplnckql, 'qf' : dplnckqf, 'qn' : dplnckqn}

################################################################
##
def planck(spectral, temperature, type='el'):
    """   Planck function.
    Parameters:
        spectral == spectral vector
        temperature == Temperature scalar in [K]
        type = 'e' signifies Radiant values in [W/m2]
                  'q' signifies photon rate values  [quanta/(s.m2)]
                  'l' signifies wavelength spectral vector  [micrometer]
                  'n' signifies wavenumber spectral vector [cm-1]
                  'f' signifies frequency spectral vecor [Hz]
        
    Return:
        returns spectral radiant emitance (not radiance) in units selected.
        For type = 'el' units will be [W/(m2.um)]
        For type = 'qf' units will be [q/(s.m2.Hz)]
        returns zeros if illegal type is requested
    """
    if type in plancktype.keys():
        #select the appropriate fn as requested by user
        emittance = plancktype[type](spectral, temperature)
    else:
        # return all zeros if illegal type
        emittance = numpy.zeros(spectral.shape)
    
    return emittance



################################################################
##
def dplanck(spectral, temperature, type='el'):
    """   Temperature derivative of Planck function.
    Parameters:
        spectral == spectral vector
        temperature == Temperature scalar in [K]
        type = 'e' signifies Radiant values in [W/m2]
                  'q' signifies photon rate values  [quanta/(s.m2)]
                  'l' signifies wavelength spectral vector  [micrometer]
                  'n' signifies wavenumber spectral vector [cm-1]
                  'f' signifies frequency spectral vecor [Hz]
        
    Return:
        returns spectral radiant emitance (not radiance) in units selected.
        For type = 'el' units will be [W/(m2.um)]
        For type = 'qf' units will be [q/(s.m2.Hz)]
        returns zeros if illegal type is requested
    """

    if type in dplancktype.keys():
        #select the appropriate fn as requested by user
        emittance = dplancktype[type](spectral, temperature)
    else:
        # return all zeros if illegal type
        emittance = numpy.zeros(spectral.shape)
    
    return emittance

################################################################
##

if __name__ == '__init__':
    pass
    
if __name__ == '__main__':
    import ryplot
    
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
    peakM =  1.28665e-11*tmprtr*tmprtr*tmprtr*tmprtr*tmprtr
    spectralPeakM = 2897.9/tmprtr
    spectralPeak = wl[M.argmax()]
    I=sum(M)*wld
    sbl=5.67033e-8*tmprtr*tmprtr*tmprtr*tmprtr
    dMe = ( planckel(spectralPeak, tmprtr+dTmprtr)  - planckel(spectralPeak,tmprtr))/dTmprtr
    dMf = dplnckel(spectralPeak,tmprtr) 
    print('                            function       equations')
    print('peak emittance             {0:e}   {1:e}  [W/(m^2.um)]'.format(max(M),peakM))
    print('peak emittance at          {0:e}   {1:e}  [um]'.format(spectralPeak,spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}  [W/m^2]'.format(I, sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}  [W/(m^2.um.K)]'.format(dMf,  dMe))
        
    print('\nplanckql WAVELENGTH DOMAIN, PHOTON EMITTANCE');
    M =planckql(wl,tmprtr);
    peakM =  2.1010732e11*tmprtr*tmprtr*tmprtr*tmprtr
    spectralPeakM = 3669.7/tmprtr
    spectralPeak = wl[M.argmax()]
    I=sum(M)*wld
    sbl=1.5204e+15*tmprtr*tmprtr*tmprtr
    dMe = ( planckql(spectralPeak,tmprtr+dTmprtr)  - planckql(spectralPeak,tmprtr))/dTmprtr
    dMf = dplnckql(spectralPeak,tmprtr) 
    print('                            function       equations')
    print('peak emittance             {0:e}   {1:e}  [q/(s.m^2.um)]'.format(max(M),peakM))
    print('peak emittance at          {0:e}   {1:e}  [um]'.format(spectralPeak,spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}  [q/(s.m^2)]'.format(I, sbl))
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
    sbl=5.67033e-8*tmprtr*tmprtr*tmprtr*tmprtr
    dMe = ( planckef(spectralPeak,tmprtr+dTmprtr)  - planckef(spectralPeak,tmprtr))/dTmprtr
    dMf = dplnckef(spectralPeak,tmprtr) 
    print('                            function       equations')
    print('peak emittance             {0:e}   {1:e}  [W/(m^2.Hz)]'.format(max(M),peakM))
    print('peak emittance at          {0:e}   {1:e}  [Hz]'.format(spectralPeak,spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}  [W/m^2]'.format(I, sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}  [W/(m^2.Hz.K)]'.format(dMf,  dMe))

        
    print('\nplanckqf FREQUENCY DOMAIN, PHOTON EMITTANCE');
    M =planckqf(f,tmprtr);
    peakM = 1.965658e4*tmprtr*tmprtr
    spectralPeakM = 3.32055239e10*tmprtr
    spectralPeak = f[M.argmax()]
    I=sum(M)*fd
    sbl=1.5204e+15*tmprtr*tmprtr*tmprtr
    dMe = ( planckqf(spectralPeak,tmprtr+dTmprtr)  - planckqf(spectralPeak,tmprtr))/dTmprtr
    dMf = dplnckqf(spectralPeak,tmprtr) 
    print('                            function       equations')
    print('peak emittance             {0:e}   {1:e}  [q/(s.m^2.Hz)]'.format(max(M),peakM))
    print('peak emittance at          {0:e}   {1:e}  [Hz]'.format(spectralPeak,spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}  [q/(s.m^2)]'.format(I, sbl))
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
    sbl=5.67033e-8*tmprtr*tmprtr*tmprtr*tmprtr
    dMe = ( plancken(spectralPeak,tmprtr+dTmprtr)  - plancken(spectralPeak,tmprtr))/dTmprtr
    dMf = dplncken(spectralPeak,tmprtr) 
    print('                            function       equations')
    print('peak emittance             {0:e}   {1:e}  [W/(m^2.cm-1)]'.format(max(M),peakM))
    print('peak emittance at          {0:e}   {1:e}  [cm-1]'.format(spectralPeak,spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}  [W/m^2]'.format(I, sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}  [W/(m^2.cm-1.K)]'.format(dMf,  dMe))

        
    print('\nplanckqn WAVENUMBER DOMAIN, PHOTON EMITTANCE');
    M =planckqn(n,tmprtr);
    peakM = 5.892639e14*tmprtr*tmprtr
    spectralPeakM = 1.1076170659*tmprtr
    spectralPeak =  n[M.argmax()]
    I=sum(M)*nd
    sbl=1.5204e+15*tmprtr*tmprtr*tmprtr
    dMe = ( planckqn(spectralPeak,tmprtr+dTmprtr)  - planckqn(spectralPeak,tmprtr))/dTmprtr
    dMf = dplnckqn(spectralPeak,tmprtr) 
    print('                            function       equations')
    print('peak emittance             {0:e}   {1:e}  [q/(s.m^2.cm-1)]'.format(max(M),peakM))
    print('peak emittance at          {0:e}   {1:e}  [cm-1]'.format(spectralPeak,spectralPeakM))
    print('radiant emittance (int)    {0:e}   {1:e}  [q/(s.m^2)]'.format(I, sbl))
    print('radiant emittance dM/dT    {0:e}   {1:e}  [q/(s.m^2.cm-1.K)]'.format(dMf,  dMe))

    
    #--------------------------------------------------------------------------------------
    #now plot a number of graphs
    







