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
        temperature == emperature scalar in [K]
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
        temperature == emperature scalar in [K]
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
        temperature == emperature scalar in [K]
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
        temperature == emperature scalar in [K]
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
        temperature == emperature scalar in [K]
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
        temperature == emperature scalar in [K]
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
        temperature == emperature scalar in [K]
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
        temperature == emperature scalar in [K]
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
        temperature == emperature scalar in [K]
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
        temperature == emperature scalar in [K]
    Return:
        returns spectral radiant emitance in  q/(s.m^2.cm^-1)
    """
    xx=(1.438786 * wavenumber /temperature)
    f=xx*numpy.exp(xx)/(temperature*(numpy.exp(xx)-1))
    y=1.883635e15 * wavenumber **2 / (numpy.exp(1.438786 * wavenumber /\
            temperature)-1)
    return f*y

################################################################

##
if __name__ == '__init__':
    pass

if __name__ == '__main__':
    pass # for now
   
