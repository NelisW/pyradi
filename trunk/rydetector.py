# -*- coding: utf-8 -*-

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

# Initial Dev of the Original Code is Ricardo Augusto Tavares Santos, D.Sc.
# Instituto Tecnologico de Aeronautica Laboratorio de Guerra Eletronica - Brazil
# Portions created by Ricardo Augusto Tavares Santos are Copyright (C) 2012
# All Rights Reserved.

# Contributor(s): CJ Willers (refactored original code).
###############################################################

"""
This model was built to give the user a simple but reliable tool to simulate or
to understand main parameters used to design a photovoltaic (PV) infrared
photodetector.  All the work done in this model was based in classical equations
found in the literature.

See the __main__ function for examples of use.

The example suggested here uses InSb parameters found in the literature. For
every compound or material, all the parameters, as well as the bandgap equation
must be changed.

This code uses the scipy.constants physical constants. For more details see
http://docs.scipy.org/doc/scipy/reference/constants.html

This code does not yet fully comply with the coding standards

References:

[1] Infrared Detectors and Systems, EL Dereniak & GD Boreman, Wiley
[2] Infrared Detectors, A Rogalski (1st or 2nd Edition), CRC Press
[3] Band Parameters for III-V Compound Semiconductors and their Alloys,
    I. Vurgaftmann, J. R. Meyer, and L. R. Ram-Mohan,
    Journal of Applied Physics 89 11, pp. 5815–5875, 2001.
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['QuantumEfficiency', 'Responsivity', 'Detectivity', \
    'NEP', 'I0', 'EgTemp', 'IXV', 'NoiseBasic',' NoiseRogalski', 'Idark']

import scipy.constants as const
import matplotlib.pyplot as plt
import numpy as np
import pyradi.ryplot as ryplot
import pyradi.ryplanck as ryplanck
import sys

################################################################################
#
def QuantumEfficiency(wavelength, Eg, lx, tempDet, theta1, a0, a0p, \
                       nFront, nMaterial):
    """
    Calculate the spectral quantum efficiency (QE) and absorption coefficient
    for a semiconductor material with given material values.

    The model used here is based on Equations 3.4, 3.5, 3.6 in Dereniaks book.

    Args:
        | wavelength: wavelength in m.
        | Eg: bandgap energy in Ev;
        | lx: detector thickness in m;
        | tempDet: detector's temperature in K;
        | theta1: angle between the surface's normal and the radiation path
        | a0: absorption coefficient in cm-1 (Dereniak Eq 3.5 & 3.6)
        | a0p:  absorption coefficient in cm-1 (Dereniak Eq 3.5 & 3.6)
        | nFront:  index of refraction of the material in front of detector
        | nMaterial:  index of refraction of the detector material

    Returns:
        | absorption: spectral absorption coefficient in cm-1
        | quantumEffic: spectral quantum efficiency
    """

    # calculate the semiconductor's optical reflectance
    theta2 = np.arcsin(np.sin(theta1) * nFront / nMaterial) # Snell's equation
    # Reflectance for perpendicular polarization
    RS = np.abs((nFront * np.cos(theta1) - nMaterial * np.cos(theta2)) / \
        (nFront * np.cos(theta1) + nMaterial * np.cos(theta2))) **2
    # Reflectance for parallel polarization
    RP = np.abs((nFront * np.cos(theta2) - nMaterial * np.cos(theta1)) / \
        (nFront * np.cos(theta1) + nMaterial * np.cos(theta2))) ** 2
    R = (RS + RP) / 2

    #wavelength expressed as energy in Ev
    E = const.h * const.c / (wavelength * const.e )

    # the np.abs() in the following code is to prevent nan and inf values
    # the effect of the abs() is corrected further down when we select
    # only the appropriate values based on E >= Eg and E < Eg

    # Absorption coef - eq. 3.5- Dereniak
    a35 = (a0 * np.sqrt(np.abs(E - Eg))) + a0p
    # Absorption coef - eq. 3.6- Dereniak
    a36 = a0p * np.exp((- np.abs(E - Eg)) / (const.k * tempDet))
    absorption = a35 * (E >= Eg) + a36 * (E < Eg)

    # QE - eq. 3.4 - [1]
    quantumEffic = (1 - R) * (1 - np.exp( - absorption * lx))

    return (absorption, quantumEffic)



################################################################################
#
def Responsivity(wavelength, quantumEffic):
    """
    Responsivity quantifies the amount of output seen per watt of radiant
    optical power input [1]. But, for this application it is interesting to
    define spectral responsivity that is the output per watt of monochromatic
    radiation.

    The model used here is based on Equations 7.114 in Dereniak's book.

    Args:
        | wavelength: wavelength in m;
        | quantumEffic: spectral quantum efficiency

    Returns:
        | responsivity
    """

    return (const.e * wavelength * quantumEffic) / (const.h * const.c)

################################################################################
#
def Detectivity(wavelength, areaDet, deltaFreq, iNoise, responsivity):
    """
    Detectivity can be interpreted as an SNR out of a detector when 1 W of
    radiant     power is incident on the detector, given an area equal to 1 cm2
    and noise-equivalent bandwidth of 1 Hz. The spectral responsivity is the rms
    signal-to-noise output when 1 W of monochromatic radiant flux is incident on
    1 cm2 detector area, within a noise-equivalent bandwidth of 1 Hz. Its
    maximum value (peak spectral D*) corresponds to the largest potential SNR.

    Args:
        | wavelength: wavelength in m;
        | areaDet: detector's area in m2;
        | deltaFreq: measurement or desirable bandwidth - Hertz;
        | iNoise: noise prediction using Dereniaki's model in A.
        | responsivity: spectral responsivity in [A/W]

    Returns
        | detectivity
    """

    return (responsivity * np.sqrt(areaDet * deltaFreq)) / (iNoise)

################################################################################
#
def NEP(detectivity):
    """
    NEP is the radiant power incident on detector that yields SNR=1 [1].

    Args:
        | wavelength: wavelength in m;
        | iNoiseDereniak: noise prediction using Dereniaki's model in A.
        | dtectivity: spectral detectivity

    Returns
        | spectral noise equivalent power
    """

    #the strange '+ (detectivity==0)' code below is to prevent divide by zero
    nep = ((1 / (detectivity + (detectivity == 0))) * (detectivity != 0))  + \
                 sys.float_info.max/10 * (detectivity == 0)

    return nep


################################################################################
#
def I0(eMob, tauE, me, mh, na, Eg, tDetec, areaDet, equation='d'):
    """
    This function calculates the reverse saturation current.

    Args:
        | eMob: electron mobility in m2/V.s;
        | tauE: electron lifetime in s;
        | me: electron effective mass in kg;
        | mh: hole effective mass in kg;
        | na: dopping concentration in m-3;
        | Eg: energy bandgap in Ev;
        | tDetec: detector's temperature in K;
        | areaDet: detector's area in m2;
        | equation: 'd' for dereniak and 'r' for rogalski equations

    Returns:
        | I0: reverse sat current by rogalski equation
    """

    # diffusion length [m] Dereniak Eq7.20
    Le=np.sqrt(const.k * tDetec * eMob * tauE / const.e)
    # intrinsic carrier concentration - dereniak`s book eq. 7.1 - m-3
    # Eg here in eV units, multiply with q
    ni = (np.sqrt(4 * (2 * np.pi * const.k * tDetec / const.h ** 2) ** 3 *\
        np.exp( - (Eg * const.e) / (const.k * tDetec)) * (me * mh) ** 1.5))
    # donor concentration in m-3
    nd = (ni ** 2 / na)

    if equation == 'd': # dereniak's equations
        # reverse saturation current - dereniak eq. 7.34 - Ampère
        I0 = areaDet * const.e * (Le / tauE) * nd
    else: # rogalski equations
        # carrier diffusion coefficient - rogalski's book pg. 164
        De = const.k * tDetec * eMob / const.e
        # reverse saturation current - rogalski's book eq. 8.118
        I0 = areaDet * const.e * De * nd / Le

    return (I0)


################################################################################
#
def EgTemp(E0, alpha, B, tempDet):
    """
    This function calculates the bandgap at detector temperature, using the
    Varshini equation ref [3]

    Args:
        | E0: band gap at room temperature
        | alpha: Varshini parameter
        | B: Varshini parameter
        | tempDet: detector operating temperature

    Returns:
        | Eg: bandgap at stated temperature
    """

    return (E0 - (alpha * (tempDet ** 2 / (tempDet + B))))

################################################################################
#
def IXV(V, IVbeta, tDetec, iPhoto,I0):
    """
    This function provides the diode curve for a given photocurrent.

    Args:
        | V: bias in V;
        | IVbeta: diode equation non linearity factor;
        | tDetec: detector's temperature in K;
        | iPhoto: photo-induced current, added to diode curve
        | I0: reverse sat current

    Returns:
        | current from detector
    """

    # diode equation from dereniak's book eq. 7.23
    return I0 * (np.exp(const.e * V / (IVbeta * const.k * tDetec)) - 1) - iPhoto

################################################################################
#
def NoiseBasic(tempDet, deltaFreq, R0, iBackgnd):
    """
    This function calculate the total noise produced in the diode using the
    basic physical models given in the references.

    Args:
        | tempDet: detector's temperature in K;
        | deltaFreq: measurement or desirable bandwidth - Hertz;
        | R0: resistivity in Ohm;
        | iBackgnd: photocurrent generated by the background in A.

    Returns:
        | noise: noise calculated from basics
    """

    # johnson noise Dereniaki's book - eq. 5.58
    iJohnson = np.sqrt(4 * const.k * tempDet * deltaFreq / R0)

    # shot noise Dereniaki's book - eq. 5.69
    iShot = np.sqrt(2 * const.e * iBackgnd * deltaFreq)

    # total noise Dereniaki's book - eq. 5.75
    noise = np.sqrt(iJohnson ** 2 + iShot ** 2)

    return (noise)


################################################################################
#
def NoiseRogalski(I0current, tempDet, ibkg, deltaFreq, IVbeta=1):
    """
    This function calculate the total noise produced in the diode using the model
    given in Rogalski.

    Args:
        | I0current: reverse saturation current in A;
        | tempDet: detector's temperature in K;
        | ibkg: background current
        | deltaFreq: measurement or desirable bandwidth - Hertz;
        | IVbeta: 1 for only diffusion, 2 if GR current dominates(Dereniak p253)

    Returns:
        | (iNoiseDereniak,iNoiseRogalski)
    """

    # % TOTAL NOISE MODELING FROM ROGALSKI'S BOOK (V=0)
    # rogalski Eq 9.83 (2ndEdition), added the beta factor here.
    R1 = IVbeta * const.k * tempDet / (const.e * I0current)

    # Rogalski eq. 8.111  (Eq 9.100 in 2nd Edition, error in book)
    # -> noise generated by the background is ignored
    #noise = np.sqrt((2 * const.e * (const.e * avg_qe * Ebkg * areaDet ) )\
    noise = np.sqrt((2 * const.e * (ibkg) )\
           + (4 * const.k * tempDet  / R1))

    return noise * np.sqrt(deltaFreq)


################################################################################
##
def Idark(I0,V,tempDet):
    """
    This function calculates the dark current, i.e. zwero kelvin background
     from a photodiode in order to predict if the detector is working under
     BLIP or not.

    Args:
        | I0: saturation reverse current in A;
        | V: applied bias in V;
        | tempDet: detector's temperature in K

    Returns:
        | dark current for voltage levels
    """

    return I0*(np.exp(const.e*V/(1*const.k*tempDet))-1)


################################################################################
if __name__ == '__init__':
    pass

if __name__ == '__main__':
    pass

    """
    In the model application, the user must define all the detector and
    semiconductor parameters. Each material type has its own paramenters,
    """

    wavelenInit=1e-6  # wavelength in meter- can start in 0
    wavelenFinal=5.5e-6  # wavelength in meter
    wavelength=np.linspace(wavelenInit,wavelenFinal,200)

    #source properties
    tempSource=0.1000       # source temperature in K
    emisSource=1.0          # source emissivity
    areaSource=0.000033     # source area in m2

    #background properties
    tempBkg=280.0              # background temperature in K
    emisBkg=1.0                # background emissivity
    areaBkg=2*np.pi*(0.0055)**2# equal to the window area

    #test setup parameters
    distance=0.01      # distance between source and detector
    deltaFreq=100.0      # measurement or desirable bandwidth - Hertz
    theta1=0.01        # radiation incident angle in radians
    transmittance=1.0    # medium/filter/optics transmittance

    # detector device parameters
    tempDet=80.0     # detector temperature in K
    areaDet=(200e-6)**2   # detector area in m2
    lx=5e-4             # detector thickness in meter
    n1=1.0              # refraction index of the air
    V=np.linspace(-250e-3,100e-3,100) # bias voltage range

    # detector material properties for InSb
    etha2=0.45     # quantum efficieny table 3.3 dereniak's book [3]
    E0=0.24        # semiconductor bandgap at room temp in Ev [3]
    n2=3.42        # refraction index of the semiconductor being analyzed [3]
    a0=1.9e4       # absorption coefficient , Equation 3.5 & 3.6 Dereniak
    a0p=800        # absorption coefficient , Equation 3.5 & 3.6 Dereniak
    eMob=120.0    # electron mobility - m2/V.s [3]
    hMob=1.0      # hole mobility - m2/V.s  [3]
    tauE=1e-10    # electron lifetime - s [3]
    tauH=1e-6     # hole lifetime - s [3]
    m0=9.11e-31    # electron mass - kg [3]
    me=0.014*m0    # used semiconductor electron effective mass [3]
    mh=0.43*m0     # used semiconductor hole effective mass [3]
    na=1e16        # positive or negative dopping - m-3
    IVbeta=1.0     # 1 when the diffusion current is dominantand 2 when the
                   # recombination current dominates - Derinaki's book page 251
    s=5e4          # surface recombination velocity ->
        # http://www.ioffe.ru/SVA/NSM/Semicond/InSb/electric.html#Recombination
    R0=1e10        # measured resistivity  - ohm
    alpha=6e-4     # first fitting parameter for the Varshini's Equation [3]
    B=500.0        # second fitting parameter for the Varshini's Equation [3]
    Eg = EgTemp(E0, alpha, B, tempDet)   # bandgap at operating termperature



    ######################################################################

    #calculate the spectral quantum efficiency and responsivity
    (absorption, quantumEffic) = QuantumEfficiency(wavelength, Eg, lx, \
                tempDet, theta1,a0,a0p,n1,n2)
    responsivity = Responsivity(wavelength,quantumEffic)

    #spectral irradiance for test setup, for both source and background
    # in units of photon rate q/(s.m2)
    EsourceQL = (emisSource * ryplanck.planck(wavelength * 1e6, tempSource, 'ql') \
        * areaSource) / (np.pi * distance ** 2)
    EbkgQL = (emisBkg * ryplanck.planck(wavelength*1e6, tempBkg, 'ql') * \
        areaBkg ) / (np.pi * distance ** 2)
    #in radiant units W/m2
    EsourceEL = (emisSource * ryplanck.planck(wavelength*1e6, tempSource,'el') *\
        areaSource) / (np.pi * distance ** 2)
    EbkgEL = (emisBkg * ryplanck.planck(wavelength*1e6, tempBkg, 'el') * \
        areaBkg) / (np.pi * distance ** 2)

    #wideband inband irradiance (not used below, only for reference purposes)
    EsourceQ = np.trapz(EsourceQL, wavelength*1e6)
    EbkgQ =  np.trapz(EbkgQL, wavelength*1e6)
    EsourceE = np.trapz(EsourceEL, wavelength*1e6)
    EbkgE =  np.trapz(EbkgEL, wavelength*1e6)
    EtotalQ = EsourceQ + EbkgQ
    EtotalE = EsourceE + EbkgE

    print ("Detector irradiance source     = {0} q/(s.m2) {1} W/m2".\
        format(EsourceQ,EsourceE))
    print ("Detector irradiance background = {0} q/(s.m2) {1} W/m2".\
        format(EbkgQ,EbkgE))
    print ("Detector irradiance total      = {0} q/(s.m2) {1} W/m2".\
        format(EtotalQ,EtotalE))

    #photocurrent from both QE&QL and R&EL spectral data - should have same values.
    iSourceE = np.trapz(EsourceEL * areaDet * responsivity, wavelength*1e6)
    iBkgE = np.trapz(EbkgEL * areaDet * responsivity, wavelength*1e6)
    iSourceQ = np.trapz(EsourceQL * areaDet * quantumEffic * const.e, wavelength*1e6)
    iBkgQ = np.trapz(EbkgQL * areaDet * quantumEffic * const.e, wavelength*1e6)
    iTotalE = iSourceE + iBkgE
    iTotalQ = iSourceQ + iBkgQ

    print(' ')
    print ("Detector current source        = {0} A {1} A".\
        format(iSourceQ,iSourceE))
    print ("Detector current background    = {0} A {1} A".\
        format(iBkgQ,iBkgE))
    print ("Detector current total         = {0} A {1} A".\
        format(iTotalQ,iTotalE))

    I0dereniak =I0(eMob, tauE, me, mh, na, Eg, tempDet, areaDet, 'd')
    I0rogalski =I0(eMob, tauE, me, mh, na, Eg, tempDet, areaDet, 'r')

    print(' ')
    print ("I0dereniak= {0} ".format(I0dereniak))
    print ("I0rogalski= {0} ".format(I0rogalski))

    ixv = IXV(V,IVbeta, tempDet, iBkgE, I0dereniak)

    iNoiseDereniak = NoiseBasic(tempDet, deltaFreq, R0, iBkgE)
    iNoiseRogalski = NoiseRogalski(I0rogalski, tempDet, iBkgE, deltaFreq, IVbeta)
    print ("iNoiseDereniak= {0} ".format(iNoiseDereniak))
    print ("iNoiseRogalski= {0} ".format(iNoiseRogalski))

    iDarkDereniak = Idark(I0dereniak,V,tempDet)
    iDarkRogalski = Idark(I0rogalski,V,tempDet)

    detectivity = Detectivity(wavelength, areaDet, deltaFreq, iNoiseDereniak, responsivity)
    detectivity=detectivity*1e2       # units in cm

    NEPower=NEP(detectivity)



    absFig = ryplot.Plotter(1,1,1)
    absFig.plot(1,wavelength*1e6,absorption,'Absorption Coefficient',\
        r'Wavelength [$\mu$m]',r'Absorption Coefficient [cm$^{-1}$]')
    absFig.saveFig('spectralabsorption.eps')

    QE = ryplot.Plotter(1,1,1)
    QE.plot(1,wavelength*1e6,quantumEffic,'Spectral Quantum Efficiency',\
        r'Wavelength [$\mu$m]','Quantum Efficiency')
    QE.saveFig('QE.eps')

    IXVplot = ryplot.Plotter(1,1,1)
    IXVplot.plot(1,V,ixv,'IxV Curve',\
        'Voltage [V]','Current [A]',plotCol=['r'])
    IXVplot.saveFig('IXVplot.eps')

    IDark = ryplot.Plotter(1,1,1)
    IDark.plot(1,V,iDarkDereniak,plotCol=['b'])
    IDark.plot(1,V,iDarkRogalski,'Dark Current',\
        'Voltage [V]','Current [A]',plotCol=['r'])
    IDark.saveFig('IDark.eps')

    Respons = ryplot.Plotter(1,1,1)
    Respons.plot(1,wavelength*1e6,responsivity,'Spectral Responsivity',\
        r'Wavelength [$\mu$m]','Responsivity [A/W]')
    Respons.saveFig('Responsivity.eps')

    Detect = ryplot.Plotter(1,1,1)
    Detect.plot(1,wavelength*1e6,detectivity,'Spectral Detectivity',\
        r'Wavelength [$\mu$m]',r'Detectivity [cm$\sqrt{{\rm Hz}}$/W]')
    Detect.saveFig('Detectivity.eps')

    NEPf = ryplot.Plotter(1,1,1)
    NEPf.plot(1,wavelength*1e6,NEPower,'Spectral Noise Equivalent Power',\
        r'Wavelength [$\mu$m]','NEP [W]',\
        pltaxis=[wavelenInit*1e6, wavelenFinal*1e6, 0,NEPower[0]])
    NEPf.saveFig('NEP.eps')

    print('Done!')


