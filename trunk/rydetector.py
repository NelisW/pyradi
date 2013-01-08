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
__all__= ['QuantumEfficiency', 'Responsivity', 'Detectivity', 'DStar', \
    'NEP', 'I0', 'EgTemp', 'IXV', 'NoiseBasic',' NoiseRogalski', 'Idark']

import scipy.constants as const
import matplotlib.pyplot as plt
import numpy as np
import pyradi.ryplot as ryplot
import pyradi.ryplanck as ryplanck
import sys


################################################################################
#
def Absorption(wavelength, Eg, tempDet, a0, a0p):
    """
    Calculate the spectral absorption coefficient
    for a semiconductor material with given material values.

    The model used here is based on Equations 3.5, 3.6 in Dereniaks book.

    Args:
        | wavelength: spectral variable [m]
        | Eg: bandgap energy [Ev]
        | tempDet: detector's temperature in [K]
        | a0: absorption coefficient [m-1] (Dereniak Eq 3.5 & 3.6)
        | a0p:  absorption coefficient in [m-1] (Dereniak Eq 3.5 & 3.6)

    Returns:
        | absorption: spectral absorption coefficient in [m-1]
    """

    #frequency/wavelength expressed as energy in Ev
    E = const.h * const.c / (wavelength * const.e )

    # the np.abs() in the following code is to prevent nan and inf values
    # the effect of the abs() is corrected further down when we select
    # only the appropriate values based on E >= Eg and E < Eg

    # Absorption coef - eq. 3.5- Dereniak
    a35 = (a0 * np.sqrt(np.abs(E - Eg))) + a0p
    # Absorption coef - eq. 3.6- Dereniak
    a36 = a0p * np.exp((- np.abs(E - Eg)) / (const.k * tempDet))
    absorption = a35 * (E >= Eg) + a36 * (E < Eg)

    return absorption



################################################################################
#
def QuantumEfficiency(absorption, lx, theta1, nFront, nMaterial):
    """
    Calculate the spectral quantum efficiency (QE) for a semiconductor material
    with given absorption and material values.

    Args:
        | absorption: spectral absorption coefficient in [m-1]
        | lx: detector depletion layer thickness [m]
        | theta1: angle between the surface's normal and the radiation in radian
        | nFront:  index of refraction of the material in front of detector
        | nMaterial:  index of refraction of the detector material

    Returns:
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

    # QE - eq. 3.4 - [1]
    quantumEffic = (1 - R) * (1 - np.exp( - absorption * lx))

    return quantumEffic

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
        | wavelength: spectral variable [m]
        | quantumEffic: spectral quantum efficiency

    Returns:
        | responsivity in [A/W]
    """

    return (const.e * wavelength * quantumEffic) / (const.h * const.c)

################################################################################
#
def Detectivity(iNoise, responsivity):
    """
    Detectivity can be interpreted as an SNR from a detector when 1 W of
    radiant power is incident on the physical detector of specific area and
    bandwidth.

    Args:
        | iNoise: noise current [A]
        | responsivity: spectral responsivity in [A/W]

    Returns
        | detectivity [1/W]
    """

    return responsivity / iNoise

################################################################################
#
def DStar(areaDet, deltaFreq, iNoise, responsivity):
    """
    The spectral D* is the signal-to-noise output when 1 W of monochromatic
    radiant flux is incident on 1 cm2 detector area, within a
    noise-equivalent bandwidth of 1 Hz.

    Args:
        | areaDet: detector's area in [m2]
        | deltaFreq: measurement or desirable bandwidth - [Hz]
        | iNoise: noise current [A]
        | responsivity: spectral responsivity in [A/W]

    Returns
        | detectivity [m \sqrt[Hz] / W] (note units)
    """

    return (responsivity * np.sqrt(areaDet * deltaFreq)) / (iNoise)

################################################################################
#
def NEP(detectivity):
    """
    NEP is the radiant power incident on detector that yields SNR=1 [1].

    Args:
        | detectivity: spectral detectivity [1/W]

    Returns
        | spectral noise equivalent power [W]
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
        | eMob: electron mobility [m2/V.s]
        | tauE: electron lifetime [s]
        | me: electron effective mass [kg]
        | mh: hole effective mass [kg]
        | na: dopping concentration [m-3]
        | Eg: energy bandgap in [Ev]
        | tDetec: detector's temperature in [K]
        | areaDet: detector's area [m2]
        | equation: 'd' for Dereniak and 'r' for Rogalski equations

    Returns:
        | I0: reverse sat current by Rogalski equation [A]
    """

    # diffusion length [m] Dereniak Eq7.20
    Le=np.sqrt(const.k * tDetec * eMob * tauE / const.e)
    # intrinsic carrier concentration - Dereniak`s book eq. 7.1 - m-3
    # Eg here in eV units, multiply with q
    ni = (np.sqrt(4 * (2 * np.pi * const.k * tDetec / const.h ** 2) ** 3 *\
        np.exp( - (Eg * const.e) / (const.k * tDetec)) * (me * mh) ** 1.5))
    # donor concentration in m-3
    nd = (ni ** 2 / na)

    if equation == 'd': # Dereniak's equations
        # reverse saturation current - Dereniak eq. 7.34 - Ampère
        I0 = areaDet * const.e * (Le / tauE) * nd
    else: # Rogalski equations
        # carrier diffusion coefficient - Rogalski's book pg. 164
        De = const.k * tDetec * eMob / const.e
        # reverse saturation current - Rogalski's book eq. 8.118
        I0 = areaDet * const.e * De * nd / Le

    return (I0)


################################################################################
#
def EgTemp(E0, VarshniA, VarshniB, tempDet):
    """
    This function calculates the bandgap at detector temperature, using the
    Varshni equation ref [3]

    Args:
        | E0: band gap at room temperature [eV]
        | VarshniA: Varshni parameter
        | VarshniB: Varshni parameter
        | tempDet: detector operating temperature [K]

    Returns:
        | Eg: bandgap at stated temperature [eV]
    """

    return (E0 - (VarshniA * (tempDet ** 2 / (tempDet + VarshniB))))

################################################################################
#
def IXV(V, IVbeta, tDetec, iPhoto,I0):
    """
    This function provides the diode curve for a given photocurrent.

    Args:
        | V: bias [V]
        | IVbeta: diode equation non linearity factor;
        | tDetec: detector's temperature [K]
        | iPhoto: photo-induced current, added to diode curve [A]
        | I0: reverse sat current [A]

    Returns:
        | current from detector [A]
    """

    # diode equation from Dereniak's book eq. 7.23
    return I0 * (np.exp(const.e * V / (IVbeta * const.k * tDetec)) - 1) - iPhoto

################################################################################
#
def NoiseBasic(tempDet, deltaFreq, R0, iBackgnd):
    """
    This function calculate the total noise produced in the diode using the
    basic physical models given in the references.

    Args:
        | tempDet: detector's temperature [K]
        | deltaFreq: measurement or desirable bandwidth [Hz]
        | R0: resistivity [Ohm]
        | iBackgnd: photocurrent generated by the background [A]

    Returns:
        | detector noise [A] over bandwidth deltaFreq
    """

    # johnson noise Dereniaki's book - eq. 5.58
    iJohnson = np.sqrt(4 * const.k * tempDet * deltaFreq / R0)

    # shot noise Dereniaki's book - eq. 5.69
    iShot = np.sqrt(2 * const.e * iBackgnd * deltaFreq)

    # total noise Dereniaki's book - eq. 5.75
    noise = np.sqrt(iJohnson ** 2 + iShot ** 2)

    return noise

################################################################################
#
def NoiseRogalski(I0current, tempDet, ibkg, deltaFreq, IVbeta=1):
    """
    This function calculate the total noise produced in the diode using the model
    given in Rogalski.

    Args:
        | I0current: reverse saturation current [A]
        | tempDet: detector's temperature in [K]
        | ibkg: background current [A]
        | deltaFreq: measurement or desirable bandwidth [Hz]
        | IVbeta: 1 for only diffusion, 2 if GR current dominates(Dereniak p253)

    Returns:
        | detector noise [A] over bandwidth deltaFreq
    """

    # % TOTAL NOISE MODELING FROM Rogalski'S BOOK (V=0)
    # Rogalski Eq 9.83 (2ndEdition), added the beta factor here.
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
    This function calculates the dark current, i.e. zero kelvin background
     from a photodiode in order to predict if the detector is working under
     BLIP or not.

    Args:
        | I0: saturation reverse current [A]
        | V: applied bias [V]
        | tempDet: detector's temperature [K]

    Returns:
        | dark current for voltage levels [A]
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

    #wavelength in micrometers, remember to scale down in functions.
    wavelenInit = 1  # wavelength in um
    wavelenFinal = 5.5  # wavelength in um
    wavelength = np.linspace(wavelenInit, wavelenFinal, 200)

    #source properties
    tempSource = 2000       # source temperature in K
    emisSource = 1.0          # source emissivity
    areaSource = 0.000033     # source area in m2

    #background properties
    tempBkg = 280.0              # background temperature in K
    emisBkg = 1.0                # background emissivity
    areaBkg = 2 * np.pi * (0.0055) ** 2 # equal to the window area

    #test setup parameters
    distance = 0.01      # distance between source and detector
    deltaFreq = 100.0      # measurement or desirable bandwidth - Hertz
    theta1 = 0.01        # radiation incident angle in radians
    transmittance = 1.0    # medium/filter/optics transmittance

    # detector device parameters
    tempDet = 80.0     # detector temperature in K
    areaDet = (200e-6) ** 2   # detector area in m2
    lx = 5e-6             # detector depletion layer thickness in meter
    n1 = 1.0              # refraction index of the air
    V = np.linspace(-250e-3, 75e-3, 100) # bias voltage range

    # detector material properties for InSb
    E0 = 0.24        # semiconductor bandgap at room temp in Ev
    VarshniA = 6e-4     # first fitting parameter for the Varshni's Equation
    VarshniB = 500.0        # second fitting parameter for the Varshni's Equation
    Eg = EgTemp(E0, VarshniA, VarshniB, tempDet)   # bandgap at operating termperature
    n2 = 3.42        # refraction index of the semiconductor being analyzed
    a0 = 1.9e4 * 100   # absorption coefficient [m-1], Eq3.5 & 3.6 Dereniak
    a0p = 800 * 100    # absorption coefficient [m-1] Eq3.5 & 3.6 Dereniak
    eMob = 120.0    # electron mobility - m2/V.s
    tauE = 1e-10    # electron lifetime - s
    me = 0.014 * const.m_e    # electron effective mass
    mh = 0.43 * const.m_e     # hole effective mass
    na = 1e16        # positive or negative dopping - m-3
    IVbeta = 1.0     # 1 when the diffusion current is dominant and 2 when the
                   # recombination current dominates - Dereniak's book page 251
    R0 = 1e10        # measured resistivity  - ohm

    ######################################################################

    #calculate the spectral absorption, quantum efficiency and responsivity
    absorption = Absorption(wavelength / 1e6, Eg, tempDet, a0, a0p)
    quantumEffic = QuantumEfficiency(absorption, lx, theta1, n1, n2)
    responsivity = Responsivity(wavelength / 1e6,quantumEffic)

    #spectral irradiance for test setup, for both source and background
    # in units of photon rate q/(s.m2)
    EsourceQL =(emisSource * ryplanck.planck(wavelength,tempSource,'ql') \
        * areaSource) / (np.pi * distance ** 2)
    EbkgQL = (emisBkg * ryplanck.planck(wavelength, tempBkg, 'ql') * \
        areaBkg ) / (np.pi * distance ** 2)
    #in radiant units W/m2
    EsourceEL = (emisSource * ryplanck.planck(wavelength, tempSource,'el')*\
        areaSource) / (np.pi * distance ** 2)
    EbkgEL = (emisBkg * ryplanck.planck(wavelength, tempBkg, 'el') * \
        areaBkg) / (np.pi * distance ** 2)

    #wideband inband irradiance (not used below, only for reference purposes)
    EsourceQ = np.trapz(EsourceQL, wavelength)
    EbkgQ =  np.trapz(EbkgQL, wavelength)
    EsourceE = np.trapz(EsourceEL, wavelength)
    EbkgE =  np.trapz(EbkgEL, wavelength)
    EtotalQ = EsourceQ + EbkgQ
    EtotalE = EsourceE + EbkgE

    print ("Detector irradiance source     = {0} q/(s.m2) {1} W/m2".\
        format(EsourceQ,EsourceE))
    print ("Detector irradiance background = {0} q/(s.m2) {1} W/m2".\
        format(EbkgQ,EbkgE))
    print ("Detector irradiance total      = {0} q/(s.m2) {1} W/m2".\
        format(EtotalQ,EtotalE))

    #photocurrent from both QE&QL and R&EL spectral data should have same values.
    iSourceE = np.trapz(EsourceEL * areaDet * responsivity, wavelength)
    iBkgE = np.trapz(EbkgEL * areaDet * responsivity, wavelength)
    iSourceQ = np.trapz(EsourceQL * areaDet * quantumEffic * const.e,wavelength)
    iBkgQ = np.trapz(EbkgQL * areaDet * quantumEffic * const.e, wavelength)
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

    ixv1 = IXV(V, IVbeta, tempDet, iBkgE, I0dereniak)
    ixv2 = IXV(V, IVbeta, tempDet, iTotalE, I0dereniak)

    iNoiseDereniak = NoiseBasic(tempDet, deltaFreq, R0, iBkgE)
    iNoiseRogalski = NoiseRogalski(I0rogalski, tempDet, iBkgE, deltaFreq, IVbeta)
    print ("iNoiseDereniak= {0} ".format(iNoiseDereniak))
    print ("iNoiseRogalski= {0} ".format(iNoiseRogalski))

    iDarkDereniak = Idark(I0dereniak,V,tempDet)
    iDarkRogalski = Idark(I0rogalski,V,tempDet)

    dStar = DStar(areaDet, deltaFreq, iNoiseDereniak, responsivity)
    dStar = dStar * 1e2       # units in cm

    detectivity = Detectivity(iNoiseDereniak, responsivity)

    NEPower=NEP(detectivity)

    absFig = ryplot.Plotter(1,1,1)
    absFig.plot(1,wavelength,absorption,'Absorption Coefficient',\
        r'Wavelength [$\mu$m]',r'Absorption Coefficient [m$^{-1}$]')
    absFig.saveFig('spectralabsorption.eps')

    QE = ryplot.Plotter(1,1,1)
    QE.plot(1,wavelength,quantumEffic,'Spectral Quantum Efficiency',\
        r'Wavelength [$\mu$m]','Quantum Efficiency')
    QE.saveFig('QE.eps')

    IXVplot = ryplot.Plotter(1,1,1)
    IXVplot.plot(1,V,ixv1,plotCol=['r'])
    IXVplot.plot(1,V,ixv2,'IxV Curve',\
        'Voltage [V]','Current [A]',plotCol=['b'])
    IXVplot.saveFig('IXVplot.eps')

    IDark = ryplot.Plotter(1,1,1)
    IDark.plot(1,V,iDarkDereniak,plotCol=['b'])
    IDark.plot(1,V,iDarkRogalski,'Dark Current','Voltage [V]','Current [A]',\
        plotCol=['r'])
    IDark.saveFig('IDark.eps')

    Respons = ryplot.Plotter(1,1,1)
    Respons.plot(1,wavelength,responsivity,'Spectral Responsivity',\
        r'Wavelength [$\mu$m]','Responsivity [A/W]')
    Respons.saveFig('Responsivity.eps')

    DStarfig = ryplot.Plotter(1,1,1)
    DStarfig.plot(1,wavelength,dStar,'Spectral normalized detectivity',\
        r'Wavelength [$\mu$m]',r'D* [cm$\sqrt{{\rm Hz}}$/W]')
    DStarfig.saveFig('dStar.eps')

    Detecfig = ryplot.Plotter(1,1,1)
    Detecfig.plot(1,wavelength,detectivity,'Detectivity',\
        r'Wavelength [$\mu$m]',r'Detectivity [W$^{-1}$]')
    Detecfig.saveFig('Detectivity.eps')

    NEPfig = ryplot.Plotter(1,1,1)
    NEPfig.plot(1,wavelength,NEPower,'Spectral Noise Equivalent Power',\
        r'Wavelength [$\mu$m]','NEP [W]',\
        pltaxis=[wavelenInit, wavelenFinal, 0,NEPower[0]])
    NEPfig.saveFig('NEP.eps')

    print('Done!')


