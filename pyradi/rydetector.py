# -*- coding: utf-8 -*-

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

# Initial Dev of the Original Code is Ricardo Augusto Tavares Santos, D.Sc.
# Instituto Tecnologico de Aeronautica Laboratorio de Guerra Eletronica - Brazil
# Portions created by Ricardo Augusto Tavares Santos are Copyright (C) 2012
# All Rights Reserved.

# Contributor(s): CJ Willers (refactored and updated original code).
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

This package was partly developed to provide additional material in support of students 
and readers of the book Electro-Optical System Analysis and Design: A Radiometry 
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""





__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['FermiDirac', 'JouleTeEv', 'eVtoJoule',
        'Absorption', 'AbsorptionFile', 'QuantumEfficiency',
        'Responsivity', 'DStar', 'NEP','Isaturation',
        'EgVarshni', 'IXV', 'Noise','DstarSpectralFlatPhotonLim']

import sys
import scipy.constants as const
import matplotlib.pyplot as plt
import numpy as np
import pyradi.ryplanck as ryplanck

################################################################################
#
def eVtoJoule(EeV):
    """
    Convert energy in eV to Joule.

    Args:
        | E: Energy in eV

    Returns:
        | EJ: Energy in J
    """

    return EeV * const.e


################################################################################
#
def JouleTeEv(EJ):
    """
    Convert energy in Joule to eV.

    Args:
        | EJ: Energy in J

    Returns:
        | EeV: Energy in eV
    """

    return EJ / const.e




################################################################################
#
def FermiDirac(Ef, EJ, T):
    """
    Returns the Fermi-Dirac probability distribution, given the crystal's
    Fermi energy, the temperature and the energy where the distribution values
    is required.

    Args:
        | Ef: Fermi energy in J
        | EJ: Energy in J
        | T : Temperature in K

    Returns:
        | fermiD : the Fermi-Dirac distribution
    """
    #prevent divide by zero
    den = (1 + np.exp( ( EJ - Ef ) / ( T * const.k) ) )
    return 1 / den



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
def AbsorptionFile(wavelength, filename):
    """
    Read the absorption coefficient from a data file and interpolate on the
    input spectral range.

    The data file must have the wavelength in the first column and absorption
    coefficient in [m-1] in the second column.

    Args:
        | wavelength: spectral variable [m]
        | filename: file containing the data


    Returns:
        | wavelength: values where absorption is defined
        | absorption: spectral absorption coefficient in [m-1]
    """

    #load data and build interpolation table
    from scipy.interpolate import interp1d
    filedata = np.loadtxt(filename)
    Table = interp1d(filedata[:,0], filedata[:,1], kind = 'linear')
    #check for valid wavelengths values wrt the input file
    mask = np.all([ wavelength >= np.min(filedata[:,0]) ,
        wavelength <= np.max(filedata[:,0])], axis=0)
    #select valid wavelengths
    wavelenOut = wavelength[mask]
    return wavelenOut, Table(wavelenOut)



################################################################################
#
def QuantumEfficiency(absorption, d1, d2, theta1, nFront, nMaterial):
    """
    Calculate the spectral quantum efficiency (QE) for a semiconductor material
    with given absorption and material values.

    Args:
        | absorption: spectral absorption coefficient in [m-1]
        | d1: depth where the detector depletion layer starts [m]
        | d2: depth where the detector depletion layer ends [m]
        | theta1: angle between the surface's normal and the radiation in radians
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
    quantumEffic = (1 - R) * \
                (np.exp( - absorption * d1) - np.exp( - absorption * d2))

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
        | detectivity [cm \sqrt[Hz] / W] (note units)
    """

    return (1e2 * responsivity * np.sqrt(areaDet * deltaFreq)) / (iNoise)

################################################################################
#
def NEP(iNoise, responsivity):
    """
    NEP is the radiant power incident on detector that yields SNR=1 [1].

    Args:
        | iNoise: noise current [A]
        | responsivity: spectral responsivity in [A/W]

    Returns
        | spectral noise equivalent power [W]
    """

    detectivity = responsivity / iNoise

    #the strange '+ (detectivity==0)' code below is to prevent divide by zero
    nep = ((1 / (detectivity + (detectivity == 0))) * (detectivity != 0))  + \
                 sys.float_info.max/10 * (detectivity == 0)

    return nep, detectivity


################################################################################
#
def Isaturation(mobE, tauE, mobH, tauH, me, mh, na, nd, Eg, tDetec, areaDet):
    """
    This function calculates the reverse saturation current, by
    Equation 7.22 in Dereniak's book

    Args:
        | mobE: electron mobility [m2/V.s]
        | tauE: electron lifetime [s]
        | mobH: hole mobility [m2/V.s]
        | tauH: hole lifetime [s]
        | me: electron effective mass [kg]
        | mh: hole effective mass [kg]
        | na: acceptor concentration [m-3]
        | nd: donor concentration [m-3]
        | Eg: energy bandgap in [Ev]
        | tDetec: detector's temperature in [K]
        | areaDet: detector's area [m2]

    Returns:
        | I0: reverse sat current [A]
    """

    # diffusion length [m] Dereniak Eq7.19
    Le=np.sqrt(const.k * tDetec * mobE * tauE / const.e)
    Lh=np.sqrt(const.k * tDetec * mobH * tauH / const.e)
    # intrinsic carrier concentration - Dereniak`s book eq. 7.1 - m-3
    # Eg here in eV units, multiply with q
    ni = (np.sqrt(4 * (2 * np.pi * const.k * tDetec / const.h ** 2) ** 3 *\
        np.exp( - (Eg * const.e) / (const.k * tDetec)) * (me * mh) ** 1.5))

    # reverse saturation current - Dereniak eq. 7.22 - Ampère
    if na == 0 or tauE == 0:
        elec = 0
    else:
        elec = Le / (na * tauE)
    if nd == 0 or tauH == 0:
        hole = 0
    else:
        hole = Lh / ( nd * tauH )

    I0 = areaDet * const.e * (ni ** 2) *( elec + hole )

    return (I0,ni)


################################################################################
#
def EgVarshni(E0, VarshniA, VarshniB, tempDet):
    """
    This function calculates the bandgap at detector temperature, using the
    Varshni equation

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

    The same function is also used to calculate the dark current, using
    IVbeta=1 and iPhoto=0

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
def Noise(tempDet, IVbeta, Isat, iPhoto, vBias=0):
    """
    This function calculates the noise power spectral density produced in the
    diode: shot noise and thermal noise. The assumption is that all noise
    sources are white noise PSD.

    Eq 5.143 plus thermal noise, see Eq 5.148

    Args:
        | tempDet: detector's temperature [K]
        | IVbeta: detector nonideal factor [-]
        | Isat: reverse saturation current [A]
        | iPhoto: photo current [A]
        | vBias: bias voltage on the detector [V]

    Returns:
        | detector noise power spectral density [A/Hz1/2]
        | R0: dynamic resistance at zero bias.
        | Johnson noise only noise power spectral density [A/Hz1/2]
        | Shot noise only noise power spectral density [A/Hz1/2]
    """

    R0 = IVbeta * const.k * tempDet / (Isat * const.e)

    # johnson noise
    iJohnson = 4 * const.k * tempDet / R0

    # shot noise for thermal component Isat
    iShot1 = 2 * const.e * Isat

    # shot noise for thermal component Isat
    iShot2 = 2 * const.e * Isat *np.exp(const.e * vBias /(const.k * tempDet * IVbeta))

    # shot noise for photocurrent
    iShot3 = 2 * const.e * iPhoto

    # total noise
    noise = np.sqrt(iJohnson + iShot1 + iShot2 + iShot3 )

    return noise, R0, np.sqrt(iJohnson), np.sqrt(iShot1 + iShot2 + iShot3 )


################################################################################
#
def DstarSpectralFlatPhotonLim(Tdetec, Tenvironment,epsilon):
    """
    This function calculates the photon noise limited D* of a detector with
    unlimited spectral response. This case does not apply to photon detectors.
    The absorption is assumed spectrally flat.

    Args:
        | Tdetec: detector temperature [K]
        | Tenvironment: environment temperature [K]
        | epsilon: emissivity/absorption

    Returns:
        | D* [cm \sqrt[Hz] / W] (note units)
    """

    # Kruse, P. W., Uncooled Thermal Imaging Arrays, Systems, and Applications ,
    #no. TT51, SPIE Press (2001).
    return 100 * np.sqrt(epsilon/(8 * const.k * const.sigma * \
        (Tdetec ** 5 + Tenvironment ** 5)))



################################################################################
if __name__ == '__init__':
    pass

if __name__ == '__main__':
    pass

    """
    In the model application, the user must define all the detector and
    semiconductor parameters. Each material type has its own paramenters,
    """
    import pyradi.ryplot as ryplot

    #calculate the theoretical D* for a spectrally flat detector
    Tenvironment = np.linspace(1,600,100)
    Tdetector = [0, 77, 195, 290]
    dstarP = ryplot.Plotter(1,1,1,figsize=(5,2))
    for Tdetec in Tdetector:
        dStar = DstarSpectralFlatPhotonLim(Tdetec,Tenvironment,1)
        dstarP.semilogY(1,Tenvironment,dStar,'',\
            r'Environmental temperature [K]','D* [cm.\sqrt{Hz}/W]',
            pltaxis=[0, 600, 1e9, 1e13])
    currentP = dstarP.getSubPlot(1)
    for xmaj in currentP.xaxis.get_majorticklocs():
        currentP.axvline(x=xmaj,ls='-')
    for ymaj in currentP.yaxis.get_majorticklocs():
        currentP.axhline(y=ymaj,ls='-')
    dstarP.saveFig('dstarFlat.eps')


    ######################################################################
    # tempBack = np.asarray([1])
    tempBack = np.asarray([1, 2, 4, 10, 25, 77, 195, 300, 500])
    eta = 1
    wavelength = np.logspace(0, 3, 100, True, 10)
    dstarwlc = np.zeros(wavelength.shape)

    def deeStarPeak(wavelength,temperature,eta,halfApexAngle):
        i = 0
        for wlc in wavelength:
            wl =  np.linspace(wlc/100, wlc, 1000).reshape(-1, 1)
            LbackLambda = ryplanck.planck(wl,temperature, type='ql')/np.pi
            Lback = np.trapz(LbackLambda.reshape(-1, 1),wl, axis=0)[0]
            Eback = Lback * np.pi * (np.sin(halfApexAngle)) ** 2
            # funny construct is to prevent divide by zero
            tempvar = np.sqrt(eta/(Eback+(Eback==0))) * (Eback!=0) + 0 * (Eback==0)
            dstarwlc[i] = 1e-6 * wlc * tempvar/(const.h * const.c * np.sqrt(2))
            #print(Eback)
            i = i + 1
        return dstarwlc * 100. # to get cm units

    dstarP = ryplot.Plotter(1,1,1)
    for temperature in tempBack:
        #halfApexAngle = 90 deg is for hemisphere
        halfApexAngle = (90 / 180.) * np.pi
        dstar = deeStarPeak(wavelength,temperature,eta,halfApexAngle)
        dstarP.logLog(1,wavelength,dstar,'',\
            r'Peak wavelength [$\mu$m]',r'D* [cm.$\sqrt{\rm Hz}$/W]',pltaxis=[1, 1e3, 1e10, 1e20])

    currentP = dstarP.getSubPlot(1)
    for xmaj in currentP.xaxis.get_majorticklocs():
        currentP.axvline(x=xmaj,ls='-')
    for ymaj in currentP.yaxis.get_majorticklocs():
        currentP.axhline(y=ymaj,ls='-')
    dstarP.saveFig('dstarPeak.eps')

    #####################################################################

    #demonstrate and test absorption data read from file
    absFile = ryplot.Plotter(1,1,1)
    wavelength = np.linspace(0.2, 2, 600)
    filenames = [
        'data/absorptioncoeff/Si.txt',
        'data/absorptioncoeff/GaAs.txt',
        'data/absorptioncoeff/Ge.txt',
        'data/absorptioncoeff/In07Ga03As64P36.txt',
        'data/absorptioncoeff/InP.txt',
        'data/absorptioncoeff/In53Ga47As.txt',
        'data/absorptioncoeff/piprekGe.txt',
        'data/absorptioncoeff/piprekGaAs.txt',
        'data/absorptioncoeff/piprekSi.txt',
        'data/absorptioncoeff/piprekInP.txt'
        ]
    for filename in filenames:
        wl, absorb = AbsorptionFile(wavelength, filename)
        absFile.semilogY(1,wl,absorb,'Absorption coefficient',\
            r'Wavelength [$\mu$m]','Absorptance [m-1]')
    currentP = absFile.getSubPlot(1)
    for xmaj in currentP.xaxis.get_majorticklocs():
        currentP.axvline(x=xmaj,ls='-')
    for ymaj in currentP.yaxis.get_majorticklocs():
        currentP.axhline(y=ymaj,ls='-')
    absFile.saveFig('absorption.eps')

    #######################################################################
    #plot the Fermi-Dirac distribution
    temperature = [0.001, 77, 300]
    EevR = np.linspace(-0.2, 0.2, 500)

    fDirac = FermiDirac(0, eVtoJoule(EevR),  temperature[0]).reshape(-1, 1)
    legend = ["{0:.0f} K".format(temperature[0])]
    for temp in temperature[1:] :
        fDirac = np.hstack((fDirac, FermiDirac(0, eVtoJoule(EevR), temp).reshape(-1, 1)))
        legend.append("{0:.0f} K".format(temp))

   # Mel = planck(wl, temperature[0], type='el').reshape(-1, 1) # [W/(m$^2$.$\mu$m)]


    fDfig = ryplot.Plotter(1,1,1)
    fDfig.plot(1,EevR,fDirac,'Fermi-Dirac distribution function',\
        r'Energy [eV]','Occupancy probability', label=legend)
    fDfig.saveFig('FermiDirac.eps')

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################

    #now calculate the characteristics of an InSb detector

    #wavelength in micrometers, remember to scale down in functions.
    wavelenInit = 1  # wavelength in um
    wavelenFinal = 5.5  # wavelength in um
    wavelength = np.linspace(wavelenInit, wavelenFinal, 200)
    Vbias = np.linspace(-200e-3, 100e-3, 200) # bias voltage range

    #source properties
    tempSource = 2000       # source temperature in K
    emisSource = 1.0          # source emissivity
    distance = 0.1      # distance between source and detector
    areaSource = 0.000033     # source area in m2
    solidAngSource = areaSource / (distance ** 2)

    #background properties
    tempBkg = 280.0              # background temperature in K
    emisBkg = 1.0                # background emissivity
    solidAngBkg = np.pi       # background is hemisphere

    #test setup parameters
    deltaFreq = 100.0      # measurement or desirable bandwidth - Hertz
    theta1 = 0.0        # radiation incident angle in radians

    # detector device parameters
    tempDet = 80.0     # detector temperature in K
    areaDet = (200e-6) ** 2   # detector area in m2
    d1 = 0                # start of the detector depletion layer in meter
    d2 = 5e-6             # end of the detector depletion layer in meter
    n1 = 1.0              # refraction index of the air

    # detector material properties for InSb
    E0 = 0.24        # semiconductor bandgap at room temp in Ev
    VarshniA = 6e-4     # first fitting parameter for the Varshni's Equation
    VarshniB = 500.0        # second fitting parameter for the Varshni's Equation
    #Varshni parameters taken from:
    #Group IV Elements, IV--IV and III--V Compounds. Part b --- Electronic,
    #Transport, Optical and Other Properties, O. Madelung and U. R\"{o}ssler and M. Schulz,
    #Springer-Verlag, 2002.
    n2 = 3.42        # refraction index of the semiconductor being analyzed
    a0 = 1.9e4 * 100   # absorption coefficient [m-1], Eq3.5 & 3.6 Dereniak
    a0p = 800 * 100    # absorption coefficient [m-1] Eq3.5 & 3.6 Dereniak
    eMob = 100.0    # electron mobility - m2/V.s
    hMob = 1.0    # hole mobility - m2/V.s
    tauE = 1e-8    # electron lifetime - s
    tauH = 1e-8    # electron lifetime - s
    me = 0.014 * const.m_e    # electron effective mass
    mh = 0.43 * const.m_e     # hole effective mass
    na = 1e16        # doping = acceptor concentration - m-3
    nd = 1e16        # doping = donor concentration - m-3
    IVbeta = 1.7     # 1 = dominanr diffusion current, 2 = g-r dominant

    #######################################################################
    # bandgap at operating termperature
    Eg = EgVarshni(E0, VarshniA, VarshniB, tempDet)
    print('Bandgap = {0:.6f}'.format(Eg))
    print('Lambda_c= {0:.6f}'.format(1.24/Eg))

    #calculate the spectral absorption, quantum efficiency and responsivity
    absorption = Absorption(wavelength / 1e6, Eg, tempDet, a0, a0p)
    quantumEffic = QuantumEfficiency(absorption, d1, d2, theta1, n1, n2)
    responsivity = Responsivity(wavelength / 1e6,quantumEffic)

    print ("\nSource temperature             = {0} K".format(tempSource))
    print ("Source emissivity              = {0} ".format(emisSource))
    print ("Source distance                = {0} m".format(distance))
    print ("Source area                    = {0} m2".format(areaSource))
    print ("Source solid angle             = {0:.6f} sr".format(solidAngSource))
    print ("\nBackground temperature         = {0} K".format(tempBkg))
    print ("Background emissivity          = {0} ".format(emisBkg))
    print ("Background solid angle         = {0:.6f} sr".format(solidAngBkg))

    #spectral irradiance for test setup, for both source and background
    # in units of photon rate q/(s.m2)
    EsourceQL =(emisSource * ryplanck.planck(wavelength,tempSource,'ql') \
        * solidAngSource) / (np.pi )
    EbkgQL = (emisBkg * ryplanck.planck(wavelength, tempBkg, 'ql') * \
        solidAngBkg ) / (np.pi )
    #in radiant units W/m2
    EsourceEL = (emisSource * ryplanck.planck(wavelength, tempSource,'el')*\
        solidAngSource) / (np.pi )
    EbkgEL = (emisBkg * ryplanck.planck(wavelength, tempBkg, 'el') * \
        solidAngBkg) / (np.pi )

    #photocurrent from both QE&QL and R&EL spectral data should have same values.
    iSourceE = np.trapz(EsourceEL * areaDet * responsivity, wavelength)
    iBkgE = np.trapz(EbkgEL * areaDet * responsivity, wavelength)
    iSourceQ = np.trapz(EsourceQL * areaDet * quantumEffic * const.e,wavelength)
    iBkgQ = np.trapz(EbkgQL * areaDet * quantumEffic * const.e, wavelength)
    iTotalE = iSourceE + iBkgE
    iTotalQ = iSourceQ + iBkgQ

    print ("\nDetector current source        = {0:.6f} A   {1:.6f} A".\
        format(iSourceQ,iSourceE))
    print ("Detector current background    = {0:.6f} A   {1:.6f} A".\
        format(iBkgQ,iBkgE))
    print ("Detector current total         = {0:.6f} A   {1:.6f} A".\
        format(iTotalQ,iTotalE))

    #saturation current from material and detector parameters
    Isat,ni = Isaturation(eMob, tauE, hMob, tauH, me, mh, na, nd, Eg, tempDet, areaDet)
    print ("\nI0               = {0:.6f} [A] ".format(Isat))
    print ("ni               = {0:.6f} [m-3] ".format(ni))

    #calculate the current-voltage curve for dark, background only and total signal
    ixvDark = IXV(Vbias, IVbeta, tempDet, 0, Isat)
    ixvBackgnd = IXV(Vbias, IVbeta, tempDet, iBkgE, Isat)
    ixvTotalE = IXV(Vbias, IVbeta, tempDet, iTotalE, Isat)

    #now calculate the noise
    iNoisePsd, R0, johnsonPsd, shotPsd = Noise(tempDet, IVbeta, Isat, iBkgE)
    iNoise = iNoisePsd * np.sqrt(deltaFreq)

    print('R0=              = {0:.6e} Ohm'.format(R0))
    print('NBW=             = {0:.6e} Hz'.format(deltaFreq))
    print("iNoiseTotal      = {0:.6e} A".format(iNoise))
    print("iNoise Johnson=  = {0:.6e} A".format(johnsonPsd * np.sqrt(deltaFreq)))
    print("iNoise shot      = {0:.6e} A".format(shotPsd * np.sqrt(deltaFreq)))

    #calculate the spectral noise equivalent power
    NEPower, detectivity = NEP(iNoise, responsivity)

    #use either of the following two forms:
    #dStar = DStar(areaDet, deltaFreq, iNoise, responsivity) # units in cm
    dStar = DStar(areaDet, 1, iNoisePsd, responsivity) # units in cm

    absFig = ryplot.Plotter(1,1,1)
    absFig.semilogY(1,wavelength,absorption,'Absorption Coefficient',\
        r'Wavelength [$\mu$m]',r'Absorption Coefficient [m$^{-1}$]')
    absFig.saveFig('spectralabsorption.eps')

    QE = ryplot.Plotter(1,1,1)
    QE.plot(1,wavelength,quantumEffic,'Spectral Quantum Efficiency',\
        r'Wavelength [$\mu$m]','Quantum Efficiency')
    QE.saveFig('QE.eps')

    IXVplot = ryplot.Plotter(1,1,1)
    IXVplot.semilogY(1,Vbias,np.abs(ixvDark) * 1e6, plotCol=['g'],label=['Dark'])
    IXVplot.semilogY(1,Vbias,np.abs(ixvBackgnd) * 1e6, plotCol=['r'],label=['Background'])
    IXVplot.semilogY(1,Vbias,np.abs(ixvTotalE) * 1e6,'IxV Curve',\
        'Voltage [V]','|Current| [uA]',plotCol=['b'],label=['Target'])
    IXVplot.saveFig('IXVplot.eps')

    Respons = ryplot.Plotter(1,1,1)
    Respons.plot(1,wavelength,responsivity,'Spectral Responsivity',\
        r'Wavelength [$\mu$m]','Responsivity [A/W]')
    Respons.saveFig('Responsivity.eps')

    DStarfig = ryplot.Plotter(1,1,1)
    DStarfig.plot(1,wavelength,dStar,'Spectral normalized detectivity',\
        r'Wavelength [$\mu$m]',r'D* [cm$\sqrt{{\rm Hz}}$/W]')
    DStarfig.saveFig('dStar.eps')

    NEPfig = ryplot.Plotter(1,1,1)
    NEPfig.plot(1,wavelength,NEPower,'Spectral Noise Equivalent Power',\
        r'Wavelength [$\mu$m]','NEP [W]',\
        pltaxis=[wavelenInit, wavelenFinal, 0,NEPower[0]])
    NEPfig.saveFig('NEP.eps')

    print('\nDone!')


