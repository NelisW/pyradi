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

# Contributor(s): ______________________________________.
###############################################################

"""
This model was built to give the user a simple but reliable tool to simulate or
to understand main parameters used to design a photovoltaic (PV) infrared
photodetector.  All the work done in this model was based in classical equations
found in the literature.

See the __main__ function for examples of use.

The code in this file does not conform to the minimum quality standards set out
in pyradi but it is offered here because its value, and coding standards should
not prevent  release. The user will have to expend some effort to make it work
in his situation.

The example suggested here uses InSb parameters found in the literature. For
every compound or material, all the parameters, as well as the bandgap equation
must be changed.

This code uses the CODATA physical constants. For more details see
http://physics.nist.gov/cuu/pdf/RevModPhysCODATA2010.pdf

References:

[1] Infrared Detectors and Systems, EL Dereniak & GD Boreman, Wiley
[2] Infrared Detectors, A Rogalski (1st or 2nd Edition), CRC Press
[3] ???

"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['QuantumEfficiency','Irradiance', 'Photocurrent', 'IXV', 'EgTemp' \
          'Detectivity', 'NEP', 'NoiseBasic', 'NoiseRogalski', \
          'Idark', 'Responsivity','I0']

from scipy.constants import codata
import scipy.constants as const
import matplotlib.pyplot as plt
import numpy as np
import pyradi.ryplot as ryplot
import sys

# IMPORTANT CONSTANTS
sigma = codata.value(u'Stefan-Boltzmann constant')   #W/(m2 K4)
sigma_photon=1.52e15    # boltzmann constant for photons- photons/(s.m2.K3)


################################################################################
#
def QuantumEfficiency(lambda_vector, Eg, lx, T_detector, theta1, a0, a0p, \
                       nFront, nMaterial):
    """
    Calculate the spectral quantum efficiency (QE) and absorption coefficient
    for a semiconductor material with given material values.

    The model used here is based on Equations 3.4, 3.5, 3.6 in Dereniaks book.

    Args:
        | lambda_vector: wavelength in m.
        | Eg: bandgap energy in Ev;
        | lx: detector thickness in m;
        | T_detector: detector's temperature in K;
        | theta1: angle between the surface's normal and the radiation path
        | a0: absorption coefficient in cm-1 (Dereniak Eq 3.5 & 3.6)
        | a0p:  absorption coefficient in cm-1 (Dereniak Eq 3.5 & 3.6)
        | nFront:  index of refraction of the material in front of detector
        | nMaterial:  index of refraction of the detector material

    Returns:
        | a_vector: spectral absorption coefficient in cm-1
        | etha_vector: spectral quantum efficiency
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
    E = const.h * const.c / (lambda_vector * const.e )

    # the np.abs() in the following code is to prevent nan and inf values
    # the effect of the abs() is corrected further down when we select
    # only the appropriate values based on E >= Eg and E < Eg

    # Absorption coef - eq. 3.5- Dereniak
    a35 = (a0 * np.sqrt(np.abs(E - Eg))) + a0p
    # Absorption coef - eq. 3.6- Dereniak
    a36 = a0p * np.exp((- np.abs(E - Eg)) / (const.k * T_detector))
    a_vector = a35 * (E >= Eg) + a36 * (E < Eg)

    # QE - eq. 3.4 - [1]
    etha_vector = (1 - R) * (1 - np.exp( - a_vector * lx))

    return (a_vector, etha_vector)



################################################################################
#
def Responsivity(lambda_vector, etha_vector):
    """
    Responsivity quantifies the amount of output seen per watt of radiant
    optical power input [1]. But, for this application it is interesting to
    define spectral responsivity that is the output per watt of monochromatic
    radiation.

    The model used here is based on Equations 7.114 in Dereniak's book.

    Args:
        | lambda_vector: wavelength in m;
        | etha_vector: spectral quantum efficiency

    Returns:
        | (Responsivity_vector)

    """

    return (const.e * lambda_vector * etha_vector) / (const.h * const.c)

################################################################################
#
def Detectivity(lambda_vector, A_det, delta_f, I_noise, responsivity):
    """
    Detectivity can be interpreted as an SNR out of a detector when 1 W of
    radiant     power is incident on the detector, given an area equal to 1 cm2
    and noise-equivalent bandwidth of 1 Hz. The spectral responsivity is the rms
    signal-to-noise output when 1 W of monochromatic radiant flux is incident on
    1 cm2 detector area, within a noise-equivalent bandwidth of 1 Hz. Its
    maximum value (peak spectral D*) corresponds to the largest potential SNR.

    Args:
        | lambda_vector: wavelength in m;
        | A_det: detector's area in m2;
        | delta_f: measurement or desirable bandwidth - Hertz;
        | I_noise: noise prediction using Dereniaki's model in A.
        | Responsivity_vector: spectral responsivity in [A/W]

    Returns
        | (Detectivity_vector)

    """

    return (responsivity * np.sqrt(A_det * delta_f)) / (I_noise)


################################################################################
#
def NEP(detectivity):
    """
    NEP is the radiant power incident on detector that yields SNR=1 [1].

    Args:
        | lambda_vector: wavelength in m;
        | I_noise_dereniak: noise prediction using Dereniaki's model in A.
        | dtectivity: spectral detectivity

    Returns
        | NEPower(spectral_nep)
    """

    #the strange '+ (detectivity==0)' code below is to prevent divide by zero
    nep = ((1 / (detectivity + (detectivity == 0))) * (detectivity != 0))  + \
                 sys.float_info.max/10 * (detectivity == 0)

    return nep


################################################################################
#
def I0(e_mob, tau_e, me, mh, na, Eg, tDetec, A_det, equation='d'):
    """
    This function calculates the reverse saturation current.

    Args:
        | e_mob: electron mobility in m2/V.s;
        | tau_e: electron lifetime in s;
        | me: electron effective mass in kg;
        | mh: hole effective mass in kg;
        | na: dopping concentration in m-3;
        | Eg: energy bandgap in Ev;
        | tDetec: detector's temperature in K;
        | A_det: detector's area in m2;
        | equation: 'd' for dereniak and 'r' for rogalski equations

    Returns:
        | I0: reverse sat current by rogalski equation
    """

    # diffusion length [m] Dereniak Eq7.20
    Le=np.sqrt(const.k * tDetec * e_mob * tau_e / const.e)
    # intrinsic carrier concentration - dereniak`s book eq. 7.1 - m-3
    # Eg here in eV units, multiply with q
    ni = (np.sqrt(4 * (2 * np.pi * const.k * tDetec / const.h ** 2) ** 3 *\
        np.exp( - (Eg * const.e) / (const.k * tDetec)) * (me * mh) ** 1.5))
    # donor concentration in m-3
    nd = (ni ** 2 / na)

    if equation == 'd': # dereniak's equations
        # reverse saturation current - dereniak eq. 7.34 - Ampère
        I0 = A_det * const.e * (Le / tau_e) * nd
    else: # rogalski equations
        # carrier diffusion coefficient - rogalski's book pg. 164
        De = const.k * tDetec * e_mob / const.e
        # reverse saturation current - rogalski's book eq. 8.118
        I0 = A_det * const.e * De * nd / Le

    return (I0)


################################################################################
#
def EgTemp(E0, alpha, B, T_detector):
    """
    This function calculates the bandgap at detector temperature, using the
    Varshini equation ref [3]

    Args:
        | E0: band gap at room temperature
        | alpha: Varshini parameter
        | B: Varshini parameter
        | T_detector: detector operating temperature

    Returns:
        | Eg: bandgap at stated temperature
    """

    return (E0 - (alpha * (T_detector ** 2 / (T_detector + B))))

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
        | I_vector: current from detector
    """

    # diode equation from dereniak's book eq. 7.23
    return I0 * (np.exp(const.e * V / (IVbeta * const.k * tDetec)) - 1) - iPhoto

################################################################################
#
def NoiseBasic(T_detector, delta_f, R0, I1_bkg):
    """
    This function calculate the total noise produced in the diode using the
    basic physical models given in the references.

    Args:
        | T_detector: detector's temperature in K;
        | delta_f: measurement or desirable bandwidth - Hertz;
        | R0: resistivity in Ohm;
        | I1_bkg: photocurrent generated by the background in A.

    Returns:
        | noise: noise calculated from basics
    """

    # johnson noise Dereniaki's book - eq. 5.58
    i_johnson = np.sqrt(4 * const.k * T_detector * delta_f / R0)

    # shot noise Dereniaki's book - eq. 5.69
    i_shot = np.sqrt(2 * const.e * I1_bkg * delta_f)

    # total noise Dereniaki's book - eq. 5.75
    noise = np.sqrt(i_johnson ** 2 + i_shot ** 2)

    return (noise)


################################################################################
#
def NoiseRogalski(I0current, T_detector, A_det, Ebkg, delta_f, avg_qe,IVbeta=1):
    """
    This function calculate the total noise produced in the diode using the model
    given in Rogalski.

    Args:
        | I0current: reverse saturation current in A;
        | T_detector: detector's temperature in K;
        | A_det: detector's area in m2;
        | Ebkg: irradiance generated by the background in W/m2;
        | delta_f: measurement or desirable bandwidth - Hertz;
        | avg_qe: calculated average quantum efficiency;
        | IVbeta: 1 for only diffusion, 2 if GR current dominates(Dereniak p253)

    Returns:
        | (I_noise_dereniak,I_noise_rogalski)

    """

    # % TOTAL NOISE MODELING FROM ROGALSKI'S BOOK (V=0)
    # rogalski Eq 9.83 (2ndEdition)
    R1 = IVbeta * const.k * T_detector / (const.e * I0current)

    # Rogalski eq. 8.111  (Eq 9.99 in 2nd Edition)
    # -> noise generated by the background is ignored
    noise = np.sqrt((2 * const.e ** 2 * avg_qe * Ebkg * A_det * delta_f)\
           + (4 * const.k * T_detector * delta_f / R1))

    return noise


################################################################################
##
def Idark(I0,V,T_detector):
    """
    This function calculates the dark current, i.e. zwero kelvin background
     from a photodiode in order to predict if the detector is working under
     BLIP or not.

    Args:
        | I0: saturation reverse current in A;
        | V: applied bias in V;
        | T_detector: detector's temperature in K

    Returns:
        | I_dark: dark current
    """

    return I0*(np.exp(const.e*V/(1*const.k*T_detector))-1)

################################################################################
#
def Photocurrent(A_det,etha2,avg_qe,Etotal,Ebkg):
    """
    The photocurrent is the the current generated by a photodetector given its
    quantum efficiency, irradiance and area.

    The result is given in current or tension (dependant on the transipedance
    used     in the calculation or measurement)

    Args:
        | A_det: detector´s area in m2;
        | Etha: quantum efficiency
        | etha2: average theoretical QE given by the literature;
        | Ebkg: background irradiance on the detector
        | Etotal: total irradiance on the detector

    Returns:
        | I1_wide: wideband total signal current, using model data
        | I1_wide_theoretical:  wideband total signal current, using theoretical data
        | I1_bkg: wideband background current, using model data
        | I1_bkg_theoretical: wideband background  current, using theoretical data
    """

    I1_wide=avg_qe*Etotal*A_det*const.e      # Photocurrent - eq. 3.10 - Infrared Detectors and Systems - Dereniak and Boreman

    I1_wide_theoretical=etha2*Etotal*A_det*const.e

    I1_bkg=avg_qe*Ebkg*A_det*const.e        # Photocurrent - eq. 3.10 - Infrared Detectors and Systems - Dereniak and Boreman

    I1_bkg_theoretical=etha2*Ebkg*A_det*const.e      # Photocurrent - eq. 3.10 - Infrared Detectors and Systems - Dereniak and Boreman

    V1_wide=np.mean(transipedance*I1_wide)

    return (I1_wide,I1_wide_theoretical,I1_bkg,I1_bkg_theoretical)


################################################################################
# calculate the radiance from a source and from the background
def Irradiance(lambda_vector,epsilon,T_source,T_bkg,A_source,A_bkg):
    """
    This function calculates the quantity of energy produced by a source and the
    background for a specific temperature in terms of number of photons and in
    terms of Watt for the inverval among the wavelengths defined by lambda_inicial
    and lambda_final. Must be understood that this amount of energy is only a fraction
    from the total calculated using the Stefann-Boltzmann Law.

    After to calculate the Radiance, the irradiance calculation is done in order
    to be able to calculate the photocurrente generated by a photodetector. So,
    it is necessary to calculate how much energy is reaching the detector given
    the energy emited by the source plus the backgorund and considering the
    distance among them and the detector. This is the irradiance calculation.

    All the equations used here are easily found in the literature.

    Args:
        | lambda_vector: wavenlength in m;
        | epsilon: source emissivity (non-dimensional);
        | T_source: source's temperature in K;
        | T_kbg: background's temperature in K;
        | A_source: soource's area in m2;
        | A_bkg: background's area in m2.

    Returns:
        | tuple of numpy arrays:
        | Esource: source irradiance on the detector
        | Ebkg: background irradiance on the detector
        | Etotal: total irradiance on the detector

    """

    # TOTAL PHOTON RADIANCE FROM THE SOURCE AND BACKGROUND

    # source total radiance in one hemisphere (adapted for photons) - eq. 2.9 - Infrared Detectors and Systems - [1]
    L_source=epsilon*sigma_photon*T_source**3/np.pi
    # bkg total radiance in one hemisphere (adapted for photons) - eq. 2.9 - Infrared Detectors and Systems - [1]
    L_bkg=epsilon*sigma_photon*T_bkg**3/np.pi

    # TOTAL EMITTED POWER FOR BKG AND SOURCE
    Total_source_energy=sigma*T_source**4               # Stefann-Boltzmann Law
    Total_bkg_energy=sigma*T_bkg**4                     # Stefann-Boltzmann Law

    # EMITTED ENERGY RATE
    # Planck Distribution - W/m2 - eq. 2.83 - Infrared Detectors and Systems - Dereniak and Boreman
    M_bkg_band_vector=((2*np.pi*const.h*const.c**2)/((lambda_vector)**5*(np.exp((const.h*const.c)/(lambda_vector*const.k*T_bkg))-1)))
    # Planck Distribution - W/m2 - eq. 2.83 - Infrared Detectors and Systems - Dereniak and Boreman
    M_source_band_vector=((2*np.pi*const.h*const.c**2)/((lambda_vector)**5*(np.exp((const.h*const.c)/(lambda_vector*const.k*T_source))-1)))

    M_total=(M_bkg_band_vector+M_source_band_vector)

    Band_source_energy=np.trapz(M_source_band_vector,lambda_vector)        # rate between the band and the total
    Band_bkg_energy=np.trapz(M_bkg_band_vector,lambda_vector)
    Total_energy=np.trapz(M_total,lambda_vector)

    rate_source=Band_source_energy/Total_source_energy
    rate_bkg=Band_bkg_energy/Total_bkg_energy

    # BAND PHOTON RADIANCE FROM THE SOURCE AND FROM THE BACKGROUND

    Radiance_final_source=L_source*rate_source
    Radiance_final_bkg=L_bkg*rate_bkg
    Radiance_total=Radiance_final_source+Radiance_final_bkg

    # IRRADIANCE CALCULUS
    Esource=(Radiance_final_source*A_source)/(d**2)     # photons/m2

    Ebkg=(Radiance_final_bkg*A_bkg)/(d**2)

    Etotal=Esource+Ebkg;

    #Irradiance_vector=np.r_[Esource,Ebkg,Etotal]

    return (Esource,Ebkg,Etotal)


################################################################################
if __name__ == '__init__':
    pass

if __name__ == '__main__':
    pass

    """
    In this step, all the parameters referring to the semiconductor material used
    to build the photodetector must be defined. Must be remembered that each material
    has its own paramenters, and each time the material is changed, the parameters
    must be changed as well.
    In the same step, the important semiconductors basic
    parameters are also calculated and defined.
    """

    lambda_initial=1e-6                      # wavelength in meter- can start in 0
    lambda_final=5.5e-6                      # wavelength in meter
    lambda_vector=np.linspace(lambda_initial,lambda_final,200)


    #source properties
    T_source=0.1                             # source temperature in K
    epsilon=1.0                              # source emissivity
    A_source=0.000033                        # source area in m2

    #background properties
    T_bkg=280.0                              # background temperature in K
    A_bkg=2*np.pi*(0.0055)**2                # bkg area in m2 - this area must be considered equals to the window area

    #test setup parameters
    d=0.01                                   # distance between source and detector or between window and detector
    delta_f=100.0                            # measurement or desirable bandwidth - Hertz
    theta1=0.01                              # radiation incident angle in degrees - angle between the surface's normal and the radiation path
    transipedance=10e7                       # transimpedance value used during the measurement
    final_trans=1.0                          # medium/filter/optics transmittance

    # detector device parameters
    T_detector=80.0                          # detector temperature in K
    A_det=(200e-6)**2                        # detector area in m2
    lx=5e-4                                  # detector thickness in meter
    n1=1.0                                   # refraction index of the air
    V=np.linspace(-250e-3,100e-3,100)        # bias voltage range

    # detector material properties for InSb
    etha2=0.45                               # quantum efficieny table 3.3 dereniak's book [3]
    E0=0.24                                  # semiconductor bandgap at room temp in Ev [3]
    n2=3.42                                  # refraction index of the semiconductor being analyzed [3]
    a0=1.9e4                                 # absorption coefficient , Equation 3.5 & 3.6 Dereniak
    a0p=800                                  # absorption coefficient , Equation 3.5 & 3.6 Dereniak
    e_mob=120.0                              # electron mobility - m2/V.s [3]
    h_mob=1.0                                # hole mobility - m2/V.s  [3]
    tau_e=1e-10                              # electron lifetime - s [3]
    tau_h=1e-6                               # hole lifetime - s [3]
    m0=9.11e-31                              # electron mass - kg [3]
    me=0.014*m0                              # used semiconductor electron effective mass [3]
    mh=0.43*m0                               # used semiconductor hole effective mass [3]
    na=1e16                                  # positive or negative dopping - m-3
    IVbeta=1.0                               # 1 when the diffusion current is dominantand 2 when the recombination current dominates - Derinaki's book page 251
    s=5e4                                    # surface recombination velocity -> http://www.ioffe.ru/SVA/NSM/Semicond/InSb/electric.html#Recombination
    R0=1e10                                  # measured resistivity  - ohm
    alpha=6e-4                                # first fitting parameter for the Varshini's Equation [3]
    B=500.0                                  # second fitting parameter for the Varshini's Equation [3]
    Eg = EgTemp(E0, alpha, B, T_detector)    # bandgap at operating termperature



    ######################################################################

    (a_vector,etha_vector) = QuantumEfficiency(lambda_vector,Eg,lx,T_detector,theta1,a0,a0p,n1,n2)

    avg_qe=np.mean(etha_vector)

    (Esource,Ebkg,Etotal) = Irradiance(lambda_vector,epsilon,T_source,T_bkg,A_source,A_bkg)

    (I1_wide,I1_wide_theoretical,I1_bkg,I1_bkg_theoretical)=Photocurrent(A_det,etha2,avg_qe,Etotal,Ebkg)

    I0_dereniak =I0(e_mob,tau_e,me,mh,na,Eg,T_detector,A_det,'d')
    I0_rogalski =I0(e_mob,tau_e,me,mh,na,Eg,T_detector,A_det,'r')

    IXV_vector1 = IXV(V,IVbeta,T_detector,I1_bkg,I0_dereniak)
    IXV_vector2 = IXV(V,IVbeta,T_detector,I1_bkg_theoretical,I0_dereniak)

    I_noise_dereniak =NoiseBasic(T_detector,delta_f,R0,I1_bkg)
    I_noise_rogalski =NoiseRogalski(I0_rogalski,T_detector,A_det,Ebkg,delta_f,avg_qe,IVbeta)

    I_dark_dereniak = Idark(I0_dereniak,V,T_detector)
    I_dark_rogalski = Idark(I0_rogalski,V,T_detector)

    Responsivity_vector = Responsivity(lambda_vector,etha_vector)

    Detectivity_vector = Detectivity(lambda_vector,A_det,delta_f,I_noise_dereniak,Responsivity_vector)
    Detectivity_vector=Detectivity_vector*1e2       # units in cm

    NEPower=NEP(Detectivity_vector)


    print ("Detector irradiance source=  {0}".format(Esource))
    print ("Detector irradiance background= {0} ".format(Ebkg))
    print ("Detector irradiance total= {0} ".format(Etotal))
    print ("I1_wide= {0} ".format(I1_wide))
    print ("I1_wide_theoretical= {0} ".format(I1_wide_theoretical))
    print ("I0_dereniak= {0} ".format(I0_dereniak))
    print ("I0_rogalski= {0} ".format(I0_rogalski))
    print ("I_noise_dereniak= {0} ".format(I_noise_dereniak))
    print ("I_noise_rogalski= {0} ".format(I_noise_rogalski))

    absFig = ryplot.Plotter(1,1,1)
    absFig.plot(1,lambda_vector*1e6,a_vector,'Absorption Coefficient',\
        r'Wavelength [$\mu$m]',r'Absorption Coefficient [cm$^{-1}$]')
    absFig.saveFig('spectralabsorption.eps')

    QE = ryplot.Plotter(1,1,1)
    QE.plot(1,lambda_vector*1e6,etha_vector,'Spectral Quantum Efficiency',\
        r'Wavelength [$\mu$m]','Quantum Efficiency')
    QE.saveFig('QE.eps')

    IXVplot = ryplot.Plotter(1,1,1)
    IXVplot.plot(1,V,IXV_vector1,plotCol=['b'])
    IXVplot.plot(1,V,IXV_vector2,'IxV Curve',\
        'Voltage [V]','Current [A]',plotCol=['r'])
    IXVplot.saveFig('IXVplot.eps')

    IDark = ryplot.Plotter(1,1,1)
    IDark.plot(1,V,I_dark_dereniak,plotCol=['b'])
    IDark.plot(1,V,I_dark_rogalski,'Dark Current',\
        'Voltage [V]','Current [A]',plotCol=['r'])
    IDark.saveFig('IDark.eps')

    Respons = ryplot.Plotter(1,1,1)
    Respons.plot(1,lambda_vector*1e6,Responsivity_vector,'Spectral Responsivity',\
        r'Wavelength [$\mu$m]','Responsivity [A/W]')
    Respons.saveFig('Responsivity.eps')

    Detect = ryplot.Plotter(1,1,1)
    Detect.plot(1,lambda_vector*1e6,Detectivity_vector,'Spectral Detectivity',\
        r'Wavelength [$\mu$m]',r'Detectivity [cm$\sqrt{{\rm Hz}}$/W]')
    Detect.saveFig('Detectivity.eps')

    NEPf = ryplot.Plotter(1,1,1)
    NEPf.plot(1,lambda_vector*1e6,NEPower,'Spectral Noise Equivalent Power',\
        r'Wavelength [$\mu$m]','NEP [W]',\
        pltaxis=[lambda_initial*1e6, lambda_final*1e6, 0,NEPower[0]])
    NEPf.saveFig('NEP.eps')

    print('Done!')


