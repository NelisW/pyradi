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

# The Initial Developer of the Original Code is Ricardo Augusto Tavares Santos, D.Sc.
# Instituto Tecnologico de Aeronautica - Laboratorio de Guerra Eletronica - Brazil
# Portions created by Ricardo Augusto Tavares Santos are Copyright (C) 2012
# All Rights Reserved.

# Contributor(s): ______________________________________.
###############################################################

"""
This module was built to give the user a simple but reliable tool to simulate or
to understand main parameters used to design a infrared photodetector.

All the work done in this module was based in classical equations found in the
literature but the used references are:

See the __main__ function for examples of use.


The code in this file does not conform to the minimum quality standards set out in pyradi
but it is offered here because its value, and coding standards should not prevent  release.
The user will have to expend some effort to make it work in his situation.

This module uses the CODATA physical constants. For more details see
http://physics.nist.gov/cuu/pdf/RevModPhysCODATA2010.pdf

References:

[1] Infrared Detectors and Systems, E L Dereniak & G D Boreman, Wiley.

The example suggested here uses InSb parameters found in the literature. For every
compound or material, all the parameters, as well as the bandgap equation must be
changed.
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['QuantumEfficiency','Irradiance', 'Photocurrent', 'IXV', 'Detectivity', 'NEP', 'Noise', 'Idark', 'Responsivity']

from scipy.constants import codata
import scipy.constants as const
import matplotlib.pyplot as plt
import numpy as np
import pyradi.ryplot as ryplot



################################################################################
#
def QuantumEfficiency(E,Eg,lx,T_detector,lambda_vector,theta1,a0,a0p):
    """
    Calculate the spectral quantum efficiency (QE) and absorption coefficient
    for a semiconductor material with given values

    Args:
        | E: energy in Ev;
        | Eg: bandgap energy in Ev;
        | lx: detector thickness in m;
        | T_detector: detector's temperature in K;
        | lambda_vector: wavelength in m.
        | a0: absorption coefficient in cm-1 (Dereniak Eq 3.5 & 3.6)
        | a0p:  absorption coefficient in cm-1 (Dereniak Eq 3.5 & 3.6)

    Returns:
        | a_vector: spectral absorption coefficient in cm-1
        | etha_vector: spectral quantum efficiency
    """

    # CALCULATING THE SEMICONDUCTOR'S OPTICAL REFLECTANCE
    theta2=np.arcsin(np.sin(theta1)*n1/n2)         # Snell's equation
    # Reflectance for perpendicular polarization
    RS=np.abs((n1*np.cos(theta1)-n2*np.cos(theta2))/(n1*np.cos(theta1)+n2*np.cos(theta2)))**2
    # Reflectance for parallel polarization
    RP=np.abs((n1*np.cos(theta2)-n2*np.cos(theta1))/(n1*np.cos(theta1)+n2*np.cos(theta2)))**2
    R=(RS+RP)/2

    a_vector=[]
    etha_vector=[]
    for i in range(0,np.size(lambda_vector)):      # Calculating the absorption coefficient and QE
        if E[i]>Eg:
            a=(a0 * np.sqrt(E[i]-Eg))+a0p
            a_vector=np.r_[a_vector,a]             # Absorption coef - eq. 3.5 - [1]
            etha1=(1-R)*(1-np.exp(-a*lx))           # QE - eq. 3.4 - [1]
            etha_vector=np.r_[etha_vector,etha1]
        else:
            a=a0p*np.exp((E[i]-Eg)/(const.k*T_detector))   # Absorption coef - eq. 3.6- [1]
            a_vector=np.r_[a_vector,a]
            etha1=(1-R)*(1-np.exp(-a*lx))
            etha_vector=np.r_[etha_vector,etha1]

    return (a_vector,etha_vector)


################################################################################
#
def Responsivity(lambda_vector,E,Eg,lx,T_detector, etha_vector):
    """
    Responsivity quantifies the amount of output seen per watt of radiant optical
    power input [1]. But, for this application it is interesting to define spectral
    responsivity that is the output per watt of monochromatic radiation. This is
    calculated in this function [1].

    Args:
        | lambda_vector: wavelength in m;
        | E: energy in Ev;
        | Eg: bandgap energy in Ev;
        | lx: detector thickness in m;
        | T_detector: detector's temperature in K;
        | lambda_vector: wavelength in m.
        | etha_vector: spectral quantum efficiency

    Returns:
        | (Responsivity_vector)

    """

    Responsivity_vector=[]
    for i in range(0,np.size(lambda_vector)):
        Responsivity_vector1=(const.e*lambda_vector[i]*etha_vector[i])/(const.h*const.c)              #  responsivity model from dereniak's book eq. 7.114
        Responsivity_vector=np.r_[Responsivity_vector,Responsivity_vector1]

    return (Responsivity_vector)

################################################################################
#
def Detectivity(lambda_vector,E,Eg,lx,T_detector,A_det,delta_f,I_noise_dereniak, Responsivity_vector):
    """
    Detectivity can be interpreted as an SNR out of a detector when 1 W of radiant
    power is incident on the detector, given an area equal to 1 cm2 and noise-
    equivalent bandwidth of 1 Hz. The spectral responsivity is the rms signal-to-
    noise output when 1 W of monochromatic radiant flux is incident on 1 cm2
    detector area, within a noise-equivalent bandwidth of 1 Hz. Its maximum value
    (called the peak spectral D*) corresponds to the largest potential SNR.

    Args:
        | lambda_vector: wavelength in m;
        | E: energy in Ev;
        | Eg: bandgap energy in Ev;
        | lx: detector thickness in m;
        | T_detector: detector's temperature in K;
        | A_det: detector's area in m2;
        | delta_f: measurement or desirable bandwidth - Hertz;
        | I_noise_dereniak: noise prediction using Dereniaki's model in A.
        | etha_vector: spectral quantum efficiency

    Returns
        | (Detectivity_vector)

    """

    Detectivity_vector=[]
    for i in range(0,np.size(lambda_vector)):
        Detectivity_vector1=(Responsivity_vector[i]*np.sqrt(A_det*delta_f))/(I_noise_dereniak)       #I_noise_rogalski - Callie's model
        Detectivity_vector=np.r_[Detectivity_vector,Detectivity_vector1]

    return (Detectivity_vector)


################################################################################
#
def NEP(lambda_vector,E,Eg,lx,T_detector,A_det,delta_f,I_noise_dereniak,Detectivity_vector):
    """
    NEP is the radiant power incident on detector that yields SNR=1 [1].

    Args:
        | lambda_vector: wavelength in m;
        | E: energy in Ev;
        | Eg: bandgap energy in Ev;
        | lx: detector thickness in m;
        | T_detector: detector's temperature in K;
        | A_det: detector's area in m2;
        | delta_f: measurement or desirable bandwidth - Hertz;
        | I_noise_dereniak: noise prediction using Dereniaki's model in A.
        | Detectivity_vector: spectral detectivity

    Returns
        | NEPower(spectral_nep)

    """

    NEPower=[]
    lambda_vector2=[]
    for i in range(0,np.size(lambda_vector)):
        if Detectivity_vector[i]>0:
           NEPower1=1/Detectivity_vector[i]
           NEPower=np.r_[NEPower,NEPower1]
           lambda_vector2=np.r_[lambda_vector2,lambda_vector[i]]
        else:
            break

    return (NEPower,lambda_vector2)


################################################################################
# CALCULATING THE RADIANCE FROM A SOURCE AND FROM THE BACKGROUND
def Irradiance(epsilon,sigma_photon,T_source,T_bkg,lambda_vector,A_source,A_bkg):
    """
    This module calculates the quantity of energy produced by a source and the
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
        | epsilon: source emissivity (non-dimensional);
        | sigma_photon: boltzmann constant for photons in photons/(s.m2.K3);
        | T_source: source's temperature in K;
        | T_kbg: background's temperature in K;
        | lambda_vector: wavenlength in m;
        | A_source: soource's area in m2;
        | A_bkg: background's area in m2.

    Returns:
        | tuple of numpy arrays: (Detector_irradiance_source,Detector_irradiance_bkg,Detector_total)

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
    Detector_irradiance_source=(Radiance_final_source*A_source)/(d**2)     # photons/m2

    Detector_irradiance_bkg=(Radiance_final_bkg*A_bkg)/(d**2)#;epsilon,sigma_photon,T_source,T_bkg,lambda_vector,A_source,A_bkg

    Detector_irradiance_total=Detector_irradiance_source+Detector_irradiance_bkg;

    #Irradiance_vector=np.r_[Detector_irradiance_source,Detector_irradiance_bkg,Detector_irradiance_total]

    return (Detector_irradiance_source,Detector_irradiance_bkg,Detector_irradiance_total)


################################################################################
#
def Photocurrent(epsilon,sigma_photon,T_source,T_bkg,A_source,A_bkg,A_det,etha2,avg_qe):
    """
    The photocurrent is the the current generated by a photodetector given its
    quantum efficiency, irradiance and area.

    The result is given in current or tension (dependant on the transipedance used
    in the calculation or measurement)

    Args:

        | E: energy in Ev;
        | Eg: bandgap energy in Ev;
        | lx: detector thickness in m;
        | T_detector: detector's temperature in K;
        | lambda_vector: wavenlength in m;
        | epsilon: source emissivity (non-dimensional);
        | sigma_photon: boltzmann constant for photons in photons/(s.m2.K3);
        | T_source: source´s temperature in K;
        | T_kbg: background´s temperature in K;
        | A_source: soource´s area in m2;
        | A_bkg: background´s area in m2.
        | A_det: detector´s area in m2;
        | Etha: quantum efficiency
        | etha2: average theoretical QE given by the literature;

    Returns:
        | Photocurrent_vector(I1_wide,I1_wide_thoeretical,V1_wide)
    """

    Irradiance_vector=Irradiance(epsilon,sigma_photon,T_source,T_bkg,lambda_vector,A_source,A_bkg)

    Detector_irradiance_total=Irradiance_vector[2]

    Detector_irradiance_bkg=Irradiance_vector[1]

    I1_wide=avg_qe*Detector_irradiance_total*A_det*const.e      # Photocurrent - eq. 3.10 - Infrared Detectors and Systems - Dereniak and Boreman

    I1_wide_theoretical=etha2*Detector_irradiance_total*A_det*const.e

    I1_bkg=avg_qe*Detector_irradiance_bkg*A_det*const.e        # Photocurrent - eq. 3.10 - Infrared Detectors and Systems - Dereniak and Boreman

    I1_bkg_theoretical=etha2*Detector_irradiance_bkg*A_det*const.e      # Photocurrent - eq. 3.10 - Infrared Detectors and Systems - Dereniak and Boreman

    V1_wide=np.mean(transipedance*I1_wide)

    return (I1_wide,I1_wide_theoretical,I1_bkg,I1_bkg_theoretical)


################################################################################
# IxV Characteristic Calculation
def IXV(e_mob,tau_e,me,mh,na,V,b,alfa,B,E,Eg,lx,T_detector,lambda_vector,epsilon,sigma_photon,T_source,T_bkg,A_source,A_bkg,A_det,etha2):
    """
    This module provides the diode curve for a given irradiance.

    Args:
        | e_mob: electron mobility in m2/V.s;
        | tau_e: electron lifetime in s;
        | me: electron effective mass in kg;
        | mh: hole effective mass in kg;
        | na: dopping concentration in m-3;
        | V: bias in V;
        | b: diode equation non linearity factor;
        | alfa: first fitting parameter for the Varshini's Equation
        | B: second fitting parameter for the Varshini's Equation
        | Eg: energy bandgap in Ev;
        | lx: detector thickness in m;
        | T_detector: detector's temperature in K;
        | lambda_vector: wavenlength in m;
        | epsilon: source emissivity (non-dimensional);
        | sigma_photon: boltzmann constant for photons in photons/(s.m2.K3);
        | T_source: source's temperature in K;
        | T_kbg: background's temperature in K;
        | A_source: soource's area in m2;
        | A_bkg: background's area in m2.
        | A_det: detector's area in m2;
        | Etha: quantum efficiency
        | etha2: average theoretical QE given by the literature;

    Returns:
        | (IXV_vector1,IXV_vector2,I0_dereniak,I0_rogalski)

    """

    # SATURATION REVERSE CURRENT CALCULATION
    # note: in this procedure the bandgap calculation must be a specific equation for the used semiconductor. It means, for each different semiconductor material used the equation must be changed.

    Photocurrent_vector=Photocurrent(epsilon,sigma_photon,T_source,T_bkg,A_source,A_bkg,A_det,etha2,avg_qe)
    I_bkg1=Photocurrent_vector[2]
    I_bkg2=Photocurrent_vector[3]

    # DIFFUSION lENGTH
    Le=np.sqrt(const.k*T_detector*e_mob*tau_e/const.e)                  # diffusion length - m - calculated using dereniak's book page 250

    ni=(np.sqrt(4*(2*np.pi*const.k*T_detector/const.h**2)**3*np.exp(-(Eg*const.e)/(const.k*T_detector))*(me*mh)**1.5)) # intrinsic carrier concentration - dereniak`s book eq. 7.1 - m-3
    nd=(ni**2/na)                                           # donnors concentration in m-3

    # REVERSE CURRENT USING DERENIAK'S MODEL
    I0_dereniak=A_det*const.e*(Le/tau_e)*nd                       # reverse saturation current - dereniak eq. 7.34 - Ampère


    # REVERSE CURRENT USING ROGALSKI'S MODEL
    De=const.k*T_detector*e_mob/const.e                               # carrier diffusion coefficient - rogalski's book pg. 164
    I0_rogalski=const.e*De*nd*A_det/Le                            # reverse saturation current - rogalski's book eq. 8.118


    # IxV CHARACTERISTIC
    I_vector=[]
    for i in range(0,np.size(lambda_vector)):
        I=I0_dereniak*(np.exp(const.e*V[i]/(b*const.k*T_detector))-1)     # diode equation from dereniak'book eq. 7.23
        I_vector=np.r_[I_vector,I]

    IXV_vector1=I_vector-I_bkg1
    IXV_vector2=I_vector-I_bkg2

    return (IXV_vector1,IXV_vector2,I0_dereniak,I0_rogalski)


################################################################################
# NOISE CALCULUS
def Noise(I0_rogalski,T_detector,A_det,Detector_irradiance_bkg,delta_f,avg_qe,R0,I1_bkg):
    """
    This module calculate the total noise produced in the diode using the models
    given in the references.

    Args:
        | I0_rogalski: reverse saturation current in A;
        | T_detector: detector's temperature in K;
        | A_det: detector's area in m2;
        | Detector_irradiance_bkg: irradiance generated by the background in W/m2;
        | delta_f: measurement or desirable bandwidth - Hertz;
        | avg_etha: calculated average quantum efficiency;
        | R0: resistivity in Ohm;
        | I1_bkg: photocurrent generated by the background in A.

    Returns:
        | (I_noise_dereniak,I_noise_rogalski)

    """


    # TOTAL NOISE MODELING FROM DERENIAK'S BOOK
    # JOHNSON NOISE
    i_johnson=np.sqrt(4*const.k*T_detector*delta_f/R0)   # Dereniaki's book - eq. 5.58

    # SHOT NOISE
    i_shot=np.sqrt(2*const.e*I1_bkg*delta_f)          # Dereniaki's book - eq. 5.69

    # TOTAL NOISE
    I_noise_dereniak=np.sqrt(i_johnson**2+i_shot**2)    # Dereniaki's book - eq. 5.75


    # % TOTAL NOISE MODELING FROM ROGALSKI'S BOOK (V=0)
    R1=const.k*T_detector/(const.e*I0_rogalski)

    I_noise_rogalski=np.sqrt((2*const.e**2*avg_qe*Detector_irradiance_bkg*A_det*delta_f)+(4*const.k*T_detector*delta_f/R1)) # Rogalski's book - eq. 8.111  -> the amount generated by the background was desconsidered.

    return (I_noise_dereniak,I_noise_rogalski)


################################################################################
## DARK CURRENT CALCULUS
def Idark(I0_dereniak,I0_rogalski,V,T_detector):
    """
    This module calculates the dark current from a photodiode in order to predict
    if the detector is submited to work under BLIP or not.

    Args:
        | I0_derinak: saturation reverse current from Dereniaki's model in A;
        | I0_rogalski: saturation reverse current from Rogalski's model in A;
        | V: applied bias in V;
        | T_detector: detector's temperature in K

    Returns:
        | (I_dark_derinak,I_dark_rogalski)
    """

    I_dark_dereniak=I0_dereniak*(np.exp(const.e*V/(1*const.k*T_detector))-1)
    I_dark_rogalski=I0_rogalski*(np.exp(const.e*V/(1*const.k*T_detector))-1)

    return (I_dark_dereniak,I_dark_rogalski)


################################################################################
if __name__ == '__init__':
    pass

if __name__ == '__main__':
    pass

    # OPERATIONAL PARAMETERS

    """
    In this step, all the parameters referring to the semiconductor material used
    to build the photodetector must be defined. Must be remembered that each material
    has its own paramenters, and each time the material is changed, the parameters
    must be changed as well.
    In the same step, the important classical constants and semiconductors basic
    parameters are also calculated and defined.
    Finally, there is loop to establish if the radiance comes from a source or from
    only the background.
    """

    T_source=0.1                             #source temperature in K
    T_bkg=280.0                              #background temperature in K
    T_detector=80.0                          # detector temperature in K
    lambda_initial=1e-6                      # wavelength in meter- can start in 0
    lambda_final=5.5e-6                      # wavelength in meter
    A_det=(200e-6)**2                        # detector area in m2
    A_source=0.000033                        # source area in m2
    A_bkg=2*np.pi*(0.0055)**2                # bkg area in m2 - this area must be considered equals to the window area
    d=0.01                                   # distance between source and detector or between window and detector
    delta_f=100.0                            # measurement or desirable bandwidth - Hertz
    alfa=6e-4                                # first fitting parameter for the Varshini's Equation [3]
    B=500.0                                  # second fitting parameter for the Varshini's Equation [3]
    n1=1.0                                   # refraction index of the air
    theta1=0.01                              # radiation incident angle in degrees - angle between the surface's normal and the radiation path
    lx=5e-4                                  # detector thickness in meter
    transipedance=10e7                       # transimpedance value used during the measurement

    #material properties for InSb
    etha2=0.45                               # quantum efficieny table 3.3 dereniak's book [3]
    E0=0.24                                  # semiconductor bandgap at room temp in Ev [3]
    nr=3.42                                  # refraction index of the semiconcuctor being simulated [3]
    n2=3.42                                  # refraction index of the semiconductor being analyzed [3]
    a0=1.9e4                                 #absorption coefficient , Equation 3.5 & 3.6 Dereniak
    a0p=800                                  #absorption coefficient , Equation 3.5 & 3.6 Dereniak
    e_mob=120.0                              # electron mobility - m2/V.s [3]
    h_mob=1.0                                # hole mobility - m2/V.s  [3]
    tau_e=1e-10                              # electron lifetime - s [3]
    tau_h=1e-6                               # hole lifetime - s [3]
    m0=9.11e-31                              # electron mass - kg [3]
    me=0.014*m0                              # used semiconductor electron effective mass [3]
    mh=0.43*m0                               # used semiconductor hole effective mass [3]
    na=1e16                                  # positive or negative dopping - m-3
    b=1.0                                    # b=1 when the diffusion current is dominantand b=2 when the recombination current dominates - Derinaki's book page 251
    s=5e4                                    # surface recombination velocity -> http://www.ioffe.ru/SVA/NSM/Semicond/InSb/electric.html#Recombination
    R0=1e10                                  # measured resistivity  - ohm

    # IMPORTANT CONSTANTS
    sigma = codata.value(u'Stefan-Boltzmann constant')   #W/(m2 K4)
    sigma_photon=1.52e15                     # boltzmann constant for photons- photons/(s.m2.K3)
    epsilon=1.0                                # source emissivity

    if T_source> T_bkg:
        r=np.sqrt(A_source/np.pi)            # source radius if it is a circle and plane source
    else:
        r=np.sqrt(A_bkg/np.pi)               # source radius if it is a circle and plane source


    # DEFINING THE WAVELENGTH VECTOR
    lambda_vector=np.linspace(lambda_initial,lambda_final,1000)

    # OPTICS TRANSMITTANCE PLUS THE ATMOSPHERIC TRANSMITTANCE
    # IN THIS CASE THE SPECTRAL TRANSMITTANCE OR THE AVERAGE TRANSMITTANCE VALUE MUST
    #BE USED OR ASSUMED EQUALS TO 1 IF IT IS DESCONSIDERED

    final_trans=1.0

    # DEFINIG THE BIAS TO BE USED IN THE SIMULATIONS (IF NECESSARY)

    V=np.linspace(-250e-3,100e-3,np.size(lambda_vector))

    # CALCULATING THE FREQUENCY RANGE GIVEN THE WAVELENGTH'S RANGE AND CHANGING IT TO
    # ENERGY

    f=const.c/lambda_vector                              # frequency in Hz
    E=const.h*f                                          # Einstein's equation in Joules
    E=E/const.e                                          # Energy in Ev

    # CALCULATING THE SEMICONDUCTOR BANDGAP
    # IT IS IMPORTANT TO NOTICE THAT FOR EACH SEMICONDUCTOR BANGAP CALCULATED HERE
    # THE EQUATION AS WELL AS IT PARAMETERS MUST BE CHANGED

    Eg=(E0-(alfa*(T_detector**2/(T_detector+B))))  # Varshini's Equation to calculate the bandgap dependant on the temp - eV [3]

    ######################################################################

    (a_vector,etha_vector) = QuantumEfficiency(E,Eg,lx,T_detector,lambda_vector,theta1,a0,a0p)

    avg_qe=np.mean(etha_vector)

    (Detector_irradiance_source,Detector_irradiance_bkg,Detector_irradiance_total) = Irradiance(epsilon,sigma_photon,T_source,T_bkg,lambda_vector,A_source,A_bkg)

    (I1_wide,I1_wide_theoretical,I1_bkg,I1_bkg_theoretical)=Photocurrent(epsilon,sigma_photon,T_source,T_bkg,A_source,A_bkg,A_det,etha2,avg_qe)

    (IXV_vector1,IXV_vector2,I0_dereniak,I0_rogalski)=IXV(e_mob,tau_e,me,mh,na,V,b,alfa,B,E,Eg,lx,T_detector,lambda_vector,epsilon,sigma_photon,T_source,T_bkg,A_source,A_bkg,A_det,etha2)

    (I_noise_dereniak,I_noise_rogalski)=Noise(I0_rogalski,T_detector,A_det,Detector_irradiance_bkg,delta_f,avg_qe,R0,I1_bkg)

    (I_dark_dereniak,I_dark_rogalski)=Idark(I0_dereniak,I0_rogalski,V,T_detector)

    Responsivity_vector = Responsivity(lambda_vector,E,Eg,lx,T_detector,etha_vector)

    Detectivity_vector = Detectivity(lambda_vector,E,Eg,lx,T_detector,A_det,delta_f,I_noise_dereniak,Responsivity_vector)
    Detectivity_vector=Detectivity_vector*1e2       # units in cm

    (NEPower,lambda_vector2)=NEP(lambda_vector,E,Eg,lx,T_detector,A_det,delta_f,I_noise_dereniak,Detectivity_vector)


    print ("Detector_irradiance_source=  {0}".format(Detector_irradiance_source))
    print ("Detector_irradiance_bkg= {0} ".format(Detector_irradiance_bkg))
    print ("Detector_irradiance_total= {0} ".format(Detector_irradiance_total))
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

    IXV = ryplot.Plotter(1,1,1)
    IXV.plot(1,V,IXV_vector1,plotCol=['b'])
    IXV.plot(1,V,IXV_vector2,'IxV Curve',\
        'Voltage [V]','Current [A]',plotCol=['r'])
    IXV.saveFig('IXV.eps')

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
    NEPf.plot(1,lambda_vector2*1e6,NEPower,'Spectral Noise Equivalent Power',\
        r'Wavelength [$\mu$m]','NEP [W]')
    NEPf.saveFig('NEP.eps')


#    plt.show()


    print('Done!')


