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
This file provides a simple worked example of a small sensor observing a flame on a smokestack.

The objective is to 
calculate the signal of a simple sensor, detecting the presence or absence of a flame in the sensor 
field of view. The sensor is pointed to an area just outside a furnace vent, against a clear sky 
background. The sensor must detect a change in signal indicating the presence of a flame at 
the vent.

The sensor has an aperture area of $7.8 \times 10^{-3}$ m$^2$ and a field of view 
of $1 \times 10^{-4}$ sr. The sensor filter spectral transmittance is shown in Figure~\ref{flame1}.
The InSb or PbSe detector has a peak responsivity of 2.5 A/W and spectral response shown in
Figure~\ref{flame1}. The preamplifier transimpedance is 10000 V/A. 

The flame area is  1 m$^2$, the flame temperature is 1000~$^\circ$C, and the emissivity is 
shown in Figure~\ref{flame1}. The emissivity is 0.1 over most of the spectral band, due to 
carbon particles in the flame. At 4.3~$\mu$m there is a strong emissivity rise due to the hot 
CO$_2$ in the flame.

The distance between the flame and the sensor is 1000~m. We use the {\sc Modtran} 
Tropical climatic model. The path is oriented such that the sensor stares out to space, 
at a zenith angle of 88.8~$^\circ$. The spectral transmittance is shown in Figure~\ref{flame1}.
The path radiance along this path is shown in Figure~\ref{flame2}.

The peak in the flame emissivity and the dip in atmospheric transmittance are both 
centered around the 4.3~$\mu$m CO$_2$ band. In order to determine the flux transferred 
one must perform a spectral calculation taking these strong spectral variations into account.
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__='CJ Willers'

import numpy
import ryfiles
import ryplot
import ryplanck
import ryutils

#this example is somewhat contrived, but serves to show toolkit use

#load atmospheric transmittance from file created in Modtran in wavenumbers
# the transmittance is specified in the wavenumber domain with
# 5 cm-1 intervals, but we want to work in wavelength with 2.5 cm-1
waven = numpy.arange(2000.0,  3300.0,  2.5).reshape(-1, 1)
wavel= ryutils.convertSpectralDomain(waven,  type='nw')

#remove comment lines, and scale path radiance from W/cm2.sr.cm-1  to W/m2.sr.cm-1 
tauA = ryfiles.loadColumnTextFile('data/path1kmflamesensor.txt', [1],abscissaOut=waven,  comment='%')
lpathwn = ryfiles.loadColumnTextFile('data/pathspaceflamesensor.txt', [9],abscissaOut=waven,  ordinateScale=1.0e4,  comment='%')
#convert path radiance spectral density from 1/cm^-1 to 1/um, at the sample wavenumber points
(dum, lpathwl) = ryutils.convertSpectralDensity(waven,  lpathwn, type='nw')

#load the detector file in wavelengths, and interpolate on required values
detR = ryfiles.loadColumnTextFile('data/detectorflamesensor.txt', [1],abscissaOut=wavel,  comment='%')

#construct the flame emissivity from parameters
emis = ryutils.sfilter(wavel,center=4.33, width=0.45, exponent=6, taupass=0.8, taustop=0.1 )

#construct the sensor filter from parameters
filter = ryutils.sfilter(wavel,center=4.3, width=0.8, exponent=12, taupass=0.9, taustop=0.0001)

#plot the data
plot1= ryplot.Plotter(1, 2, 2,'Flame sensor',figsize=(12,8))
#it seems that all attempts to plot in same subplot space must use same ptitle.
plot1.plot(1, "Spectral","Wavelength [$\mu$m]", "Relative magnitude", wavel, detR,plotCol=['b'], label=['Detector'])
plot1.plot(1, "Spectral","Wavelength [$\mu$m]", "Relative magnitude", wavel, emis,plotCol=['g'], label=['Emissivity'])
plot1.plot(1, "Spectral","Wavelength [$\mu$m]", "Relative magnitude", wavel, filter,plotCol=['r'], label=['filter'])
plot1.plot(1, "Spectral","Wavelength [$\mu$m]", "Relative magnitude", wavel, tauA,plotCol=['c'], \
       label=['Atmosphere transmittance'], maxNX=10, maxNY=10)
#check path radiance against Planck's Law for atmo temperature
LbbTropical = ryplanck.planck(wavel, 273+27, type='el').reshape(-1, 1)/numpy.pi
plot1.plot(2, "Radiance","Wavelength [$\mu$m]", "Radiance [W/(m$^2$.sr.$\mu$m)]", wavel, LbbTropical,plotCol=['r'], label=['300 K Planck Law'])
plot1.plot(2, "Radiance","Wavelength [$\mu$m]", "Radiance [W/(m$^2$.sr.$\mu$m)]", wavel, lpathwl,plotCol=['b'], label=['Tropical path radiance'])

##########################################################
# define sensor scalar parameters
opticsArea=7.8e-3 # optical aperture area [m2]
opticsFOV=1.0e-4 # sensor field of view [sr]
transZ=1.0e4 # amplifier transimpedance gain [V/A]
responsivity=2.5 # detector peak responsivity =A/W]

# define the  flame properties
flameTemperature = 1000+273.16    # temperature in [K]
flameArea = 1  # in [m2]
distance = 1000  # [m]
fill = (flameArea /distance**2) /  opticsFOV # how much of FOV is filled
fill = 1 if fill > 1 else fill # limit target solid angle to sensor FOV

# first do for flame
# get spectral radiance in  W/m^2.sr.cm-1
radianceFlame = ryplanck.planck(waven, flameTemperature,  type='en').reshape(-1, 1)/numpy.pi
inbandirradianceFlame = radianceFlame * detR * tauA * emis * filter * fill * opticsFOV
totalirradianceFlame = numpy.trapz(inbandirradianceFlame.reshape(-1, 1),waven, axis=0)[0]
signalFlame = totalirradianceFlame *transZ*responsivity *opticsArea

print('Optics   : area={0} m^2 FOV={1} [sr]'.format(opticsArea, opticsFOV ))
print('Amplifier: gain={0} [V/A]'.format(transZ))
print('Detector : peak responsivity={0} [A/W]'.format(responsivity))
print('Flame    : temperature={0} [K] area={1} [m^2] distance={2} [m] fill={3} [-]'.format(flameTemperature, flameArea, distance, fill))
print('Flame    : irradiance={0:9.2e} [W/m^2] signal={1:7.4f} [V]'.format(totalirradianceFlame, signalFlame))

# now do path
inbandirradiancePath =  lpathwn * detR * filter * opticsFOV
totalirradiancePath = numpy.trapz(inbandirradiancePath.reshape(-1, 1),waven, axis=0)[0]
signalPath = totalirradiancePath * transZ*responsivity *opticsArea
print('Path    : irradiance={0:9.2e} [W/m^2] signal={1:7.4f} [V]'.format(totalirradiancePath, signalPath))

(dum, iFlamewl) = ryutils.convertSpectralDensity(waven,  inbandirradianceFlame, type='nw')
(dum, iPathwl) = ryutils.convertSpectralDensity(waven,  inbandirradiancePath, type='nw')
plot1.plot(3, "Irradiance","Wavelength [$\mu$m]", "Iradiance [W/(m$^2$.$\mu$m)]", wavel, iFlamewl,plotCol=['r'], label=['Flame'])
plot1.plot(3, "Irradiance","Wavelength [$\mu$m]", "Iradiance [W/(m$^2$.$\mu$m)]", wavel, iPathwl,plotCol=['b'], label=['Path'])
plot1.plot(4, "Irradiance","Wavenumber [cm$^{-1}$]", "Irradiance [W/(m$^2$.cm$^{-1}$)]", waven, inbandirradianceFlame,plotCol=['r'], label=['Flame'])
plot1.plot(4, "Irradiance","Wavenumber [cm$^{-1}$]", "Irradiance [W/(m$^2$.cm$^{-1}$)]", waven, inbandirradiancePath,plotCol=['b'], \
       label=['Path'], maxNX=10, maxNY=10)

plot1.saveFig('flamesensor01.eps')

