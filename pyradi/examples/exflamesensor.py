#  $Id$
#  $HeadURL$

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

# The Initial Developer of the Original Code is CJ Willers,
# Portions created by CJ Willers are Copyright (C) 2006-2012
# All Rights Reserved.

# Contributor(s): ______________________________________.
################################################################
"""
This file provides a simple worked example of a small sensor observing a flame
on a smokestack.

The objective is to calculate the signal of a simple sensor, detecting the
presence or absence of a flame in the sensor field of view. The sensor is
pointed to an area just outside a furnace vent, against a clear sky
background. The sensor must detect a change in signal indicating the presence
of a flame at the vent.

The sensor has an aperture area of 7.8e-3 m^2 and a field of view of 1e-4 sr.
The InSb or PbSe detector has a peak responsivity of 2.5 A/W and spectral
response The preamplifier transimpedance is 10000 V/A.

The flame area is 1 m^2, the flame temperature is 1000 C. The emissivity is
0.1 over most of the spectral band, due to carbon particles in the flame. At
4.3 um there is a strong emissivity rise due to the hot CO2 in the flame.

The distance between the flame and the sensor is 1000~m. We use the  Modtran
Tropical climatic model. The path is oriented such that the sensor stares out to space,
at a zenith angle of 88.8 deg

The peak in the flame emissivity and the dip in atmospheric transmittance are
both centered around the 4.3 um CO2 band. In order to determine the flux
transferred one must perform a spectral calculation taking these strong
spectral variations into account.
"""

if sys.version_info[0] > 2:
    pass
else:
    from __future__ import division
    from __future__ import print_function
    from __future__ import unicode_literals

__version__= "$Revision$"
__author__='CJ Willers'

import numpy
import pyradi.ryfiles as ryfiles
import pyradi.ryplot as ryplot
import pyradi.ryplanck as ryplanck
import pyradi.ryutils as ryutils

#this example is somewhat contrived, but serves to show toolkit use

#load atmospheric transmittance from file created in Modtran in wavenumbers
# the transmittance is specified in the wavenumber domain with
# 5 cm-1 intervals, but we want to work in wavelength with 2.5 cm-1
waven = numpy.arange(2000.0,  3300.0,  2.5).reshape(-1, 1)
wavel= ryutils.convertSpectralDomain(waven,  type='nl')

#remove comment lines, and scale path radiance from W/cm2.sr.cm-1  to W/m2.sr.cm-1
tauA = ryfiles.loadColumnTextFile('../data/path1kmflamesensor.txt',
    [1],abscissaOut=waven,  comment='%')
lpathwn = ryfiles.loadColumnTextFile('../data/pathspaceflamesensor.txt',
    [9],abscissaOut=waven,  ordinateScale=1.0e4,  comment='%')

#convert path radiance spectral density from 1/cm^-1 to 1/um, at the sample
#wavenumber points
(dum, lpathwl) = ryutils.convertSpectralDensity(waven,  lpathwn, type='nl')

#load the detector file in wavelengths, and interpolate on required values
detR = ryfiles.loadColumnTextFile('../data/detectorflamesensor.txt',
    [1],abscissaOut=wavel,  comment='%')

#construct the flame emissivity from parameters
emis = ryutils.sfilter(wavel,center=4.33, width=0.45, exponent=6, taupass=0.8,
    taustop=0.1 )

#construct the sensor filter from parameters
sfilter = ryutils.sfilter(wavel,center=4.3, width=0.8, exponent=12,
taupass=0.9, taustop=0.0001)

#plot the data
plot1= ryplot.Plotter(1, 2, 2,'Flame sensor',figsize=(24,16))


plotdata = detR
plotdata = numpy.hstack((plotdata,emis))
plotdata = numpy.hstack((plotdata,sfilter))
plotdata = numpy.hstack((plotdata,tauA))
label = ['Detector','Emissivity','Filter','Atmosphere transmittance']
plot1.plot(1, wavel, plotdata, "Spectral","Wavelength [$\mu$m]",
    "Relative magnitude",
    label = label, maxNX=10, maxNY=10)

#check path radiance against Planck's Law for atmo temperature
LbbTropical = ryplanck.planck(wavel, 273+27, type='el').reshape(-1, 1)/numpy.pi
plotdata = LbbTropical
plotdata = numpy.hstack((plotdata,lpathwl))
label=['300 K Planck Law','Tropical path radiance']
plot1.plot(2, wavel, plotdata, label = label)

currentP = plot1.getSubPlot(2)
currentP.set_xlabel('Wavelength [$\mu$m]')
currentP.set_ylabel('Radiance [W/(m$^2$.sr.$\mu$m)]')
currentP.set_title('Path Radiance')

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


# do case path
inbandirradiancePath =  lpathwn * detR * sfilter * opticsFOV
totalirradiancePath = numpy.trapz(inbandirradiancePath.reshape(-1, 1),waven, axis=0)[0]
signalPath = totalirradiancePath * transZ*responsivity *opticsArea

# do case flame
# get spectral radiance in  W/m^2.sr.cm-1
radianceFlame = ryplanck.planck(waven, flameTemperature,  type='en')\
    .reshape(-1, 1)/numpy.pi
inbandirradianceFlame = radianceFlame * detR * tauA * emis * sfilter *\
    fill * opticsFOV
totalirradianceFlame = numpy.trapz(inbandirradianceFlame.reshape(-1, 1),
    waven, axis=0)[0]
totalirradiancePathremainder = totalirradiancePath * (1-fill)
signalFlameOnly = totalirradianceFlame *transZ*responsivity *opticsArea
signalFlame = (totalirradianceFlame + totalirradiancePathremainder ) *\
    transZ*responsivity *opticsArea


print('Optics    : area={0} m^2 FOV={1} [sr]'.format(opticsArea, opticsFOV ))
print('Amplifier : gain={0} [V/A]'.format(transZ))
print('Detector  : peak responsivity={0} [A/W]'.format(responsivity))
print('Flame     : temperature={0} [K] area={1} [m^2] distance={2} [m]'.\
    format(flameTemperature, flameArea, distance))
print('          : fill={0} [-]'.format(fill))
print('Flame only: irradiance={0:9.2e} [W/m^2] signal={1:7.4f} [V]'.\
    format(totalirradianceFlame, signalFlameOnly))
print('Path      : irradiance={0:9.2e} [W/m^2] signal={1:7.4f} [V]'.\
    format(totalirradiancePath, signalPath))
print('Flame+Path: irradiance={0:9.2e} [W/m^2] signal={1:7.4f} [V]'.\
    format(totalirradianceFlame + totalirradiancePathremainder , \
        signalFlame))

(dum, iFlamewl) = ryutils.convertSpectralDensity(waven,  inbandirradianceFlame, type='nl')
(dum, iPathwl) = ryutils.convertSpectralDensity(waven,  inbandirradiancePath, type='nl')
plot1.plot(3, wavel, iFlamewl, "Irradiance","Wavelength [$\mu$m]",
    "Iradiance [W/(m$^2$.$\mu$m)]",plotCol=['r'], label=['Flame'])
plot1.plot(3, wavel, iPathwl, "Irradiance","Wavelength [$\mu$m]",
    "Iradiance [W/(m$^2$.$\mu$m)]",plotCol=['b'], label=['Path'])
plot1.plot(4, waven, inbandirradianceFlame, "Irradiance","Wavenumber [cm$^{-1}$]",
    "Irradiance [W/(m$^2$.cm$^{-1}$)]",plotCol=['r'], label=['Flame'])
plot1.plot(4, waven, inbandirradiancePath, "Irradiance","Wavenumber [cm$^{-1}$]",
    "Irradiance [W/(m$^2$.cm$^{-1}$)]",plotCol=['b'], label=['Path'], maxNX=10, maxNY=10)

plot1.saveFig('flamesensor01.eps')

