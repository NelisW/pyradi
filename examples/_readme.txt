This directory contains a number of example studies.

1. exflamesensor.py
====================
This file provides a simple worked example of a small sensor observing a flame
on a smokestack.

The objective is to calculate the signal of a simple sensor, detecting the
presence or absence of a flame in the sensor field of view. The sensor is
pointed to an area just outside a furnace vent, against a clear sky
background. The sensor must detect a change in signal indicating the presence
of a flame at the vent.


2. objectinimage.zip
====================
This script calculates the appearance of a target object in an image,
through the atmosphere, against a background.
Three bands are considered: 0.4-0.75 um, 3.5-4.5 um and 8-12 um.
Different target properties are used in each of the three bands.
In all three cases are the sun irradiance part of the signature,
but it only plays a significant role in the visible band.

This code expects the sun transmittance to be pre-calculated, but
the path radiance and transmittance for the path between the source
and receiver are calculated in this code, using Modtran.
For this purpose, this script must be placed in the same directory
as the modtran executable. This code was tested on Windows.

