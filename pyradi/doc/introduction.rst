Introduction 
****************************

.. include global.rst


Overview
--------


Electro-optical system design, data analysis and modelling involve a significant 
amount of calculation and processing. Many of these calculations are of a 
repetitive and general nature, suitable for including in a generic toolkit. 
The availability of such a toolkit facilitates and increases productivity 
during subsequent tool development: 'develop once and use many times'. The 
concept of an extendible toolkit lends itself naturally to the open-source 
philosophy, where the toolkit user-base develops the capability cooperatively, 
for mutual benefit. This paper covers the underlying philosophy to the toolkit 
development, brief descriptions and examples of the various tools and an 
overview of the electro-optical toolkit.

The pyradi toolbox can be applied towards many different applications. An example
is included in the pyradi website (see the file exflamesensor.py). This example 
was first published in a SPIE conference paper [SPIE8543Pyradi]_.


Toolkit approach
----------------


The development of this toolkit is following the Unix philosophy for software 
development, summarised in the words of Doug McIlroy: 'Write programs that do 
one thing and do it well. Write programs to work together.' In broader terms the
philosophy was stated by Eric Raymond, but only selected items shown here
(http://en.wikipedia.org/wiki/Unix_philosophy): 

1. Rule of Modularity: Write simple parts connected by clean interfaces. 
2. Rule of Clarity: Clarity is better than cleverness. 
3. Rule of Composition: Design programs to be connected to other programs. 
4. Rule of Simplicity: Design for simplicity; add complexity only where you must. 
5. Rule of Parsimony: Write a big program only when it is clear by demonstration 
   that nothing else will do. 
6. Rule of Transparency: Design for visibility to make inspection and 
   debugging easier. 
7. Rule of Robustness: Robustness is the child of transparency and simplicity. 
8. Rule of Representation: Fold knowledge into data so program logic can 
   be stupid and robust.
9. Rule of Economy: Programmer time is expensive; conserve it in preference 
   to machine time. 
10. Rule of Generation: Avoid hand-hacking; write programs to write programs 
    when you can. 
11. Rule of Optimisation: Prototype before polishing. Get it working before 
    you optimise it. 
12. Rule of Extensibility: Design for the future, because it will be here sooner 
    than you think.


Example application 
--------------------

A typical radiometry toolkit requirement (very much simplified) is the calculation
of the detector current of an electro-optical sensor viewing a target object. 
The system can be conceptually modelled as shown in the figure below, 
comprising a radiating source with 
spectral radiance, an intervening medium (e.g. the atmosphere), a spectral filter, 
optics, a detector and an amplifier. 

.. image:: _images/simplesystem.png
    :width: 812px
    :align: center
    :height: 244px
    :alt: pyradi logo
    :scale: 50 %

The amplifier output signal
can be calculated in the following equation,  by integrating  over 
all wavelengths, over the full source area :math:`A_0` and over the optical 
aperture area :math:`A_1`,

.. math::
 v=
 Z_t 
 \int_{A_0}
 \int_{A_1}
 \frac{1}{r_{01}^2}
 \int_0^\infty
 \epsilon_\lambda L_\lambda(T,A_0)\tau_{a\lambda}\tau_{s\lambda}(A_1){\cal R}_\lambda
 \;d\lambda
 \;d(\cos\theta_0 A_0)
 \;d(\cos\theta_1 A_1)


where
:math:`v` is the output signal voltage,
:math:`r_{01}` is the distance between elemental areas 
:math:`d(\cos\theta_0 A_0)` and 
:math:`d(\cos\theta_1 A_1)`,
:math:`\epsilon_\lambda` is the source spectral emissivity,
:math:`L_\lambda(T,A_0)` is the Planck Law radiation at temperature 
:math:`T` at location :math:`A_0`,
:math:`\tau_{a\lambda}` is the atmospheric spectral transmittance,
:math:`\tau_{s\lambda}(A_1)` is the sensor spectral transmittance at location :math:`A_1`,
:math:`{\cal R}_\lambda` is the spectral detector responsivity in [A/W],
:math:`Z_t` is the amplifier transimpedance gain in [V/A]. 
The spectral integral :math:`\int_0^\infty d\lambda` accounts for the total 
flux for all wavelengths, the spatial integral 
:math:`\int_{A_0}d(\cos\theta_0 A_0)`
accounts for flux over the total area of the source, and 
the spatial integral 
:math:`\int_{A_1}d(\cos\theta_1 A_1)` accounts for the total area of the receiving area.

The top graphic in the following figure illustrates the 
reasoning behind the spectral integral as a product, followed by an integral (summation),

.. image:: _images/multispectral.png
    :width: 797px
    :align: center
    :height: 584px
    :alt: pyradi logo
    :scale: 50 %
   
.. math::
 \int_0^\infty
 \epsilon_\lambda L_\lambda(T)\tau_{a\lambda}\tau_{s\lambda}{\cal R}_\lambda
 \;d\lambda,
 
where the spectral variability of the  source, medium and sensor parameters 
are multiplied as spectral variables and afterwards integrated over all wavelengths 
to yield the total in-band signal. The domain of spectral quantities can be 
stated in terms of a wavelength, wavenumber, or less often, temporal frequency. 

Likewise, the source radiance is integrated over the two respective areas of the 
target :math:`A_0`, and the sensor aperture :math:`A_1`.  Note that if the 
sensor field of  view footprint at the source is smaller than the physical 
source area, only the flux emanating from the footprint area is integrated.


This example is a relatively complete worked example. The objective is to 
calculate the signal of a simple sensor, detecting the presence or absence of 
a flame in the sensor field of view. The sensor is pointed to an area just 
outside a furnace smokestack, against a clear sky background. The sensor 
must detect a change in signal, to indicate the presence or absence of a flame.

The sensor has an aperture area of :math:`7.8 \times 10^{-3}` :math:`{\rm m}^2` 
and  a field of view of :math:`1 \times 10^{-4}` sr. The sensor filter spectral 
transmittance is shown below. The InSb detector has a peak responsivity of 2.5 
A/W and normalised spectral response shown below. The preamplifier transimpedance 
is 10000 V/A. 

The flame area is  1 :math:`{\rm m}^2`, the flame temperature is 
:math:`1000^\circ` C, and the emissivity is shown below. The emissivity 
is 0.1 over most of the spectral band, due to carbon particles in the flame. 
At 4.3 :math:`\mu{\rm m}` there is a strong emissivity rise due to the hot 
carbon dioxide :math:`{\rm CO}_2` in the flame.

The distance between the flame and the sensor is 1000~m. The atmospheric 
properties are calculated with the Modtran Tropical climatic model. The 
path is oriented such that the sensor stares out to space, at a zenith angle 
of :math:`88^\circ`. The spectral transmittance and path radiance along this 
path is shown in below.

The peak in the flame emissivity and the dip in atmospheric transmittance are 
both centered around the :math:`4.3\mu{\rm m}` :math:`{\rm CO}_2` band. The calculation
of flux 
radiative transfer through the atmosphere must account for the strong spectral 
variation, by using a spectral integral.

The signal caused by the flame is given by the equation above, where the 
integrals over the surfaces of the flame and sensor are just their respective 
areas. The signal caused by the atmospheric path radiance is given by 

.. math::
 v=
 Z_t 
 \omega_{\rm optics}
 A_{\rm optics}
 \int_0^\infty
 L_{{\rm path}\lambda}
 \tau_{s\lambda}{\cal R}_\lambda
 \;d\lambda,


where 
:math:`\omega_{\rm optics}` is the sensor field of view,
:math:`A_{\rm optics}` is the optical aperture area, 
:math:`L_{{\rm path}\lambda}` is the spectral path radiance
and the rest of the symbols are as defined above.


.. image:: _images/flamesensor.png
    :width: 736px
    :align: center
    :height: 540px
    :alt: pyradi logo
    :scale: 70 %
   
The pyradi code to model this sensor is available as exflamesensor.py_.
The output from this script is as follows:

::

 Optics   : area=0.0078 m^2 FOV=0.0001 [sr]
 Amplifier: gain=10000.0 [V/A]
 Detector : peak responsivity=2.5 [A/W]
 Flame    : temperature=1273.16 [K] area=1 [m^2] distance=1000 [m] fill=0.01 [-]
 Flame    : irradiance= 3.29e-04 [W/m^2] signal= 0.0641 [V]
 Path     : irradiance= 5.45e-05 [W/m^2] signal= 0.0106 [V]


It is clear that the flame signal is six times larger than the path radiance
signal, even though the flame fills only 0.01 of the sensor field of view. 


.. [SPIE8543Pyradi] *Pyradi: an open-source toolkit for infrared calculation 
   and data processing*,  SPIE Proceedings Vol 8543, Security+Defence 2011,  
   Technologies for Optical Countermeasures, Edinburgh, 24-27 September, 
   C.J. Willers, M. S. Willers, R.A.T. Santos, P.J. van der Merwe, J.J. Calitz, 
   A de Waal and A.E. Mudau.
   
.. _exflamesensor.py: http://code.google.com/p/pyradi/source/browse/trunk/exflamesensor.py
