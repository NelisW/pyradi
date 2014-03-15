
---------
Overview
---------

The PyRadi toolkit provides utilities for radiometry (flux flow) calculations, supporting electro-optical and infrared system design.

The toolkit is an extendable, integrated and coherent collection of basic functions, code modules, documentation, example templates, unit tests and resources, that can be applied towards diverse calculations in the electro-optics domain. The toolkit covers:
* Models of physical concepts (e.g. Planck's Law)
  
* Mathematical operations (e.g. spectral integrals, spatial integrals, spectral convolution)
  
* Data manipulation (e.g. file input/output, interpolation, spectral quantity conversions, reading Flir Inc PTW files)
  
* Detector modelling from physical parameters
  
* 3-D noise analysis of image sequences
  
* Modtran5 tape7 read functions
  
* Graphical visualization(2-D and 3-D graphs)
  
* All these in an interactive development environment

The individual scripts in the toolkit is supported by examples, test cases and documentation.

For more information see [SPIE8543Pyradi_].

   
Prerequisites
-------------

This  toolkit requires (current versions in brackets)  
Python (2.7),
Numpy (1.7 or later),
SciPy (0.13 or later),
Matplotlib (1.2 or later).
Mayavi (4.1) is required only for one file to do three-dimensional rendering.


Status
------

This project is *stable beta*. Current content is tested, stable and usable. With time and active use the scope of the pyradi offering will grow and expand. The current version is already quite useful in our labs and lectures.


    
Example application 
-------------------

A typical radiometry toolkit requirement (very much simplified) is the calculation
of the detector current of an electro-optical sensor viewing a target object through the atmosphere. 
The system can be conceptually modelled as  
comprising a radiating source with 
spectral radiance, an intervening medium (e.g. the atmosphere), a spectral filter, 
optics, a detector and an amplifier. The pyradi toolkit provides several classes and
functions to implement this model with minimal code.
An example solution is given in this script_ and is further explained with results in SPIE8543Pyradi_.

    
.. _script: https://code.google.com/p/pyradi/source/browse/trunk/examples/exflamesensor.py

.. [SPIE8543Pyradi] *Pyradi: an open-source toolkit for infrared calculation 
   and data processing*,  SPIE Proceedings Vol 8543, Security+Defence 2011,  
   Technologies for Optical Countermeasures, Edinburgh, 24-27 September, 
   C.J. Willers, M. S. Willers, R.A.T. Santos, P.J. van der Merwe, J.J. Calitz, 
   A de Waal and A.E. Mudau.
   