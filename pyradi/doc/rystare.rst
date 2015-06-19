Staring Array Module (CCD)
****************************
.. include global.rst

Introduction 
****************************


Overview
--------

This module provides a high level model for CCD and CMOS staring array 
signal chain modelling.  The model accepts an input image in photon rate irradiance units 
and then proceeds to calculate the various noise components and 
signal components along the signal flow chain.

The code in this module serves as an example of implementation of a high-level 
CCD/CMOS photosensor signal chain model. The model is described in the article 
'High-level numerical simulations of noise in solid-state photosensors:  
review and tutorial' by Mikhail Konnik and James Welsh. 
The code was originally written in Matlab and used for the Adaptive Optics 
simulations and study of noise propagation in wavefront sensors, but can be 
used for many other applications involving light registration on CCD/CMOS
photosensors.  The original files are available at:

- Paper: http://arxiv.org/pdf/1412.4031.pdf
- Matlab code: https://bitbucket.org/aorta/highlevelsensorsim

The original Matlab code was ported to Python and extended
in a number of ways.  The core of the model remains the original Konnik model
as implemented in the Matlab code.  The  Python code was validated 
against results obtained with the Matlab code, up to a point 
and then substantially reworked and refactored.  During the refactoring
due diligence was applied with regression testing, checking the new
results against the previous results.

The documentation in the code was copied from Konnik's Matlab code, so 
he deserves all credit for the very detailed documentation.  His documentation 
was extracted from the paper quoted above.

The sample code in the repository models two different cases (from Konnik's code)

- a simple model: which is completely linear (no non-linearities), 
  where all noise are basically Gaussian, and without 
  source follower noise, 
- an advanced model: which has V/V and V/e non-linearities, 
  Wald or lognormal noise, source follower and sense node noise 
  sources and even ADC non-linearities.

The code supports enabling/disabling of key components by using flags.

In the documentation for the Matlab code Konnik expressed the hope "that this 
model will be useful for somebody, or at least save someone's time.
The model can be (and should be) criticized."  Indeed it has, thanks Mikhail!
Konnik quotes George E. P. Box, the famous statistician, and who said that 
"essentially, all models are wrong, but some are useful".


Signal Flow
------------

The process from incident photons to the digital numbers appearing in the 
image is outlined in the picture below. 
First the input image is provided in photon rate irradiance, 
with photon noise already present in the image.  The count of photons 
captured in the detector is determined from the irradiance by accounting 
for the detector area and integration time.
Then, the code models the process of conversion from photons to 
electrons and subsequently to signal voltage. Various noise sources 
are modelled to derive at a realistic image model.
Finally, the ADC converts the voltage signal into digital numbers. 
The whole process is depicted in the figure below.
 
.. image:: _images/camerascheme_horiz.png
    :width: 812px
    :align: center
    :height: 244px
    :alt: camerascheme_horiz.png
    :scale: 100 %

Many noise sources contribute to the resulting noise image that is produced by
the sensor. Noise sources can be broadly classified as either
*fixed-pattern (time-invariant)* or *temporal (time-variant)*
noise. Fixed-pattern noise refers to any spatial pattern that does not change
significantly from frame to frame. Temporal noise, on the other hand, changes
from one frame to the next.  All these noise sources are modelled in the code.
For more details see Konnik's original paper or the docstrings present in the code.

Example Code
-------------

The two examples provided by Konnik are merged into a single code, with flags to 
select between the two options.  The code is found at the end of the module file
in the `__main__` part of the module file.  Set `doTest = 'Simple'` or `doTest = 'Advanced'`
depending on which model. 
Either example will run the `photosensor` function thoroughly documented in the Python code.
The two prepared image files are both 256x256 in size.  New images can be generated
following the example shown  in the `__main__` part of the module file (using the function 
`create_HDF5_image`).

The easiest way to run the code is to open a command window in the installation directory 
and run the `run_example` function in the module code.  This will load the module and 
execute the example code function. This will create files with names similar to 

Towards the end of the code there are several 
commented lines that can be uncommented to create plots and graphs. 

Some time in future an IPython notebook will be released on 
https://github.com/NelisW/ComputationalRadiometry.


HDF5 File
---------

The Python implementation of the model uses an HDF5 file to capture the
input and output data for record keeping or subsequent analysis. 
HDF5 files provide for hierarchical data structures and easy read/save to disk. 
See the file `hdf5-as-data-format.md` for more detail.

Input images are written to and read from HDF5 files as well.  These files store the
image as well as the images' dimensional scaling in the focal plane.  
The intent is to later create test targets with specific spatial 
frequencies in these files.


Example application 
--------------------
todo


Code Overview
---------------
.. automodule:: CcdCmosSim.ccd


Module functions
------------------

.. autofunction:: CcdCmosSim.ccd.photosensor	

.. autofunction:: CcdCmosSim.ccd.source_follower	

.. autofunction:: CcdCmosSim.ccd.cds

.. autofunction:: CcdCmosSim.ccd.adc

.. autofunction:: CcdCmosSim.ccd.sense_node_chargetovoltage

.. autofunction:: CcdCmosSim.ccd.sense_node_reset_noise

.. autofunction:: CcdCmosSim.ccd.dark_current_and_dark_noises

.. autofunction:: CcdCmosSim.ccd.source_follower_noise

.. autofunction:: CcdCmosSim.ccd.set_photosensor_constants

.. autofunction:: CcdCmosSim.ccd.create_data_arrays

.. autofunction:: CcdCmosSim.ccd.image_irradiance_to_flux

.. autofunction:: CcdCmosSim.ccd.convert_to_electrons

.. autofunction:: CcdCmosSim.ccd.shotnoise

.. autofunction:: CcdCmosSim.ccd.responsivity_FPN_light

.. autofunction:: CcdCmosSim.ccd.responsivity_FPN_dark

.. autofunction:: CcdCmosSim.ccd.FPN_models

.. autofunction:: CcdCmosSim.ccd.create_HDF5_image

.. autofunction:: CcdCmosSim.ccd.define_metrics

.. autofunction:: CcdCmosSim.ccd.limitzero

.. autofunction:: CcdCmosSim.ccd.distribution_exp

.. autofunction:: CcdCmosSim.ccd.distribution_lognormal

.. autofunction:: CcdCmosSim.ccd.distribution_inversegauss

.. autofunction:: CcdCmosSim.ccd.distribution_logistic

.. autofunction:: CcdCmosSim.ccd.distribution_wald

.. autofunction:: CcdCmosSim.ccd.distributions_generator

.. autofunction:: CcdCmosSim.ccd.validateParam

.. autofunction:: CcdCmosSim.ccd.checkParamsNum

.. autofunction:: CcdCmosSim.ccd.run_example



