Target Radiance Image Generator (rytarggen)
**********************************************


.. include global.rst

Overview
----------

This module provides a simple model for an optical target image radiance calculator,
given the radiometric sensor characteristics and the target spatial temperature 
and emissivity distribution.
The calculator accounts for 

#. the sensor spectral band 
#. target self-emission 
#. target reflected sunlight.

The model does not account for reflected ambient radiance.

HDF5 File
---------

The Python implementation of the model uses an HDF5 file to capture the
input and output data for record keeping or subsequent analysis. 
HDF5 files provide for hierarchical data structures and easy read/save to disk. 
See the file `hdf5-as-data-format.md` ([hdf5asdataformat]_) in the pyradi root directory for more detail.

Input images are written to and read from HDF5 files as well.  These files store the
image as well as the images' dimensional scaling in the focal plane.  
The intent is to later create test targets with specific spatial 
frequencies in these files.


Code Overview
---------------
.. automodule:: pyradi.rytarggen


Module functions
------------------

.. autofunction:: pyradi.rytarggen.create_HDF5_image

.. autofunction:: pyradi.rytarggen.hdf_Raw

.. autofunction:: pyradi.rytarggen.hdf_Uniform

.. autofunction:: pyradi.rytarggen.hdf_disk_photon

.. autofunction:: pyradi.rytarggen.hdf_stairs



.. [hdf5asdataformat] https://github.com/NelisW/pyradi/blob/master/pyradi/hdf5-as-data-format.md