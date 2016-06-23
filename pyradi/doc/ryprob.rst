Staring Array Module (rystare)
*******************************
.. include global.rst

Overview
----------

The code in this module is based on extracts of code originally written in Matlab and used 
for the Adaptive Optics simulation models, 
but can be used for many other applications involving light registration on CCD/CMOS
photosensors.  The original files are available at:

- Paper: http://arxiv.org/pdf/1412.4031.pdf
- Matlab code: https://bitbucket.org/aorta/highlevelsensorsim

The original Matlab code was ported to Python and extended
in a number of ways.  

In the documentation for the Matlab code Konnik expressed the hope "that this 
model will be useful for somebody, or at least save someone's time.
The model can be (and should be) criticized."  Indeed it has, thanks Mikhail!
Konnik quotes George E. P. Box, the famous statistician, and who said that 
"essentially, all models are wrong, but some are useful".



Code Overview
---------------
.. automodule:: pyradi.ryprob


Module functions
------------------


.. autofunction:: pyradi.rystare.distribution_exp

.. autofunction:: pyradi.rystare.distribution_lognormal

.. autofunction:: pyradi.rystare.distribution_inversegauss

.. autofunction:: pyradi.rystare.distribution_logistic

.. autofunction:: pyradi.rystare.distribution_wald

.. autofunction:: pyradi.rystare.distributions_generator

.. autofunction:: pyradi.rystare.validateParam

.. autofunction:: pyradi.rystare.checkParamsNum

