The purpose of computing is insight, not numbers.<BR>
_-- Richard Hamming_

## Objective

The PyRadi toolkit is a Python toolkit to perform optical and infrared computational radiometry (flux flow) calculations.

The toolkit is an extendable, integrated and coherent collection of basic functions, code modules, documentation, example templates, unit tests and resources, that can be applied towards diverse calculations in the electro-optics domain. The toolkit covers

  * Models of physical radiators (e.g., Planck's Law) and conversion between values expressed in different units.

  * Mathematical operations for radiometry (e.g., spectral integrals, spatial integrals, spectral convolution)

  * Data manipulation (e.g., file input/output, interpolation, spectral quantity conversions, reading Flir Inc PTW files)

  * Detector modelling from physical parameters: including single element detectors and staring arrays

  * 3-D noise analysis of image sequences

  * Modtran tape7 read functions

  * Graphical visualization(2-D and 3-D graphs) in compact format, including cartesian, polar, image and mesh plots.

  * Spectral variables are expressed in Numpy arrays to ease spectral operations and integrals.

The individual scripts in the toolkit is supported by examples, test cases and documentation. These examples are included at the end of each script in the `__main__`  section.  If you just run the script, the code will be executed and results will be available in graphs or text files.


## Prerequisites

If you install the [Anaconda](https://store.continuum.io/cshop/anaconda/) distribution (Python 2.7 or 3.5)  you can ignore the rest of this paragraph.  It works well, is easy to do and, unless you have conflicting requirements, just download Anaconda and focus on the work.

This specific toolkit is implemented in
[Python (2.7 or 3.5)](http://www.python.org/) and its associated modules
[Numpy](https://www.scipy.org/),
[SciPy](https://www.scipy.org/),
[Matplotlib](http://matplotlib.sourceforge.net/), and
[Mayavi](http://code.enthought.com/projects/mayavi/).
The links provided in this section should get you going, except maybe for Mayavi. Mayavi is only required for the three-dimensional spherical plots, you may ignore the requirement otherwise.

Most sources advise users to install Mayavi as part of a Python distribution, such as the
[Enthought Tool Set](http://www.lfd.uci.edu/~gohlke/pythonlibs/#ets) or
[Anaconda](https://store.continuum.io/cshop/anaconda/) distributions. This is certainly an option for newcomers, but  these distributions are not always current with the latest code package versions. Of the two, Anaconda appears to be quicker with the updates.
Look for
[Anaconda](https://store.continuum.io/cshop/anaconda/) (under the supervision of Travis Oliphant, one of the driving forces behind Numpy) or
[Enthought Tool Set](http://www.lfd.uci.edu/~gohlke/pythonlibs/#ets), both of which contains contains Mayavi and other tools.
You can find Mayavi (and other) installation packages on
[this site](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikits-image).

pyradi requires Python version 2.7 or Python 3.5.  

pyradi was last tested on 2016-11-20 using Python 2.7.12 and Python 3.5.2.


## Learning Python

<a href="http://lorenabarba.com/blog/why-i-push-for-python"><img src="http://pyradi.googlecode.com/svn/trunk/pyradi/doc/_images/keep-calm-and-code-python_BW.png" alt="Keep calm and code Python" width="150" height="175"/></a>


To use pyradi you would have to know Python and Numpy. Getting acquainted with a new tool or computer language takes time and practice. Invest your precious time in learning Python and its modules, you will not be disappointed!

There are many free
[books](http://docs.python-guide.org/en/latest/intro/learning/),
[classes](https://developers.google.com/edu/python/),
[getting started ](http://www.python.org/about/gettingstarted/) blogs websites and tutorials
[videos](https://www.youtube.com/playlist?list=PLEA1FEF17E1E5C0DA),  
[more videos](http://pyvideo.org/) and
[conferences](http://www.python.org/community/workshops/).  Material for Numpy is less bountiful, but the
[numpy reference](http://docs.scipy.org/doc/numpy/numpy-ref-1.8.0.pdf) and StackOverflow are good sources.
Just google some variation of 'learning python' and make your choice.

A very good introduction to Python for scientific work are the
[two books](http://folk.uio.no/hpl/scripting/book_comparison.html) by Hans Petter Langtangen.


## Status
This project has Production status. Current content is tested, stable and usable. The scope of the pyradi is continually growing as new functionality and examples are added. 

The development is ongoing as and when new needs arise.  We are open for feature requests as well.

## Documentation
- Module documentation is available in  
- [HTML](http://nelisw.github.io/pyradi-docs/_build/html/index.html) and  [PDF](https://raw.githubusercontent.com/NelisW/pyradi-docs/gh-pages/_build/latex/pyradi.pdf).
- A number of IPython notebooks demonstrate pyradi use. Head on over to
[Computational Radiometry](https://github.com/NelisW/ComputationalRadiometry) to download the notebooks. If you are not using the IPython notebook, the are also [HTML renderings](https://github.com/NelisW/ComputationalRadiometry#computational-optical-radiometry-with-pyradi).
- At the end of each of the pyradi files, in the __main__ section, you will find test and example code demonstrating the use of that specific file. Some day, these will find their way into a tutorial, but for now, please study the example code.
- ["Pyradi: an open-source toolkit for infrared calculation and data processing"](http://pyradi.googlecode.com/svn/trunk/pyradi/documentation/SPIE-8543-Pyradi-an-open-source-toolkit-for-infrared-85430J.pdf), SPIE   Proceedings Vol 8543, Security+Defence 2012,  Technologies for Optical Countermeasures, Edinburgh, 24-27 September, C.J. Willers, M. S. Willers,    R.A.T. Santos, P.J. van der Merwe, J.J. Calitz, A de Waal and A.E. Mudau.
- ["Pyradi radiometry toolkit"](http://pyradi.googlecode.com/svn/trunk/pyradi/documentation/pyradi-SPIE-Newsroom.pdf),   C.J. Willers, M. S. Willers,    R.A.T. Santos, P.J. van der Merwe, J.J. Calitz, A de Waal and A.E. Mudau,  SPIE Newsroom, DOI:10.1117/2.1201211.004568, 2012.
- [Electro-Optical System Analysis and Design: A Radiometry Perspective](http://spie.org/x648.html?product_id=2021423&origin_id=x646)  provides detailed examples of pyradi use.  The 528-page book provides detailed coverage of radiometry and infrared system analysis and design.  It uses pyradi in the extensively documented computational examples. _Electro-Optical System Analysis and Design: A Radiometry Perspective_, Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume PM236, SPIE Press, 2013. The book is available from [SPIE](http://spie.org/x648.html?product_id=2021423&origin_id=x646) and [Amazon](http://www.amazon.com/Electro-optical-System-Analysis-Design-Perspective/dp/0819495697/ref=sr_1_13?ie=UTF8&qid=1371620238&sr=8-13&keywords=willers).

[<img src="https://raw.githubusercontent.com/NelisW/pyradi/master/pyradi/doc/_images/PM236.jpg"/>](http://spie.org/x648.html?product_id=2021423&origin_id=x646)


## Get the code via Python pip or easy_install

**Note that the PyPi version is not up to date with the github version.**

You can download the pyradi package using pip, from the command line by typing

    pip install --upgrade pyradi

This command will install or upgrade pyradi to the latest version written to  <https://pypi.python.org/pypi/pyradi/>. If the install is successful, pyradi is available for use immediately, no further action is required.  The pip install may initiate the download of numpy, scipy, matplotlib or scikit-image if these are not presently in your Python distribution.

The pyradi version in pip is **not** built at regular intervals from the subversion repository, but definitely does not have all the very latest version.

Instructions on how to use pip is available at <http://www.pythonforbeginners.com/basics/python-pip-usage> and  <http://www.pip-installer.org/en/latest/installing.html>.

## Get the code from GitHub
You can download the very latest version of pyradi from the [pyradi repository](https://github.com/NelisW/pyradi) on GitHub.  I am somewhat slow to update PyPi with the latest GitHub version.

Note that the subversion download only installs pyradi and not any of its dependency packages matplotlib, numpy, scipy or scikit-image.  These can be installed using pip with commands of the form:

    pip install --upgrade matplotlib

If you install the Anaconda distribution all the dependency packages should already be present.

## Repositories associated with pyradi

In order to keep the size of the pyradi repository smaller, some pyradi information has been moved to separate repositories.  If you require to work on regression testing and documentation, check out the tree repositories as follows:


    ..
    +-pyradi  [https://github.com/NelisW/pyradi]
      +-.git
      |-setup.py (this file)
      | ...
      +-pyradi
        + ... all the pyradi files

    +-pyradi-docs [https://github.com/NelisW/pyradi-data] (this is an optional clone) 
      +-.git
      +-_build

    +-pyradi-data [https://github.com/NelisW/pyradi-docs] (this is an optional clone) 
      +-.git
      +-regression
      +-images


## Related toolkits and resources

For image segmentation and analysis, please see
<http://scikit-image.org/ scikit-image>.

For hyperspectral image processing see
<http://spectralpython.sourceforge.net/ Spectral Python>.

## Acknowledgement

The authors gratefully acknowledge the [CSIR](http://www.csir.co.za/) and [Denel Dynamics](http://www.deneldynamics.co.za/) for support in the development of the code.


## Contact

You can contact the pyradi community by emailing the repository owner through github or at neliswillers at gmail.
