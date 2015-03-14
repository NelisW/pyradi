The purpose of computing is insight, not numbers.<BR>
_- Richard Hamming_

##Objective

The PyRadi toolkit is a Python toolkit to perform optical and infrared computational radiometry (flux flow) calculations.

The toolkit is an extendable, integrated and coherent collection of basic functions, code modules, documentation, example templates, unit tests and resources, that can be applied towards diverse calculations in the electro-optics domain. The toolkit covers 

  * Models of physical radiators (e.g., Planck's Law) and conversion between values expressed in different units.
  
  * Mathematical operations (e.g., spectral integrals, spatial integrals, spectral convolution)
  
  * Data manipulation (e.g., file input/output, interpolation, spectral quantity conversions, reading Flir Inc PTW files)
  
  * Detector modelling from physical parameters
  
  * 3-D noise analysis of image sequences
  
  * Modtran tape7 read functions
  
  * Graphical visualization(2-D and 3-D graphs) in compact format, including cartesian, polar, image and mesh plots. 

  * Spectral variables are expressed in Numpy arrays to ease spectral operations and integrals.

The individual scripts in the toolkit is supported by examples, test cases and documentation. These examples are included at the end of each script in the `__main__`  section.  If you just run the script, the code will be executed and results will be available in graphs or text files.


##Prerequisites

If you install the Anaconda distribution (based on Python 2.7) from https://store.continuum.io/cshop/anaconda/ you can ignore the rest of this paragraph.  It works well, is easy to do and, unless you have conflicting requirements, just download Anaconda and focus on the work.

This specific toolkit is implemented (current versions in brackets) in [http://www.python.org/ Python (2.7) ]
and its associated modules [https://www.scipy.org/ Numpy (1.6 or 1.7)], [https://www.scipy.org/ SciPy (0.12)], [http://matplotlib.sourceforge.net/ Matplotlib (1.2 or 1.3)] and [http://code.enthought.com/projects/mayavi/ Mayavi (4.1+)]. The links provided in this section should get you going, except maybe for Mayavi. Mayavi is only required for the three-dimensional spherical plots, you may ignore the requirement otherwise.

Most sources advise users to install Mayavi as part of a Python distribution, such as the [http://www.lfd.uci.edu/~gohlke/pythonlibs/#ets Enthought Tool Set] or [https://store.continuum.io/cshop/anaconda/ Anaconda] distributions. This is certainly an option for newcomers, but  these distributions are not always current with the latest code package versions. Of the two, Anaconda appears to be quicker with the updates.
Look for [https://store.continuum.io/cshop/anaconda/ Anaconda] (under the supervision of Travis Oliphant, one of the driving forces behind Numpy) or [http://www.lfd.uci.edu/~gohlke/pythonlibs/#ets Enthought Tool Set], both of which contains contains Mayavi and other tools.
You can find Mayavi (and other) installation packages on [http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikits-image this site]. 

pyradi requires Python version 2.7, *it does not work with Python 3.x*.  
An investigation on 2014-03-03 indicated that not all the prerequisite modules are available in Python 3 (most notably Mayavi).

pyradi was last tested on 2015-03-14 using Python 2.7.9, Numpy 1.9.0, SciPy 0.14 and Matplotlib 1.4.


##Learning Python

<a href="http://lorenabarba.com/blog/why-i-push-for-python"><img src="http://pyradi.googlecode.com/svn/trunk/pyradi/doc/_images/keep-calm-and-code-python_BW.png" alt="Keep calm and code Python" width="150" height="175"/></a>


To use pyradi you would have to know Python and Numpy. Getting acquainted with a new tool or computer language takes time and practice. Invest your precious time in learning Python and its modules, you will not be disappointed!

There are many free [http://docs.python-guide.org/en/latest/intro/learning/ books], [https://developers.google.com/edu/python/ classes], blogs [http://www.python.org/about/gettingstarted/ getting started] websites and tutorial [https://www.youtube.com/playlist?list=PLEA1FEF17E1E5C0DA videos],  [http://pyvideo.org/ more videos] and [http://www.python.org/community/workshops/ conferences].  Material for Numpy is less bountiful, but the [http://docs.scipy.org/doc/numpy/numpy-ref-1.8.0.pdf numpy reference] and StackOverflow are good sources.
Just google some variation of 'learning python' and make your choice.

A very good introduction to Python for scientific work are the [http://folk.uio.no/hpl/scripting/book_comparison.html two books] by Hans Petter Langtangen.


##Status
This project is *stable beta*. Current content is tested, stable and usable. With time and active use the scope of the pyradi offering will grow and expand. The current version is already quite useful in our labs and lectures.

The development is ongoing as and when new needs arise.  We are open for feature requests as well.

##Documentation
- Module documentation is given in  [http://pyradi.googlecode.com/svn//trunk/pyradi/doc/_build/html/index.html html docs] and   [http://pyradi.googlecode.com/svn//trunk/pyradi/doc/_build/latex/pyradi.pdf pdf doc.]
- A number of IPython notebooks demonstrate pyradi use. Head on over to [https://github.com/NelisW/ComputationalRadiometry Computational Radiometry] to download the notebooks. If you are not using the IPython notebook, the are also [https://github.com/NelisW/ComputationalRadiometry#computational-optical-radiometry-with-pyradi HTML renderings]. 
- At the end of each of the pyradi files, in the __main__ section, you will find test and example code demonstrating the use of that specific file. Some day, these will find their way into a tutorial, but for now, please study the example code.
- [http://pyradi.googlecode.com/svn/trunk/pyradi/documentation/SPIE-8543-Pyradi-an-open-source-toolkit-for-infrared-85430J.pdf  "Pyradi: an open-source toolkit for infrared calculation and data processing"], SPIE   Proceedings Vol 8543, Security+Defence 2012,  Technologies for Optical Countermeasures, Edinburgh, 24-27 September, C.J. Willers, M. S. Willers,    R.A.T. Santos, P.J. van der Merwe, J.J. Calitz, A de Waal and A.E. Mudau. 
- [http://pyradi.googlecode.com/svn/trunk/pyradi/documentation/pyradi-SPIE-Newsroom.pdf  "Pyradi radiometry toolkit"],   C.J. Willers, M. S. Willers,    R.A.T. Santos, P.J. van der Merwe, J.J. Calitz, A de Waal and A.E. Mudau,  SPIE Newsroom, DOI:10.1117/2.1201211.004568, 2012. 
- [http://spie.org/x648.html?product_id=2021423&origin_id=x646 Electro-Optical System Analysis and Design: A Radiometry Perspective]  provides detailed examples of pyradi use.  The 528-page book provides detailed coverage of radiometry and infrared system analysis and design.  It uses pyradi in the extensively documented computational examples. _Electro-Optical System Analysis and Design: A Radiometry Perspective_, Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume PM236, SPIE Press, 2013. The book is available from [http://spie.org/x648.html?product_id=2021423&origin_id=x646 SPIE] and [http://www.amazon.com/Electro-optical-System-Analysis-Design-Perspective/dp/0819495697/ref=sr_1_13?ie=UTF8&qid=1371620238&sr=8-13&keywords=willers Amazon]. 


[http://spie.org/x648.html?product_id=2021423&origin_id=x646 http://pyradi.googlecode.com/svn/trunk/pyradi/doc/_images/PM236.jpg]


##Get the code via Python pip or easy_install
You can download the pyradi package using pip, from the command line by typing

    pip install --upgrade pyradi

This command will install or upgrade pyradi to the latest version in https://pypi.python.org/pypi/pyradi/. If the install is successful, pyradi is available for use immediately, no further action is required.  The pip install may initiate the download of numpy, scipy, matplotlib or scikit-image if these are not presently in your Python distribution.

The pyradi version in pip is built at regular intervals from the subversion repository, but may not have all the very latest updates.

Instructions on how to use pip is available at http://www.pythonforbeginners.com/basics/python-pip-usage and  http://www.pip-installer.org/en/latest/installing.html.

##Get the code from GitHub
You can download the very latest version of pyradi from the pyradi repository on GitHub.

Note that the subversion download only installs pyradi and not any of its dependency packages matplotlib, numpy, scipy or scikit-image.  These can be installed using pip with commands of the form:

 pip install --upgrade matplotlib


==Checking out pyradi for building PyPI packages==

This is only relevant for developers building a pyradi installation package.
Do a subversion checkout of the pyradi trunk to anywhere on your computer (except Python's site-packages) from 
{{{
 https://pyradi.googlecode.com/svn/trunk/
}}}
To allow Python to reach the pyradi code where it was checked out, add the fully qualified path {{{somewhere-on-your-PC/pyradi}}}  to  your {{{sys.path}}} e.g., adding the {{{somewhere-on-your-PC/pyradi}}} directory to your {{{PYTHONPATH}}} environmental variable.  For more information on how Python searches for packages, see http://docs.python.org/2/tutorial/modules.html#the-module-search-path and https://stackoverflow.com/questions/1893598/pythonpath-vs-sys-path.

Building a package is somewhat confusing at best. A few pointers are given in the {{{setup.py}}} script in the root directory.

##Related toolkits and resources

For image segmentation and analysis, please see 
[http://scikit-image.org/ scikit-image].

For hyperspectral image processing see
[http://spectralpython.sourceforge.net/ Spectral Python].

##Acknowledgement

The authors gratefully acknowledge the [http://www.csir.co.za/ CSIR] and [http://www.deneldynamics.co.za/ Denel Dynamics] for support in the development of the code.


##Contact

You can contact the pyradi community by emailing the repository owner through github or at neliswillers at gmail.

