#  $Id$
#  $HeadURL$
################################################################
# The contents of this file are subject to the BSD 3Clause (New)cense
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

# Contributor(s): MS Willers

################################################################
"""


For more examples of use see:
https://github.com/NelisW/ComputationalRadiometry

See the __main__ function for examples of use.

This package was partly developed to provide additional material in support of students
and readers of the book Electro-Optical System Analysis and Design: A Radiometry
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""

__version__ = "$Revision$"
__author__ = 'pyradi team'
__all__ = ['PFlux']

import sys
if sys.version_info[0] > 2:
    print("pyradi is not yet ported to Python 3, because imported modules are not yet ported")
    exit(-1)

import numpy as np
import math
import sys
import itertools

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.font_manager import FontProperties
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.dates as mdates
# from mpl_toolkits.axes_grid1 import make_axes_locatable


class PFlux:
    """ 
    """

    ############################################################
    ##
    def __init__(self):
        """Class constructor

        The constructor defines the 

            Args:
                | fignumber (int): the plt figure number, must be supplied

            Returns:
                | Nothing. Creates the figure for subequent use.

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', ]




    ############################################################
    ##
    def dummy(self, var1=None, var2=None):
        """

            Args:
                | var1 ([strings]): User-supplied list
                |    of plotting styles(can be empty []).
                | var2 (int): Length of required sequence.

            Returns:
                | Whatever.

            Raises:
                | No exception is raised.
        """




################################################################
################################################################
##

if __name__ == '__main__':

    doAll = False

    if True:  
        pass


    print('module rypflux done!')
