#  $Id$
#  $HeadURL$

################################################################
# The contents of this file are subject to the Mozilla Public License
# Version 1.1 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/

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


import numpy

##############################################################################
##
def chromaticityforSpectralL(spectral,radiance,xbar,ybar,zbar):
    """ Calculate the CIE chromaticity coordinates for an arbitrary spectrum.
    
    Given a spectral radiance vector and CIE tristimulus curves, 
    calculate the CIE chromaticity coordinates. It is assumed that the 
    radiance spectral density is given in the same units as the spectral
    vector (i.e. [1/um] or [1/cm-1], corresponding to [um] or [cm-1] respectively.
    It is furthermore accepted that the tristimulus curves are also sampled at
    the same spectral intervals as the radiance. See 
    http://en.wikipedia.org/wiki/CIE_1931_color_space 
    for more information on CIE tristimulus spectral curves.
    
    Args:
        | spectral: spectral vector in  [um] or [cm-1].
        | radiance: the spectral radiance (any units), (sampled at spectral).
        | xbar: CIE x tristimulus spectral curve (sampled at spectral values).
        | ybar: CIE y tristimulus spectral curve (sampled at spectral values).
        | zbar: CIE z tristimulus spectral curve (sampled at spectral values).
        
    Returns:
        | A list with color coordinates and Y [x,y,Y].
        
    Raises:
        | No exception is raised.
    """
    
    X=numpy.trapz(radiance*xbar.reshape(-1, 1),spectral, axis=0)
    Y=numpy.trapz(radiance*ybar.reshape(-1, 1),spectral, axis=0)
    Z=numpy.trapz(radiance*zbar.reshape(-1, 1),spectral, axis=0)
    
    x=X/(X+Y+Z)
    y=Y/(X+Y+Z)
    
    return [x[0], y[0], Y[0]]
