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
def chromaticityforSpectralL(wavelength,radiance,xbar,ybar,zbar):
    """ Calculate the CIE chromaticity coordinates for an arbitrary spectrum
    Parameters:
        wavelength  == wavelength vector in  [um]
        radiance == the spectral radiance (any units), (sampled at wavelength)
        xbar == CIE x tristimulus spectral curve (sampled at wavelength values)
        ybar == CIE y tristimulus spectral curve (sampled at wavelength values)
        zbar == CIE z tristimulus spectral curve (sampled at wavelength values)
    Return:
        a list with color coordinates and Y [x,y,Y]
    """
    
    X=numpy.trapz(radiance*xbar.reshape(-1, 1),wavelength, axis=0)
    Y=numpy.trapz(radiance*ybar.reshape(-1, 1),wavelength, axis=0)
    Z=numpy.trapz(radiance*zbar.reshape(-1, 1),wavelength, axis=0)
    
    x=X/(X+Y+Z)
    y=Y/(X+Y+Z)
    
    return [x[0], y[0], Y[0]]
