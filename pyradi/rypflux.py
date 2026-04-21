# -*- coding: utf-8 -*-

################################################################
# The contents of this file are subject to the BSD 3Clause (New) License
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
Provides a simple, order-of-magnitude estimate of the photon flux and
electron count in a detector for various sources and scene lighting.
All models are based on published information or derived herein, so you
can check their relevancy and suitability for your work.

For a detailed theoretical derivation and more examples of use see:
http://nbviewer.jupyter.org/github/NelisW/ComputationalRadiometry/blob/master/07-Optical-Sources.ipynb

See the __main__ function for examples of use.

This package was partly developed to provide additional material in support of students
and readers of the book Electro-Optical System Analysis and Design: A Radiometry
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""

__version__ = '0.1.0'
__author__ = 'pyradi team'
__all__ = ['PFlux']


import numpy as np
import pandas as pd
import pyradi.ryutils as ryutils
import pyradi.ryplanck as ryplanck


##############################################################################################
class PFlux:
    """Provides a combined thermal and reflected photon irradiance estimate.

    The class name derives from (P)hoton (Flux).

    See here:
    https://github.com/NelisW/ComputationalRadiometry/blob/master/07-Optical-Sources.ipynb
    for mathematical derivations and more detail.
    """

    ############################################################
    def __init__(self):
        """Initialise the PFlux instance with built-in illumination conditions and spectral bands.

        Populates:
        - ``lllux``: dict mapping illumination condition name to
          ``[irradiance_lm_m2, colour_temp_K, photopic_fraction]``
        - ``llluxCols``: column names for the lllux values
        - ``specranges``: dict of default spectral bands used by ``lllPhotonrates``

        Args:
            | None

        Returns:
            | Nothing. Sets up the class for use.

        Raises:
            | No exception is raised.
        """

        # Low-light-level lux, colour temperature, and photopic fraction.
        # Fraction predicts the ratio between photopic and scotopic response
        # and is used later to weight spectrally.
        # Source: RCA/Burle electro-optics handbook.
        self.lllux = {}
        self.lllux['Sun light']       = [107527, 5700,  1.0]
        self.lllux['Full sky light']  = [10752,  12000, 1.0]
        self.lllux['Overcast day']    = [1075,   6000,  1.0]
        self.lllux['Very dark day']   = [107,    7000,  1.0]
        self.lllux['Twilight']        = [10.8,   10000, 1.0]
        self.lllux['Deep twilight']   = [1.08,   10000, 0.8]
        self.lllux['Full moon']       = [0.108,  4150,  0.6]
        self.lllux['Quarter moon']    = [0.0108, 4150,  0.4]
        self.lllux['Star light']      = [0.0011, 5000,  0.2]
        self.lllux['Overcast night']  = [0.0001, 5000,  0.0]

        self.llluxCols = ['Irradiance-lm/m2', 'ColourTemp', 'FracPhotop']

        numpts = 300
        self.specranges = {
            'VIS':  [np.linspace(0.43, 0.69, numpts).reshape(-1, 1), np.ones((numpts, 1)), 'wl'],
            'NIR':  [np.linspace(0.7,  0.9,  numpts).reshape(-1, 1), np.ones((numpts, 1)), 'wl'],
            'SWIR': [np.linspace(1.0,  1.7,  numpts).reshape(-1, 1), np.ones((numpts, 1)), 'wl'],
            'MWIR': [np.linspace(3.6,  4.9,  numpts).reshape(-1, 1), np.ones((numpts, 1)), 'wl'],
            'LWIR': [np.linspace(7.5,  10,   numpts).reshape(-1, 1), np.ones((numpts, 1)), 'wl'],
        }

    ############################################################
    def lllPhotonrates(self, specranges=None):
        """Calculate the approximate photon rate radiance for low-light conditions.

        The colour temperature of various sources is used to predict the photon flux.
        The calculation uses the colour temperature of the source and the ratio of the
        real low-light luminance to the luminance of a Planck radiator at the same
        colour temperature.

        This procedure critically depends on the sources' spectral radiance in the
        various different spectral bands.  For natural scenes the spectral shape is
        modelled by a Planck curve at the appropriate colour temperature.

        Steps followed:

        1. Calculate the photon rate for the scene at the appropriate colour temperature,
           spectrally weighted by the eye's luminous efficiency response, for both
           photopic and scotopic vision.
        2. Weight the photopic and scotopic photon rates according to illumination level.
        3. Determine the ratio k of low-light-level scene illumination to photon
           irradiance.  k is calculated in the visual band and then applied to scale all
           other spectral bands.
        4. Use Planck radiation at the appropriate colour temperature to calculate radiance
           in any spectral band, scaled by k.

        The ``specranges`` dict maps band name to a three-element list
        ``[spectral_vector, response_vector, type_str]`` where ``type_str`` is
        ``'wl'`` (wavelength in µm) or ``'wn'`` (wavenumber in cm⁻¹).  Example:

        .. code-block:: python

            numpts = 300
            specranges = {
                'VIS':  [np.linspace(0.43, 0.69, numpts).reshape(-1, 1), np.ones((numpts, 1)), 'wl'],
                'NIR':  [np.linspace(0.7,  0.9,  numpts).reshape(-1, 1), np.ones((numpts, 1)), 'wl'],
                'SWIR': [np.linspace(1.0,  1.7,  numpts).reshape(-1, 1), np.ones((numpts, 1)), 'wl'],
                'MWIR': [np.linspace(3.6,  4.9,  numpts).reshape(-1, 1), np.ones((numpts, 1)), 'wl'],
                'LWIR': [np.linspace(7.5,  10,   numpts).reshape(-1, 1), np.ones((numpts, 1)), 'wl'],
            }

        If ``specranges`` is ``None`` the predefined values above are used.

        The function returns a Pandas DataFrame with columns for the integrated
        spectral radiance in each band, plus ``'Irradiance-lm/m2'``, ``'ColourTemp'``,
        ``'FracPhotop'``, and ``'k'``.  Rows are indexed by illumination-condition name
        and sorted in ascending order of irradiance.

        Args:
            | specranges (dict): User-supplied spectral-band definition dict.
            |   See format description above. Pass ``None`` to use built-in defaults.

        Returns:
            | pd.DataFrame: Pandas DataFrame with radiance in the specified spectral bands.

        Raises:
            | No exception is raised.
        """

        self.dfPhotRates = pd.DataFrame(self.lllux).transpose()
        self.dfPhotRates.columns = ['Irradiance-lm/m2', 'ColourTemp', 'FracPhotop']
        self.dfPhotRates.sort_values(by='Irradiance-lm/m2', inplace=True)

        wl = np.linspace(0.3, 0.8, 100)
        photLumEff, wl = ryutils.luminousEfficiency(vlamtype='photopic',  wavelen=wl, eqnapprox=False)
        scotLumEff, wl = ryutils.luminousEfficiency(vlamtype='scotopic',  wavelen=wl, eqnapprox=False)

        self.dfPhotRates['k'] = (
            self.dfPhotRates['Irradiance-lm/m2']
            / (
                self.dfPhotRates['FracPhotop'] * 683
                * np.trapezoid(
                    photLumEff.reshape(-1, 1) * ryplanck.planckel(wl, self.dfPhotRates['ColourTemp']),
                    wl, axis=0)
                + (1 - self.dfPhotRates['FracPhotop']) * 1700
                * np.trapezoid(
                    scotLumEff.reshape(-1, 1) * ryplanck.planckel(wl, self.dfPhotRates['ColourTemp']),
                    wl, axis=0)
            )
        )

        if specranges is not None:
            self.specranges = specranges
            for key in self.specranges:
                for idx in [0, 1]:
                    self.specranges[key][idx] = self.specranges[key][idx].reshape(-1, 1)

        for specrange in self.specranges.keys():
            spec = self.specranges[specrange][0]

            if 'wl' in self.specranges[specrange][2]:
                stype = 'ql'
            else:
                stype = 'qn'

            self.dfPhotRates[f'Radiance-q/(s.m2.sr)-{specrange}'] = (
                (self.dfPhotRates['k'] / np.pi)
                * np.trapezoid(
                    self.specranges[specrange][1] * ryplanck.planck(spec, self.dfPhotRates['ColourTemp'], stype),
                    spec, axis=0)
            )

        self.dfPhotRates.sort_values(by='Irradiance-lm/m2', inplace=True)
        self.dfPhotRates = self.dfPhotRates.reindex(sorted(self.dfPhotRates.columns), axis=1)

        return self.dfPhotRates


################################################################

if __name__ == '__main__':

    doAll = True

    if doAll:
        pf = PFlux()
        keys = sorted(list(pf.lllux.keys()))
        for key in keys:
            print('{}: {}'.format(key, pf.lllux[key]))

        dfPhotRates = pd.DataFrame(pf.lllux).transpose()
        dfPhotRates.columns = ['Irradiance-lm/m2', 'ColourTemp', 'FracPhotop']
        dfPhotRates.sort_values(by='Irradiance-lm/m2', inplace=True)
        print(dfPhotRates)
        print(pf.lllPhotonrates())

    print('\n---------------------\n')
    print('module rypflux done!')
