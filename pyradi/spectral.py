import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
import os
import pkg_resources
from StringIO import StringIO
import pyradi.ryfiles as ryfiles
from numbers import Number


##############################################################################################
class Spectral(object):
    """Generic spectral can be used for any spectral vector
    """
    ############################################################
    ##
    def __init__(self, ID, value, wl=None, wn=None, desc=None):
        """Defines a spectral variable of property vs wavelength or wavenumber

        One of wavelength or wavenunber must be supplied, the other is calculated.
        No assumption is made of the sampling interval on eiher wn or wl.

        The constructor defines the 

            Args:
                | ID (str): identification string
                | wl (np.array (N,) or (N,1)): vector of wavelength values
                | wn (np.array (N,) or (N,1)): vector of wavenumber values
                | value (np.array (N,) or (N,1)): vector of property values
                | desc (str): description string

            Returns:
                | None

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', ]

        self.ID = ID
        self.desc = desc

        if wn is not None:
            self.wn =  wn.reshape(-1,1)
            self.wl = 1e4 /  self.wn
        elif wl is not None:
            self.wl =  wl.reshape(-1,1)
            self.wn = 1e4 /  self.wl
        else:
            print('Spectral {} has both wn and wl as None'.format(ID))

        if isinstance(value, Number):
            self.value = value * np.ones(self.wn.shape)
        elif isinstance(value, np.ndarray):
            self.value = value.reshape(-1,1)


    ############################################################
    ##
    def __str__(self):
        """Returns string representation of the object

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        numpts = self.wn.shape[0]
        stride = numpts / 3
        strn = 'ID: {}\n'.format(self.ID)
        strn += 'desc: {}\n'.format(self.desc)
 
        # for all numpy arrays, provide subset of values
        for var in self.__dict__:
            # then see if it is an array
            if isinstance(eval('self.{}'.format(var)), np.ndarray):
                svar = (np.vstack((eval('self.{}'.format(var))[0::stride], eval('self.{}'.format(var))[-1] ))).T
                strn += '{} (subsampled.T): {}\n'.format(var, svar)


        return strn


    ############################################################
    ##
    def __mul__(self, other):
        """Returns a spectral product

        it is not intended that the function will be called directly by the user

            Args:
                | other (Spectral): the other Spectral to be used in multiplication

            Returns:
                | str

            Raises:
                | No exception is raised.
        """

        if isinstance(other, Number):
            other = Spectral('{}'.format(other),value=other, wn=self.wn,desc='{}'.format(other))

        if isinstance(other.value, Number):
            other.value = other.value * np.ones(other.wn.shape).reshape(-1,1)

        # create new spectral in wn wider than either self or other.
        wnmin = min(np.min(self.wn),np.min(other.wn))
        wnmax = max(np.max(self.wn),np.max(other.wn))
        wninc = min(np.min(np.abs(np.diff(self.wn,axis=0))),np.min(np.abs(np.diff(other.wn,axis=0))))
        wn = np.linspace(wnmin, wnmax, (wnmax-wnmin)/wninc)
        wl = 1e4 / self.wn

        if np.mean(np.diff(self.wn)) > 0:
            s = np.interp(wn,self.wn[:,0], self.value[:,0])
            o = np.interp(wn,other.wn[:,0], other.value[:,0])
        else:
            s = np.interp(wn,np.flipud(self.wn[:,0]), np.flipud(self.value[:,0]))
            o = np.interp(wn,np.flipud(other.wn[:,0]), np.flipud(other.value[:,0]))

        return Spectral(ID='{}*{}'.format(self.ID,other.ID), value=s * o, wl=wl, wn=wn,
             desc='{}*{}'.format(self.desc,other.desc))

    ############################################################
    ##
    def __pow__(self, power):
        """Returns a spectral to some power

        it is not intended that the function will be called directly by the user

            Args:
                | power (number): spectral raised to power

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        if isinstance(self.value, Number):
            self.value = self.value * np.ones(self.wn.shape)

        return Spectral(ID='{}**{}'.format(self.ID,power), value=self.value ** power, 
            wl=self.wl, wn=self.wn,
         desc='{}**{}'.format(self.desc,power))

    ############################################################
    ##
    def plot(self, filename, heading, ytitle=''):
        """Do a simple plot of spectral variable(s)

            Args:
                | filename (str): filename for png graphic
                | heading (str): graph heading
                | ytitle (str): graph y-axis title

            Returns:
                | Nothing, writes png file to disk

            Raises:
                | No exception is raised.
        """
        import pyradi.ryplot as ryplot
        p = ryplot.Plotter(1,2,1,figsize=(8,5))

        if isinstance(self.value, np.ndarray):
            p.plot(1,self.wl,self.value,heading,'Wavelength $\mu$m',
                ytitle)
            p.plot(2,self.wn,self.value,heading,'Wavenumber cm$^{-1}$',
                ytitle)

        p.saveFig(ryfiles.cleanFilename(filename))


################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':
    import math
    import sys
    from scipy.interpolate import interp1d
    import pyradi.ryplanck as ryplanck
    import pyradi.ryplot as ryplot
    import pyradi.ryfiles as ryfiles
    import os

    figtype = ".png"  # eps, jpg, png
    # figtype = ".eps"  # eps, jpg, png

    doAll = False

    if True:
        spectrals = {}
        atmos = {}
        sensors = {}
        targets = {}

        # test loading of spectrals
        print('\n---------------------Spectrals:')
        spectral = np.loadtxt('data/MWIRsensor.txt')
        spectrals['ID1'] = Spectral('ID1',value=.3,wl=spectral[:,0],desc="MWIR transmittance")
        spectrals['ID2'] = Spectral('ID2',value=1-spectral[:,1],wl=spectral[:,0],desc="MWIR absorption")
        spectrals['ID3'] = spectrals['ID1'] * spectrals['ID2']
        spectrals['ID4'] = spectrals['ID1'] ** 3
        spectrals['ID5'] = spectrals['ID2'] * 1.67

        for key in spectrals:
            print(spectrals[key])
        for key in spectrals:
            filename ='{}-{}'.format(key,spectrals[key].desc)
            spectrals[key].plot(filename=filename,heading=spectrals[key].desc,ytitle='Value')

