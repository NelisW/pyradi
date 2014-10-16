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

# The Initial Developers of the Original Code are CJ Willers & L van der Westhuizen. 
# Portions created by the authors are Copyright (C) 2014
# All Rights Reserved.

# Contributor(s): ______________________________________.
################################################################
"""
This class provides lookup functionality between source temperature, 
source radiance and sensor signal.  In this class the sensor signal is called 
digital level, but it represents any atribrary sensor signal unit.  

See the __main__ function for examples of use.

This package was partly developed to provide additional material in support of students 
and readers of the book Electro-Optical System Analysis and Design: A Radiometry 
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""
### this file uses four  spaces for one tab 

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['RadLookup']

import sys
if sys.version_info[0] > 2:
    print("pyradi is not yet ported to Python 3, because imported modules are not yet ported")
    exit(-1)

import numpy as np
import collections
import os.path

import pyradi.ryplot as ryplot
import pyradi.ryfiles as ryfiles
import pyradi.ryutils as ryutils
import pyradi.ryplanck as ryplanck

class RadLookup:
    """Performs radiometric lookup capability between temperature and radiance, given 
       camera spectral and calibration data.

    Given spectral and temperature data this class calculates loopup tables and 
    provide lookup functions operating on the tables.

    The class provides two parallel functional capabilities:

    * given spectral data and no calibration data it calculates a simple radiance-based
      lookup between temperture and radiance, assuming Planck-law relationships.

    * given camera calibration data it relates between signal value, temperature and
      radiance.  It accounts for the effect of hot optics that cause a lower asymptotic
      radiance level.  The calibration mode supports linear interpolation between two
      calibration curves, to account for the instrument internal temperature. This 
      mode requires the sigMin, sigMax, sigInc, and dicCaldat parameters.

    By not passing the calibration parameters simply means that that part of the code is 
    not executed and only the simple radiance-based lookup is available. 

    Spectral data parameter may be either a filename (data read from file) or a 
    numpy array np.array(:,1) with the data on (nuMin, nuMax,nuInc) scale.  The data file
    must have two colums: first column is wavelength, and second
    column is the spectral value at this wavelength.  Data read in from the file
    will be interpolated to (nuMin, nuMax,nuInc) scale. If the parameter is None, 
    then unity spectral values will be used.

    The camera calibration data set requires the following data:

    *  sigMin, sigMax, sigInc: the parameters to define the signal magnitude vector.

    *  dicCaldata:  the dictionary containing the camera calibration data.  The dictionary
       key is the instrument internal temperature [deg-C] (one or two values required).  
       For each key, provide a numpy array where the *first column* is the calibration 
       source temperature [deg-C], and the *second column* is the signal measured on 
       the instrument and the *third column* is the radiance for this temperature 
       (only after the tables have been calculated).

    *  dicPower: the dictionary containing the camera calibration curve lower knee
       sharpness.  The dictionary key is the instrument internal temperature 
       (one or two values required). 

    *  dicFloor: the dictionary containing the camera calibration curve lower asymptotic
       signal value.  The dictionary key is the instrument internal temperature 
       (one or two values required). 

    Error handling is simply to test for certain conditions and then execute the 
    task if conditions are met, otherwise do nothing.

    Args:
        |  specName (string): Name for this lookup data set, used in graphs.
        |  nu (np.array(N,1)): wavenumber vector to be used in spectral calcs.
        |  tmprLow (float): Lower temperature bound [K] for lookup tables.
        |  tmprHi (float):  Upper temperature bound [K] for lookup tables.
        |  tmprInc (float):  Increment temperature [K] for lookup tables.
        |  sensorResp (string/np.array(N,1)): sensor/detector spectral filename or array data (optional).
        |  opticsTau (string/np.array(N,1)): opticsTransmittance spectral filename or array data (optional).
        |  filterTau ((string/np.array(N,1)): filter spectral filename or array data (optional). 
        |  atmoTau (string/np.array(N,1)): atmoTau spectral filename or array data (optional).
        |  sourceEmis (string/np.array(N,1)): sourceEmis spectral filename or array data (optional).
        |  sigMin (float): minimum signal value, typically 2**0 (optional). 
        |  sigMax (float): maximum signal value, typically 2**14 (optional). 
        |  sigInc(float): signal increment, typically 2**8 (optional).
        |  dicCaldata (dict): calibration data for sensor.
        |  dicPower (dict): cal curve lower asymptote knee sharpness parameter (optional).
        |  dicFloor (dict): cal curve lower asymptote  parameter (optional).

    Returns:
        |  Nothing, but init() loads data and  calculates the tables on instantiation.

    Raises:
        | No exception is raised.

    """

    ################################################################################
    def __init__(self, specName, nu, tmprLow, tmprHi, tmprInc,
                sensorResp=None, opticsTau=None, filterTau=None, atmoTau=None, 
                sourceEmis=None, 
                sigMin=None, sigMax=None, sigInc=None, dicCaldata=None, 
                dicPower=None, dicFloor=None):
        """Initialise and loads the camera calibration data from files and calculate the lookup tables.

        """

        __all__ = ['info', 
                    'LookupDLRad', 'LookupDLTemp', 'LookupTempRad',  'LookupRadTemp', 
                    'PlotTempRadiance',
                    'PlotSpectrals', 'PlotCalSpecRadiance', 'PlotCalDLRadiance', 
                    'PlotCalTempRadiance', 'PlotCalTintRad', 'PlotCalDLTemp'
                    ]

        self.specName = specName
        self.nu = nu
        self.sigMin = sigMin
        self.sigMax = sigMax
        self.sigInc = sigInc
        self.tmprLow = tmprLow
        self.tmprHi = tmprHi
        self.tmprInc = tmprInc
        self.sensorResp = sensorResp
        self.opticsTau = opticsTau
        self.filterTau = filterTau
        self.atmoTau = atmoTau
        self.sourceEmis = sourceEmis
        self.dicCaldata = dicCaldata
        self.dicPower = dicPower
        self.dicFloor = dicFloor

        #flags to signal task completion
        self.spectralsLoaded = False
        self.hiresTablesCalculated = False
        self.calTablesCalculated = False

        #set up the spectral domain in wavelength
        self.wl = ryutils.convertSpectralDomain(self.nu,  type='nl')

        #load the spectral files
        self.LoadSpectrals()

        # multiply to calculate the total spectral shape.
        self.specEff = self.specEmis * self.specAtmo * \
                         self.specFilter * self.specSensor * self.specOptics

        # convert calibration temperatures to K
        if self.dicCaldata:
            self.calLokey = min(self.dicCaldata.keys())
            self.calHikey = max(self.dicCaldata.keys())
            for tmprInstr in self.dicCaldata:
                self.dicCaldata[tmprInstr][:,0] += 273.16

        #calculate the lookup tables
        self.CalculateDataTables()

    ################################################################################
    def Info(self):
        """Write the calibration data file data to a string and return string.

        Args:
            | None.
     
        Returns:
            |  (string) containing the key information for this class.

        Raises:
            | No exception is raised.
        """

        str = ""
        str += 'Lookup Name          = {}\n'.format(self.specName)

        if type(self.sensorResp) == type('str'):
            str += 'Sensor Response      = {}\n'.format(self.sensorResp)
        if type(self.opticsTau) == type('str'):
            str += 'Optics Transmittance = {}\n'.format(self.opticsTau)
        if type(self.filterTau) == type('str'):
            str += 'Filter               = {}\n'.format(self.filterTau)
        if type(self.sourceEmis) == type('str'):
            str += 'Source Emissivity    = {}\n'.format(self.sourceEmis)
        if type(self.atmoTau) == type('str'):
            str += 'Atmospheric Transm   = {}\n'.format(self.atmoTau)

        str += 'Nu (Min, Max, Inc)   = ({}, {}, {})\n'.format(
            np.min(self.nu),np.max(self.nu),self.nu[1]-self.nu[0])

        if self.dicCaldata:
            str += 'Calibration data set'
            for i,tmprInstr in enumerate(sorted(self.dicCaldata)):
                str += '\n\nInstrument temperature = {} C\n'.format(tmprInstr)
                str += 'DL floor               = {}\n'.format(self.dicFloor[tmprInstr])
                str += 'DL power               = {}\n'.format(self.dicPower[tmprInstr])
                str += '      T Deg-C         DL         L'
                str += '\n'
                str += np.array_str(self.dicCaldata[tmprInstr], max_line_width=120, precision=4, )
                str += '\nstraight line fit L = {} DL + {}'.format(self.DlRadCoeff[tmprInstr][0],
                      self.DlRadCoeff[tmprInstr][1])

        return str


    ################################################################
    def LoadSpectrals(self):
        """Load the five required spectral parameters.

        If the spectral parameters are strings, the strings are used as filenames 
        and the data loaded from file.  If None, unity values are assumed. If not 
        a string or None, the parameters are used as is, and must be numpy arrays
        with shape (N,1) where the N vector exactly matches to spectral samples 

        Args:
            |  None.

        Returns:
            |  None. Side-effect of loaded spectrals.


        Raises:
            | No exception is raised.
        """
        #load the various input files, interpolate on the fly to local spectrals.
        if self.sourceEmis is not None:
            if type(self.sourceEmis) == type('str'):
                self.specEmis= ryfiles.loadColumnTextFile(self.sourceEmis, 
                    loadCol=[1], normalize=0, abscissaOut=self.wl)
        else:
            self.specEmis= np.ones(self.nu.shape)

        if self.atmoTau is not None:
            if type(self.sourceEmis) == type('str'):
                self.specAtmo = ryfiles.loadColumnTextFile(self.atmoTau, 
                    loadCol=[1], normalize=0, abscissaOut=self.wl)
        else:
            self.specAtmo= np.ones(self.nu.shape)

        if self.filterTau is not None:
            if type(self.sourceEmis) == type('str'):
                self.specFilter = ryfiles.loadColumnTextFile(self.filterTau, 
                    loadCol=[1], normalize=0, abscissaOut=self.wl)
        else:
            self.specFilter= np.ones(self.nu.shape)

        if self.sensorResp is not None:
            if type(self.sourceEmis) == type('str'):
                self.specSensor = ryfiles.loadColumnTextFile(self.sensorResp, 
                    loadCol=[1], normalize=0, abscissaOut=self.wl)
        else:
            self.specSensor= np.ones(self.nu.shape)

        if self.opticsTau is not None:
            if type(self.sourceEmis) == type('str'):
                self.specOptics = ryfiles.loadColumnTextFile(self.opticsTau, 
                    loadCol=[1], normalize=0, abscissaOut=self.wl)
        else:
            self.specOptics= np.ones(self.nu.shape)

        self.spectralsLoaded = True


    ################################################################
    def CalculateDataTables(self):
        """Calculate the mapping functions between sensor signal, radiance and temperature.

           Using the spectral curves and DL vs. temperature calibration inputs
           calculate the various mapping functions between digital level, radiance
           and temperature. Set up the various tables for later conversion.

        Args:
            |  None.

        Returns:
            |  None. Side effect of all tables calculated.

        Raises:
            | No exception is raised.

        """

        if  self.spectralsLoaded:

            # calculate a high resolution lookup table between temperature and radiance
            # this is independent from calibration data, no keyTInstr dependence
            tempHiRes = np.linspace(self.tmprLow, self.tmprHi,  1 + \
                                      (self.tmprHi - self.tmprLow )/self.tmprInc)
            xx = np.ones(tempHiRes.shape)
            _, spectrlHR = np.meshgrid(xx,self.specEff)
            specLHiRes = spectrlHR * ryplanck.planck(self.nu, tempHiRes, type='en')
            LHiRes = np.trapz(specLHiRes, x=self.nu, axis=0) / np.pi
            self.TableTempRad = np.hstack((tempHiRes.reshape(-1,1),LHiRes.reshape(-1,1) ))
            self.hiresTablesCalculated = True

            # this part is only done if there is calibration data
            if self.dicCaldata is not None:

                #set up the DL range
                self.interpDL = np.linspace(self.sigMin, self.sigMax, 1 + \
                                         (self.sigMax - self.sigMin )/self.sigInc )
                # print(np.min(self.interpDL))

                #create containers to store
                self.dicSpecRadiance = collections.defaultdict(float)
                self.dicRadiance = collections.defaultdict(float)
                self.DlRadCoeff = collections.defaultdict(float)
                self.dicinterpDL = collections.defaultdict(float)
                self.dicTableDLRad = collections.defaultdict(float)

                #step through all instrument temperatures
                for tmprInstr in self.dicCaldata:
                    #temperature is in 0'th column of dicCaldata[tmprInstr]
                    #digital level is in 1'st column of dicCaldata[tmprInstr]
                    #radiance is in 2'nd column of dicCaldata[tmprInstr]
                    #planck array has shape (nu,temp), now get spectrals to same shape, then
                    #integrate along nu axis=0

                    self.dicCaldata[tmprInstr] = self.dicCaldata[tmprInstr].astype(np.float64)

                    xx = np.ones(self.dicCaldata[tmprInstr][:,0].shape)
                    _, spectrl = np.meshgrid(xx,self.specEff)
                    #now spectrl has the same shape as the planck function return

                    self.dicSpecRadiance[tmprInstr] = \
                      spectrl * ryplanck.planck(self.nu, self.dicCaldata[tmprInstr][:,0], type='en')

                    self.dicRadiance[tmprInstr] = \
                      np.trapz(self.dicSpecRadiance[tmprInstr], x=self.nu,  axis=0) / np.pi

                    #if first time, stack horizontally to the array, otherwise overwrite
                    if self.dicCaldata[tmprInstr].shape[1] < 3:
                        self.dicCaldata[tmprInstr] = np.hstack((self.dicCaldata[tmprInstr],
                                self.dicRadiance[tmprInstr].reshape(-1,1)))
                    else:
                        self.dicCaldata[tmprInstr][:,2] = self.dicRadiance[tmprInstr].reshape(-1,)

                    # now determine the best fit between radiance and DL
                    # the relationship should be linear y = mx + c
                    # x=digital level, y=radiance
                    coeff = np.polyfit(self.dicCaldata[tmprInstr][:,1],
                                      self.dicRadiance[tmprInstr].reshape(-1,), deg=1)
                    # print('straight line fit DL = {} L + {}'.format(coeff[0],coeff[1]))
                    self.DlRadCoeff[tmprInstr] = coeff
                    self.interpL = np.polyval(coeff, self.interpDL)  #.astype(np.float64)

                    # add the DL floor due to instrument & optics temperature
                    pow = self.dicPower[tmprInstr]
                    self.dicinterpDL[tmprInstr] = \
                         (self.interpDL ** pow + self.dicFloor[tmprInstr] ** pow) ** (1./pow)

                    #now save a lookup table for tmprInstr value
                    self.dicTableDLRad[tmprInstr] = \
                          np.hstack(([self.dicinterpDL[tmprInstr].reshape(-1,1), self.interpL.reshape(-1,1)]))
                    # for some strange reason this is necessary to force float64
                    self.dicTableDLRad[tmprInstr] = self.dicTableDLRad[tmprInstr].astype(np.float64)

                #calculate the radiance in the optics and sensor for later interpolation of Tinternal
                tempTint = 273.15 + np.linspace(np.min(self.dicCaldata.keys())-20,
                                              np.max(self.dicCaldata.keys())+20, 101)
                xx = np.ones(tempTint.shape)
                _, spectrlHR = np.meshgrid(xx,self.specEff)
                specLTint = spectrlHR * ryplanck.planck(self.nu, tempTint, type='en')
                LTint = np.trapz(specLTint, x=self.nu, axis=0) / np.pi
                self.TableTintRad = np.hstack((tempTint.reshape(-1,1),LTint.reshape(-1,1) ))

                self.calTablesCalculated = True

        else:
            #reset containers 
            self.TableTempRad = None
            self.interpDL = None
            self.dicTableDLRad = None
            self.DlRadCoeff = None
            self.dicinterpDL = None
            self.dicSpecRadiance = None
            self.dicRadiance = None
            self.TableTintRad = None
            
            self.calTablesCalculated = False
            self.hiresTablesCalculated = False


    ################################################################
    def LookupDLRad(self, DL, Tint):
        """Calculate the radiance associated with a DL and Tint pair.
           Interpolate linearly on Tint radiance not temperature.

        Args:
            |  DL (float, np.array[N,]): scalar, list or array of sensor signal values.
            |  Tint (float): scalar, internal temperature of the sensor.

        Returns:
            |  (np.array[N,]) radiance W/(sr.m2) values associated with sensor signals.

        Raises:
            | No exception is raised.
        """

        if self.calTablesCalculated:

            DL = np.asarray(DL).astype(np.float64)

            #get radiance values for lower and upper Tint and the actual Tin
            Llo = np.interp(self.calLokey+273.15, self.TableTintRad[:,0], self.TableTintRad[:,1])

            #find the parametric value for Tint radiance, do this once
            #If only one calibration temperature is available then skip prametric
            if((self.calHikey - self.calLokey)==0):
                paraK = 0
            else:
                Lhi = np.interp(self.calHikey+273.15, self.TableTintRad[:,0], self.TableTintRad[:,1])
                Lti = np.interp(Tint+273.15, self.TableTintRad[:,0], self.TableTintRad[:,1])
                paraK = (Lti - Llo) / (Lhi - Llo)

            rad =  self.LookupDLRadHelper(DL, paraK)

        else:
            rad = None

        return rad


    ################################################################
    def LookupDLRadHelper(self, DL, paraK):
        """Calculate the radiance associated with a DL and parametric pair. The
           parametric variable was calculated once and used for all DL values.

        Args:
            |  DL (float, np.array[N,]): scalar, list or array of sensor signal values.
            |  paraK (float): scalar, internal temperature parametric values.

        Returns:
            |  (np.array[N,]) radiance W/(sr.m2) values associated with sensor signals.

        Raises:
            | No exception is raised.
        """


        # numpy's interp supports arrays as input, but only does linear interpolation
        lo = np.interp(DL, self.dicTableDLRad[self.calLokey][:,0], self.dicTableDLRad[self.calLokey][:,1])

        if (paraK != 0):
            hi = np.interp(DL, self.dicTableDLRad[self.calHikey][:,0], self.dicTableDLRad[self.calHikey][:,1])
            return  lo + (hi - lo) * paraK
        else:
            return lo

    ################################################################
    def LookupDLTemp(self, DL, Tint):
        """Calculate the temperature associated with a DL and Tint pair.
           Interpolate linearly on Tint temperature - actually we must
           interpolate linearly on radiance - to be done later.

        Args:
            |  DL (float, np.array[N,]): scalar, list or array of sensor signal values.
            |  Tint (float): scalar, internal temperature of the sensor.

        Returns:
            |  (np.array[N,]) temperature K values associated with sensor signals.

        Raises:
            | No exception is raised.
        """

        if self.calTablesCalculated:

            L = self.LookupDLRad(DL, Tint)
            t = np.interp(L, self.TableTempRad[:,1], self.TableTempRad[:,0])
        else:
            t = None

        return t

    ###############################################################################
    def LookupRadTemp(self, radiance):
        """Calculate the temperature associated with a radiance value for the 
        given spectral shapes (no calibration involved).

        Args:
            |  radiance (float, np.array[N,]): scalar, list or array of radiance W/(sr.m2) values.

        Returns:
            |  (np.array[N,]) temperature K values associated with radiance values.

        Raises:
            | No exception is raised.
        """

        if self.hiresTablesCalculated:
            t = np.interp(radiance, self.TableTempRad[:,1], self.TableTempRad[:,0])
        else:
            t = None

        return t

    ###############################################################################
    def LookupTempRad(self, temperature):
        """Calculate the radiance associated with a temperature for the 
        given spectral shapes (no calibration involved).

        Args:
            |  temperature(np.array[N,])  scalar, list or array of temperature K values.

        Returns:
            |  radiance (float, np.array[N,]): radiance W/(sr.m2) associated with temperature values..

        Raises:
            | No exception is raised.
        """

        if self.hiresTablesCalculated:

            rad = np.interp(temperature, self.TableTempRad[:,0], self.TableTempRad[:,1])
        else:
            rad = None

        return rad


    ################################################################
    def PlotSpectrals(self, savePath=None, saveExt='png'):
        """Plot all spectral curve data to a single graph.

        The filename is constructed from the given object name, save path, and
        the word 'spectrals'.

        Args:
            | savePath (string): Path to where the plots must be saved (optional).
            | saveExt (string) : Extension to save the plot as, default of 'png' (optional).
        Returns:
            | None, the images are saved to a specified location or in the location
            | from which the script is running.
        Raises:
            | No exception is raised.
        """

        if  self.spectralsLoaded == True:

            p = ryplot.Plotter(1, figsize=(10,5))
            p.semilogY(1,self.wl,self.specEmis,label=['Emissivity'])
            p.semilogY(1,self.wl,self.specAtmo,label=['Atmosphere'])
            p.semilogY(1,self.wl,self.specFilter,label=['Filter'])
            p.semilogY(1,self.wl,self.specSensor,label=['Sensor'])
            p.semilogY(1,self.wl,self.specOptics,label=['Optics'])
            p.semilogY(1,self.wl,self.specEff,label=['Effective'])
            currentP = p.getSubPlot(1)
            currentP.set_xlabel('Wavelength {}m'.format(ryutils.upMu(False)))
            currentP.set_ylabel('Response')
            currentP.set_title('Spectral Response for {}'.format(self.specName))

            if savePath==None:
                savePath=self.specName 
            else:
                savePath = os.path.join(savePath,self.specName)

            p.saveFig('{}-spectrals.{}'.format(savePath, saveExt))

    ################################################################
    def PlotCalSpecRadiance(self, savePath=None, saveExt='png'):
        """Plot spectral radiance data for the calibration temperatures.

        The filename is constructed from the given object name, save path, and
        the word 'CalRadiance'.

        Args:
            | savePath (string): Path to where the plots must be saved (optional).
            | saveExt (string) : Extension to save the plot as, default of 'png' (optional).
        Returns:
            | None, the images are saved to a specified location or in the location
            | from which the script is running.
        Raises:
            | No exception is raised.
        """

        if self.calTablesCalculated:

            p = ryplot.Plotter(1,2,1, figsize=(10,10))
            for j,tmprInstr in enumerate(self.dicCaldata):
                #print(tmprInstr,j)

                # build temperature labels
                labels = []
                for temp in self.dicCaldata[tmprInstr][:,0]:
                    labels.append('{:.0f} $^\circ$C, {:.0f} K'.format(temp-273.15, temp))

                p.plot(1+j,self.wl,self.dicSpecRadiance[tmprInstr],label=labels)
                currentP = p.getSubPlot(1+j)
                currentP.set_xlabel('Wavelength {}m'.format(ryutils.upMu(False)), fontsize=12)
                currentP.set_ylabel('Radiance W/(m$^2$.sr.cm$^{-1}$)', fontsize=12)
                title = '{} at Tinstr={} $^\circ$C'.format(self.specName, tmprInstr)
                currentP.set_title(title, fontsize=12)

            if savePath==None:
                savePath=self.specName 
            else:
                savePath = os.path.join(savePath,self.specName)

            p.saveFig('{}-CalRadiance.{}'.format(savePath, saveExt))


    ################################################################
    def PlotCalDLRadiance(self, savePath=None, saveExt='png'):
        """Plot DL level versus radiance for both camera temperatures

        The filename is constructed from the given object name, save path, and
        the word 'CaldlRadiance'.

        Args:
            | savePath (string): Path to where the plots must be saved (optional).
            | saveExt (string) : Extension to save the plot as, default of 'png' (optional).
        Returns:
            | None, the images are saved to a specified location or in the location
            | from which the script is running.
        Raises:
            | No exception is raised.
        """

        if self.calTablesCalculated:

            p = ryplot.Plotter(1,1,1, figsize=(10,5))
            for j,tmprInstr in enumerate(self.dicCaldata):
              if j == 0:
                plotColB = ['r']
                plotColM = ['b--']
              else:
                plotColB = ['c']
                plotColM = ['g--']
              p.plot(1,self.dicTableDLRad[tmprInstr][:,0],self.dicTableDLRad[tmprInstr][:,1],
                label=['Best fit line {}$^\circ$C'.format(tmprInstr)],plotCol=plotColB)
              p.plot(1,self.dicCaldata[tmprInstr][:,1],self.dicRadiance[tmprInstr],
                label=['Measured {}$^\circ$C'.format(tmprInstr)],markers=['x'],plotCol=plotColM)
            currentP = p.getSubPlot(1)
            currentP.set_xlabel('Digital level')
            currentP.set_ylabel(' Radiance at sensor W/(m$^2$.sr)')
            currentP.set_title('{}'.format(self.specName))

            if savePath==None:
                savePath=self.specName 
            else:
                savePath = os.path.join(savePath,self.specName)

            p.saveFig('{}-CaldlRadiance.{}'.format(savePath, saveExt))


    ################################################################
    def PlotTempRadiance(self, savePath=None, saveExt='png'):
        """Plot temperature versus radiance for both camera temperatures

        The filename is constructed from the given object name, save path, and
        the word 'TempRadiance'.

        Args:
            | savePath (string): Path to where the plots must be saved (optional).
            | saveExt (string) : Extension to save the plot as, default of 'png' (optional).
        Returns:
            | None, the images are saved to a specified location or in the location
            | from which the script is running.
        Raises:
            | No exception is raised.
        """

        if self.hiresTablesCalculated:

            p = ryplot.Plotter(1,1,1, figsize=(10,5))
            p.plot(1,self.TableTempRad[:,0],self.TableTempRad[:,1])
            currentP = p.getSubPlot(1)
            currentP.set_xlabel('Temperature K')
            currentP.set_ylabel('Radiance at sensor W/(m$^2$.sr)')
            currentP.set_title('{}'.format(self.specName))
            currentP.set_ylim([0,np.max(self.TableTempRad[:,1])])

            if savePath==None:
                savePath=self.specName 
            else:
                savePath = os.path.join(savePath,self.specName)

            p.saveFig('{}-TempRadiance.{}'.format(savePath, saveExt))


    ################################################################
    def PlotCalTempRadiance(self, savePath=None, saveExt='png'):
        """Plot temperature versus radiance for both camera temperatures

        The filename is constructed from the given object name, save path, and
        the word 'CalTempRadiance'.

        Args:
            | savePath (string): Path to where the plots must be saved (optional).
            | saveExt (string) : Extension to save the plot as, default of 'png' (optional).
        Returns:
            | None, the images are saved to a specified location or in the location
            | from which the script is running.
        Raises:
            | No exception is raised.
        """

        if self.calTablesCalculated:

            p = ryplot.Plotter(1,1,1, figsize=(10,5))
            for j,tmprInstr in enumerate(self.dicCaldata):
                p.plot(1,self.dicCaldata[tmprInstr][:,0],self.dicCaldata[tmprInstr][:,2],
                    label=['Tint {} $^\circ$C'.format(tmprInstr)])
                currentP = p.getSubPlot(1)
                currentP.set_xlabel('Temperature K')
                currentP.set_ylabel('Radiance at sensor W/(m$^2$.sr)')
                currentP.set_title('{}'.format(self.specName))
                currentP.set_ylim([0,np.max(self.dicCaldata[tmprInstr][:,2])])

            if savePath==None:
                savePath=self.specName 
            else:
                savePath = os.path.join(savePath,self.specName)

            p.saveFig('{}-CalTempRadiance.{}'.format(savePath, saveExt))


    ################################################################
    def PlotCalTintRad(self, savePath=None, saveExt='png'):
        """Plot optics radiance versus instrument temperature
        
        The filename is constructed from the given object name, save path, and
        the word 'CalInternal'.

        Args:
            | savePath (string): Path to where the plots must be saved (optional).
            | saveExt (string) : Extension to save the plot as, default of 'png' (optional).
        Returns:
            | None, the images are saved to a specified location or in the location
            | from which the script is running.
        Raises:
            | No exception is raised.
        """
        
        if self.calTablesCalculated:

            p = ryplot.Plotter(1,1,1, figsize=(10,5))
            ptitle = '{} sensor internal radiance'.format(self.specName)
            p.plot(1,self.TableTintRad[:,0]-273.15,self.TableTintRad[:,1], ptitle=ptitle,
              xlabel='Internal temperature $^\circ$C', ylabel='Radiance W/(m$^2$.sr)')

            if savePath==None:
                savePath=self.specName 
            else:
                savePath = os.path.join(savePath,self.specName)

            p.saveFig('{}-CalInternal.{}'.format(savePath, saveExt))


    ################################################################
    def PlotCalDLTemp(self, savePath=None, saveExt='png'):
        """Plot digital level versus temperature for both camera temperatures

        The filename is constructed from the given object name, save path, and
        the word 'CalDLTemp'.

        Args:
            | savePath (string): Path to where the plots must be saved (optional).
            | saveExt (string) : Extension to save the plot as, default of 'png' (optional).
        Returns:
            | None, the images are saved to a specified location or in the location
            | from which the script is running.
        Raises:
            | No exception is raised.
        """
        
        if self.calTablesCalculated:

            p = ryplot.Plotter(1,1,1, figsize=(10,5))
            for j,tmprInstr in enumerate(self.dicCaldata):
                DL = self.dicTableDLRad[tmprInstr][:,0]
                temp = self.LookupDLTemp(DL, tmprInstr)
                if j == 0:
                    plotColB = ['r']
                    plotColM = ['b--']
                else:
                    plotColB = ['c']
                    plotColM = ['g--']
                p.plot(1,DL,temp,label=['Best fit line {}$^\circ$C'.format(tmprInstr)],plotCol=plotColB)
                p.plot(1,self.dicCaldata[tmprInstr][:,1],self.dicCaldata[tmprInstr][:,0],
                   label=['Measured {}$^\circ$C'.format(tmprInstr)],markers=['x'],plotCol=plotColM)
                currentP = p.getSubPlot(1)
                currentP.set_xlabel('Digital level')
                currentP.set_ylabel('Temperature K')
                currentP.set_title('{} at Tinstr={} $^\circ$C'.format(self.specName, tmprInstr))

            if savePath==None:
                savePath=self.specName 
            else:
                savePath = os.path.join(savePath,self.specName)

            p.saveFig('{}-CalDLTemp.{}'.format(savePath, saveExt))


################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    doPlots = True

    nuMin = 700
    nuMax = 1665
    nuInc = 5
    nu = np.linspace(nuMin, nuMax, 1 + (nuMax - nuMin )/nuInc )


    tmprMin = 300.
    tmprMax = 800.
    tmprInc = 10.

    # print('-------------------------------------------------------')
    # # first test is simple test with unity spectrals, no camera cal data
    # lut1 = RadLookup('lut1', nu, tmprMin,tmprMax,tmprInc)
    # print(lut1.Info())
    # if doPlots:
    #     lut1.PlotSpectrals()
    #     lut1.PlotTempRadiance()
    #     lut1.PlotCalSpecRadiance()
    #     lut1.PlotCalDLTemp()
    #     lut1.PlotCalDLRadiance()
    #     lut1.PlotCalTempRadiance()
    #     lut1.PlotCalTintRad()
    # tempr1 = [300., 307.5, 505., 792.5, 800.]
    # rad1 = lut1.LookupTempRad(tempr1)
    # tempr1i =  lut1.LookupRadTemp(rad1)
    # print('Input temperature values = {}'.format(tempr1))
    # print('Lut radiance values = {}'.format(rad1))
    # print('Lut temperature values = {}'.format(tempr1i))
    # # now calculate from first principles
    # L1stP1 = np.trapz(ryplanck.planck(nu, tempr1, type='en'), x=nu, axis=0) / np.pi
    # print('First principles radiance values = {}'.format(L1stP1))

    # print('\n-------------------------------------------------------')
    # # second test has non-unity spectrals, no camera cal data
    # lut2 = RadLookup('lut2', nu, tmprMin,tmprMax,tmprInc,
    #      'data/LWIRsensor.txt', 'data/LW100mmLens.txt', 'data/LWND10.txt', 
    #          'data/Unity.txt', 'data/Unity.txt')
    # print(lut2.Info())
    # if doPlots:
    #     lut2.PlotSpectrals()
    #     lut2.PlotTempRadiance()
    #     lut2.PlotCalSpecRadiance()
    #     lut2.PlotCalDLTemp()
    #     lut2.PlotCalDLRadiance()
    #     lut2.PlotCalTempRadiance()
    #     lut2.PlotCalTintRad()
    # tempr2 = [300., 307.5, 505., 792.5, 800.]
    # rad2 = lut2.LookupTempRad(tempr2)
    # tempr2i =  lut2.LookupRadTemp(rad2)
    # print('Input temperature values = {}'.format(tempr2))
    # print('Lut radiance values = {}'.format(rad2))
    # print('Lut temperature values = {}'.format(tempr2i))

    print('\n-------------------------------------------------------')
    # third test has non-unity spectrals, and camera cal data

    arrLo = np.asarray([
        [50., 100., 150., 200., 250., 300., 350., 400., 450. ],
        [4571., 5132., 5906., 6887., 8034., 9338., 10834., 12386., 14042.]
        ]).T
    arrHi = np.asarray([
        [50., 100., 150., 200., 250., 300., 350., 400., 450. ],
        [5477, 6050, 6817, 7789, 8922, 10262, 11694, 13299, 14921]
        ]).T

    Tlo = 17.1
    Thi = 34.4
    dicCaldata = {Tlo: arrLo, Thi: arrHi}
    dicPower = {Tlo: 10., Thi: 10}
    dicFloor = {Tlo: 3625, Thi: 4210}

    print('dicPower = {}'.format(dicPower))
    print('dicFloor = {}'.format(dicFloor))
    print('dicCaldata = {}'.format(dicCaldata))

    lut3 = RadLookup('lut3', nu, tmprMin,tmprMax,tmprInc,
         'data/LWIRsensor.txt', 'data/LW100mmLens.txt', 'data/LWND10.txt', 
             'data/Unity.txt', 'data/Unity.txt',
             sigMin=0, sigMax=2.**14, sigInc=2.**6, dicCaldata=dicCaldata, 
                dicPower=dicPower, dicFloor=dicFloor)
    print(lut3.Info())
    if doPlots:
        lut3.PlotSpectrals()
        lut3.PlotTempRadiance()
        lut3.PlotCalSpecRadiance()
        lut3.PlotCalDLTemp()
        lut3.PlotCalDLRadiance()
        lut3.PlotCalTempRadiance()
        lut3.PlotCalTintRad()
    tempr3 = [300., 307.5, 505., 792.5, 800.]
    rad3 = lut3.LookupTempRad(tempr3)
    tempr3i =  lut3.LookupRadTemp(rad3)
    print('Input temperature values = {}'.format(tempr3))
    print('Lut radiance values = {}'.format(rad3))
    print('Lut temperature values = {}'.format(tempr3i))
    print(' ')   
    print('Use the cal tables tp look up the original input values:')
    print(lut3.LookupDLTemp(arrLo[:,1], Tlo) - 273.16)
    print(arrLo[:,0] - 273.16)
    print(' ')   
    print('Convert DL->Radiance->Temperature (at low instrunent temperature:')
    calRad = lut3.LookupDLRad(arrLo[:,1], Tlo)
    print(calRad)
    print(lut3.LookupRadTemp(calRad) - 273.16)
    print(' ')   
    print('Convert DL->Radiance->Temperature (at high instrunent temperature:')
    print(lut3.LookupDLTemp(arrHi[:,1], Thi) - 273.16)
    print(arrHi[:,0] - 273.16)
    calRad = lut3.LookupDLRad(arrHi[:,1], Thi)
    print(calRad)
    print(lut3.LookupRadTemp(calRad) - 273.16)


