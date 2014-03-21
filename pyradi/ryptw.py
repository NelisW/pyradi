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

# The Initial Developer of the Original Code is JJ Calitz,
# Portions created by JJ Calitz are Copyright (C) 2011-2012
# All Rights Reserved.

#The author wishes to thank FLIR Advanced Thermal Solutions for the permission to 
#publicly release our Python version of the *.ptw file reader. Note that the
#copyright to the proprietary *.ptw file format remains the property of FLIR Inc.

# Contributor(s):  JJ Calitz, MS Willers & CJ Willers.
################################################################
"""
This module provides functionality to read the contents of files in the
PTW file format and convert the raw data to source radiance or souorce 
temperature (provided that the instrument calibration data is available).

Functions are provided to read the binary Agema/Cedip/FLIR Inc PTW format
into data structures for further processing.

The following functions are available to read PTW files:

    | readPTWHeader(ptwfilename)
    | showHeader(header)
    | getPTWFrame (header, frameindex)


readPTWHeader(ptwfilename) :
Returns a class object defining all the header information in ptw file.

showHeader(header) :
Returns nothing.  Prints the PTW header content to the screen.

getPTWFrame (header, frameindex) :
Return the raw DL levels of the frame defined by frameindex.

The authors wish to thank FLIR Advanced Thermal Solutions for the permission
to publicly release our Python version of the ptw file reader.  Please note that the
copyright to the proprietary ptw file format remains the property of FLIR Inc.


This package was partly developed to provide additional material in support of students 
and readers of the book Electro-Optical System Analysis and Design: A Radiometry 
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""

#prepare so long for Python 3
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

__version__= "$Revision$"
__author__='JJ Calitz'
__all__=['myint','mylong','myfloat','mybyte',  'ReadPTWHeader', 'GetPTWFrameFromFile',
'ShowHeader', 'GetPTWFrame', 'GetPTWFrames']

import sys
if sys.version_info[0] > 2:
    print("pyradi is not yet ported to Python 3, because imported modules are not yet ported")
    exit(-1)

import collections
import os.path

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import struct
import numpy as np
from scipy.interpolate import interp1d

import pyradi.ryutils as ryutils
import pyradi.ryplanck as ryplanck
import pyradi.ryfiles as ryfiles
import pyradi.ryplot as ryplot


################################################################
# Miscelaneous functions to read bytes, int, long, float values.

def myint(x):
    # two bytes length
    ans = ord(x[0])+(ord(x[1])<<8)
    return ans

def mylong(x):
    # four bytes length
    ans = ord(x[0])+(ord(x[1])<<8)+(ord(x[2])<<16)+(ord(x[3])<<32)
    return ans

def myfloat(x):
    ans = struct.unpack('f',x)
    ans = ans[0]
    return ans

def mydouble(x):
    ans = struct.unpack('d',x)
    ans = ans[0]
    return ans

def mybyte(x):
    # will return an error if x is more than a byte length
    # TODO need to write error catch routine
    ans = ord(x)
    return ans

def terminateStrOnZero (str):
    """Iterate through string and terminate on first zero
    """
    idx=0
    while idx < len(str) and str[idx] != '\00':
        idx += 1
    return str[:idx]

################################################################
class PTWFrameInfo:
  """Class to store the ptw file header information.
  """
  
  def __init__(self):

    self.FileName = ''
    self.h_Signature = '' #[0:5]
    self.h_format = 'unknown'
    self.h_unit = ''
    self.h_Version = '' #[5:10]
    self.h_MainHeaderSize = 0 #[11:15]
    self.h_FrameHeaderSize = 0 #[15:19]
    self.h_SizeOfOneFrameAndHeader = 0 #[19:23]
    self.h_SizeOfOneFrame = 0 #[23:27]
    self.h_NumberOfFieldInFile = 0 #[27:31]
    self.h_CurrentFieldNumber = 0 #[31:35]

    #self.h_FileSaveDate = '' #[35:39]
    self.h_FileSaveYear = 0
    self.h_FileSaveMonth = 0
    self.h_FileSaveDay = 0

    #self.h_FileSaveTime = '' #[39:43]
    self.h_FileSaveHour = 0
    self.h_FileSaveMinute = 0
    self.h_FileSaveSecond = 0

    self.h_Millieme = 0 #[43:44]
    self.h_CameraName = '' #[44:64]
    self.h_LensName = '' #[64:84]
    self.h_FilterName = '' #[84:104]
    self.h_ApertureName = '' #[104:124]
    self.h_IRUSBilletSpeed = 0 #[124:128] # IRUS
    self.h_IRUSBilletDiameter = 0 #myfloat(headerinfo[128:132]) # IRUS
    self.h_IRUSBilletShape = 0 #myint(headerinfo[132:134]) #IRUS
    self.h_Emissivity = 0 #myfloat(headerinfo[141:145])
    self.h_Ambiant = 0 #myfloat(headerinfo[145:149])
    self.h_Distance = 0 #myfloat(headerinfo[149:153])
    self.h_IRUSInductorCoil = 0 #ord(everthing[153:154]) # IRUS
    self.h_IRUSInductorPower = 0 #mylong(headerinfo[154:158]) # IRUS
    self.h_IRUSInductorVoltage = 0 #myint(headerinfo[158:160]) # IRUS
    self.h_IRUSInductorFrequency = 0 #mylong(headerinfo[160:164]) # IRUS
    self.h_IRUSSynchronization = 0 #ord(headerinfo[169:170]) # IRUS
    self.h_AtmTransmission = 0 #myfloat(headerinfo[170:174])
    self.h_ExtinctionCoeficient = 0 #myfloat(headerinfo[174:178])
    self.h_Object = 0 #myint(headerinfo[178:180])
    self.h_Optic = 0 #myint(headerinfo[180:182])
    self.h_Atmo = 0 #myint(headerinfo[182:184])
    self.h_AtmosphereTemp = 0 #myfloat(headerinfo[184:188])
    self.h_CutOnWavelength = 0 #myfloat(headerinfo[188:192])
    self.h_CutOffWavelength = 0 #myfloat(headerinfo[192:196])
    self.h_PixelSize = 0 #myfloat(headerinfo[196:200])
    self.h_PixelPitch = 0 #myfloat(headerinfo[200:204])
    self.h_DetectorApperture = 0 #myfloat(headerinfo[204:208])
    self.h_OpticsFocalLength = 0 #myfloat(headerinfo[208:212])
    self.h_HousingTemperature1 = 0 #myfloat(headerinfo[212:216])
    self.h_HousingTemperature2 = 0 #myfloat(headerinfo[216:220])
    self.h_CameraSerialNumber = '' #headerinfo[220:231])
    self.h_MinimumLevelThreshold = 0 #myint(headerinfo[245:247])
    self.h_MaximumLevelThreshold = 0 #myint(headerinfo[247:249])
    self.h_EchelleSpecial = 0 #myint(headerinfo[277:279])
    self.h_EchelleUnit = 0 #headerinfo[279:289]
    self.h_EchelleValue = 0 #(headerinfo[289:357]) # 16 float values
    self.h_Units = ''
    self.h_Lockin = 0
    self.h_LockinGain = 0 #myfloat(headerinfo[357:361])
    self.h_LockinOffset = 0 #myfloat(headerinfo[361:365])
    self.h_HorizontalZoom = 0 #myfloat(headerinfo[365:369])
    self.h_VerticalZoom = 0 #myfloat(headerinfo[369:373])
    self.h_PixelsPerLine = 0 #myint(headerinfo[377:379])
    self.h_LinesPerField = 0 #myint(headerinfo[379:381])
    self.h_Rows = 0
    self.h_Cols = 0

    self.h_framepointer=1
    self.h_firstframe=1
    self.h_cliprect=[0,0,1,1]
    self.h_lastframe=0
    self.h_FrameSize = 0

    self.h_ADDynamic = 0 #myint(headerinfo[381:383])
    self.h_SATIRTemporalFrameDepth = 0 #myint(headerinfo[383:385]) # SATIR
    self.h_SATIRLocationLongitude = 0 #myfloat(headerinfo[385:389]) # SATIR
    self.h_SATIRLocationLatitude = 0 #myfloat(headerinfo[389:393]) # SATIR South is negative
    self.h_SATIRLocationAltitude = 0 #myfloat(headerinfo[393:397]) # SATIR
    self.h_ExternalSynch = 0 #ord(headerinfo[397]) # 1=External 0 = Internal
    self.h_CEDIPAquisitionPeriod = 0 #myfloat(headerinfo[403:407]) # CEDIP seconds
    self.h_CEDIPIntegrationTime = 0 #myfloat(headerinfo[407:411]) # CEDIP seconds
    self.h_WOLFSubwindowCapability = 0 #myint(headerinfo[411:413]) # WOLF
    self.h_ORIONIntegrationTime = 0 #myfloat(headerinfo[431:437]) # ORION (6 values)
    self.h_ORIONFilterNames = '' #headerinfo[437:557]) # ORION 6 fields of 20 chars each
    self.h_NucTable = 0 #myint(headerinfo[557:559])
    self.h_Reserve6 = '' #headerinfo[559:563]
    self.h_Comment = '' #headerinfo[563:1563]
    self.h_CalibrationFileName = '' #headerinfo[1563:1663]
    self.h_ToolsFileName = '' #headerinfo[1663:1919]
    self.h_PaletteIndexValid = 0 #ord(headerinfo[1919:1920])
    self.h_PaletteIndexCurrent = 0 #myint(1920:1922])
    self.h_PaletteToggle = 0 #ord(headerinfo[1922:1923])
    self.h_PaletteAGC = 0 #ord(headerinfo[1923:1924])
    self.h_UnitIndexValid = 0 #ord(headerinfo[1923:1924])
    self.h_CurrentUnitIndex = 0 #myint(headerinfo[1925:1927])
    self.h_ZoomPosition = 0 #(headerinfo[1927:1935]) # unknown format POINT
    self.h_KeyFrameNumber = 0 #ord(headerinfo[1935:1936])
    self.h_KeyFramesInFilm = 0 #headerinfo[1936:2056] # set of 30 frames
    self.h_PlayerLocked = 0 # ord(headerinfo[2057:2057])
    self.h_FrameSelectionValid = 0 #ord(headerinfo[2057:2058])
    self.h_FrameofROIStart = 0 #mylong(headerinfo[2058:2062])
    self.h_FrameofROIEnd = 0 #mylong(headerinfo[2062:2066])
    self.h_PlayerInfinitLoop = 0 #ord(headerinfo[2067:2068])
    self.h_PlayerInitFrame = 0 #mylong(headerinfo[2068:2072])

    self.h_Isoterm0Active = 0 #ord(headerinfo[2072:2073])
    self.h_Isoterm0DLMin = 0 #myint(headerinfo[2073:2075])
    self.h_Isoterm0DLMax = 0 #myint(headerinfo[2075:2077])
    self.h_Isoterm0Color = 0 #headerinfo[2077:2081]

    self.h_Isoterm1Active = 0 #ord(headerinfo[2081:2082])
    self.h_Isoterm1DLMin = 0 #myint(headerinfo[2082:2084])
    self.h_Isoterm1DLMax = 0 #myint(headerinfo[2084:2086])
    self.h_Isoterm1Color = 0 #headerinfo[2086:2090]

    self.h_Isoterm2Active = 0 #ord(headerinfo[2090:2091])
    self.h_Isoterm2DLMin = 0 #myint(headerinfo[2091:2093])
    self.h_Isoterm2DLMax = 0 #myint(headerinfo[2093:2095])
    self.h_Isoterm2Color = 0 #headerinfo[2095:2099]

    self.h_ZeroActive = 0 #ord(headerinfo[2099:2100])
    self.h_ZeroDL = 0 #myint(headerinfo[2100:2102])
    self.h_PaletteWidth = 0 #myint(headerinfo[2102:2104])
    self.h_PaletteFull = 0 #ord(headerinfo[2104:2105])
    self.h_PTRFrameBufferType = 0 #ord(headerinfo[2105:2106]) # 0=word 1=double
    self.h_ThermoElasticity = 0 #headerinfo[2106:2114] # type double (64 bits)
    self.h_DemodulationFrequency = 0 #myfloat(headerinfo[2114:2118])
    self.h_CoordinatesType = 0 #mylong(headerinfo[2118:2122])
    self.h_CoordinatesXorigin = 0 #mylong(headerinfo[2122:2126])
    self.h_CoordinatesYorigin = 0 #mylong(headerinfo[2126:2130])
    self.h_CoordinatesShowOrigin = 0 #ord(headerinfo[2130:2131])
    self.h_AxeColor = 0 #headerinfo[2131:2135]
    self.h_AxeSize = 0 #mylong(headerinfo[2135:2139])
    self.h_AxeValid = 0 #ord(headerinfo[2139:2140])
    self.h_DistanceOffset = 0 #myfloat(headerinfo[2140:2144])
    self.h_HistoEqualizationEnabled = 0 #ord(headerinfo[2144:2145])
    self.h_HistoEqualizationPercent = 0 #myint(headerinfo[2145:2147])
    self.h_CalibrationFileName = '' #headerinfo[2147:2403]
    self.h_PTRTopFrameValid = 0 #ord(headerinfo[2403:2404])
    self.h_SubSampling = 0 #myint(headerinfo[2404:2408])
    self.h_CameraHFlip = 0 #ord(headerinfo[2408:2409])
    self.h_CameraHVFlip = 0 #ord(headerinfo[2409:2410])
    self.h_BBTemp = 0 #myfloat(headerinfo[2410:2414])
    self.h_CaptureWheelIndex = 0 #ord(headerinfo[2414:2415])
    self.h_CaptureFocalIndex = 0 #ord(headerinfo[2415:2416])
    self.h_Reserved7 = '' #headerinfo[2416:3028]
    self.h_Reserved8 = '' #headerinfo[3028:3076]
    self.h_Framatone = 0 #ord(headerinfo[3076:3077]

    # Container for a single frame
    self.data = []
    self.minval = 0
    self.maxval = 0

# End of header definition

################################################################

def readPTWHeader(ptwfilename):
    """Given a ptw filename, read the header and return the header to caller

    Args:
        | filename (string) with full path to the ptw file.

    Returns:
        | Header (class) containing all PTW header information.

    Raises:
        | No exception is raised.

    Reference:
       h_variables of the header and byte positions are obtained 
       from DL002U-D Altair Reference Guide
     """


    # Define the variables holding the header values
    headerinfo = '' #the vector holding the file header
    Header = PTWFrameInfo()

    # Read file to get the header size
    Header.FileName = ptwfilename
    fid = open(ptwfilename,'rb')
    headerinfo = fid.read(16)
    fid.close()
    MainHeaderSize = mylong(headerinfo[11:15])

    # Open file again and read the header information using the header size
    fid = open(ptwfilename,'rb')
    headerinfo = fid.read(MainHeaderSize)

    Header.h_Signature = headerinfo[0:3]
    if Header.h_Signature == 'AIO': #AGEMA
        Header.h_format = 'agema'
    elif Header.h_Signature == 'CED':
        Header.h_format = 'cedip'
        Header.h_unit = 'dl'

    Header.h_Version = headerinfo[5:10]
    Header.h_MainHeaderSize = mylong(headerinfo[11:15])
    Header.h_FrameHeaderSize = mylong(headerinfo[15:19])
    Header.h_SizeOfOneFrameAndHeader = mylong(headerinfo[19:23])
    Header.h_SizeOfOneFrame = mylong(headerinfo[23:27])
    Header.h_NumberOfFieldInFile = mylong(headerinfo[27:31])
    Header.h_CurrentFieldNumber = myint(headerinfo[31:35])

    #Header.h_FileSaveDate = '' #[35:39] decoded below
    Header.h_FileSaveYear = myint(headerinfo[35:37])
    Header.h_FileSaveMonth = ord(headerinfo[37:38])
    Header.h_FileSaveDay = ord(headerinfo[38:39])

    #Header.h_FileSaveTime = '' #[39:43] decoded below
    Header.h_FileSaveHour = ord(headerinfo[39:40])
    Header.h_FileSaveMinute = ord(headerinfo[40:41])
    Header.h_FileSaveSecond = ord(headerinfo[41:42])

    Header.h_Millieme = ord(headerinfo[43:44])

    Header.h_CameraName = terminateStrOnZero(headerinfo[44:64])
    Header.h_LensName = terminateStrOnZero(headerinfo[64:84])
    Header.h_FilterName = terminateStrOnZero(headerinfo[84:104])
    Header.h_ApertureName = terminateStrOnZero(headerinfo[104:124])


    Header.h_IRUSBilletSpeed = myfloat(headerinfo[124:128]) # IRUS
    Header.h_IRUSBilletDiameter = myfloat(headerinfo[128:132]) # IRUS
    Header.h_IRUSBilletShape = myint(headerinfo[132:134]) #IRUS
    Header.h_Emissivity = myfloat(headerinfo[141:145])
    Header.h_Ambiant = myfloat(headerinfo[145:149])
    Header.h_Distance = myfloat(headerinfo[149:153])
    Header.h_IRUSInductorCoil = ord(headerinfo[153:154]) # IRUS
    Header.h_IRUSInductorPower = mylong(headerinfo[154:158]) # IRUS
    Header.h_IRUSInductorVoltage = myint(headerinfo[158:160]) # IRUS
    Header.h_IRUSInductorFrequency = mylong(headerinfo[160:164]) # IRUS
    Header.h_IRUSSynchronization = ord(headerinfo[169:170]) # IRUS
    Header.h_AtmTransmission = myfloat(headerinfo[170:174])
    Header.h_ExtinctionCoeficient = myfloat(headerinfo[174:178])
    Header.h_Object = myint(headerinfo[178:180])
    Header.h_Optic = myint(headerinfo[180:182])
    Header.h_Atmo = myint(headerinfo[182:184])
    Header.h_AtmosphereTemp = myfloat(headerinfo[184:188])
    Header.h_CutOnWavelength = myfloat(headerinfo[188:192])
    Header.h_CutOffWavelength = myfloat(headerinfo[192:196])
    Header.h_PixelSize = myfloat(headerinfo[196:200])
    Header.h_PixelPitch = myfloat(headerinfo[200:204])
    Header.h_DetectorApperture = myfloat(headerinfo[204:208])
    Header.h_OpticsFocalLength = myfloat(headerinfo[208:212])
    Header.h_HousingTemperature1 = myfloat(headerinfo[212:216])
    Header.h_HousingTemperature2 = myfloat(headerinfo[216:220])
    Header.h_CameraSerialNumber = terminateStrOnZero(headerinfo[220:231])
    Header.h_MinimumLevelThreshold = myint(headerinfo[245:247])
    Header.h_MaximumLevelThreshold = myint(headerinfo[247:249])
    Header.h_EchelleSpecial = myint(headerinfo[277:279])
    Header.h_EchelleUnit = headerinfo[279:289]
    Header.h_EchelleValue = headerinfo[289:357] # 16 float values

    if(Header.h_EchelleSpecial==0):
        Header.h_Units='dl' # [dl T rad]
    else:
        Header.h_Units= Header.h_EchelleUnit # [dl T rad]

    Header.h_LockinGain = myfloat(headerinfo[357:361])
    Header.h_LockinOffset = myfloat(headerinfo[361:365])
    Header.h_HorizontalZoom = myfloat(headerinfo[365:369])
    Header.h_VerticalZoom = myfloat(headerinfo[369:373])

    Header.h_PixelsPerLine = myint(headerinfo[377:379])
    Header.h_LinesPerField = myint(headerinfo[379:381])
    if Header.h_LinesPerField==0:
        Header.h_LinesPerField=128
    if Header.h_PixelsPerLine==0:
        Header.h_PixelsPerLine=128

    Header.h_Rows = Header.h_LinesPerField
    Header.h_Cols = Header.h_PixelsPerLine

    Header.h_cliprect = [0,0,Header.h_Cols-1,Header.h_Rows-1]
    Header.h_lastframe = Header.h_NumberOfFieldInFile
    Header.h_FrameSize = Header.h_FrameHeaderSize + Header.h_Cols * Header.h_Rows * 2

    Header.h_ADDynamic = myint(headerinfo[381:383])
    Header.h_SATIRTemporalFrameDepth = myint(headerinfo[383:385]) # SATIR
    Header.h_SATIRLocationLongitude = myfloat(headerinfo[385:389]) # SATIR
    Header.h_SATIRLocationLatitude = myfloat(headerinfo[389:393]) # SATIR South is negative
    Header.h_SATIRLocationAltitude = myfloat(headerinfo[393:397]) # SATIR
    Header.h_ExternalSynch = ord(headerinfo[397]) # 1=External 0 = Internal
    Header.h_CEDIPAquisitionPeriod = myfloat(headerinfo[403:407]) # CEDIP seconds
    Header.h_CEDIPIntegrationTime = myfloat(headerinfo[407:411]) # CEDIP seconds
    Header.h_WOLFSubwindowCapability = myint(headerinfo[411:413]) # WOLF
    Header.h_ORIONIntegrationTime = headerinfo[431:437] # ORION (6 values)
    Header.h_ORIONFilterNames = headerinfo[437:557] # ORION 6 fields of 20 chars each
    Header.h_NucTable = myint(headerinfo[557:559])
    Header.h_Reserve6 = headerinfo[559:563]
    Header.h_Comment = terminateStrOnZero(headerinfo[563:1563])
    Header.h_CalibrationFileName = terminateStrOnZero(headerinfo[1563:1663])
    Header.h_ToolsFileName = terminateStrOnZero(headerinfo[1663:1919])
    Header.h_PaletteIndexValid = ord(headerinfo[1919:1920])
    Header.h_PaletteIndexCurrent = myint(headerinfo[1920:1922])
    Header.h_PaletteToggle = ord(headerinfo[1922:1923])
    Header.h_PaletteAGC = ord(headerinfo[1923:1924])
    Header.h_UnitIndexValid = ord(headerinfo[1923:1924])
    Header.h_CurrentUnitIndex = myint(headerinfo[1925:1927])
    Header.h_ZoomPosition = terminateStrOnZero(headerinfo[1927:1935]) # unknown format POINT
    Header.h_KeyFrameNumber = ord(headerinfo[1935:1936])
    Header.h_KeyFramesInFilm = terminateStrOnZero(headerinfo[1936:2056]) # set of 30 frames
    Header.h_PlayerLocked =  ord(headerinfo[2056:2057])
    Header.h_FrameSelectionValid = ord(headerinfo[2057:2058])
    Header.h_FrameofROIStart = mylong(headerinfo[2058:2062])
    Header.h_FrameofROIEnd = mylong(headerinfo[2062:2066])
    Header.h_PlayerInfinitLoop = ord(headerinfo[2067:2068])
    Header.h_PlayerInitFrame = mylong(headerinfo[2068:2072])

    Header.h_Isoterm0Active = ord(headerinfo[2072:2073])
    Header.h_Isoterm0DLMin = myint(headerinfo[2073:2075])
    Header.h_Isoterm0DLMax = myint(headerinfo[2075:2077])
    Header.h_Isoterm0Color = headerinfo[2077:2081]

    Header.h_Isoterm1Active = ord(headerinfo[2081:2082])
    Header.h_Isoterm1DLMin = myint(headerinfo[2082:2084])
    Header.h_Isoterm1DLMax = myint(headerinfo[2084:2086])
    Header.h_Isoterm1Color = headerinfo[2086:2090]

    Header.h_Isoterm2Active = ord(headerinfo[2090:2091])
    Header.h_Isoterm2DLMin = myint(headerinfo[2091:2093])
    Header.h_Isoterm2DLMax = myint(headerinfo[2093:2095])
    Header.h_Isoterm2Color = headerinfo[2095:2099]

    Header.h_ZeroActive = ord(headerinfo[2099:2100])
    Header.h_ZeroDL = myint(headerinfo[2100:2102])
    Header.h_PaletteWidth = myint(headerinfo[2102:2104])
    Header.h_PaletteFull = ord(headerinfo[2104:2105])
    Header.h_PTRFrameBufferType = ord(headerinfo[2105:2106]) # 0=word 1=double
    Header.h_ThermoElasticity = mydouble(headerinfo[2106:2114]) # type double (64 bits)
    Header.h_DemodulationFrequency = myfloat(headerinfo[2114:2118])
    Header.h_CoordinatesType = mylong(headerinfo[2118:2122])
    Header.h_CoordinatesXorigin = mylong(headerinfo[2122:2126])
    Header.h_CoordinatesYorigin = mylong(headerinfo[2126:2130])
    Header.h_CoordinatesShowOrigin = ord(headerinfo[2130:2131])
    Header.h_AxeColor = headerinfo[2131:2135]
    Header.h_AxeSize = mylong(headerinfo[2135:2139])
    Header.h_AxeValid = ord(headerinfo[2139:2140])
    Header.h_DistanceOffset = myfloat(headerinfo[2140:2144])
    Header.h_HistoEqualizationEnabled = ord(headerinfo[2144:2145])
    Header.h_HistoEqualizationPercent = myint(headerinfo[2145:2147])
    Header.h_CalibrationFileName = terminateStrOnZero(headerinfo[2147:2403])
    Header.h_PTRTopFrameValid = ord(headerinfo[2403:2404])
    Header.h_SubSampling = myint(headerinfo[2404:2408])
    Header.h_CameraHFlip = ord(headerinfo[2408:2409])
    Header.h_CameraHVFlip = ord(headerinfo[2409:2410])
    Header.h_BBTemp = myfloat(headerinfo[2410:2414])
    Header.h_CaptureWheelIndex = ord(headerinfo[2414:2415])
    Header.h_CaptureFocalIndex = ord(headerinfo[2415:2416])
    Header.h_Reserved7 = headerinfo[2416:3028]
    Header.h_Reserved8 = headerinfo[3028:3076]
    Header.h_Framatone = ord(headerinfo[3076:3077])

    # Read the first video frame info, not the data
    # to determine lockin information

    fid.seek(Header.h_MainHeaderSize,0)#,'bof')  %skip main header
    fid.seek(Header.h_FrameHeaderSize,1)#'cof')  %skip frame header
    firstline = fid.read(Header.h_Cols)#, 'uint16')  %read one line

    # look if first line contains lockin information
    if(firstline[1:4]==[1220,3907,1204,2382]):
        Header.h_Lockin=1
        Header.h_Rows=Header.h_Rows-1
        print ('* LOCKIN')
    else:
        Header.h_Lockin=0

    fid.close()

    return Header

################################################################
def GetPTWFrameFromFile(header):
    """From the ptw file, load the frame specified in the header variable
       header.h_framepointer

    Args:
        | header (class object) header of the ptw file, with framepointer set

    Returns:
        | header.data plus newly added information: 
          requested frame DL values, dimensions (rows,cols)

    Raises:
        | No exception is raised.
    """

    # for debugging
    #print ('.....Loading frame', header.m_framepointer , 'from', header.m_filename,'.....')
    #print (header.m_cols,'x', header.m_rows, 'data points')

    fid = open(header.FileName,'rb')
    # skip main header
    fid.seek (header.h_MainHeaderSize,0)  #bof

    # for debugging
    #print ('EndHeader =',fid.tell())

    if(header.h_Lockin): # lockin -> skip first line
        fid.seek ((header.h_framepointer-1) * (header.h_FrameSize + 2*header.h_Cols),1)#, 'cof'
    else:
        fid.seek ((header.h_framepointer-1) * (header.h_FrameSize),1)#, 'cof'

    # for debugging
    #print ('StartFrameHeader =',fid.tell())

    #fid.seek(header.m_FrameHeaderSize,1)#,'cof') #skip frame header
    FrameHeader = fid.read(header.h_FrameHeaderSize)

    # for debugging
    #print ('Start FrameData at',fid.tell())

    header.data = np.eye(header.h_Cols, header.h_Rows)

    #datapoints = header.m_cols * header.m_rows
    for y in range(header.h_Rows):
        for x in range(header.h_Cols):
            header.data[x][y] = myint(fid.read(2))

    # for debugging
    #print ('Data read',len(header.m_data), 'points')
    #print ('End FrameData at',fid.tell())

    # if a special scale is given then transform the data
    if(header.h_EchelleSpecial):
        low = min(header.h_EchelleScaleValue)
        high = max(header.h_EchelleScaleValue)
        header.data = header.data * (high-low)/ 2.0**16 + low
        #clear low high
    if(header.h_Lockin): # lockin -> skip first line
        header.h_cliprect = [0,1,header.h_Cols-1,header.h_Rows]

    header.h_minval = header.data.min()
    header.h_maxval = header.data.max()

    # for debugging
    #print ('DL values min', header.m_minval)
    #print ('DL values max', header.m_maxval)

    fid.close()  #close file
    return header


################################################################
def getPTWFrame (header, frameindex):
    """Retrieve a single PTW frame, given the header and frame index

    This routine also stores the data array as part of the header. This may 
    change - not really needed to have both a return value and header stored 
    value for the DL valueheader. This for a historical reason due to the way 
    GetPTWFrameFromFile was written
 
    Args:
        | header (class object)
        | frameindex (integer): The frame to be extracted

    Returns:
        | header.data (numpy.ndarray): requested frame DL values, dimensions (rows,cols)

    Raises:
        | No exception is raised.
    """

    # Check if this is  a cedip file
    errorresult = np.asarray([0])
    if header.h_format!='cedip':
        print('ReadJade Error: file format is not supported')
        return errorresult
    if (frameindex <= header.h_lastframe):
        if frameindex>0:
            header.h_framepointer = frameindex
            header = GetPTWFrameFromFile(header)
        else:
            print ('frameindex smaller than 0')
            return errorresult
    else:                           # frameindex exceeds no of frames
        print ('ReadJade Error: cannot load frame. Frameindex exceeds sequence length.')
        return errorresult

    return header.data.conj().transpose()


################################################################
def getPTWFrames (header, loadFrames=[]):
    """Retrieve a number of PTW frames, given in a list of frameheader.

    Args:
        | header (class object)
        | loadFrames ([int]): List of indices for frames to be extracted

    Returns:
        | data (numpy.ndarray): requested image frame DL values, dimensions (frames,rows,cols)

    Raises:
        | No exception is raised.
    """

    # error checking on inputs
    errorresult = np.asarray([0])
    # Check if this is  a cedip file
    if header.h_format!='cedip':
        print('getPTWFrames Error: file format is not supported')
        return errorresult
    #check for legal frame index values
    npFrames = np.asarray(loadFrames)
    if np.any( npFrames < 1 ) or np.any ( npFrames > header.h_lastframe ):
        print('getPTWFrames Error: at least one requested frame not in file')
        print('legal frames for this file are: {0} to {1}'.format(1,header.h_lastframe))
        return errorresult

    data = getPTWFrame (header, loadFrames[0])
    for frame in loadFrames[1:]:
        data = np.concatenate((data, getPTWFrame (header, frame)))

    rows = header.h_Rows
    cols = header.h_Cols
    return data.reshape(len(loadFrames), rows ,cols)


################################################################
def showHeader(Header):
    """Utility function to print the PTW header information to stdout

    Args:
        | header (class object) ptw file header structure

    Returns:
        | None

    Raises:
        | No exception is raised.
    """

    print (Header.h_Signature, 'version', Header.h_Version)
    print ('Main Header Size',Header.h_MainHeaderSize)
    print ('Frame Header Size',Header.h_FrameHeaderSize)
    print ('Frame + Frame Header Size',Header.h_SizeOfOneFrameAndHeader)
    print ('Frame Size',Header.h_SizeOfOneFrame)
    print ('Number of Frames', Header.h_NumberOfFieldInFile)
    #print Header.h_CurrentFieldNumber

    print ('Year',Header.h_FileSaveYear, 'Month',Header.h_FileSaveMonth, 'Day', Header.h_FileSaveDay,)
    print ('(',str(Header.h_FileSaveYear).zfill(2), '/',str(Header.h_FileSaveMonth).zfill(2), '/', str(Header.h_FileSaveDay).zfill(2),')')

    print ('Hour',Header.h_FileSaveHour, 'Minute',Header.h_FileSaveMinute, 'Second',Header.h_FileSaveSecond,)
    print ('(',str(Header.h_FileSaveHour).zfill(2), ':',str(Header.h_FileSaveMinute).zfill(2), ':',str(Header.h_FileSaveSecond).zfill(2),')')

    #print Header.h_Millieme
    print ('Camera Name',Header.h_CameraName)
    print ('Lens',Header.h_LensName)
    print ('Filter',Header.h_FilterName)
    print ('Aperture Name', Header.h_ApertureName)
    if Header.h_Signature == 'IRUS':
        print (Header.h_IRUSBilletSpeed)
        print (Header.h_IRUSBilletDiameter)
        print (Header.h_IRUSBilletShape)
    print ('Emissivity',Header.h_Emissivity)
    print ('Ambient Temperature', Header.h_Ambiant,'(K)')
    print ('Ambient Temperature', Header.h_Ambiant-273.15,'(degC)')
    print ('Distance to target',Header.h_Distance)
    if Header.h_Signature == 'IRUS':
        print (Header.h_IRUSInductorCoil)
        print (Header.h_IRUSInductorPower)
        print (Header.h_IRUSInductorVoltage)
        print (Header.h_IRUSInductorFrequency)
        print (Header.h_IRUSSynchronization)
    print ('Atm Transmission', Header.h_AtmTransmission)
    print ('Ext Coef',Header.h_ExtinctionCoeficient)
    print ('Target', Header.h_Object)
    print ('Optic',Header.h_Optic)
    print ('Atmo',Header.h_Atmo)
    print ('Atm Temp', Header.h_AtmosphereTemp)
    print ('Cut on Wavelength', Header.h_CutOnWavelength)
    print ('Cut off Wavelength', Header.h_CutOffWavelength)
    print ('PixelSize', Header.h_PixelSize)
    print ('PixelPitch', Header.h_PixelPitch)
    print ('Detector Apperture', Header.h_DetectorApperture)
    print ('Optic Focal Length', Header.h_OpticsFocalLength)
    print ('Housing Temp1', Header.h_HousingTemperature1, '(K)')
    print ('Housing Temp2', Header.h_HousingTemperature2, '(K)')
    print ('Camera Serial Number', Header.h_CameraSerialNumber)
    print ('Min Threshold', Header.h_MinimumLevelThreshold)
    print ('Max Threshold', Header.h_MaximumLevelThreshold)
    #print Header.h_EchelleSpecial
    #print Header.h_EchelleUnit
    #print Header.h_EchelleValue
    print ('Gain', Header.h_LockinGain)
    print ('Offset', Header.h_LockinOffset)
    print ('HZoom', Header.h_HorizontalZoom)
    print ('VZoom', Header.h_VerticalZoom)
    print ('Field', Header.h_PixelsPerLine,'X',Header.h_LinesPerField)
    print ('AD converter',Header.h_ADDynamic, 'bit')
    if Header.h_Signature == 'SATIR':
        print (Header.h_SATIRTemporalFrameDepth)
        print (Header.h_SATIRLocationLongitude)
        print (Header.h_SATIRLocationLatitude)
        print (Header.h_SATIRLocationAltitude)
    if Header.h_ExternalSynch:
        print ('Ext Sync ON')
    else:
        print ('Ext Sync OFF')
    print('Header.h_Signature = {}'.format(Header.h_Signature))
    if Header.h_Signature == 'CED':
        print ('CEDIP Period', 1./Header.h_CEDIPAquisitionPeriod,'Hz')
        print ('CEDIP Integration', Header.h_CEDIPIntegrationTime*1000,'msec')
    if Header.h_Signature == 'WOLF':
        print (Header.h_WOLFSubwindowCapability)
    if Header.h_Signature == 'ORI':
        print (Header.h_ORIONIntegrationTime)
        print (Header.h_ORIONFilterNames)
    print ('NUC ', Header.h_NucTable)
    #print Header.h_Reserve6
    print ('Comment', Header.h_Comment)

    print ('Calibration File Name')
    print (Header.h_CalibrationFileName)

    print ('Tools File Name')
    print (Header.h_ToolsFileName)

    print ('Palette Index?', Header.h_PaletteIndexValid)
    print ('Palette Current',Header.h_PaletteIndexCurrent)
    print ('Palette Toggle', Header.h_PaletteToggle)
    print ('Palette AGC', Header.h_PaletteAGC)
    print ('Unit Index?', Header.h_UnitIndexValid)
    print ('Current Unit Index', Header.h_CurrentUnitIndex)
    print ('Zoom Pos', Header.h_ZoomPosition)
    print ('Key Framenum', Header.h_KeyFrameNumber)
    print ('Num Keyframes', Header.h_KeyFramesInFilm)
    print ('Player lock', Header.h_PlayerLocked)
    print ('Frame Select?', Header.h_FrameSelectionValid)
    print ('ROI Start', Header.h_FrameofROIStart)
    print ('ROI Stop', Header.h_FrameofROIEnd)
    print ('Player inf loop?', Header.h_PlayerInfinitLoop)
    print ('Player Init Frame', Header.h_PlayerInitFrame)

    print ('Isoterm0?', Header.h_Isoterm0Active)
    print ('Isoterm0 DL Min', Header.h_Isoterm0DLMin)
    print ('Isoterm0 DL Max', Header.h_Isoterm0DLMax)
    print ('Isoterm0 Color', Header.h_Isoterm0Color)

    print ('Isoterm1?', Header.h_Isoterm1Active)
    print ('Isoterm1 DL Min', Header.h_Isoterm1DLMin)
    print ('Isoterm1 DL Max', Header.h_Isoterm1DLMax)
    print ('Isoterm1 Color', Header.h_Isoterm1Color)

    print ('Isoterm2?', Header.h_Isoterm2Active)
    print ('Isoterm2 DL Min', Header.h_Isoterm2DLMin)
    print ('Isoterm2 DL Max', Header.h_Isoterm2DLMax)
    print ('Isoterm2 Color', Header.h_Isoterm2Color)

    print ('Zero?' ,Header.h_ZeroActive)
    print ('Zero DL', Header.h_ZeroDL)
    print ('Palette Width', Header.h_PaletteWidth)
    print ('PaletteF Full', Header.h_PaletteFull)
    print ('PTR Frame Buffer type', Header.h_PTRFrameBufferType)
    print ('Thermoelasticity', Header.h_ThermoElasticity)
    print ('Demodulation', Header.h_DemodulationFrequency)
    print ('Coordinate Type', Header.h_CoordinatesType)
    print ('X Origin',Header.h_CoordinatesXorigin)
    print ('Y Origin', Header.h_CoordinatesYorigin)
    print ('Coord Show Orig', Header.h_CoordinatesShowOrigin)
    print ('Axe Colour', Header.h_AxeColor)
    print ('Axe Size', Header.h_AxeSize)
    print ('Axe Valid?',Header.h_AxeValid)
    print ('Distance offset', Header.h_DistanceOffset)
    print ('Histogram?', Header.h_HistoEqualizationEnabled)
    print ('Histogram %', Header.h_HistoEqualizationPercent)

    print ('Calibration File Name')
    print (Header.h_CalibrationFileName)

    print ('PTRFrame Valid?', Header.h_PTRTopFrameValid)
    print ('Subsampling?', Header.h_SubSampling)
    print ('Camera flip H', Header.h_CameraHFlip)
    print ('Camera flip V', Header.h_CameraHVFlip)
    print ('BB Temp',Header.h_BBTemp)
    print ('Capture Wheel Index', Header.h_CaptureWheelIndex)
    print ('Capture Wheel Focal Index', Header.h_CaptureFocalIndex)
    #print Header.h_Reserved7
    #print Header.h_Reserved8
    #print Header.h_Framatone


################################################################################
class JadeCalibrationData:
  """Container to describe the calibration data of a Jade camera.
  """
  dicCaldata = collections.defaultdict(float)
  dicSpectrals = collections.defaultdict(str)
  dicRadiance = collections.defaultdict(str)
  dicRadiance = collections.defaultdict(str)
  dicIrradiance = collections.defaultdict(str)
  dicSpecRadiance = collections.defaultdict(str)
  diclFloor = collections.defaultdict(str)
  dicPower = collections.defaultdict(str)
 
  def __init__(self,filename, datafileroot):
    self.pathtoXML = os.path.dirname(filename)
    self.datafileroot = datafileroot

    ftree = ET.parse(filename)
    froot = ftree.getroot()
    self.name = froot.find(".").attrib["Name"]
    self.id = froot.find(".").attrib["ID"]
    self.version = froot.find(".").attrib["Version"]
    self.summary = froot.find(".").attrib["Summary"]
    self.sensorResponseFilename = '/'.join([self.datafileroot,froot.find(".//SensorResponse").attrib["Filename"]])
    self.opticsTransmittanceFilename = '/'.join([self.datafileroot,froot.find(".//OpticsTransmittance").attrib["Filename"]])
    self.filterFilename = '/'.join([self.datafileroot,froot.find(".//Filter").attrib["Filename"]])
    self.filterName = os.path.basename(self.filterFilename)[:-4]
    self.sourceEmisFilename = '/'.join([self.datafileroot,froot.find(".//SourceEmis").attrib["Filename"]])
    self.atmoTauFilename = '/'.join([self.datafileroot,froot.find(".//AtmoTau").attrib["Filename"]])

    self.detectorPitch = float(froot.find(".//DetectorPitch").attrib["Value"])
    self.fillFactor = float(froot.find(".//FillFactor").attrib["Value"])
    self.focallength = float(froot.find(".//Focallength").attrib["Value"])
    self.integrationTime = float(froot.find(".//IntegrationTime").attrib["Value"])
    self.fnumber = float(froot.find(".//Fnumber").attrib["Value"])
    self.nuMin = float(froot.find(".//Nu").attrib["Min"])
    self.nuMax = float(froot.find(".//Nu").attrib["Max"])
    self.nuInc = float(froot.find(".//Nu").attrib["Inc"])
    #read the nested 
    for child in froot.findall(".//Caldatas"):
      for childA in child.findall(".//Caldata"):
        InstrTemp = float(childA.attrib["InstrTemperature"])
        self.diclFloor[InstrTemp] = float(childA.attrib["DlFloor"])
        self.dicPower[InstrTemp] = float(childA.attrib["Power"])
        self.dicCaldata[InstrTemp] = []
        for i,childC in enumerate(childA.findall("CalPoint")):
          #build a numpy array row-wise as the data is read in
          if i==0:
            data = np.asarray([273.15 + float(childC.attrib["Temp"]),int(childC.attrib["DL"])])
          else:
            data = np.vstack((data, np.asarray([273.15 + float(childC.attrib["Temp"]),int(childC.attrib["DL"])] )))
        self.dicCaldata[InstrTemp] = data
    self.lokey = min(self.dicCaldata.keys())
    self.hikey = max(self.dicCaldata.keys())
    # print(self.Print())


  ################################################################
  def Print(self):
    """Write the calibration data file data to a string and return string.
    """
    str = ""
    str += 'Path to XML                  = {}\n'.format(self.pathtoXML)
    str += 'Path to datafiles            = {}\n'.format(self.datafileroot)
    str += 'Calibration Name             = {}\n'.format(self.name)
    str += 'ID                           = {}\n'.format(self.id)
    str += 'Version                      = {}\n'.format(self.version)
    str += 'Summary                      = {}\n'.format(self.summary)
    str += 'Sensor Response Filename     = {}\n'.format(self.sensorResponseFilename)
    str += 'Optics Transmittance Filename= {}\n'.format(self.opticsTransmittanceFilename)
    str += 'Filter Filename              = {}\n'.format(self.filterFilename)
    str += 'Source Emissivity Filename   = {}\n'.format(self.sourceEmisFilename)
    str += 'Atmospheric Transm Filename  = {}\n'.format(self.atmoTauFilename)
    str += 'DetectorPitch                = {}\n'.format(self.detectorPitch)
    str += 'FillFactor                   = {}\n'.format(self.fillFactor)
    str += 'Focallength                  = {}\n'.format(self.focallength)
    str += 'integrationTime              = {}\n'.format(self.integrationTime)
    str += 'Fnumber                      = {}\n'.format(self.fnumber)
    str += 'Nu (Min, Max, Inc)           = ({}, {}, {})\n'.format(self.nuMin,self.nuMax,self.nuInc)

    str += 'Calibration data set'
    for i,key in enumerate(sorted(self.dicCaldata.keys())):
      str += '\n\nInstrument temperature = {} C\n'.format(key)
      str += 'DL floor               = {}\n'.format(self.diclFloor[key])
      str += '    Deg-C      DL   '
      if self.dicCaldata[key].shape[1]>2:
        for item in self.spectrals:
          str += '  L-{0}  E-{0}'.format(item)
      str += '\n'
      str += np.array_str(self.dicCaldata[key], max_line_width=120, precision=4, )
      str += '\nstraight line fit L = {} DL + {}'.format(self.DlRadCoeff[key][0],self.DlRadCoeff[key][1])
    return str

  ################################################################
  def LoadPrepareData(self):
    """Load the camera calibration data from files and preprocess spectrals data 
    """
    self.spectrals = ['CalFilter','NoFilter']
    #set up the spectral domain in both wavenumber and wavelength
    self.nu = np.linspace(self.nuMin, self.nuMax, 1 + (self.nuMax -self.nuMin )/self.nuInc )
    self.wl = ryutils.convertSpectralDomain(self.nu,  type='nl')

    #load the various input files, interpolate on the fly to local spectrals.
    self.calEmis = ryfiles.loadColumnTextFile(self.sourceEmisFilename, loadCol=[1], 
                    normalize=0, abscissaOut=self.wl)
    self.calAtmo = ryfiles.loadColumnTextFile(self.atmoTauFilename, loadCol=[1], 
                    normalize=0, abscissaOut=self.wl)
    self.calFilter = ryfiles.loadColumnTextFile(self.filterFilename, loadCol=[1], 
                    normalize=0, abscissaOut=self.wl)
    self.calSensor = ryfiles.loadColumnTextFile(self.sensorResponseFilename, loadCol=[1], 
                    normalize=0, abscissaOut=self.wl)
    self.calOptics = ryfiles.loadColumnTextFile(self.opticsTransmittanceFilename, loadCol=[1], 
                    normalize=0, abscissaOut=self.wl)
    #get the conversion factor from radiance to irradiance
    self.radiancetoIrradiance = self.fillFactor * (self.detectorPitch / self.focallength ) ** 2

    #build a number of spectral cases, with filter variations.
    self.dicSpectrals['NoFilter'] = self.calEmis * self.calAtmo * self.calSensor * self.calOptics
    self.dicSpectrals['CalFilter'] = self.dicSpectrals['NoFilter'] * self.calFilter

    #normalise the spectral response
    # for spectral in self.dicSpectrals:
    #   self.dicSpectrals[spectral]  /= np.max(self.dicSpectrals[spectral])

  ################################################################
  def PlotSpectrals(self):
    """Plot all spectral curve data to a single graph.
    """
    #load data if not yet loaded
    if not 'NoFilter' in self.dicSpectrals:
      self.LoadPrepareData()

    p = ryplot.Plotter(1, figsize=(10,5))
    p.semilogY(1,self.wl,self.calEmis,label=['Emissivity'])
    p.semilogY(1,self.wl,self.calAtmo,label=['Atmosphere'])
    p.semilogY(1,self.wl,self.calFilter,label=['Filter'])
    p.semilogY(1,self.wl,self.calSensor,label=['Sensor'])
    p.semilogY(1,self.wl,self.calOptics,label=['Optics'])
    p.semilogY(1,self.wl,self.dicSpectrals['NoFilter'],label=['Eff. no filter'])
    p.semilogY(1,self.wl,self.dicSpectrals['CalFilter'],label=['Eff. with filter'])
    currentP = p.getSubPlot(1)
    currentP.set_xlabel('Wavelength {}m'.format(ryutils.upMu(False)))
    currentP.set_ylabel('Normalised Response')
    currentP.set_title('Spectral Response for {}'.format(self.name))
    p.saveFig('{}-spectrals.png'.format(self.name))


  ################################################################
  def PlotRadiance(self):
    """Plot all spectral radiance data for the calibration temperatures.
    """
    # calculate radiance values if not yet done
    if not 'NoFilter' in self.dicRadiance:
      self.CalculateCalibrationTables()

    p = ryplot.Plotter(1,2,1)
    for i,spectral in enumerate(self.spectrals):
      for j,key in enumerate(self.dicCaldata):
        if j == 0:
          #get temperature labels
          labels = []
          for temp in self.dicCaldata[key][:,0]:
            labels.append('{:.0f} $^\circ$C, {:.0f} K'.format(temp-273.15, temp))

          # print(self.wl.shape, self.dicSpecRadiance[spectral][key].shape)
          p.plot(i,self.wl,self.dicSpecRadiance[spectral][key],label=labels)
          currentP = p.getSubPlot(i)
          currentP.set_xlabel('Wavelength {}m'.format(ryutils.upMu(False)))
          currentP.set_ylabel('Radiance W/(m$^2$.sr.cm$^{-1}$)')
          if spectral == 'NoFilter':
            filname = 'no filter'
            currentP.set_ylabel('Radiance at source W/(m$^2$.sr.cm$^{-1}$)')
          else:
            filname = self.filterName
            currentP.set_ylabel('Radiance at sensor W/(m$^2$.sr.cm$^{-1}$)')
          currentP.set_title('{}-{} at Tinstr={} $^\circ$C'.format(self.name, filname, key))

    p.saveFig('{}-radiance.png'.format(self.name,spectral))

  ################################################################
  def PlotDLRadiance(self):
    """Plot DL level versus radiance for both camera temperatures
    """
    # calculate radiance values if not yet done
    if not 'NoFilter' in self.dicRadiance:
      self.CalculateCalibrationTables()

    p = ryplot.Plotter(1,1,1, figsize=(10,5))
    for j,key in enumerate(self.dicCaldata):
      if j == 0:
        plotColB = ['r']
        plotColM = ['b--']
      else:
        plotColB = ['c']
        plotColM = ['g--']
      p.plot(1,self.dicTableDLRad[key][:,0],self.dicTableDLRad[key][:,1],
        label=['Best fit line {}$^\circ$C'.format(key)],plotCol=plotColB)
      p.plot(1,self.dicCaldata[key][:,1],self.dicRadiance['NoFilter'][key],
        label=['Measured {}$^\circ$C'.format(key)],markers=['x'],plotCol=plotColM)
    currentP = p.getSubPlot(1)
    currentP.set_xlabel('Digital level')
    currentP.set_ylabel(' Radiance at source W/(m$^2$.sr)')
    currentP.set_title('{}-{}'.format(self.name, self.filterName))
    p.saveFig('{}-dlRadiance.png'.format(self.name,'CalFilter'))


  ################################################################
  def PlotTempRadiance(self):
    """Plot temperature versus radiance for both camera temperatures
    """
   # calculate radiance values if not yet done
    if not 'NoFilter' in self.dicRadiance:
      self.CalculateCalibrationTables()

    p = ryplot.Plotter(1,1,1, figsize=(10,5))
    for j,key in enumerate(self.dicCaldata):
      p.plot(1,self.dicTableTempRad[key][:,0],self.dicTableTempRad[key][:,1])
      # p.plot(1,self.dicCaldata[key][:,0],self.dicCaldata[key][:,1],label=['Measured'],markers=['x'])
      currentP = p.getSubPlot(1)
      currentP.set_xlabel('Temperature K')
      currentP.set_ylabel('Radiance at source W/(m$^2$.sr)')
      currentP.set_title('{}-{}'.format(self.name, self.filterName))
      currentP.set_ylim([0,np.max(self.dicTableTempRad[key][:,1])])
    p.saveFig('{}-TempRadiance.png'.format(self.name,'CalFilter'))


  ################################################################
  def PlotTintRad(self):
    """Plot optics radiance versus instrument temperature
    """
    # calculate radiance values if not yet done
    if not 'NoFilter' in self.dicRadiance:
      self.CalculateCalibrationTables()
    p = ryplot.Plotter(1,1,1, figsize=(10,5))
    ptitle = '{} sensor internal radiance'.format(self.name)
    p.plot(1,self.TableTintRad[:,0]-273.15,self.TableTintRad[:,1], ptitle=ptitle,
      xlabel='Internal temperature $^\circ$C', ylabel='Radiance W/(m$^2$.sr)')
    p.saveFig('{}-Internal.png'.format(self.name))


  ################################################################
  def PlotDLTemp(self):
    """Plot digital level versus temperature for both camera temperatures
    """
    # calculate radiance values if not yet done
    if not 'NoFilter' in self.dicRadiance:
      self.CalculateCalibrationTables()

    p = ryplot.Plotter(1,1,1, figsize=(10,5))
    for j,key in enumerate(self.dicCaldata):
      DL = self.dicTableDLRad[key][:,0]
      temp = self.LookupDLTemp(DL, key)
      if j == 0:
        plotColB = ['r']
        plotColM = ['b--']
      else:
        plotColB = ['c']
        plotColM = ['g--']
      p.plot(1,DL,temp,label=['Best fit line {}$^\circ$C'.format(key)],plotCol=plotColB)
      p.plot(1,self.dicCaldata[key][:,1],self.dicCaldata[key][:,0],label=['Measured {}$^\circ$C'.format(key)],markers=['x'],plotCol=plotColM)
      currentP = p.getSubPlot(1)
      currentP.set_xlabel('Digital level')
      currentP.set_ylabel('Temperature K')
      currentP.set_title('{}-{} at Tinstr={} $^\circ$C'.format(self.name, self.filterName, key))
    p.saveFig('{}-DLTemp.png'.format(self.name,'CalFilter'))


  ################################################################
  def CalculateCalibrationTables(self):
    """Calculate the mapping functions between digital level, radiance and temperature

       Using the spectral curves and DL vs. temperature calibration inputs
       calculate the various mapping functions between digital level, radiance 
       and temperature. Set up the various tables for later conversion.
    """
    #load data if not yet loaded
    if not 'NoFilter' in self.dicSpectrals:
      self.LoadPrepareData()

    self.dicTableDLRad = collections.defaultdict(float)
    self.dicLookupDLRad = collections.defaultdict(float)
    self.dicLookupRadDL = collections.defaultdict(float)
    self.dicTableTempRad = collections.defaultdict(float)
    self.dicLookupTempRad = collections.defaultdict(float)
    self.dicLookupRadTemp = collections.defaultdict(float)
    self.DlRadCoeff = collections.defaultdict(float)
    self.dicinterpDL = collections.defaultdict(float)

    # calculate radiance values if not yet done
    if not 'NoFilter' in self.dicRadiance:

      #set up the DL range
      interpDL = np.linspace(0,2**14, 2**8 + 1)
      # print(interpDL)

      # do for all instrument temperatures and spectral variations  
      for i,spectral in enumerate(self.spectrals):
        self.dicSpecRadiance[spectral] = collections.defaultdict(float)

        self.dicRadiance[spectral] = collections.defaultdict(float)
        self.dicIrradiance[spectral] = collections.defaultdict(float)

        for key in self.dicCaldata:
          #temperature is in 0'th column of dicCaldata[key]
          #planck array has shape (nu,temp), now get spectrals to same shape, then
          #integrate along nu axis=0
          xx = np.ones(self.dicCaldata[key][:,0].shape)
          _, spectrl = np.meshgrid(xx,self.dicSpectrals[spectral])
          #now spectrl has the same shape as the planck function return

          self.dicSpecRadiance[spectral][key] = \
          spectrl * ryplanck.planck(self.nu, self.dicCaldata[key][:,0], type='en')
          self.dicRadiance[spectral][key] = \
          np.trapz(self.dicSpecRadiance[spectral][key], x=self.nu,  axis=0) / np.pi
          self.dicIrradiance[spectral][key] = \
          self.dicRadiance[spectral][key] * self.radiancetoIrradiance

          #if first time, stack horizontally to the calibration array, otherwise overwrite
          if self.dicCaldata[key].shape[1] < 6:
            self.dicCaldata[key] = np.hstack((self.dicCaldata[key], 
                                              self.dicRadiance[spectral][key].reshape(-1,1)))
            self.dicCaldata[key] = np.hstack((self.dicCaldata[key], 
                                              self.dicIrradiance[spectral][key].reshape(-1,1)))
          else:
            self.dicCaldata[key][:,2*i+2] = self.dicRadiance[spectral][key].reshape(-1,)
            self.dicCaldata[key][:,2*i+3] = self.dicIrradiance[spectral][key].reshape(-1,)

          if spectral == 'NoFilter':
            # now determine the best fit between radiance and DL
            # the relationship should be linear y = mx + c
            # x=digital level, y=radiance
            coeff = np.polyfit(self.dicCaldata[key][:,1], 
                              self.dicRadiance[spectral][key].reshape(-1,), deg=1)
            # print('straight line fit DL = {} L + {}'.format(coeff[0],coeff[1]))
            self.DlRadCoeff[key] = coeff
            interpL = np.polyval(coeff, interpDL)
            # add the DL floor due to instrument & optics temperature
            pow = self.dicPower[key]
            self.dicinterpDL[key] = (interpDL ** pow + self.diclFloor[key] ** pow) ** (1./pow)
            #now save a lookup table for key value
            self.dicTableDLRad[key] = \
            np.hstack(([self.dicinterpDL[key].reshape(-1,1), interpL.reshape(-1,1)]))

            #now calculate a high resolution lookup table between temperature and radiance
            tempHiRes = np.linspace(np.min(self.dicCaldata[key][:,0])-100, 
                                    np.max(self.dicCaldata[key][:,0])+100, 101)
            xx = np.ones(tempHiRes.shape)
            _, spectrlHR = np.meshgrid(xx,self.dicSpectrals[spectral])
            specLHiRes = spectrlHR * ryplanck.planck(self.nu, tempHiRes, type='en')
            LHiRes = np.trapz(specLHiRes, x=self.nu, axis=0) / np.pi
            self.dicTableTempRad[key] = \
                                np.hstack((tempHiRes.reshape(-1,1),LHiRes.reshape(-1,1) ))

        if spectral == 'NoFilter':
          #calculate the radiance in the optics and sensor for later interpolation of Tinternal
          tempTint = 273.15 + np.linspace(np.min(self.dicCaldata.keys())-20, 
                                          np.max(self.dicCaldata.keys())+20, 101)
          xx = np.ones(tempTint.shape)
          _, spectrlHR = np.meshgrid(xx,self.dicSpectrals[spectral])
          specLTint = spectrlHR * ryplanck.planck(self.nu, tempTint, type='en')
          LTint = np.trapz(specLTint, x=self.nu, axis=0) / np.pi
          self.TableTintRad = np.hstack((tempTint.reshape(-1,1),LTint.reshape(-1,1) ))
          

  ################################################################
  def LookupDLRad(self, DL, Tint):
    """ Calculate the radiance associated with a DL and Tint pair.
        Interpolate linearly on Tint radiance not temperature.
    """
    # calculate radiance values if not yet done
    if not 'NoFilter' in self.dicRadiance:
      self.CalculateCalibrationTables()

    #get radiance values for lower and upper Tint and the actual Tin
    Llo = np.interp(self.lokey+273.15, self.TableTintRad[:,0], self.TableTintRad[:,1])
    Lhi = np.interp(self.hikey+273.15, self.TableTintRad[:,0], self.TableTintRad[:,1])
    Lti = np.interp(Tint+273.15, self.TableTintRad[:,0], self.TableTintRad[:,1])

    #find the parametric value for Tint radiance, do this once
    paraK = (Lti - Llo) / (Lhi - Llo)
    return self.LookupDLRadHelper(DL, paraK)


  ################################################################
  def LookupDLRadHelper(self, DL, paraK):
    """Calculate the radiance associated with a DL and parametric pair. The 
       parametric variable was calculated once and used for all DL values.
    """

    # numpy's interp supports arrays as input, but only does linear interpolation
    lo = np.interp(DL, self.dicTableDLRad[self.lokey][:,0], self.dicTableDLRad[self.lokey][:,1])
    hi = np.interp(DL, self.dicTableDLRad[self.hikey][:,0], self.dicTableDLRad[self.hikey][:,1])
    return  lo + (hi - lo) * paraK


  ################################################################
  def LookupDLTemp(self, DL, Tint):
    """ Calculate the temperature associated with a DL and Tint pair.
        Here we interpolate linearly on Tint temperature - actually we must 
        interpolate linearly on radiance - to be done later.
        Note that dicLookupRadTemp is available for both Tint values, but
        it has the same value in both cases.
    """
    # calculate radiance values if not yet done
    if not 'NoFilter' in self.dicRadiance:
      self.CalculateCalibrationTables()

    L = self.LookupDLRad(DL, Tint)
    t = np.interp(L, self.dicTableTempRad[self.hikey][:,1], self.dicTableTempRad[self.hikey][:,0])

    return t


################################################################
################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    #--------------------------------------------------------------
    # first read the ptw file
    ptwfile  = 'data/PyradiSampleLWIR.ptw'
    outfilename = 'PyradiSampleLWIR.txt'

    header = readPTWHeader(ptwfile)
    showHeader(header)

    #loading sequence of frames
    framesToLoad = [3,4,10]
    data = getPTWFrames (header, framesToLoad)
    print(data.shape)

    #loading sequence of frames, with an error in request
    framesToLoad = [0,4,10]
    data = getPTWFrames (header, framesToLoad)
    print(data.shape)

    #loading single frames
    framesToLoad = range(1, 101, 1)
    frames = len(framesToLoad)
    data = getPTWFrame (header, framesToLoad[0])
    for frame in framesToLoad[1:]:
        f = (getPTWFrame (header, frame))
        data = np.concatenate((data, f))

    rows = header.h_Rows
    cols = header.h_Cols
    img = data.reshape(frames, rows ,cols)
    print(img.shape)


    #--------------------------------------------------------------
    # first read the calibration file file
    calData = JadeCalibrationData('./data/lwir100mm150us10ND20090910.xml', './data')

    #presumably calibration was done at short range, with blackbody.
    calData.CalculateCalibrationTables()
    calData.PlotSpectrals()
    calData.PlotRadiance()
    calData.PlotDLRadiance()
    calData.PlotTempRadiance()
    calData.PlotDLTemp()
    calData.PlotTintRad()
    print(calData.Print())

    for Tint in [17.1]:
      # for DL in [5477, 6050, 6817, 7789, 8922, 10262, 11694, 13299, 14921 ]:
      for DL in [4571, 5132, 5906, 6887, 8034, 9338, 10834, 12386, 14042 ]:
        print('Tint={} DL={} T={:.1f} L={:.0f}'.format(Tint, DL, calData.LookupDLTemp(DL, Tint)-273.15, calData.LookupDLRad(DL, Tint)))
        # print('Tint={} DL={}  L={:.1f}'.format(Tint, DL,  calData.LookupDLRad(DL, Tint)))
    print(' ')
    for Tint in [34.4]:
      for DL in [5477, 6050, 6817, 7789, 8922, 10262, 11694, 13299, 14921 ]:
        print('Tint={} DL={} T={:.1f} L={:.0f}'.format(Tint, DL, calData.LookupDLTemp(DL, Tint)-273.15, calData.LookupDLRad(DL, Tint)))
        # print('Tint={} DL={}  L={:.1f}'.format(Tint, DL,  calData.LookupDLRad(DL, Tint)))
    print(' ')
    for Tint in [25]:
      for DL in [5477, 6050, 6817, 7789, 8922, 10262, 11694, 13299, 14921 ]:
        print('Tint={} DL={} T={:.1f} L={:.0f}'.format(Tint, DL, calData.LookupDLTemp(DL, Tint)-273.15, calData.LookupDLRad(DL, Tint)))
        # print('Tint={} DL={}  L={:.1f}'.format(Tint, DL,  calData.LookupDLRad(DL, Tint)))

    """The following measurement observed a large-area blackbody in the lab.
       The blackbody set point was 150C. The Cedip Altair s/w calculated 157C.
       This code calculates 149C.
    """
    ptwfile  = 'data/LWIR-BBref-150C-150us.ptw'
    header = readPTWHeader(ptwfile)
    # showHeader(header)
    #loading sequence of frames
    framesToLoad = [1]
    for frame in framesToLoad:
        data = getPTWFrame (header, frame)
        #get the internal temperature from the header and use here
        tempIm = calData.LookupDLTemp(data, header.h_HousingTemperature1-273.15)
        print('Temperature at ({},{})={} C'.format(160,120,tempIm[160,120]-273.15))
        print('Temperature at ({},{})={} C'.format(140,110,tempIm[140,110]-273.15))
        I = ryplot.Plotter(4, 1, 1,'', figsize=(8, 8))
        I.showImage(1, tempIm, ptitle='{}, frame {}, temperature in K'.format(ptwfile[:-4],frame), titlefsize=7, cbarshow=True, cbarfontsize=7)
        I.saveFig('{}-{}.png'.format(os.path.basename(ptwfile)[:-4],frame))

    print('ryptw Done!')
