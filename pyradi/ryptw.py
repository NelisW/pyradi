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





__version__= "$Revision$"
__author__='JJ Calitz'
__all__=['myint','mylong','myfloat','mybyte', 'mydouble', 'ReadPTWHeader',
'LoadPrepareDataSpectrals', 'calculateDataTables',
'GetPTWFrameFromFile', 'terminateStrOnZero',
'ShowHeader', 'GetPTWFrame', 'GetPTWFrames']

import sys
import collections
import os.path
import string

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
import pyradi.rylookup as rylookup

import copy


################################################################
# Miscelaneous functions to read bytes, int, long, float values.
# 
# https://docs.python.org/3/library/struct.html

def myint(x):
    # two bytes length
    # print(type(x), type(x[0]), type(ord(x[0])))

    # ans = ord(x[0])+(ord(x[1])<<8)

    ans = struct.unpack('h',x)[0]
    return ans

def mylong(x):
    # four bytes length
    if sys.version_info[0] > 2:
        ans = x[0] + (x[1]<<8) + (x[2]<<16) + (x[3]<<32)
    else:
        ans = ord(x[0]) + (ord(x[1])<<8) + (ord(x[2])<<16) + (ord(x[3])<<32)
    return ans

def myfloat(x):
    ans = struct.unpack('f',x)[0]
    # ans = ans[0]
    return ans

def mydouble(x):
    ans = struct.unpack('d',x)[0]
    # ans = ans[0]
    return ans

def mybyte(x):
    # will return an error if x is more than a byte length
    # TODO need to write error catch routine
    ans = ord(x)
    return ans

def myRGB(x):
    # three bytes with RGB values 
    # TODO need to write error catch routine
    R = x[0] if sys.version_info[0] > 2 else ord(x[0])
    G = x[1] if sys.version_info[0] > 2 else ord(x[1])
    B = x[2] if sys.version_info[0] > 2 else ord(x[2])

    ans = '#{:02x}{:02x}{:02x}'.format(R,G,B)
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
    self.h_EofAsciiCode = 0 #[10]
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
    self.h_FileSaveCent = 0

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
    self.h_ORIONIntegrationTime = 0 #myfloat(headerinfo[413:437]) # ORION (6 values)
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
    self.h_UnitIndexValid = 0 #ord(headerinfo[1924:1925])
    self.h_CurrentUnitIndex = 0 #myint(headerinfo[1925:1927])
    self.h_ZoomPosition = 0 #(headerinfo[1927:1935]) # unknown format POINT
    self.h_KeyFrameNumber = 0 #ord(headerinfo[1935:1936])
    self.h_KeyFramesInFilm = 0 #headerinfo[1936:2056] # set of 30 frames
    self.h_PlayerLocked = 0 # ord(headerinfo[2057:2057])
    self.h_FrameSelectionValid = 0 #ord(headerinfo[2057:2058])
    self.h_FrameofROIStart = 0 #mylong(headerinfo[2058:2062])
    self.h_FrameofROIEnd = 0 #mylong(headerinfo[2062:2066])
    self.h_PlayerLockedROI = 0# ord(headerinfo[2066:2067])
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

    # Frame time
    self.h_frameMinute = 0 #FrameHeader[80:81]
    self.h_frameHour = 0 #FrameHeader[81:82]
    self.h_frameSecond = 0 #h_second+(h_thousands+h_hundred)/1000.0

    # detector / FPA temperature
    self.h_detectorTemp = 0.0 #FrameHeader[228:232]
    self.h_sensorTemp4 = 0.0 #FrameHeader[232:236]

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

    if sys.version_info[0] > 2:
        Header.h_Signature = headerinfo[0:3].decode('utf-8')
    else:
        Header.h_Signature = headerinfo[0:3]
    if Header.h_Signature == 'AIO': #AGEMA
        Header.h_format = 'agema'
    elif Header.h_Signature == 'CED':
        Header.h_format = 'cedip'
        Header.h_unit = 'dl'

    if sys.version_info[0] > 2:
        Header.h_Version = headerinfo[5:10].decode('utf-8')
    else:
        Header.h_Version = headerinfo[5:10]
    if not Header.h_Version[-1] in string.printable:
        Header.h_Version =  Header.h_Version[:-1]
    Header.h_EofAsciiCode = mybyte(headerinfo[10:11])
    Header.h_MainHeaderSize = mylong(headerinfo[11:15])
    Header.h_FrameHeaderSize = mylong(headerinfo[15:19])
    Header.h_SizeOfOneFrameAndHeader = mylong(headerinfo[19:23])
    Header.h_SizeOfOneFrame = mylong(headerinfo[23:27])
    Header.h_NumberOfFieldInFile = mylong(headerinfo[27:31])
    # Header.h_CurrentFieldNumber = myint(headerinfo[31:35])
    Header.h_CurrentFieldNumber = mylong(headerinfo[31:35])

    #Header.h_FileSaveDate = '' #[35:39] decoded below
    Header.h_FileSaveYear = myint(headerinfo[35:37])
    Header.h_FileSaveDay = ord(headerinfo[37:38])
    Header.h_FileSaveMonth = ord(headerinfo[38:39])

    #Header.h_FileSaveTime = '' #[39:43] decoded below
    Header.h_FileSaveMinute = ord(headerinfo[39:40])
    Header.h_FileSaveHour = ord(headerinfo[40:41])
    Header.h_FileSaveCent = ord(headerinfo[41:42])
    Header.h_FileSaveSecond = ord(headerinfo[42:43])

    Header.h_Millieme = ord(headerinfo[43:44])


    if sys.version_info[0] > 2:
        stripchar = terminateStrOnZero(headerinfo[44:64]).decode('utf-8')[-1]
        Header.h_CameraName = terminateStrOnZero(headerinfo[44:64]).decode('utf-8').rstrip(stripchar)
        Header.h_LensName = terminateStrOnZero(headerinfo[64:84]).decode('utf-8').rstrip(stripchar)
        Header.h_FilterName = terminateStrOnZero(headerinfo[84:104]).decode('utf-8').rstrip(stripchar)
        Header.h_ApertureName = terminateStrOnZero(headerinfo[104:124]).decode('utf-8').rstrip(stripchar)
    else:
        Header.h_CameraName = terminateStrOnZero(headerinfo[44:64])
        Header.h_LensName = terminateStrOnZero(headerinfo[64:84])
        Header.h_FilterName = terminateStrOnZero(headerinfo[84:104])
        Header.h_ApertureName = terminateStrOnZero(headerinfo[104:124])

    Header.h_IRUSBilletSpeed = myfloat(headerinfo[124:128]) # IRUS
    Header.h_IRUSBilletDiameter = myfloat(headerinfo[128:132]) # IRUS
    Header.h_IRUSBilletShape = myint(headerinfo[132:134]) #IRUS
    Header.h_Reserved134 = mybyte(headerinfo[134:141])
    Header.h_Emissivity = myfloat(headerinfo[141:145])
    Header.h_Ambiant = myfloat(headerinfo[145:149])
    Header.h_Distance = myfloat(headerinfo[149:153])
    Header.h_IRUSInductorCoil = ord(headerinfo[153:154]) # IRUS
    Header.h_IRUSInductorPower = mylong(headerinfo[154:158]) # IRUS
    Header.h_IRUSInductorVoltage = myint(headerinfo[158:160]) # IRUS
    Header.h_IRUSInductorFrequency = mylong(headerinfo[160:164]) # IRUS
    Header.h_Reserved164 = mybyte(headerinfo[164:169])
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
    if sys.version_info[0] > 2:
        stripchar = terminateStrOnZero(headerinfo[220:231]).decode('utf-8')[-1]
        Header.h_CameraSerialNumber = terminateStrOnZero(headerinfo[220:231]).decode('utf-8').rstrip(stripchar)
    else:
        Header.h_CameraSerialNumber = terminateStrOnZero(headerinfo[220:231])
        
        
    Header.h_Reserved231 = mybyte(headerinfo[231:239])
    Header.h_DetectorCode = myint(headerinfo[239:243])
    Header.h_DetectorGain = my(headerinfo[245:247])

    
    
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
    if sys.version_info[0] > 2:
        Header.h_ExternalSynch = headerinfo[397] # 1=External 0 = Internal
    else:
        Header.h_ExternalSynch = ord(headerinfo[397]) # 1=External 0 = Internal

    Header.h_CEDIPAquisitionPeriod = myfloat(headerinfo[403:407]) # CEDIP seconds
    Header.h_CEDIPIntegrationTime = myfloat(headerinfo[407:411]) # CEDIP seconds
    Header.h_WOLFSubwindowCapability = myint(headerinfo[411:413]) # WOLF
    Header.h_ORIONIntegrationTime = headerinfo[413:437] # ORION (6 values)
    Header.h_ORIONFilterNames = headerinfo[437:557] # ORION 6 fields of 20 chars each
    Header.h_NucTable = myint(headerinfo[557:559])
    Header.h_Reserve6 = headerinfo[559:563]
    if sys.version_info[0] > 2:
        stripchar = terminateStrOnZero(headerinfo[563:1563]).decode('utf-8')[-1]
        Header.h_Comment = terminateStrOnZero(headerinfo[563:1563]).decode('utf-8').rstrip(stripchar)
        stripchar = terminateStrOnZero(headerinfo[1563:1663]).decode('utf-8')[-1]
        Header.h_CalibrationFileName = terminateStrOnZero(headerinfo[1563:1663]).decode('utf-8').rstrip(stripchar)
        stripchar = terminateStrOnZero(headerinfo[1663:1919]).decode('utf-8')[-1]
        Header.h_ToolsFileName = terminateStrOnZero(headerinfo[1663:1919]).decode('utf-8').rstrip(stripchar)
    else:
        Header.h_Comment = terminateStrOnZero(headerinfo[563:1563])
        Header.h_CalibrationFileName = terminateStrOnZero(headerinfo[1563:1663])
        Header.h_ToolsFileName = terminateStrOnZero(headerinfo[1663:1919])

    Header.h_PaletteIndexValid = ord(headerinfo[1919:1920])
    Header.h_PaletteIndexCurrent = myint(headerinfo[1920:1922])
    Header.h_PaletteToggle = ord(headerinfo[1922:1923])
    Header.h_PaletteAGC = ord(headerinfo[1923:1924])
    Header.h_UnitIndexValid = ord(headerinfo[1924:1925])
    Header.h_CurrentUnitIndex = myint(headerinfo[1925:1927])
    if sys.version_info[0] > 2:
        stripchar = terminateStrOnZero(headerinfo[1927:1935]).decode('utf-8')[-1]
        Header.h_ZoomPosition = terminateStrOnZero(headerinfo[1927:1935]).decode('utf-8').rstrip(stripchar) # unknown format POINT
        Header.h_KeyFramesInFilm = terminateStrOnZero(headerinfo[1936:2056]).decode('utf-8').rstrip(stripchar) # set of 30 frames
    else:
        Header.h_ZoomPosition = terminateStrOnZero(headerinfo[1927:1935]) # unknown format POINT
        Header.h_KeyFramesInFilm = terminateStrOnZero(headerinfo[1936:2056]) # set of 30 frames
    Header.h_KeyFrameNumber = ord(headerinfo[1935:1936])
    Header.h_PlayerLocked =  ord(headerinfo[2056:2057])
    Header.h_FrameSelectionValid = ord(headerinfo[2057:2058])
    Header.h_FrameofROIStart = mylong(headerinfo[2058:2062])
    Header.h_FrameofROIEnd = mylong(headerinfo[2062:2066])
    Header.h_PlayerLockedROI =  ord(headerinfo[2066:2067])
    Header.h_PlayerInfinitLoop = ord(headerinfo[2067:2068])
    Header.h_PlayerInitFrame = mylong(headerinfo[2068:2072])

    Header.h_Isoterm0Active = ord(headerinfo[2072:2073])
    Header.h_Isoterm0DLMin = myint(headerinfo[2073:2075])
    Header.h_Isoterm0DLMax = myint(headerinfo[2075:2077])
    Header.h_Isoterm0Color = myRGB(headerinfo[2077:2081])

    Header.h_Isoterm1Active = ord(headerinfo[2081:2082])
    Header.h_Isoterm1DLMin = myint(headerinfo[2082:2084])
    Header.h_Isoterm1DLMax = myint(headerinfo[2084:2086])
    Header.h_Isoterm1Color = myRGB(headerinfo[2086:2090])

    Header.h_Isoterm2Active = ord(headerinfo[2090:2091])
    Header.h_Isoterm2DLMin = myint(headerinfo[2091:2093])
    Header.h_Isoterm2DLMax = myint(headerinfo[2093:2095])
    Header.h_Isoterm2Color = myRGB(headerinfo[2095:2099])

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
    Header.h_AxeColor = myRGB(headerinfo[2131:2135])
    Header.h_AxeSize = mylong(headerinfo[2135:2139])
    Header.h_AxeValid = ord(headerinfo[2139:2140])
    Header.h_DistanceOffset = myfloat(headerinfo[2140:2144])
    Header.h_HistoEqualizationEnabled = ord(headerinfo[2144:2145])
    Header.h_HistoEqualizationPercent = myint(headerinfo[2145:2147])
    if sys.version_info[0] > 2:
        stripchar = terminateStrOnZero(headerinfo[2147:2403]).decode('utf-8')[-1]
        Header.h_CalibrationFileName = terminateStrOnZero(headerinfo[2147:2403]).decode('utf-8').rstrip(stripchar)
    else:
        Header.h_CalibrationFileName = terminateStrOnZero(headerinfo[2147:2403])


    Header.h_PTRTopFrameValid = ord(headerinfo[2403:2404])
    # Header.h_SubSampling = myint(headerinfo[2404:2408])
    Header.h_SubSampling = mylong(headerinfo[2404:2408])
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

    #Get the frame time
    header.h_frameMinute = ord(FrameHeader[80:81])
    header.h_frameHour = ord(FrameHeader[81:82])
    h_hundred = ord(FrameHeader[82:83])*10
    h_second = ord(FrameHeader[83:84])
    h_thousands = ord(FrameHeader[160:161])
    header.h_frameSecond = h_second+(h_hundred+h_thousands)/1000.0
	#detector FPA temperature
    header.h_detectorTemp = myfloat(FrameHeader[228:232]) 
    header.h_sensorTemp4 = myfloat(FrameHeader[232:236]) 
    if header.h_sensorTemp4 is None:
        header.h_sensorTemp4 = 0.0
    if header.h_detectorTemp is None:
        header.h_detectorTemp = 0.0

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
    GetPTWFrameFromFile was written.  

    The contents of the header is changed (new information added: frame time 
    and detector temperature).  The header is returned from the function to
    make it explicit that the contents have changed from the header passed into
    the function.

    Args:
        | header (class object)
        | frameindex (integer): The frame to be extracted

    Returns:
        | header.data (np.ndarray): requested frame DL values, dimensions (rows,cols)
        | header (class): updated header now with frame time and FPA temperature

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

    return header.data.conj().transpose(), header


################################################################
def getPTWFrames (header, loadFrames=[]):
    """Retrieve a number of PTW frames, given in a list of frameheader.
    The function returns the image data as well as the file and image header 
    data (time and FPA temperature) valid for the particular frame.
    The frame header data is written in the same class as is the file
    header, in order to keep all the information together for the frame.

    Args:
        | header (class object)
        | loadFrames ([int]): List of indices for frames to be extracted

    Returns:
        | data (np.ndarray): requested image frame DL values, dimensions (frames,rows,cols)
        | fheaders (object): requested image frame header values

    Raises:
        | No exception is raised.
    """

    fheaders = []

    # error checking on inputs
    errorresult = np.asarray([0])
    # Check if this is  a cedip file
    if header.h_format!='cedip':
        print('getPTWFrames Error: file format is not supported')
        return errorresult,None
    #check for legal frame index values
    npFrames = np.asarray(loadFrames)
    if np.any( npFrames < 1 ) or np.any ( npFrames > header.h_lastframe ):
        print('getPTWFrames Error: at least one requested frame not in file')
        print('legal frames for this file are: {0} to {1}'.format(1,header.h_lastframe))
        return errorresult, None

    data, headerx = getPTWFrame (header, loadFrames[0])
    fheaders.append(headerx)

    for frame in loadFrames[1:]:
        datax, headerx = getPTWFrame (header, frame)
        data = np.concatenate((data, datax))
        fheaders.append(headerx)

    rows = header.h_Rows
    cols = header.h_Cols

    return data.reshape(len(loadFrames), rows ,cols), fheaders


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

    print('{} version {}'.format(Header.h_Signature, Header.h_Version))
    print('Main Header Size {}'.format(Header.h_MainHeaderSize))
    print('Frame Header Size {}'.format(Header.h_FrameHeaderSize))
    print('Frame + Frame Header Size {}'.format(Header.h_SizeOfOneFrameAndHeader))
    print('Frame Size {}'.format(Header.h_SizeOfOneFrame))
    print('Number of Frames {}'.format(Header.h_NumberOfFieldInFile))
    #print Header.h_CurrentFieldNumber

    print('Year {} Month {} Day {}'.format(Header.h_FileSaveYear, Header.h_FileSaveMonth, 
        Header.h_FileSaveDay))
    print('( {} / {} / {} )'.format(str(Header.h_FileSaveYear).zfill(2), 
        str(Header.h_FileSaveMonth).zfill(2), str(Header.h_FileSaveDay).zfill(2)))
    print('Hour {} Minute {} Second {}'.format(Header.h_FileSaveHour,Header.h_FileSaveMinute,
        Header.h_FileSaveSecond))
    print('( {} : {} : {} )'.format(str(Header.h_FileSaveHour).zfill(2),
        str(Header.h_FileSaveMinute).zfill(2),str(Header.h_FileSaveSecond).zfill(2)))

    #print Header.h_Millieme
    print ('Camera Name {}'.format(Header.h_CameraName))
    print ('Lens {}'.format(Header.h_LensName))
    print ('Filter {}'.format(Header.h_FilterName))
    print ('Aperture Name {}'.format( Header.h_ApertureName))
    if Header.h_Signature == 'IRUS':
        print ('{}'.format(Header.h_IRUSBilletSpeed))
        print ('{}'.format(Header.h_IRUSBilletDiameter))
        print ('{}'.format(Header.h_IRUSBilletShape))
    print ('Emissivity {:.6f}'.format(Header.h_Emissivity))
    print ('Ambient Temperature {:.6f} (K)'.format(Header.h_Ambiant))
    print ('Ambient Temperature {:.6f} (degC)'.format(Header.h_Ambiant-273.15))
    print ('Distance to target {}'.format(Header.h_Distance))
    if Header.h_Signature == 'IRUS':
        print ('{}'.format(Header.h_IRUSInductorCoil))
        print ('{}'.format(Header.h_IRUSInductorPower))
        print ('{}'.format(Header.h_IRUSInductorVoltage))
        print ('{}'.format(Header.h_IRUSInductorFrequency))
        print ('{}'.format(Header.h_IRUSSynchronization))
    print ('Atm Transmission {}'.format(Header.h_AtmTransmission))
    print ('Ext Coef {}'.format(Header.h_ExtinctionCoeficient))
    print ('Target {}'.format(Header.h_Object))
    print ('Optic {}'.format(Header.h_Optic))
    print ('Atmo {}'.format(Header.h_Atmo))
    print ('Atm Temp {:.6f}'.format(Header.h_AtmosphereTemp))
    print ('Cut on Wavelength {:.6f}'.format(Header.h_CutOnWavelength))
    print ('Cut off Wavelength {:.6f}'.format(Header.h_CutOffWavelength))
    print ('PixelSize {}'.format(Header.h_PixelSize))
    print ('PixelPitch {}'.format(Header.h_PixelPitch))
    print ('Detector Apperture {}'.format(Header.h_DetectorApperture))
    print ('Optic Focal Length {}'.format(Header.h_OpticsFocalLength))
    print ('Housing Temp1 {:.6f} (K)'.format(Header.h_HousingTemperature1))
    print ('Housing Temp2 {:.6f} (K)'.format(Header.h_HousingTemperature2))

    print ('Sensor Temp4 {:.6f} (K)'.format(Header.h_sensorTemp4))
    print ('Detector/FPA Temp {:.6f} (K)'.format(Header.h_detectorTemp))
    
    print ('Camera Serial Number {}'.format(Header.h_CameraSerialNumber))
    print ('Min Threshold {}'.format(Header.h_MinimumLevelThreshold))
    print ('Max Threshold {}'.format(Header.h_MaximumLevelThreshold))
    #print Header.h_EchelleSpecial
    #print Header.h_EchelleUnit
    #print Header.h_EchelleValue
    print ('Gain {}'.format(Header.h_LockinGain))
    print ('Offset {}'.format(Header.h_LockinOffset))
    print ('HZoom {}'.format(Header.h_HorizontalZoom))
    print ('VZoom {}'.format(Header.h_VerticalZoom))
    print ('Field {}'.format(Header.h_PixelsPerLine,'X',Header.h_LinesPerField))
    print ('AD converter {} bit'.format(Header.h_ADDynamic))




    if Header.h_Signature == 'SATIR':
        print ('{}'.format(Header.h_SATIRTemporalFrameDepth))
        print ('{}'.format(Header.h_SATIRLocationLongitude))
        print ('{}'.format(Header.h_SATIRLocationLatitude))
        print ('{}'.format(Header.h_SATIRLocationAltitude))
    if Header.h_ExternalSynch:
        print ('Ext Sync ON')
    else:
        print ('Ext Sync OFF')
    print('Header.h_Signature = {}'.format(Header.h_Signature))
    if Header.h_Signature == 'CED':
        print ('CEDIP Period {:.6f} Hz'.format(1./Header.h_CEDIPAquisitionPeriod))
        print ('CEDIP Integration {:.6f} msec'.format(Header.h_CEDIPIntegrationTime*1000))
    if Header.h_Signature == 'WOLF':
        print ( '{}'.format(Header.h_WOLFSubwindowCapability))
    if Header.h_Signature == 'ORI':
        print ( '{}'.format(Header.h_ORIONIntegrationTime))
        print ( '{}'.format(Header.h_ORIONFilterNames))
    print ('NUC {}'.format(Header.h_NucTable))
    #print Header.h_Reserve6
    print ('Comment {}'.format(Header.h_Comment))

    print ('Calibration File Name {}'.format(Header.h_CalibrationFileName))

    print ('Tools File Name {}'.format(Header.h_ToolsFileName))

    print ('Palette Index {}'.format(Header.h_PaletteIndexValid))
    print ('Palette Current {}'.format(Header.h_PaletteIndexCurrent))
    print ('Palette Toggle {}'.format(Header.h_PaletteToggle))
    print ('Palette AGC {}'.format(Header.h_PaletteAGC))
    print ('Unit Index {}'.format(Header.h_UnitIndexValid))
    print ('Current Unit Index {}'.format(Header.h_CurrentUnitIndex))
    print ('Zoom Pos {}'.format(Header.h_ZoomPosition))
    print ('Key Framenum {}'.format(Header.h_KeyFrameNumber))
    print ('Num Keyframes {}'.format(Header.h_KeyFramesInFilm))
    print ('Player lock {}'.format(Header.h_PlayerLocked))
    print ('Frame Select {}'.format(Header.h_FrameSelectionValid))
    print ('ROI Start {}'.format(Header.h_FrameofROIStart))
    print ('ROI Stop {}'.format(Header.h_FrameofROIEnd))
    print ('Player inf loop {}'.format(Header.h_PlayerInfinitLoop))
    print ('Player Init Frame {}'.format(Header.h_PlayerInitFrame))

    print ('Isoterm0 {}'.format(Header.h_Isoterm0Active))
    print ('Isoterm0 DL Min {}'.format(Header.h_Isoterm0DLMin))
    print ('Isoterm0 DL Max {}'.format(Header.h_Isoterm0DLMax))
    print ('Isoterm0 Color RGB {}'.format(Header.h_Isoterm0Color))

    print ('Isoterm1 {}'.format(Header.h_Isoterm1Active))
    print ('Isoterm1 DL Min {}'.format(Header.h_Isoterm1DLMin))
    print ('Isoterm1 DL Max {}'.format(Header.h_Isoterm1DLMax))
    print ('Isoterm1 Color RGB {}'.format(Header.h_Isoterm1Color))

    print ('Isoterm2 {}'.format(Header.h_Isoterm2Active))
    print ('Isoterm2 DL Min {}'.format(Header.h_Isoterm2DLMin))
    print ('Isoterm2 DL Max {}'.format(Header.h_Isoterm2DLMax))
    print ('Isoterm2 Color RGB {}'.format(Header.h_Isoterm2Color))

    print ('Zero {}'.format(Header.h_ZeroActive))
    print ('Zero DL {}'.format(Header.h_ZeroDL))
    print ('Palette Width {}'.format(Header.h_PaletteWidth))
    print ('PaletteF Full {}'.format(Header.h_PaletteFull))
    print ('PTR Frame Buffer type {}'.format(Header.h_PTRFrameBufferType))
    print ('Thermoelasticity {}'.format(Header.h_ThermoElasticity))
    print ('Demodulation {}'.format(Header.h_DemodulationFrequency))
    print ('Coordinate Type {}'.format(Header.h_CoordinatesType))
    print ('X Origin {}'.format(Header.h_CoordinatesXorigin))
    print ('Y Origin {}'.format(Header.h_CoordinatesYorigin))
    print ('Coord Show Orig {}'.format(Header.h_CoordinatesShowOrigin))
    print ('Axe Colour RGB {}'.format(Header.h_AxeColor))
    print ('Axe Size {}'.format(Header.h_AxeSize))
    print ('Axe Valid {}'.format(Header.h_AxeValid))
    print ('Distance offset {}'.format(Header.h_DistanceOffset))
    print ('Histogram {}'.format(Header.h_HistoEqualizationEnabled))
    print ('Histogram % {}'.format(Header.h_HistoEqualizationPercent))

    print ('Calibration File Name {}'.format(Header.h_CalibrationFileName))

    print ('PTRFrame Valid {}'.format(Header.h_PTRTopFrameValid))
    print ('Subsampling {}'.format(Header.h_SubSampling))
    print ('Camera flip H {}'.format(Header.h_CameraHFlip))
    print ('Camera flip V {}'.format(Header.h_CameraHVFlip))
    print ('BB Temp {}'.format(Header.h_BBTemp))
    print ('Capture Wheel Index {}'.format(Header.h_CaptureWheelIndex))
    print ('Capture Wheel Focal Index {}'.format(Header.h_CaptureFocalIndex))
    #print Header.h_Reserved7
    #print Header.h_Reserved8
    #print Header.h_Framatone




################################################################################
################################################################################
################################################################################
class JadeCalibrationData:
    """Container to describe the calibration data of a Jade camera.
    """

    def __init__(self,filename, datafileroot):
    
        self.dicCaldata = collections.defaultdict(float)
        # dicSpectrals = collections.defaultdict(str)
        # dicRadiance = collections.defaultdict(str)
        # dicIrradiance = collections.defaultdict(str)
        # dicSpecRadiance = collections.defaultdict(str)
        self.dicFloor = collections.defaultdict(str)
        self.dicPower = collections.defaultdict(str)

        self.pathtoXML = os.path.dirname(filename)
        self.datafileroot = datafileroot
    
        self.wl = None
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
                self.dicFloor[InstrTemp] = float(childA.attrib["DlFloor"])
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
        # print(self.info())

        for i,tmprInstr in enumerate(self.dicCaldata):
            if i==0:
                tmprLow = np.min(self.dicCaldata[tmprInstr][:,0])-100
                tmprHi  = np.max(self.dicCaldata[tmprInstr][:,0])+100
            else:
                tmprLow = min(tmprLow, np.min(self.dicCaldata[tmprInstr][:,0])-100)
                tmprHi  = max(tmprHi, np.max(self.dicCaldata[tmprInstr][:,0])+100)

        tmprInc = (tmprHi - tmprLow) / 100.
        nu = np.linspace(self.nuMin, self.nuMax, int(1 + (self.nuMax - self.nuMin )/self.nuInc ))

        self.LU = rylookup.RadLookup(self.name, nu, tmprLow=tmprLow, tmprHi=tmprHi, tmprInc=tmprInc,
                sensorResp=self.sensorResponseFilename, opticsTau=self.opticsTransmittanceFilename,
                filterTau=self.filterFilename, atmoTau=self.atmoTauFilename, 
                sourceEmis=self.sourceEmisFilename, 
                sigMin=0., sigMax=2.0**14, sigInc=2.0**6, dicCaldata=self.dicCaldata, 
                dicPower=self.dicPower, dicFloor=self.dicFloor)

    def Info(self):
        """Write the calibration data file data to a string and return string.
        """
        str = ""
        str += 'Path to XML                  = {}\n'.format(self.pathtoXML)
        str += 'Path to datafiles            = {}\n'.format(self.datafileroot)
        str += 'Calibration Name             = {}\n'.format(self.name)
        str += 'ID                           = {}\n'.format(self.id)
        str += 'Version                      = {}\n'.format(self.version)
        str += 'Summary                      = {}\n'.format(self.summary)
        str += 'DetectorPitch                = {}\n'.format(self.detectorPitch)
        str += 'FillFactor                   = {}\n'.format(self.fillFactor)
        str += 'Focallength                  = {}\n'.format(self.focallength)
        str += 'integrationTime              = {}\n'.format(self.integrationTime)
        str += 'Fnumber                      = {}\n'.format(self.fnumber)
        str += self.LU.Info()
   
        return str

################################################################
################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    rit = ryutils.intify_tuple

    #--------------------------------------------------------------
    # first read the ptw file
    ptwfile  = 'data/PyradiSampleLWIR.ptw'
    outfilename = 'PyradiSampleLWIR.txt'

    header = readPTWHeader(ptwfile)
    showHeader(header)

    #loading sequence of frames
    print('\n{}\n'.format(30*'-'))
    framesToLoad = [3,4,10]
    data, fheaders = getPTWFrames (header, framesToLoad)
    print(ptwfile)
    print('(1) Data shape={}'.format(rit(data.shape)))
    ifrm = 0
    print('Frame {} summary data:'.format(ifrm))
    print('Image data:\n{}'.format(data[ifrm]))
    print('Sensor Temperature4 {:.6f} K '.format(fheaders[ifrm].h_sensorTemp4))
    print('FPA Temperature {:.6f} K '.format(fheaders[ifrm].h_detectorTemp))
    print('Time {}:{}:{} '.format(fheaders[ifrm].h_frameHour,
            fheaders[ifrm].h_frameMinute,fheaders[ifrm].h_frameSecond))
    
    print('\n{}\n'.format(30*'-'))
    #loading sequence of frames, with an error in request
    framesToLoad = [0,4,10]
    data, fheaders = getPTWFrames (header, framesToLoad)
    print(rit(data.shape))
    print('(2) Data shape={}'.format(rit(data.shape)))
    print('\n{}\n'.format(30*'-'))
    
    #loading single frames
    framesToLoad = list(range(1, 101, 1))
    frames = len(framesToLoad)
    data, fheaders = getPTWFrame (header, framesToLoad[0])
    for frame in framesToLoad[1:]:
        f, fheaders  = getPTWFrame (header, frame)
        data = np.concatenate((data, f))

    rows = header.h_Rows
    cols = header.h_Cols
    img = data.reshape(frames, rows ,cols)
    print(rit(img.shape))

    #--------------------------------------------------------------
    # first read the calibration file file

    calData = JadeCalibrationData('./data/lwir100mm150us10ND20090910.xml', './data')

    #presumably calibration was done at short range, with blackbody.

    calData.LU.PlotSpectrals()
    calData.LU.PlotCalSpecRadiance()
    calData.LU.PlotCalDLRadiance()
    calData.LU.PlotTempRadiance()
    calData.LU.PlotCalDLTemp()
    calData.LU.PlotCalTintRad()
    print('\ncalData.Info:')
    print(calData.Info())

    print(' ')
    for Tint in [17.1]:
        for DL in [4571, 5132, 5906, 6887, 8034, 9338, 10834, 12386, 14042 ]:
            print('Tint={}  DL={} T={:.1f} C  L(filter)={:.0f}  L(no filter)={:.0f} W/(sr.m2)'.format(Tint, DL, 
                calData.LU.LookupDLTemp(DL, Tint)-273.15, 
                calData.LU.LookupDLRad(DL, Tint),
                calData.LU.LookupTempRad(calData.LU.LookupDLTemp(DL, Tint), withFilter=False)))
    print(' ')

    for Tint in [34.4]:
      for DL in [5477, 6050, 6817, 7789, 8922, 10262, 11694, 13299, 14921 ]:
        print('Tint={}  DL={} T={:.1f} C  L(filter)={:.0f}  L(no filter)={:.0f} W/(sr.m2)'.format(Tint, DL, 
            calData.LU.LookupDLTemp(DL, Tint)-273.15, 
            calData.LU.LookupDLRad(DL, Tint),
            calData.LU.LookupTempRad(calData.LU.LookupDLTemp(DL, Tint), withFilter=False)))
    print(' ')

    for Tint in [25]:
      for DL in [5477, 6050, 6817, 7789, 8922, 10262, 11694, 13299, 14921 ]:
        print('Tint={} DL={}  T={:.1f} C  L(filter)={:.0f}  L(no filter)={:.0f} W/(sr.m2)'.format(Tint, DL, 
            calData.LU.LookupDLTemp(DL, Tint)-273.15, 
            calData.LU.LookupDLRad(DL, Tint),
            calData.LU.LookupTempRad(calData.LU.LookupDLTemp(DL, Tint), withFilter=False)))
    print(' ')

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
        data, fheader = getPTWFrame (header, frame)
        print('Sensor Temperature4 {:.6f} K '.format(fheader.h_sensorTemp4))
        print('FPA Temperature {:.6f} K '.format(fheader.h_detectorTemp))
        print('Time {}:{}:{} '.format(fheader.h_frameHour,
                fheader.h_frameMinute,fheader.h_frameSecond))

        #get the internal temperature from the header and use here
        tempIm = calData.LU.LookupDLTemp(data, header.h_HousingTemperature1-273.15)
        # print('Temperature at ({},{})={:.6f} C'.format(160,120,tempIm[160,120]-273.15))
        # print('Temperature at ({},{})={:.6f} C'.format(140,110,tempIm[140,110]-273.15))
        print('Temperature at ({},{})={:.6f} C'.format(160,120,tempIm[160,120]-273.15))
        print('Temperature at ({},{})={:.6f} C'.format(140,110,tempIm[140,110]-273.15))
        I = ryplot.Plotter(4, 1, 1,'', figsize=(8, 8))
        I.showImage(1, tempIm, ptitle='{}, frame {}, temperature in K'.format(ptwfile[:-4],frame), titlefsize=7, cbarshow=True, cbarfontsize=7)
        I.saveFig('{}-{}.png'.format(os.path.basename(ptwfile)[:-4],frame))

    print('\nryptw Done!')
