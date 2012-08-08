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

#The author wishes to thank  Mr Emmanuel Vanneau from FLIR Inc for the permission 
#to publicly release our Python version of the *.ptw file reader.  Please note that the
#copyright to the proprietary *.ptw file format remains the property of FLIR Inc.

# Contributor(s): ______________________________________.
################################################################
"""
This module provides functions to read the contents of files in the
PTW file format.


Callable functions :

    | readPTWHeader(ptwfilename)
    | showHeader(header)
    | getPTWFrame (s, frameindex)


readPTWHeader(ptwfilename) :
Returns a class object defining all the header information

showHeader(header) :
Returns nothing.  Prints the PTW header content to the screen

getPTWFrame (header, frameindex) :
Return the raw DL levels of the frame defined by frameindex
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__='JJ Calitz'
__all__=['myint','mylong','myfloat','mybyte',  'ReadPTWHeader', 'sLoadCedip',
'ShowHeader', 'GetPTWFrame']

import struct
import numpy as np


# ========================================================================

# --------------------------------------------------------------------
# Miscelaneous functions to read bytes, int, long, float values.
# --------------------------------------------------------------------

def myint(x):
    """Temperative derivative of Planck function in wavenumber domain for radiance emittance.

    Args:
        | wavelength (np.array[N,]):  wavelength vector in  [um]
        | temperature (float):  temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emitance in  q/(K.s.m^2.um)

    Raises:
        | No exception is raised.
    """
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

def mybyte(x):
    # will return an error if x is more than a byte length
    # TODO need to write error catch routine
    ans = ord(x)
    return ans

# ========================================================================

def readPTWHeader(ptwfilename):
    """Read a PTW file header

    Args:
        | filename (string) with full path

    Returns:
        | Header (class) containing all PTW header information

    Raises:
        | No exception is raised.
    """
    # --------------------------------------------------------------------
    # Notes:
    # h_ variables of the header
    # Byte positions from DL002U-D Altair Reference Guide
    # --------------------------------------------------------------------

    class frameinfo:
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

    # Define the variables holding the header values
    headerinfo = '' #the vector holding the file header
    Header = frameinfo()

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
    if Header.h_Signature == 'AI0': #AGEMA
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
    Header.h_CameraName = headerinfo[44:64]
    Header.h_LensName = headerinfo[64:84]
    Header.h_FilterName = headerinfo[84:104]
    Header.h_ApertureName = headerinfo[104:124]
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
    Header.h_CameraSerialNumber = headerinfo[220:231]
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
    Header.h_Comment = headerinfo[563:1563]
    Header.h_CalibrationFileName = headerinfo[1563:1663]
    Header.h_ToolsFileName = headerinfo[1663:1919]
    Header.h_PaletteIndexValid = ord(headerinfo[1919:1920])
    Header.h_PaletteIndexCurrent = myint(headerinfo[1920:1922])
    Header.h_PaletteToggle = ord(headerinfo[1922:1923])
    Header.h_PaletteAGC = ord(headerinfo[1923:1924])
    Header.h_UnitIndexValid = ord(headerinfo[1923:1924])
    Header.h_CurrentUnitIndex = myint(headerinfo[1925:1927])
    Header.h_ZoomPosition = headerinfo[1927:1935] # unknown format POINT
    Header.h_KeyFrameNumber = ord(headerinfo[1935:1936])
    Header.h_KeyFramesInFilm = headerinfo[1936:2056] # set of 30 frames
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
    Header.h_ThermoElasticity = headerinfo[2106:2114] # type double (64 bits)
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
    Header.h_CalibrationFileName = headerinfo[2147:2403]
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

    fid.close()#  %close file

    return Header

#
# ========================================================================
#

def sLoadCedip(s):
    # --------------------------------------------------------------------
    # AIM : Do the hard work of loading the correct frame from the PTW file
    # INPUT : PTWheader
    # RETURN : PTWheader with data added
    # --------------------------------------------------------------------

    # for debugging
    #print ('.....Loading frame', s.m_framepointer , 'from', s.m_filename,'.....')
    #print (s.m_cols,'x', s.m_rows, 'data points')

    fid = open(s.FileName,'rb')
    # skip main header
    fid.seek (s.h_MainHeaderSize,0)  #bof

    # for debugging
    #print ('EndHeader =',fid.tell())

    if(s.h_Lockin): # lockin -> skip first line
        fid.seek ((s.h_framepointer-1) * (s.h_FrameSize + 2*s.h_Cols),1)#, 'cof'
    else:
        fid.seek ((s.h_framepointer-1) * (s.h_FrameSize),1)#, 'cof'

    # for debugging
    #print ('StartFrameHeader =',fid.tell())

    #fid.seek(s.m_FrameHeaderSize,1)#,'cof') #skip frame header
    FrameHeader = fid.read(s.h_FrameHeaderSize)

    # for debugging
    #print ('Start FrameData at',fid.tell())

    s.data = np.eye(s.h_Cols, s.h_Rows)

    #datapoints = s.m_cols * s.m_rows
    for y in range(s.h_Rows):
        for x in range(s.h_Cols):
            s.data[x][y] = myint(fid.read(2))

    # for debugging
    #print ('Data read',len(s.m_data), 'points')
    #print ('End FrameData at',fid.tell())

    # if a special scale is given then transform the data
    if(s.h_EchelleSpecial):
        low = min(s.h_EchelleScaleValue)
        high = max(s.h_EchelleScaleValue)
        s.data = s.data * (high-low)/ 2.0**16 + low
        #clear low high
    if(s.h_Lockin): # lockin -> skip first line
        s.h_cliprect = [0,1,s.h_Cols-1,s.h_Rows]

    s.h_minval = s.data.min()
    s.h_maxval = s.data.max()

    # for debugging
    #print ('DL values min', s.m_minval)
    #print ('DL values max', s.m_maxval)

    fid.close()  #close file
    return s

#
# ========================================================================
#

def getPTWFrame (s, frameindex):
    """Retrieve a single PTW frame

    Args:
        | header (class object)
        | frameindex (integer) The frame to be extracted

    Returns:
        | an array with the frameindex frame DL values

    Raises:
        | No exception is raised.
    """
    # --------------------------------------------------------------------
    # This routine also stores the data array as part of the header
    # This may change - not really needed to have both a return value
    #                   and header stored value for the DL values.
    #                   historical reason due to the way sLoadCedip was written
    # --------------------------------------------------------------------

    # Check if this is  a cedip file

    if s.h_format!='cedip':
        print('ReadJade Error: file format is not supported')
        result = -1
    if (frameindex <= s.h_lastframe):
        if frameindex>0:
            s.h_framepointer = frameindex
            s = sLoadCedip(s)
        else:
            print ('frameindex smaller than 0')
    else:                           # frameindex exceeds no of frames
        print ('ReadJade Error: cannot load frame. Frameindex exceeds sequence length.')

    return s.data
#
# ========================================================================
#
def showHeader(Header):
    """Utility function to display the PTW header information

    Args:
        | header (class object)

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
    if Header.h_Signature == 'CED':
        print ('CEDIP Period', Header.h_CEDIPAquisitionPeriod*100,'Hz')
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

#
# ========================================================================
#
