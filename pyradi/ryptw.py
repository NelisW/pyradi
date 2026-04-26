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

# The author wishes to thank FLIR Advanced Thermal Solutions for the permission to
# publicly release our Python version of the *.ptw file reader. Note that the
# copyright to the proprietary *.ptw file format remains the property of FLIR Inc.

# Contributor(s):  JJ Calitz, MS Willers & CJ Willers.
################################################################
"""
This module provides functionality to read the contents of files in the
PTW file format and convert the raw data to source radiance or source
temperature (provided that the instrument calibration data is available).

Functions are provided to read the binary Agema/Cedip/FLIR Inc PTW format
into data structures for further processing.

The following functions are available to read PTW files:

    | readPTWHeader(ptwfilename)
    | showHeader(header)
    | getPTWFrame(header, frameindex)

readPTWHeader(ptwfilename):
    Returns a PTWFrameInfo object with all header information from the ptw file.

showHeader(header):
    Returns nothing. Prints the PTW header content to the screen.

getPTWFrame(header, frameindex):
    Return the raw DL levels of the frame defined by frameindex.

The authors wish to thank FLIR Advanced Thermal Solutions for the permission
to publicly release our Python version of the ptw file reader.  Please note that the
copyright to the proprietary ptw file format remains the property of FLIR Inc.


This package was partly developed to provide additional material in support of students
and readers of the book Electro-Optical System Analysis and Design: A Radiometry
Perspective, Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""

__version__ = '0.1.0'
__author__ = 'JJ Calitz'
__all__ = [
    'myint',
    'mylong',
    'myfloat',
    'mydouble',
    'mybyte',
    'mypass',
    'myRGB',
    'terminateStrOnZero',
    'PTWFrameInfo',
    'readPTWHeader',
    'GetPTWFrameFromFile',
    'getPTWFrame',
    'getPTWFrames',
    'showHeader',
    'JadeCalibrationData',
]

import os.path
import string
import xml.etree.ElementTree as ET
import struct
from collections import defaultdict

import numpy as np

import pyradi.rylookup as rylookup


################################################################
# Miscellaneous functions to read bytes, int, long, float values.
#
# https://docs.python.org/3/library/struct.html

def myint(x):
    """Decode a 2-byte little-endian signed integer from a bytes slice.

    Args:
        x: bytes of length 2.

    Returns:
        int: signed 16-bit integer value.
    """
    return struct.unpack('h', x)[0]


def mylong(x):
    """Decode a 4-byte little-endian unsigned integer from a bytes slice.

    Args:
        x: bytes of length 4.

    Returns:
        int: unsigned integer value.

    Note:
        Pre-existing bug: the shift for byte 3 is ``x[3] << 32`` instead of
        the correct ``x[3] << 24``.  When byte 3 is non-zero the result is
        larger than expected by a factor of 256.  In practice all PTW integer
        fields fit within 24 bits so byte 3 is always 0 and the result is
        correct.
    """
    return x[0] + (x[1] << 8) + (x[2] << 16) + (x[3] << 32)


def myfloat(x):
    """Decode a 4-byte IEEE 754 single-precision float from a bytes slice.

    Args:
        x: bytes of length 4.

    Returns:
        float: single-precision floating-point value.
    """
    return struct.unpack('f', x)[0]


def mydouble(x):
    """Decode an 8-byte IEEE 754 double-precision float from a bytes slice.

    Args:
        x: bytes of length 8.

    Returns:
        float: double-precision floating-point value.
    """
    return struct.unpack('d', x)[0]


def mybyte(x):
    """Decode a single unsigned byte from a bytes slice of length 1.

    Args:
        x: bytes of length 1.

    Returns:
        int: unsigned byte value (0–255).

    Note:
        Will raise ``TypeError`` if *x* is longer than one byte.
    """
    return ord(x)


def mypass(x):
    """Identity function — return *x* unchanged.

    Args:
        x: any value.

    Returns:
        The input *x* unchanged.
    """
    return x


def myRGB(x):
    """Decode three bytes as an HTML-style hex colour string.

    Args:
        x: bytes of length 3, ordered R, G, B.

    Returns:
        str: colour string of the form ``'#rrggbb'``.
    """
    R = x[0]
    G = x[1]
    B = x[2]
    return f'#{R:02x}{G:02x}{B:02x}'


def terminateStrOnZero(str):
    """Iterate through *str* and terminate on the first null byte.

    Args:
        str: bytes object to scan.

    Returns:
        bytes: prefix of *str* up to (but not including) the first null byte.

    Note:
        Pre-existing Python-3 bug: ``str[idx]`` on a bytes object returns an
        ``int``, so the comparison ``str[idx] != '\\00'`` (int vs str) is
        always ``True`` and the loop never exits early.  The function therefore
        always returns the full input slice unchanged.  Callers in
        ``readPTWHeader`` compensate by using ``.rstrip('\\x00')`` on the
        decoded string, which achieves the intended trimming for null-padded
        fields.
    """
    idx = 0
    while idx < len(str) and str[idx] != '\00':
        idx += 1
    return str[:idx]


################################################################
class PTWFrameInfo:
    """Container for PTW file header and frame information.

    Attributes are populated by :func:`readPTWHeader` (file-level fields) and
    :func:`GetPTWFrameFromFile` (per-frame fields).  Attribute names follow the
    original DL002U-D Altair Reference Guide conventions.
    """

    def __init__(self):

        self.FileName = ''
        self.h_Signature = ''   # [0:5]
        self.h_format = 'unknown'
        self.h_unit = ''
        self.h_Version = ''     # [5:10]
        self.h_EofAsciiCode = 0     # [10]
        self.h_MainHeaderSize = 0   # [11:15]
        self.h_FrameHeaderSize = 0  # [15:19]
        self.h_SizeOfOneFrameAndHeader = 0  # [19:23]
        self.h_SizeOfOneFrame = 0   # [23:27]
        self.h_NumberOfFieldInFile = 0  # [27:31]
        self.h_CurrentFieldNumber = 0   # [31:35]

        # h_FileSaveDate [35:39] decoded into separate fields below
        self.h_FileSaveYear = 0
        self.h_FileSaveMonth = 0
        self.h_FileSaveDay = 0

        # h_FileSaveTime [39:43] decoded into separate fields below
        self.h_FileSaveHour = 0
        self.h_FileSaveMinute = 0
        self.h_FileSaveSecond = 0
        self.h_FileSaveCent = 0

        self.h_Millieme = 0             # [43:44]
        self.h_CameraName = ''          # [44:64]
        self.h_LensName = ''            # [64:84]
        self.h_FilterName = ''          # [84:104]
        self.h_ApertureName = ''        # [104:124]
        self.h_IRUSBilletSpeed = 0      # [124:128] IRUS
        self.h_IRUSBilletDiameter = 0   # [128:132] IRUS
        self.h_IRUSBilletShape = 0      # [132:134] IRUS
        self.h_Emissivity = 0           # [141:145]
        self.h_Ambiant = 0              # [145:149]
        self.h_Distance = 0            # [149:153]
        self.h_IRUSInductorCoil = 0     # [153:154] IRUS
        self.h_IRUSInductorPower = 0    # [154:158] IRUS
        self.h_IRUSInductorVoltage = 0  # [158:160] IRUS
        self.h_IRUSInductorFrequency = 0  # [160:164] IRUS
        self.h_IRUSSynchronization = 0  # [169:170] IRUS
        self.h_AtmTransmission = 0      # [170:174]
        self.h_ExtinctionCoeficient = 0 # [174:178]
        self.h_Object = 0               # [178:180]
        self.h_Optic = 0                # [180:182]
        self.h_Atmo = 0                 # [182:184]
        self.h_AtmosphereTemp = 0       # [184:188]
        self.h_CutOnWavelength = 0      # [188:192]
        self.h_CutOffWavelength = 0     # [192:196]
        self.h_PixelSize = 0            # [196:200]
        self.h_PixelPitch = 0           # [200:204]
        self.h_DetectorApperture = 0    # [204:208]
        self.h_OpticsFocalLength = 0    # [208:212]
        self.h_HousingTemperature1 = 0  # [212:216]
        self.h_HousingTemperature2 = 0  # [216:220]
        self.h_CameraSerialNumber = ''  # [220:231]
        self.h_MinimumLevelThreshold = 0  # [245:247]
        self.h_MaximumLevelThreshold = 0  # [247:249]
        self.h_EchelleSpecial = 0       # [277:279]
        self.h_EchelleUnit = 0          # [279:289]
        self.h_EchelleValue = 0         # [289:357] 16 float values
        self.h_Units = ''
        self.h_Lockin = 0
        self.h_LockinGain = 0           # [357:361]
        self.h_LockinOffset = 0         # [361:365]
        self.h_HorizontalZoom = 0       # [365:369]
        self.h_VerticalZoom = 0         # [369:373]
        self.h_PixelsPerLine = 0        # [377:379]
        self.h_LinesPerField = 0        # [379:381]
        self.h_Rows = 0
        self.h_Cols = 0

        self.h_framepointer = 1
        self.h_firstframe = 1
        self.h_cliprect = [0, 0, 1, 1]
        self.h_lastframe = 0
        self.h_FrameSize = 0

        self.h_ADDynamic = 0                    # [381:383]
        self.h_SATIRTemporalFrameDepth = 0      # [383:385] SATIR
        self.h_SATIRLocationLongitude = 0       # [385:389] SATIR
        self.h_SATIRLocationLatitude = 0        # [389:393] SATIR South is negative
        self.h_SATIRLocationAltitude = 0        # [393:397] SATIR
        self.h_ExternalSynch = 0                # [397] 1=External 0=Internal
        self.h_CEDIPAquisitionPeriod = 0        # [403:407] CEDIP seconds
        self.h_CEDIPIntegrationTime = 0         # [407:411] CEDIP seconds
        self.h_WOLFSubwindowCapability = 0      # [411:413] WOLF
        self.h_ORIONIntegrationTime = 0         # [413:437] ORION (6 values)
        self.h_ORIONFilterNames = ''            # [437:557] ORION 6 fields of 20 chars
        self.h_NucTable = 0                     # [557:559]
        self.h_Reserve6 = ''                    # [559:563]
        self.h_Comment = ''                     # [563:1563]
        self.h_CalibrationFileName = ''         # [1563:1663] and [2147:2403]
        self.h_ToolsFileName = ''               # [1663:1919]
        self.h_PaletteIndexValid = 0            # [1919:1920]
        self.h_PaletteIndexCurrent = 0          # [1920:1922]
        self.h_PaletteToggle = 0                # [1922:1923]
        self.h_PaletteAGC = 0                   # [1923:1924]
        self.h_UnitIndexValid = 0               # [1924:1925]
        self.h_CurrentUnitIndex = 0             # [1925:1927]
        self.h_ZoomPosition = 0                 # [1927:1935] unknown format POINT
        self.h_KeyFrameNumber = 0               # [1935:1936]
        self.h_KeyFramesInFilm = 0              # [1936:2056] set of 30 frames
        self.h_PlayerLocked = 0                 # [2056:2057]
        self.h_FrameSelectionValid = 0          # [2057:2058]
        self.h_FrameofROIStart = 0              # [2058:2062]
        self.h_FrameofROIEnd = 0                # [2062:2066]
        self.h_PlayerLockedROI = 0              # [2066:2067]
        self.h_PlayerInfinitLoop = 0            # [2067:2068]
        self.h_PlayerInitFrame = 0              # [2068:2072]

        self.h_Isoterm0Active = 0   # [2072:2073]
        self.h_Isoterm0DLMin = 0    # [2073:2075]
        self.h_Isoterm0DLMax = 0    # [2075:2077]
        self.h_Isoterm0Color = 0    # [2077:2081]

        self.h_Isoterm1Active = 0   # [2081:2082]
        self.h_Isoterm1DLMin = 0    # [2082:2084]
        self.h_Isoterm1DLMax = 0    # [2084:2086]
        self.h_Isoterm1Color = 0    # [2086:2090]

        self.h_Isoterm2Active = 0   # [2090:2091]
        self.h_Isoterm2DLMin = 0    # [2091:2093]
        self.h_Isoterm2DLMax = 0    # [2093:2095]
        self.h_Isoterm2Color = 0    # [2095:2099]

        self.h_ZeroActive = 0               # [2099:2100]
        self.h_ZeroDL = 0                   # [2100:2102]
        self.h_PaletteWidth = 0             # [2102:2104]
        self.h_PaletteFull = 0              # [2104:2105]
        self.h_PTRFrameBufferType = 0       # [2105:2106] 0=word 1=double
        self.h_ThermoElasticity = 0         # [2106:2114] type double (64 bits)
        self.h_DemodulationFrequency = 0    # [2114:2118]
        self.h_CoordinatesType = 0          # [2118:2122]
        self.h_CoordinatesXorigin = 0       # [2122:2126]
        self.h_CoordinatesYorigin = 0       # [2126:2130]
        self.h_CoordinatesShowOrigin = 0    # [2130:2131]
        self.h_AxeColor = 0                 # [2131:2135]
        self.h_AxeSize = 0                  # [2135:2139]
        self.h_AxeValid = 0                 # [2139:2140]
        self.h_DistanceOffset = 0           # [2140:2144]
        self.h_HistoEqualizationEnabled = 0 # [2144:2145]
        self.h_HistoEqualizationPercent = 0 # [2145:2147]
        self.h_PTRTopFrameValid = 0         # [2403:2404]
        self.h_SubSampling = 0              # [2404:2408]
        self.h_CameraHFlip = 0             # [2408:2409]
        self.h_CameraHVFlip = 0            # [2409:2410]
        self.h_BBTemp = 0                   # [2410:2414]
        self.h_CaptureWheelIndex = 0        # [2414:2415]
        self.h_CaptureFocalIndex = 0        # [2415:2416]
        self.h_Reserved7 = ''              # [2416:3028]
        self.h_Reserved8 = ''              # [3028:3076]
        self.h_Framatone = 0               # [3076:3077]

        # Container for a single frame
        self.data = []
        self.minval = 0
        self.maxval = 0

        # Frame time
        self.h_frameMinute = 0     # FrameHeader[80:81]
        self.h_frameHour = 0       # FrameHeader[81:82]
        self.h_frameSecond = 0     # h_second + (h_thousands + h_hundred) / 1000.0

        # Detector / FPA temperature
        self.h_detectorTemp = 0.0  # FrameHeader[228:232]
        self.h_sensorTemp4 = 0.0   # FrameHeader[232:236]

# End of header definition

################################################################

def readPTWHeader(ptwfilename):
    """Read the PTW file header and return a populated PTWFrameInfo object.

    Args:
        ptwfilename (str): Full path to the ptw file.

    Returns:
        PTWFrameInfo: Object containing all PTW header information.

    Raises:
        No exception is raised.

    Reference:
        Header field names and byte positions are from DL002U-D Altair
        Reference Guide.
    """

    headerinfo = ''
    Header = PTWFrameInfo()

    # Read first 16 bytes to determine the main header size
    Header.FileName = ptwfilename
    with open(ptwfilename, 'rb') as fid:
        headerinfo = fid.read(16)
    MainHeaderSize = mylong(headerinfo[11:15])

    # Re-open and read the full main header
    fid = open(ptwfilename, 'rb')
    headerinfo = fid.read(MainHeaderSize)

    Header.h_Signature = headerinfo[0:3].decode('utf-8')
    if Header.h_Signature == 'AIO':     # AGEMA
        Header.h_format = 'agema'
    elif Header.h_Signature == 'CED':
        Header.h_format = 'cedip'
        Header.h_unit = 'dl'

    Header.h_Version = headerinfo[5:10].decode('utf-8')
    if not Header.h_Version[-1] in string.printable:
        Header.h_Version = Header.h_Version[:-1]
    Header.h_EofAsciiCode = mybyte(headerinfo[10:11])
    Header.h_MainHeaderSize = mylong(headerinfo[11:15])
    Header.h_FrameHeaderSize = mylong(headerinfo[15:19])
    Header.h_SizeOfOneFrameAndHeader = mylong(headerinfo[19:23])
    Header.h_SizeOfOneFrame = mylong(headerinfo[23:27])
    Header.h_NumberOfFieldInFile = mylong(headerinfo[27:31])
    Header.h_CurrentFieldNumber = mylong(headerinfo[31:35])

    # h_FileSaveDate [35:39]
    Header.h_FileSaveYear = myint(headerinfo[35:37])
    Header.h_FileSaveDay = ord(headerinfo[37:38])
    Header.h_FileSaveMonth = ord(headerinfo[38:39])

    # h_FileSaveTime [39:43]
    Header.h_FileSaveMinute = ord(headerinfo[39:40])
    Header.h_FileSaveHour = ord(headerinfo[40:41])
    Header.h_FileSaveCent = ord(headerinfo[41:42])
    Header.h_FileSaveSecond = ord(headerinfo[42:43])

    Header.h_Millieme = ord(headerinfo[43:44])

    stripchar = terminateStrOnZero(headerinfo[44:64]).decode('utf-8')[-1]
    Header.h_CameraName = terminateStrOnZero(headerinfo[44:64]).decode('utf-8').rstrip(stripchar)
    Header.h_LensName = terminateStrOnZero(headerinfo[64:84]).decode('utf-8').rstrip(stripchar)
    Header.h_FilterName = terminateStrOnZero(headerinfo[84:104]).decode('utf-8').rstrip(stripchar)
    Header.h_ApertureName = terminateStrOnZero(headerinfo[104:124]).decode('utf-8').rstrip(stripchar)

    Header.h_IRUSBilletSpeed = myfloat(headerinfo[124:128])        # IRUS
    Header.h_IRUSBilletDiameter = myfloat(headerinfo[128:132])     # IRUS
    Header.h_IRUSBilletShape = myint(headerinfo[132:134])          # IRUS
    Header.h_Reserved134 = headerinfo[134:141]
    Header.h_Emissivity = myfloat(headerinfo[141:145])
    Header.h_Ambiant = myfloat(headerinfo[145:149])
    Header.h_Distance = myfloat(headerinfo[149:153])
    Header.h_IRUSInductorCoil = ord(headerinfo[153:154])           # IRUS
    Header.h_IRUSInductorPower = mylong(headerinfo[154:158])       # IRUS
    Header.h_IRUSInductorVoltage = myint(headerinfo[158:160])      # IRUS
    Header.h_IRUSInductorFrequency = mylong(headerinfo[160:164])   # IRUS
    Header.h_Reserved164 = headerinfo[164:169]
    Header.h_IRUSSynchronization = ord(headerinfo[169:170])        # IRUS
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
    stripchar = terminateStrOnZero(headerinfo[220:231]).decode('utf-8')[-1]
    Header.h_CameraSerialNumber = terminateStrOnZero(headerinfo[220:231]).decode('utf-8').rstrip(stripchar)

    Header.h_Reserved231 = headerinfo[231:239]
    Header.h_DetectorCode1 = myint(headerinfo[239:241])
    Header.h_DetectorCode2 = myint(headerinfo[241:243])
    Header.h_DetectorGain = myint(headerinfo[245:247])

    Header.h_MinimumLevelThreshold = myint(headerinfo[245:247])
    Header.h_MaximumLevelThreshold = myint(headerinfo[247:249])
    Header.h_EchelleSpecial = myint(headerinfo[277:279])
    Header.h_EchelleUnit = headerinfo[279:289]
    Header.h_EchelleValue = headerinfo[289:357]     # 16 float values

    if Header.h_EchelleSpecial == 0:
        Header.h_Units = 'dl'   # [dl T rad]
    else:
        Header.h_Units = Header.h_EchelleUnit   # [dl T rad]

    Header.h_LockinGain = myfloat(headerinfo[357:361])
    Header.h_LockinOffset = myfloat(headerinfo[361:365])
    Header.h_HorizontalZoom = myfloat(headerinfo[365:369])
    Header.h_VerticalZoom = myfloat(headerinfo[369:373])

    Header.h_PixelsPerLine = myint(headerinfo[377:379])
    Header.h_LinesPerField = myint(headerinfo[379:381])
    if Header.h_LinesPerField == 0:
        Header.h_LinesPerField = 128
    if Header.h_PixelsPerLine == 0:
        Header.h_PixelsPerLine = 128

    Header.h_Rows = Header.h_LinesPerField
    Header.h_Cols = Header.h_PixelsPerLine

    Header.h_cliprect = [0, 0, Header.h_Cols - 1, Header.h_Rows - 1]
    Header.h_lastframe = Header.h_NumberOfFieldInFile
    Header.h_FrameSize = Header.h_FrameHeaderSize + Header.h_Cols * Header.h_Rows * 2

    Header.h_ADDynamic = myint(headerinfo[381:383])
    Header.h_SATIRTemporalFrameDepth = myint(headerinfo[383:385])       # SATIR
    Header.h_SATIRLocationLongitude = myfloat(headerinfo[385:389])      # SATIR
    Header.h_SATIRLocationLatitude = myfloat(headerinfo[389:393])       # SATIR South is negative
    Header.h_SATIRLocationAltitude = myfloat(headerinfo[393:397])       # SATIR
    Header.h_ExternalSynch = headerinfo[397]                            # 1=External 0=Internal

    Header.h_CEDIPAquisitionPeriod = myfloat(headerinfo[403:407])       # CEDIP seconds
    Header.h_CEDIPIntegrationTime = myfloat(headerinfo[407:411])        # CEDIP seconds
    Header.h_WOLFSubwindowCapability = myint(headerinfo[411:413])       # WOLF
    Header.h_ORIONIntegrationTime = headerinfo[413:437]                 # ORION (6 values)
    Header.h_ORIONFilterNames = headerinfo[437:557]                     # ORION 6 fields of 20 chars
    Header.h_NucTable = myint(headerinfo[557:559])
    Header.h_Reserve6 = headerinfo[559:563]
    stripchar = terminateStrOnZero(headerinfo[563:1563]).decode('utf-8')[-1]
    Header.h_Comment = terminateStrOnZero(headerinfo[563:1563]).decode('utf-8').rstrip(stripchar)
    stripchar = terminateStrOnZero(headerinfo[1563:1663]).decode('utf-8')[-1]
    Header.h_CalibrationFileName = terminateStrOnZero(headerinfo[1563:1663]).decode('utf-8').rstrip(stripchar)
    stripchar = terminateStrOnZero(headerinfo[1663:1919]).decode('utf-8')[-1]
    Header.h_ToolsFileName = terminateStrOnZero(headerinfo[1663:1919]).decode('utf-8').rstrip(stripchar)

    Header.h_PaletteIndexValid = ord(headerinfo[1919:1920])
    Header.h_PaletteIndexCurrent = myint(headerinfo[1920:1922])
    Header.h_PaletteToggle = ord(headerinfo[1922:1923])
    Header.h_PaletteAGC = ord(headerinfo[1923:1924])
    Header.h_UnitIndexValid = ord(headerinfo[1924:1925])
    Header.h_CurrentUnitIndex = myint(headerinfo[1925:1927])
    stripchar = terminateStrOnZero(headerinfo[1927:1935]).decode('utf-8')[-1]
    # unknown format POINT
    Header.h_ZoomPosition = terminateStrOnZero(headerinfo[1927:1935]).decode('utf-8').rstrip(stripchar)
    # set of 30 frames
    Header.h_KeyFramesInFilm = terminateStrOnZero(headerinfo[1936:2056]).decode('utf-8').rstrip(stripchar)
    Header.h_KeyFrameNumber = ord(headerinfo[1935:1936])
    Header.h_PlayerLocked = ord(headerinfo[2056:2057])
    Header.h_FrameSelectionValid = ord(headerinfo[2057:2058])
    Header.h_FrameofROIStart = mylong(headerinfo[2058:2062])
    Header.h_FrameofROIEnd = mylong(headerinfo[2062:2066])
    Header.h_PlayerLockedROI = ord(headerinfo[2066:2067])
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
    Header.h_PTRFrameBufferType = ord(headerinfo[2105:2106])    # 0=word 1=double
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
    stripchar = terminateStrOnZero(headerinfo[2147:2403]).decode('utf-8')[-1]
    Header.h_CalibrationFileName = terminateStrOnZero(headerinfo[2147:2403]).decode('utf-8').rstrip(stripchar)

    Header.h_PTRTopFrameValid = ord(headerinfo[2403:2404])
    Header.h_SubSampling = mylong(headerinfo[2404:2408])
    Header.h_CameraHFlip = ord(headerinfo[2408:2409])
    Header.h_CameraHVFlip = ord(headerinfo[2409:2410])
    Header.h_BBTemp = myfloat(headerinfo[2410:2414])
    Header.h_CaptureWheelIndex = ord(headerinfo[2414:2415])
    Header.h_CaptureFocalIndex = ord(headerinfo[2415:2416])
    Header.h_Reserved7 = headerinfo[2416:3028]
    Header.h_Reserved8 = headerinfo[3028:3076]
    Header.h_Framatone = ord(headerinfo[3076:3077])

    # Read the first video frame info (not data) to detect lockin mode
    fid.seek(Header.h_MainHeaderSize, 0)    # skip main header
    fid.seek(Header.h_FrameHeaderSize, 1)   # skip frame header
    firstline = fid.read(Header.h_Cols)     # read one line

    if firstline[1:4] == [1220, 3907, 1204, 2382]:
        Header.h_Lockin = 1
        Header.h_Rows = Header.h_Rows - 1
        print('* LOCKIN')
    else:
        Header.h_Lockin = 0

    fid.close()

    return Header


################################################################
def GetPTWFrameFromFile(header):
    """Load the frame specified by header.h_framepointer from the PTW file.

    Args:
        header (PTWFrameInfo): Header object with h_framepointer set to the
            desired 1-based frame index.

    Returns:
        PTWFrameInfo: The same *header* object, updated with:
            ``data`` (np.ndarray of shape ``(cols, rows)``),
            ``h_frameMinute``, ``h_frameHour``, ``h_frameSecond``,
            ``h_detectorTemp``, ``h_sensorTemp4``,
            ``h_minval``, ``h_maxval``.

    Raises:
        No exception is raised.
    """

    fid = open(header.FileName, 'rb')
    fid.seek(header.h_MainHeaderSize, 0)    # skip main header

    if header.h_Lockin:     # lockin -> skip first line
        fid.seek((header.h_framepointer - 1) * (header.h_FrameSize + 2 * header.h_Cols), 1)
    else:
        fid.seek((header.h_framepointer - 1) * header.h_FrameSize, 1)

    FrameHeader = fid.read(header.h_FrameHeaderSize)

    # Frame time
    header.h_frameMinute = ord(FrameHeader[80:81])
    header.h_frameHour = ord(FrameHeader[81:82])
    h_hundred = ord(FrameHeader[82:83]) * 10
    h_second = ord(FrameHeader[83:84])
    h_thousands = ord(FrameHeader[160:161])
    header.h_frameSecond = h_second + (h_hundred + h_thousands) / 1000.0

    # Detector / FPA temperature
    header.h_detectorTemp = myfloat(FrameHeader[228:232])
    header.h_sensorTemp4 = myfloat(FrameHeader[232:236])
    if header.h_sensorTemp4 is None:
        header.h_sensorTemp4 = 0.0
    if header.h_detectorTemp is None:
        header.h_detectorTemp = 0.0

    header.data = np.eye(header.h_Cols, header.h_Rows)

    for y in range(header.h_Rows):
        for x in range(header.h_Cols):
            header.data[x][y] = myint(fid.read(2))

    # Apply special scale if present
    if header.h_EchelleSpecial:
        low = min(header.h_EchelleScaleValue)
        high = max(header.h_EchelleScaleValue)
        header.data = header.data * (high - low) / 2.0 ** 16 + low

    if header.h_Lockin:     # lockin -> skip first line
        header.h_cliprect = [0, 1, header.h_Cols - 1, header.h_Rows]

    header.h_minval = header.data.min()
    header.h_maxval = header.data.max()

    fid.close()
    return header


################################################################
def getPTWFrame(header, frameindex):
    """Retrieve a single PTW frame given the header and frame index.

    The frame data array is also stored inside *header* as ``header.data``.
    The header is returned explicitly to make it clear that its contents have
    been updated (frame time, detector temperature).

    Args:
        header (PTWFrameInfo): PTW file header object.
        frameindex (int): 1-based index of the frame to extract.

    Returns:
        tuple: ``(data, header)`` where *data* is ``np.ndarray`` of shape
        ``(rows, cols)`` containing raw DL values, and *header* is the
        updated PTWFrameInfo object.  On error a bare ``np.ndarray([0])``
        is returned instead of the tuple.

    Raises:
        No exception is raised.
    """

    errorresult = np.asarray([0])
    if header.h_format != 'cedip':
        print('ReadJade Error: file format is not supported')
        return errorresult
    if frameindex <= header.h_lastframe:
        if frameindex > 0:
            header.h_framepointer = frameindex
            header = GetPTWFrameFromFile(header)
        else:
            print('frameindex smaller than 0')
            return errorresult
    else:
        print('ReadJade Error: cannot load frame. Frameindex exceeds sequence length.')
        return errorresult

    return header.data.conj().transpose(), header


################################################################
def getPTWFrames(header, loadFrames=[]):
    """Retrieve multiple PTW frames specified by a list of frame indices.

    Args:
        header (PTWFrameInfo): PTW file header object.
        loadFrames (list[int]): 1-based frame indices to extract.

    Returns:
        tuple: ``(data, fheaders)`` where *data* is ``np.ndarray`` of shape
        ``(len(loadFrames), rows, cols)`` and *fheaders* is a list of
        PTWFrameInfo objects (one per frame, carrying frame time and FPA
        temperature).  On error returns ``(np.asarray([0]), None)``.

    Raises:
        No exception is raised.
    """

    fheaders = []

    errorresult = np.asarray([0])
    if header.h_format != 'cedip':
        print('getPTWFrames Error: file format is not supported')
        return errorresult, None

    npFrames = np.asarray(loadFrames)
    if np.any(npFrames < 1) or np.any(npFrames > header.h_lastframe):
        print('getPTWFrames Error: at least one requested frame not in file')
        print(f'legal frames for this file are: {1} to {header.h_lastframe}')
        return errorresult, None

    data, headerx = getPTWFrame(header, loadFrames[0])
    fheaders.append(headerx)

    for frame in loadFrames[1:]:
        datax, headerx = getPTWFrame(header, frame)
        data = np.concatenate((data, datax))
        fheaders.append(headerx)

    rows = header.h_Rows
    cols = header.h_Cols

    return data.reshape(len(loadFrames), rows, cols), fheaders


################################################################
def showHeader(Header):
    """Print all PTW header fields to stdout.

    Args:
        Header (PTWFrameInfo): PTW file header object.

    Returns:
        None
    """

    print(f'{Header.h_Signature} version {Header.h_Version}')
    print(f'Main Header Size {Header.h_MainHeaderSize}')
    print(f'Frame Header Size {Header.h_FrameHeaderSize}')
    print(f'Frame + Frame Header Size {Header.h_SizeOfOneFrameAndHeader}')
    print(f'Frame Size {Header.h_SizeOfOneFrame}')
    print(f'Number of Frames {Header.h_NumberOfFieldInFile}')

    print(f'Year {Header.h_FileSaveYear} Month {Header.h_FileSaveMonth} Day {Header.h_FileSaveDay}')
    print(f'( {str(Header.h_FileSaveYear).zfill(2)} / '
          f'{str(Header.h_FileSaveMonth).zfill(2)} / '
          f'{str(Header.h_FileSaveDay).zfill(2)} )')
    print(f'Hour {Header.h_FileSaveHour} Minute {Header.h_FileSaveMinute} Second {Header.h_FileSaveSecond}')
    print(f'( {str(Header.h_FileSaveHour).zfill(2)} : '
          f'{str(Header.h_FileSaveMinute).zfill(2)} : '
          f'{str(Header.h_FileSaveSecond).zfill(2)} )')

    print(f'Camera Name {Header.h_CameraName}')
    print(f'Detector Code1 {Header.h_DetectorCode1}')
    print(f'Detector Code2 {Header.h_DetectorCode2}')
    print(f'Detector Gain {Header.h_DetectorGain}')

    print(f'Lens {Header.h_LensName}')
    print(f'Filter {Header.h_FilterName}')
    print(f'Aperture Name {Header.h_ApertureName}')
    if Header.h_Signature == 'IRUS':
        print(f'{Header.h_IRUSBilletSpeed}')
        print(f'{Header.h_IRUSBilletDiameter}')
        print(f'{Header.h_IRUSBilletShape}')
    print(f'Emissivity {Header.h_Emissivity:.6f}')
    print(f'Ambient Temperature {Header.h_Ambiant:.6f} (K)')
    print(f'Ambient Temperature {Header.h_Ambiant - 273.15:.6f} (degC)')
    print(f'Distance to target {Header.h_Distance}')
    if Header.h_Signature == 'IRUS':
        print(f'{Header.h_IRUSInductorCoil}')
        print(f'{Header.h_IRUSInductorPower}')
        print(f'{Header.h_IRUSInductorVoltage}')
        print(f'{Header.h_IRUSInductorFrequency}')
        print(f'{Header.h_IRUSSynchronization}')
    print(f'Atm Transmission {Header.h_AtmTransmission}')
    print(f'Ext Coef {Header.h_ExtinctionCoeficient}')
    print(f'Target {Header.h_Object}')
    print(f'Optic {Header.h_Optic}')
    print(f'Atmo {Header.h_Atmo}')
    print(f'Atm Temp {Header.h_AtmosphereTemp:.6f}')
    print(f'Cut on Wavelength {Header.h_CutOnWavelength:.6f}')
    print(f'Cut off Wavelength {Header.h_CutOffWavelength:.6f}')
    print(f'PixelSize {Header.h_PixelSize}')
    print(f'PixelPitch {Header.h_PixelPitch}')
    print(f'Detector Apperture {Header.h_DetectorApperture}')
    print(f'Optic Focal Length {Header.h_OpticsFocalLength}')
    print(f'Housing Temp1 {Header.h_HousingTemperature1:.6f} (K)')
    print(f'Housing Temp2 {Header.h_HousingTemperature2:.6f} (K)')

    print(f'Sensor Temp4 {Header.h_sensorTemp4:.6f} (K)')
    print(f'Detector/FPA Temp {Header.h_detectorTemp:.6f} (K)')

    print(f'Camera Serial Number {Header.h_CameraSerialNumber}')
    print(f'Min Threshold {Header.h_MinimumLevelThreshold}')
    print(f'Max Threshold {Header.h_MaximumLevelThreshold}')
    print(f'Gain {Header.h_LockinGain}')
    print(f'Offset {Header.h_LockinOffset}')
    print(f'HZoom {Header.h_HorizontalZoom}')
    print(f'VZoom {Header.h_VerticalZoom}')
    print(f'Field {Header.h_PixelsPerLine}')
    print(f'AD converter {Header.h_ADDynamic} bit')

    if Header.h_Signature == 'SATIR':
        print(f'{Header.h_SATIRTemporalFrameDepth}')
        print(f'{Header.h_SATIRLocationLongitude}')
        print(f'{Header.h_SATIRLocationLatitude}')
        print(f'{Header.h_SATIRLocationAltitude}')
    if Header.h_ExternalSynch:
        print('Ext Sync ON')
    else:
        print('Ext Sync OFF')
    print(f'Header.h_Signature = {Header.h_Signature}')
    if Header.h_Signature == 'CED':
        print(f'CEDIP Period {1.0 / Header.h_CEDIPAquisitionPeriod:.6f} Hz')
        print(f'CEDIP Integration {Header.h_CEDIPIntegrationTime * 1000:.6f} msec')
    if Header.h_Signature == 'WOLF':
        print(f'{Header.h_WOLFSubwindowCapability}')
    if Header.h_Signature == 'ORI':
        print(f'{Header.h_ORIONIntegrationTime}')
        print(f'{Header.h_ORIONFilterNames}')
    print(f'NUC {Header.h_NucTable}')
    print(f'Comment {Header.h_Comment}')

    print(f'Calibration File Name {Header.h_CalibrationFileName}')

    print(f'Tools File Name {Header.h_ToolsFileName}')

    print(f'Palette Index {Header.h_PaletteIndexValid}')
    print(f'Palette Current {Header.h_PaletteIndexCurrent}')
    print(f'Palette Toggle {Header.h_PaletteToggle}')
    print(f'Palette AGC {Header.h_PaletteAGC}')
    print(f'Unit Index {Header.h_UnitIndexValid}')
    print(f'Current Unit Index {Header.h_CurrentUnitIndex}')
    print(f'Zoom Pos {Header.h_ZoomPosition}')
    print(f'Key Framenum {Header.h_KeyFrameNumber}')
    print(f'Num Keyframes {Header.h_KeyFramesInFilm}')
    print(f'Player lock {Header.h_PlayerLocked}')
    print(f'Frame Select {Header.h_FrameSelectionValid}')
    print(f'ROI Start {Header.h_FrameofROIStart}')
    print(f'ROI Stop {Header.h_FrameofROIEnd}')
    print(f'Player inf loop {Header.h_PlayerInfinitLoop}')
    print(f'Player Init Frame {Header.h_PlayerInitFrame}')

    print(f'Isoterm0 {Header.h_Isoterm0Active}')
    print(f'Isoterm0 DL Min {Header.h_Isoterm0DLMin}')
    print(f'Isoterm0 DL Max {Header.h_Isoterm0DLMax}')
    print(f'Isoterm0 Color RGB {Header.h_Isoterm0Color}')

    print(f'Isoterm1 {Header.h_Isoterm1Active}')
    print(f'Isoterm1 DL Min {Header.h_Isoterm1DLMin}')
    print(f'Isoterm1 DL Max {Header.h_Isoterm1DLMax}')
    print(f'Isoterm1 Color RGB {Header.h_Isoterm1Color}')

    print(f'Isoterm2 {Header.h_Isoterm2Active}')
    print(f'Isoterm2 DL Min {Header.h_Isoterm2DLMin}')
    print(f'Isoterm2 DL Max {Header.h_Isoterm2DLMax}')
    print(f'Isoterm2 Color RGB {Header.h_Isoterm2Color}')

    print(f'Zero {Header.h_ZeroActive}')
    print(f'Zero DL {Header.h_ZeroDL}')
    print(f'Palette Width {Header.h_PaletteWidth}')
    print(f'PaletteF Full {Header.h_PaletteFull}')
    print(f'PTR Frame Buffer type {Header.h_PTRFrameBufferType}')
    print(f'Thermoelasticity {Header.h_ThermoElasticity}')
    print(f'Demodulation {Header.h_DemodulationFrequency}')
    print(f'Coordinate Type {Header.h_CoordinatesType}')
    print(f'X Origin {Header.h_CoordinatesXorigin}')
    print(f'Y Origin {Header.h_CoordinatesYorigin}')
    print(f'Coord Show Orig {Header.h_CoordinatesShowOrigin}')
    print(f'Axe Colour RGB {Header.h_AxeColor}')
    print(f'Axe Size {Header.h_AxeSize}')
    print(f'Axe Valid {Header.h_AxeValid}')
    print(f'Distance offset {Header.h_DistanceOffset}')
    print(f'Histogram {Header.h_HistoEqualizationEnabled}')
    print(f'Histogram % {Header.h_HistoEqualizationPercent}')

    print(f'Calibration File Name {Header.h_CalibrationFileName}')

    print(f'PTRFrame Valid {Header.h_PTRTopFrameValid}')
    print(f'Subsampling {Header.h_SubSampling}')
    print(f'Camera flip H {Header.h_CameraHFlip}')
    print(f'Camera flip V {Header.h_CameraHVFlip}')
    print(f'BB Temp {Header.h_BBTemp}')
    print(f'Capture Wheel Index {Header.h_CaptureWheelIndex}')
    print(f'Capture Wheel Focal Index {Header.h_CaptureFocalIndex}')


################################################################################
class JadeCalibrationData:
    """Container and loader for Jade camera calibration data.

    Reads an XML calibration file and associated spectral data files, then
    builds a :class:`pyradi.rylookup.RadLookup` table for DL ↔ temperature /
    radiance conversion.

    Args:
        filename (str): Full path to the XML calibration file.
        datafileroot (str): Root directory for spectral data files referenced
            inside the XML.
    """

    def __init__(self, filename, datafileroot):

        self.dicCaldata = defaultdict(float)
        self.dicFloor = defaultdict(str)
        self.dicPower = defaultdict(str)

        self.pathtoXML = os.path.dirname(filename)
        self.datafileroot = datafileroot

        self.wl = None
        ftree = ET.parse(filename)
        froot = ftree.getroot()
        self.name = froot.find('.').attrib['Name']
        self.id = froot.find('.').attrib['ID']
        self.version = froot.find('.').attrib['Version']
        self.summary = froot.find('.').attrib['Summary']
        self.sensorResponseFilename = '/'.join(
            [self.datafileroot, froot.find('.//SensorResponse').attrib['Filename']])
        self.opticsTransmittanceFilename = '/'.join(
            [self.datafileroot, froot.find('.//OpticsTransmittance').attrib['Filename']])
        self.filterFilename = '/'.join(
            [self.datafileroot, froot.find('.//Filter').attrib['Filename']])
        self.filterName = os.path.basename(self.filterFilename)[:-4]
        self.sourceEmisFilename = '/'.join(
            [self.datafileroot, froot.find('.//SourceEmis').attrib['Filename']])
        self.atmoTauFilename = '/'.join(
            [self.datafileroot, froot.find('.//AtmoTau').attrib['Filename']])

        self.detectorPitch = float(froot.find('.//DetectorPitch').attrib['Value'])
        self.fillFactor = float(froot.find('.//FillFactor').attrib['Value'])
        self.focallength = float(froot.find('.//Focallength').attrib['Value'])
        self.integrationTime = float(froot.find('.//IntegrationTime').attrib['Value'])
        self.fnumber = float(froot.find('.//Fnumber').attrib['Value'])
        self.nuMin = float(froot.find('.//Nu').attrib['Min'])
        self.nuMax = float(froot.find('.//Nu').attrib['Max'])
        self.nuInc = float(froot.find('.//Nu').attrib['Inc'])

        for child in froot.findall('.//Caldatas'):
            for childA in child.findall('.//Caldata'):
                InstrTemp = float(childA.attrib['InstrTemperature'])
                self.dicFloor[InstrTemp] = float(childA.attrib['DlFloor'])
                self.dicPower[InstrTemp] = float(childA.attrib['Power'])
                self.dicCaldata[InstrTemp] = []
                for i, childC in enumerate(childA.findall('CalPoint')):
                    if i == 0:
                        data = np.asarray([273.15 + float(childC.attrib['Temp']),
                                           int(childC.attrib['DL'])])
                    else:
                        data = np.vstack((data, np.asarray([273.15 + float(childC.attrib['Temp']),
                                                            int(childC.attrib['DL'])])))
                self.dicCaldata[InstrTemp] = data
        self.lokey = min(self.dicCaldata.keys())
        self.hikey = max(self.dicCaldata.keys())

        for i, tmprInstr in enumerate(self.dicCaldata):
            if i == 0:
                tmprLow = np.min(self.dicCaldata[tmprInstr][:, 0]) - 100
                tmprHi = np.max(self.dicCaldata[tmprInstr][:, 0]) + 100
            else:
                tmprLow = min(tmprLow, np.min(self.dicCaldata[tmprInstr][:, 0]) - 100)
                tmprHi = max(tmprHi, np.max(self.dicCaldata[tmprInstr][:, 0]) + 100)

        tmprInc = (tmprHi - tmprLow) / 100.
        nu = np.linspace(self.nuMin, self.nuMax,
                         int(1 + (self.nuMax - self.nuMin) / self.nuInc))

        self.LU = rylookup.RadLookup(
            self.name, nu, tmprLow=tmprLow, tmprHi=tmprHi, tmprInc=tmprInc,
            sensorResp=self.sensorResponseFilename,
            opticsTau=self.opticsTransmittanceFilename,
            filterTau=self.filterFilename,
            atmoTau=self.atmoTauFilename,
            sourceEmis=self.sourceEmisFilename,
            sigMin=0., sigMax=2.0 ** 14, sigInc=2.0 ** 6,
            dicCaldata=self.dicCaldata,
            dicPower=self.dicPower,
            dicFloor=self.dicFloor)

    def Info(self):
        """Return a summary string of the calibration data.

        Returns:
            str: Human-readable summary of all loaded calibration parameters.
        """
        str = ''
        str += f'Path to XML                  = {self.pathtoXML}\n'
        str += f'Path to datafiles            = {self.datafileroot}\n'
        str += f'Calibration Name             = {self.name}\n'
        str += f'ID                           = {self.id}\n'
        str += f'Version                      = {self.version}\n'
        str += f'Summary                      = {self.summary}\n'
        str += f'DetectorPitch                = {self.detectorPitch}\n'
        str += f'FillFactor                   = {self.fillFactor}\n'
        str += f'Focallength                  = {self.focallength}\n'
        str += f'integrationTime              = {self.integrationTime}\n'
        str += f'Fnumber                      = {self.fnumber}\n'
        str += self.LU.Info()
        return str


if __name__ == '__main__':
    import os as _os
    _clutter = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'clutter')
    _os.makedirs(_clutter, exist_ok=True)

    def _c(fname):
        """Redirect an output filename into the clutter folder."""
        return _os.path.join(_clutter, _os.path.basename(str(fname)))

    import pyradi.ryutils as ryutils
    import pyradi.ryplot as ryplot

    rit = ryutils.intify_tuple

    # ── read the PTW file ──────────────────────────────────────────────────────
    ptwfile = 'data/PyradiSampleLWIR.ptw'
    outfilename = 'PyradiSampleLWIR.txt'

    header = readPTWHeader(ptwfile)
    showHeader(header)

    # Loading a sequence of frames
    print(f"\n{30 * '-'}\n")
    framesToLoad = [3, 4, 10]
    data, fheaders = getPTWFrames(header, framesToLoad)
    print(ptwfile)
    print(f'(1) Data shape={rit(data.shape)}')
    ifrm = 0
    print(f'Frame {ifrm} summary data:')
    print(f'Image data:\n{data[ifrm]}')
    print(f'Sensor Temperature4 {fheaders[ifrm].h_sensorTemp4:.6f} K ')
    print(f'FPA Temperature {fheaders[ifrm].h_detectorTemp:.6f} K ')
    print(f'Time {fheaders[ifrm].h_frameHour}:{fheaders[ifrm].h_frameMinute}:{fheaders[ifrm].h_frameSecond} ')

    print(f"\n{30 * '-'}\n")
    # Loading a sequence of frames with an error in request
    framesToLoad = [0, 4, 10]
    data, fheaders = getPTWFrames(header, framesToLoad)
    print(rit(data.shape))
    print(f'(2) Data shape={rit(data.shape)}')
    print(f"\n{30 * '-'}\n")

    # Loading single frames
    framesToLoad = list(range(1, 101, 1))
    frames = len(framesToLoad)
    data, fheaders = getPTWFrame(header, framesToLoad[0])
    for frame in framesToLoad[1:]:
        f, fheaders = getPTWFrame(header, frame)
        data = np.concatenate((data, f))

    rows = header.h_Rows
    cols = header.h_Cols
    img = data.reshape(frames, rows, cols)
    print(rit(img.shape))

    # ── read the calibration file ──────────────────────────────────────────────
    calData = JadeCalibrationData('./data/lwir100mm150us10ND20090910.xml', './data')

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
        for DL in [4571, 5132, 5906, 6887, 8034, 9338, 10834, 12386, 14042]:
            print(f'Tint={Tint}  DL={DL} T={calData.LU.LookupDLTemp(DL, Tint) - 273.15:.1f} C  L(filter)={calData.LU.LookupDLRad(DL, Tint):.0f}  L(no filter)={calData.LU.LookupTempRad(calData.LU.LookupDLTemp(DL, Tint), withFilter=False):.0f} W/(sr.m2)')
    print(' ')

    for Tint in [34.4]:
        for DL in [5477, 6050, 6817, 7789, 8922, 10262, 11694, 13299, 14921]:
            print(f'Tint={Tint}  DL={DL} T={calData.LU.LookupDLTemp(DL, Tint) - 273.15:.1f} C  L(filter)={calData.LU.LookupDLRad(DL, Tint):.0f}  L(no filter)={calData.LU.LookupTempRad(calData.LU.LookupDLTemp(DL, Tint), withFilter=False):.0f} W/(sr.m2)')
    print(' ')

    for Tint in [25]:
        for DL in [5477, 6050, 6817, 7789, 8922, 10262, 11694, 13299, 14921]:
            print(f'Tint={Tint} DL={DL}  T={calData.LU.LookupDLTemp(DL, Tint) - 273.15:.1f} C  L(filter)={calData.LU.LookupDLRad(DL, Tint):.0f}  L(no filter)={calData.LU.LookupTempRad(calData.LU.LookupDLTemp(DL, Tint), withFilter=False):.0f} W/(sr.m2)')
    print(' ')

    """The following measurement observed a large-area blackbody in the lab.
       The blackbody set point was 150C. The Cedip Altair s/w calculated 157C.
       This code calculates 149C.
    """
    ptwfile = 'data/LWIR-BBref-150C-150us.ptw'
    header = readPTWHeader(ptwfile)
    framesToLoad = [1]

    for frame in framesToLoad:
        data, fheader = getPTWFrame(header, frame)
        print(f'Sensor Temperature4 {fheader.h_sensorTemp4:.6f} K ')
        print(f'FPA Temperature {fheader.h_detectorTemp:.6f} K ')
        print(f'Time {fheader.h_frameHour}:{fheader.h_frameMinute}:{fheader.h_frameSecond} ')

        tempIm = calData.LU.LookupDLTemp(data, header.h_HousingTemperature1 - 273.15)
        print(f'Temperature at ({160},{120})={tempIm[160, 120] - 273.15:.6f} C')
        print(f'Temperature at ({140},{110})={tempIm[140, 110] - 273.15:.6f} C')
        I = ryplot.Plotter(4, 1, 1, '', figsize=(8, 8))
        I.showImage(1, tempIm,
                    ptitle=f'{ptwfile[:-4]}, frame {frame}, temperature in K',
                    titlefsize=7, cbarshow=True, cbarfontsize=7)
        I.saveFig(_c(f'{os.path.basename(ptwfile)[:-4]}-{frame}.png'))

    print('\nryptw Done!')
