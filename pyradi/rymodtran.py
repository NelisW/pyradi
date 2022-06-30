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

# The Initial Developer of the Original Code is AE Mudau,
# Portions created by AE Mudau are Copyright (C) 2012
# All Rights Reserved.

# Contributor(s):  CJ Willers, PJ Smit.
################################################################
"""
This module provides MODTRAN tape7 file reading.
Future development may include a class to write tape5 files, but that is 
a distant target at present.

See the __main__ function for examples of use.

This package was partly developed to provide additional material in support of students 
and readers of the book Electro-Optical System Analysis and Design: A Radiometry 
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""


from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['fixHeaders', 'loadtape7','fixHeadersList','runModtranAndCopy','runModtranModrootIn','variationTape5File']

import re
import sys
import numpy as np
import os.path
import pandas as pd
import json

if sys.version_info[0] > 2:
    # from io import StringIO
    from io import BytesIO
    # from str import maketrans
else:
    from string import maketrans
    import StringIO


##########################################################################################
climatics = [
'user-defined uniform','Tropical','Mid-Latitude Summer','Mid-Latitude Winter',
'Sub-Arctic Summer','Sub-Arctic Winter','1976 US Standard','user-defined layer',
'user-defined layers by pressure']
aerosols = [
'no aerosol','Rural 23 km',
'Rural 5 km','Navy maritime','Maritime','Urban','Tropospheric',
'user defined','Fog (advective)','Fog  (radiative)','Desert']


class extracttape6:
    """Class to collate all functions reading tape6 together, automatically export to json and latex.
    """


    def __init__(self, tp6name):

        self.tp6name = tp6name
        self.tp6tojsonlatex()

    def refind(self, line,pattern,warn=True):
        try:
            found = re.search(pattern, line).group(1)
        except AttributeError:
            if warn:
                # later expand with proper error handling
                print(f'{pattern} not found')
            found = None
        return found



    def extractAerosolAlts(self, line):
        """Find lower and upper altitude for aerosol layers in standard modtran atmos.
        BOUNDARY LAYER (0-2KM)     ...............
        """
        hlo =  float(self.refind(line,r'^\s*.+\s*\((.+?)-.+\)\s+\s+.*$',warn=True))
        hhi =  float(self.refind(line,r'^\s*.+\s*\(.+-(.+?)KM\)\s+\s+.*$',warn=True))
        return hlo,hhi


    def findPathRadianceMode(self, lines,tp6data):
        var = None
        for line in lines:
            if 'PROGRAM WILL COMPUTE ' in line:
                var = self.refind(line,r'PROGRAM WILL COMPUTE\s*(.+?)\s*$')
                tp6data['path radiance mode'] = var.lower()
        return var,tp6data


    def findScatterMode(self, lines,tp6data):
        var = {'scatter mode':'single scattering'}
        tp6data['scatter mode'] = 'single scattering'
        for line in lines:
            if 'Calculations will be done using ' in line:
                var = self.refind(line,r'Calculations will be done using\s*(.+?)\.$')
                tp6data['scatter mode'] = var
                return var,tp6data
        return var,tp6data


    def findSMetRange(self, lines,tp6data):
        var = 'default value'
        for line in lines:
            if 'METEOROLOGICAL RANGE' in line:
                var = float(self.refind(line,r'METEOROLOGICAL RANGE =\s*(.+?)\s*KM.*$'))
                break
        tp6data['meteorological range'] = var
        return var,tp6data


    def findWindSpeed(self, lines,tp6data):
        var = None
        var2 = None
        tp6data['wind speed'] = None
        tp6data['wind speed 24hr'] = None
        for line in lines:
            if var is None and 'METEOROLOGICAL RANGE' in line:
                var = float(self.refind(line,r'.*WIND =\s*(.+?)\s*M/S.*$',warn=True))
                tp6data['wind speed'] = var
            if var is None and 'WIND SPEED =' in line:
                var = float(self.refind(line,r'^\s*WIND SPEED =\s+(.+?)\s+M/SEC.*$',warn=True))
                tp6data['wind speed'] = var
            if 'WIND SPEED (24 HR AVERAGE) =' in line:
                var2 =float( self.refind(line,r'^\s*WIND SPEED \(24 HR AVERAGE\) =\s+(.+?)\s+M/SEC.*$',warn=True))
                tp6data['wind speed 24hr'] = var2

            if var is not None and var2 is not None:
                return (var,var2),tp6data
        return (var,var2),tp6data

    def findProfiles(self, lines,tp6data):
        var = None
        cntr = 0
        modelnumber = None
        df = pd.DataFrame()
        # first to the case with user-supplied layers
        for line in lines:
            if 'MODEL ATMOSPHERE NO.' in line:
                modelnumber = int(self.refind(line,r'\s*MODEL ATMOSPHERE NO\.\s*(.+?)\s*$',warn=True))
                if modelnumber != 7:
                    break
        if modelnumber == 7:
            for line in lines:
                if cntr==3:
                    if len(line) < 2:
                        break
                    else:
                        if 'FOG2 (RADIATI0N)    FOG2 (RADIATI0N)' in line:
                            line = line.replace('FOG2 (RADIATI0N)    FOG2 (RADIATI0N)','FOG2_(RADIATI0N)    FOG2_(RADIATI0N)')
                        if 'FOG1 (ADVECTTION)' in line:
                            line = line.replace('FOG1 (ADVECTTION)','FOG1_(ADVECTTION)')
                        if 'USER DEFINED' in line:
                            line = line.replace('USER DEFINED','USER DEFINED')
                        lst = line.strip().split()
                        df = df.append(pd.DataFrame([lst],index=None))
                if 'Z         P        T     REL H    H2O' in line:
                    cntr += 1
                if '(KM)      (MB)     (K)     (%)  (GM / M3)  TYPE' in line:
                    cntr += 1
                if '[Before scaling]' in line:
                    cntr += 1
            if cntr > 0:
                try:
                    df.columns = ['Height km','Pressure mB','Temperature K','RelHum %','H2O g/m3','Aero Type','Aero Prof','Aero Season']
                except:
                    print('Probably spaces resulting in additional columns')
                    print(df)
                df = df.fillna('')
                for col in df.columns[:5]:
                    df[col] = df[col].astype(float)
                for col in df.columns[5:]:
                    df[col] = df[col].str.replace('_',' ')
                    df[col] = df[col].str.replace('AEROSOL','')
                    df[col] = df[col].str.lower()




            #  find aerosol extinction
            cntr = 0
            dfh = pd.DataFrame()
            for line in lines:
                if cntr==2:
                    if len(line) < 2:
                        break
                    else:
                        lst = line.strip().split()
                        dfh = dfh.append(pd.DataFrame([lst],index=None))
                elif 'AEROSOL 1 AEROSOL 2 AEROSOL 3 AEROSOL 4  AER1*RH     RH (%)    RH (%)   CIRRUS   WAT DROP  ICE PART' in line:
                    cntr += 1
                elif '(      550nm EXTINCTION [KM-1]      )  (BEFORE H2O SCALING)   (AFTER)    (-)     (550nm VIS [KM-1])' in line:
                    cntr += 1


            dfh.columns = ['layer','Height km','Pressure mB','Temperature K','AEROSOL 1','AEROSOL 2','AEROSOL 3','AEROSOL 4','AER1*RH','RH (%)','RelHum %','CIRRUS','WAT DROP','ICE PART']
            dfh = dfh.fillna('')
            dfh['aeroextinc'] = dfh['AEROSOL 1'].astype(float) +dfh['AEROSOL 2'].astype(float)+dfh['AEROSOL 3'].astype(float)+dfh['AEROSOL 4'].astype(float)
            dfh['visibility'] = np.log(50) / (dfh['aeroextinc']+0.01159)
            dfh = dfh.drop(['layer','AEROSOL 1','AEROSOL 2','AEROSOL 3','AEROSOL 4','AER1*RH','RH (%)','CIRRUS','WAT DROP','ICE PART'],axis=1)
            for col in dfh.columns:
                dfh[col] = dfh[col].astype(float)
            dfh = dfh.drop(['Pressure mB','Temperature K','RelHum %'],axis=1)

            # now attach aerosol info
            dfh = dfh.merge(df,on='Height km')
            dfh = dfh[["Height km","Pressure mB","Temperature K","RelHum %","H2O g/m3","Aero Type","Aero Prof","Aero Season","aeroextinc","visibility"]]

            var =  dfh.to_dict(orient='records')
            tp6data['Profiles'] = var
        # now do modtran-supplied layers
        else:
            aerolayers = {}
            # first find boundary layer type
            cntr = 0
            for line in lines:
                # if 'FOG2 (RADIATI0N)    FOG2 (RADIATI0N)' in line:
                #     line = line.replace('FOG2 (RADIATI0N)    FOG2 (RADIATI0N)','FOG2_(RADIATI0N)    FOG2_(RADIATI0N)')
                if cntr==1:
                    if 'BOUNDARY LAYER' in line:
                        aerotype = line[33:58].lower()
                        aerotype = aerotype if 'aerosol'not in aerotype else aerotype.replace('aerosol','')
                        aerotype = aerotype.replace('_',' ') if '_' in aerotype else aerotype
                        hlo,hhi = self.extractAerosolAlts(line)
                        aerolayers[(hlo,hhi)] = [aerotype.strip(),'','']
                    if 'TROPOSPHERE' in line:
                        aerotype = line[33:58].lower()
                        aerotype = aerotype if 'aerosol'not in aerotype else aerotype.replace('aerosol','')
                        aerotype = aerotype.replace('_',' ') if '_' in aerotype else aerotype
                        aeroprofile = line[58:83].lower()
                        aeroprofile = aeroprofile if 'aerosol'not in aeroprofile else aeroprofile.replace('aerosol','')
                        aeroprofile = aeroprofile.replace('_',' ') if '_' in aeroprofile else aeroprofile
                        aeroseason = line[83:].lower()
                        aeroseason = aeroseason if 'aerosol'not in aeroseason else aeroseason.replace('aerosol','')
                        aeroseason = aeroseason.replace('_',' ') if '_' in aeroseason else aeroseason
                        hlo,hhi = self.extractAerosolAlts(line)
                        aerolayers[(hlo,hhi)] = [aerotype.strip(),aeroprofile.strip(),aeroseason.strip()]
                    if 'STRATOSPHERE' in line:
                        aerotype = line[33:58].lower()
                        aerotype = aerotype if 'aerosol'not in aerotype else aerotype.replace('aerosol','')
                        aerotype = aerotype.replace('_',' ') if '_' in aerotype else aerotype
                        aeroprofile = line[58:83].lower()
                        aeroprofile = aeroprofile if 'aerosol'not in aeroprofile else aeroprofile.replace('aerosol','')
                        aeroprofile = aeroprofile.replace('_',' ') if '_' in aeroprofile else aeroprofile
                        aeroseason = line[83:].lower()
                        aeroseason = aeroseason if 'aerosol'not in aeroseason else aeroseason.replace('aerosol','')
                        aeroseason = aeroseason.replace('_',' ') if '_' in aeroseason else aeroseason
                        hlo,hhi = self.extractAerosolAlts(line)
                        aerolayers[(hlo,hhi)] = [aerotype.strip(),aeroprofile.strip(),aeroseason.strip()]
                    if 'UPPER ATMOS' in line:
                        aerotype = line[33:58].lower()
                        aerotype = aerotype if 'aerosol'not in aerotype else aerotype.replace('aerosol','')
                        aerotype = aerotype.replace('_',' ') if '_' in aerotype else aerotype
                        aeroprofile = line[58:83].lower()
                        aeroprofile = aeroprofile if 'aerosol'not in aeroprofile else aeroprofile.replace('aerosol','')
                        aeroprofile = aeroprofile.replace('_',' ') if '_' in aeroprofile else aeroprofile
                        aeroseason = line[83:].lower()
                        aeroseason = aeroseason if 'aerosol'not in aeroseason else aeroseason.replace('aerosol','')
                        aeroseason = aeroseason.replace('_',' ') if '_' in aeroseason else aeroseason
                        hlo,hhi = self.extractAerosolAlts(line)
                        aerolayers[(hlo,hhi)] = [aerotype.strip(),aeroprofile.strip(),aeroseason.strip()]
                        break
                if 'REGIME                      AEROSOL TYPE             PROFILE                  SEASON' in line:
                    cntr += 1

            dfh = pd.DataFrame()
            #  find height,temp, pressure,humidity
            for line in lines:
                if cntr==3:
                    if len(line) < 2:
                        break
                    else:
                        if 'FOG2 (RADIATI0N)    FOG2 (RADIATI0N)' in line:
                            line = line.replace('FOG2 (RADIATI0N)    FOG2 (RADIATI0N)','FOG2_(RADIATI0N)    FOG2_(RADIATI0N)')
                        if 'FOG1 (ADVECTTION)' in line:
                            line = line.replace('FOG1 (ADVECTTION)','FOG1_(ADVECTTION)')
                        if 'USER DEFINED' in line:
                            line = line.replace('USER DEFINED','USER DEFINED')
                        lst = line.strip().split()
                        dfh = dfh.append(pd.DataFrame([lst],index=None))
                if 'AEROSOL 1 AEROSOL 2 AEROSOL 3 AEROSOL 4  AER1*RH     RH (%)    RH (%)   CIRRUS   WAT DROP  ICE PART' in line \
                    or 'Z         P        T     REL H    H2O                          AEROSOL' in line:
                    cntr += 1
                if '(      550nm EXTINCTION [KM-1]      )  (BEFORE H2O SCALING)   (AFTER)    (-)     (550nm VIS [KM-1])' in line \
                    or '(KM)      (MB)     (K)     (%)  (GM / M3)  TYPE                 PROFILE' in line :
                    cntr += 1
                if '[Before scaling]' in line:
                    # the first bump  'BOUNDARY LAYER' would not have occurred if we reach here, both cases count to 3
                    cntr += 1
            if cntr > 0:
                # print(dfh.shape)
                if dfh.shape[1]==14:
                    dfh.columns = ['layer','Height km','Pressure mB','Temperature K','AEROSOL 1','AEROSOL 2','AEROSOL 3','AEROSOL 4','AER1*RH','RH (%)','RelHum %','CIRRUS','WAT DROP','ICE PART']
                    dfh = dfh.fillna('')
                    dfh['aeroextinc'] = dfh['AEROSOL 1'].astype(float) +dfh['AEROSOL 2'].astype(float)+dfh['AEROSOL 3'].astype(float)+dfh['AEROSOL 4'].astype(float)
                    dfh['visibility'] = np.log(50) / (dfh['aeroextinc']+0.01159)
                    dfh = dfh.drop(['layer','AEROSOL 1','AEROSOL 2','AEROSOL 3','AEROSOL 4','AER1*RH','RH (%)','CIRRUS','WAT DROP','ICE PART'],axis=1)
                    for col in dfh.columns:
                        dfh[col] = dfh[col].astype(float)
                    Tk = dfh['Temperature K'].values
                    absHum = (dfh["RelHum %"]/100.) * (216.685/Tk)*6.11657 * np.exp(24.9215*(1-273.16/Tk)) * (273.16/Tk) **5.06
                    dfh["H2O g/m3"] =  absHum
                    # prepare aerosol df for later merging
                    dfa = pd.DataFrame()
                    heights = dfh['Height km'].values
                    for height in heights:
                        for key in aerolayers.keys():
                            if height >= key[0] and height < key[1]:
                                if len( aerolayers[key]) < 4:
                                    aerolayers[key].append(height)
                                dfa = dfa.append(pd.DataFrame([aerolayers[key]]))
                    dfa.columns = ["Aero Type","Aero Prof","Aero Season","Height km"]
                    # link the two dataframes on common
                    df = dfa.merge(dfh,on='Height km')
                    # reorder columns to be same as for model 7
                    # print(df)
                    df = df[["Height km","Pressure mB","Temperature K","RelHum %","H2O g/m3","Aero Type","Aero Prof","Aero Season","aeroextinc","visibility"]]
                    var =  df.to_dict(orient='records')
                    tp6data['Profiles'] = var

                elif  dfh.shape[1] == 8:
                    dfh.columns = ['Height km','Pressure mB','Temperature K','RelHum %','H2O g/m3','Aero Type','Aero Prof','Aero Season']
                    dfh = dfh.fillna('')
                    for col in dfh.columns[:5]:
                        dfh[col] = dfh[col].astype(float)
                    for col in dfh.columns[5:]:
                        dfh[col] = dfh[col].str.replace('_',' ')
                        dfh[col] = dfh[col].str.replace('AEROSOL','')
                        dfh[col] = dfh[col].str.lower()
                    # prepare aerosol df for later merging

                    # link the two dataframes on common
                    # dfh = dfh.merge(dfa,on='Height km')
                    # reorder columns to be same as for model 7
                    # print(dfh)
                    # dfh = dfh[["Height km","Pressure mB","Temperature K","RelHum %","H2O g/m3","Aero Type","Aero Prof","Aero Season","aeroextinc","visibility"]]
                    dfh = dfh[["Height km","Pressure mB","Temperature K","RelHum %","H2O g/m3","Aero Type","Aero Prof","Aero Season"]]
                    var =  dfh.to_dict(orient='records')
                    tp6data['Profiles'] = var
                # elif  dfh.shape[1] == 9:
                #     pass

                else:
                    print('unknown table layout')
                    print(dfh.shape)
                    print(dfh)

        if modelnumber is not None:
            tp6data['climatic model'] = climatics[modelnumber]

        return var,tp6data


    def findPath(self, lines,tp6data):
        var = {}
        fields = ['SLANT PATH TO SPACE (OR GROUND)','SLANT PATH No.  1, H1ALT TO H2ALT','HORIZONTAL PATH']
        done = False
        for line in lines:
            for field in fields:
                if field in line:
                    field = field.lower()
                    var['path type'] = field.replace(' No.  1,','') if ' No.  1' in field else field
                    done = True
            if done:
                break
        cntr = 0
        fields = ['H1ALT','H2ALT','OBSZEN','HRANGE','ECA','BCKZEN','HMIN','BENDING','CKRANG','LENN']
        for line in lines:
            if cntr>0:
                for field in fields:
                    if field in line:
                        var[field] = float(self.refind(line,f'{field}\\s*=\\s*(.+?)\\s.*$'))
                if 'LENN' in line:
                    var['LENN'] = float(self.refind(line,r'LENN\s*=\s*(.+?)\s*.*$'))
                    break
            if 'SUMMARY OF LINE-OF-SIGHT No.  1 GEOMETRY CALCULATION' in line:
                cntr += 1
        tp6data['Path'] = var
        return var,tp6data


    def findSingleScatter(self, lines,tp6data):
        var = None
        cntr = 0
        fields = ['LATITUDE AT H1ALT','LONGITUDE AT H1ALT','SUBSOLAR LATITUDE','SUBSOLAR LONGITUDE','TIME (<0 UNDEF)','PATH AZIMUTH (FROM H1ALT TO H2ALT)']
        for line in lines:
            if cntr>0:
                for field in fields:
                    if field in line:
                        sfield = field.replace('(','\\(').replace(')','\\)')
                        var[field] = float(self.refind(line,f'{sfield}\\s*=\\s*(.+?)\\s.*$'))
                if 'DAY OF THE YEAR' in line:
                    var['DAY OF THE YEAR'] = float(self.refind(line,r'DAY OF THE YEAR\s*=\s*(.+?)\s*.*$'))
                    break
            if 'SINGLE SCATTERING CONTROL PARAMETERS SUMMARY' in line:
                cntr += 1
                var = {}
        tp6data['SingleScat parameters'] = var
        return var,tp6data


    def findExtraTerresSource(self, lines,tp6data):
        var = None
        for line in lines:
            if 'EXTRATERRESTIAL SOURCE IS THE' in line:
                var = self.refind(line,r'EXTRATERRESTIAL SOURCE IS THE\s*(.+?)\s*$')
                tp6data['extra terrestial source'] = var.lower()
                return var,tp6data
        return var,tp6data


    def findPhaseFunction(self, lines,tp6data):
        var = None
        for line in lines:
            if 'PHASE FUNCTION FROM' in line:
                var = self.refind(line,r'PHASE FUNCTION FROM\s*(.+?)\s*$')
                tp6data['phase function'] = var.lower()
                return var,tp6data
        return var,tp6data


    def findFreqRange(self, lines,tp6data):
        var = None
        cntr = 0
        fields = ['IV1','IV2','IDV']
        for line in lines:
            if cntr>0:
                for field in fields:
                    if field in line:
                        var[field] = float(self.refind(line,f'{field}\\s*=\\s*(.+?)\\sCM-1.*$'))
                if 'IFWHM' in line:
                    var['IFWHM'] = float(self.refind(line,r'IFWHM\s*=\s*(.+?)\sCM-1.*$'))
                    break
            if 'FREQUENCY RANGE' in line:
                cntr += 1
                var = {}
        tp6data['Frequency range'] = var
        return var,tp6data


    def findAlbedo(self, lines,tp6data):
        var = {}
        cntr = 0
        for line in lines:
            if cntr==1:
                if 'Boundary Layer' not in line:
                    var['albedo name'] = line.strip()
                cntr += 1
            elif cntr==2:
                cntr += 1
                pass
                # var['albedo line'] = line.strip()
            elif cntr==3:
                if ' Using surface' in line:
                    var['using surface'] = self.refind(line,f' Using surface:\s*(.+?)\s*$')
                cntr += 1
            elif cntr==4:
                if ' From file' in line:
                    var['from file'] = self.refind(line,f' From file:\s*(.+?)\s*$')
                break

            elif cntr==0:
                if 'IMAGED PIXEL' in line:
                    if ' The IMAGED PIXEL' in line:
                        cntr += 1
                        var['albedo model'] = line[40:].strip()

        for line in lines:
            if 'AREA-AVERAGED GROUND EMISSIVITY' in line:
                areaemis = self.refind(line,r'\s+AREA-AVERAGED GROUND EMISSIVITY\s*=\s*(.+?)\s*$')
                var['area emissivity'] = float(areaemis)
            if 'IMAGED-PIXEL' in line and 'DIRECTIONAL EMISSIVITY' in line:
                directemis = self.refind(line,r'\s+IMAGED-PIXEL.+DIRECTIONAL EMISSIVITY\s*=\s*(.+?)\s*$')
                var['directional emissivity'] = float(directemis)

        tp6data['albedo'] = var
        return var,tp6data

#  AREA-AVERAGED GROUND EMISSIVITY =      1.000
#  IMAGED-PIXEL No.  1 DIRECTIONAL EMISSIVITY =      1.000

    def writeLaTeX(self, tp6data,filel):
        llines = []
        if len(tp6data['notes']) > 2:
            llines.append("\\textit{{{}}} ".format(tp6data['notes']))
        if 'climatic model' in tp6data.keys():
            llines.append(f"Using the {tp6data['climatic model']} climatic model, ")
        if 'Profiles' in tp6data.keys():
            llines.append(f"{tp6data['Profiles'][0]['Aero Type']} aerosol, ")
        else:
            print('*** No profile data in json file')
        if 'meteorological range' in tp6data.keys(): 
            if isinstance(tp6data['meteorological range'], str):
                llines.append(f"meteorological range {tp6data['meteorological range']}, ")
            else:
                llines.append(f"meteorological range {tp6data['meteorological range']:.3f}~""\si{\kilo\metre}, ")
        if 'wind speed 24hr' in tp6data.keys():
            if tp6data['wind speed 24hr'] is not None:
                llines.append(f" 24 h wind speed {tp6data['wind speed 24hr']}~""\si{\metre\per\second}, ")
        if tp6data['wind speed'] is not None:
            llines.append(f"wind speed {tp6data['wind speed']:.1f}~""\si{\metre\per\second}. ")
        if 'Profiles' in tp6data.keys():
            llines.append(f"At the first layer altitude of {tp6data['Profiles'][0]['Height km']:.3f}~""\si{\kilo\metre} ")
            llines.append("the conditions are:  ")
            llines.append(f"pressure {tp6data['Profiles'][0]['Pressure mB']:.1f}~""\si{\milli\\bar}, ")
            llines.append(f"temperature {tp6data['Profiles'][0]['Temperature K']:.1f}~""\si{\kelvin}, ")
            llines.append(f"relative humidity {tp6data['Profiles'][0]['RelHum %']:.1f}""\\%, ")
            llines.append(f"absolute humidity {tp6data['Profiles'][0]['H2O g/m3']:.2f}~""\si{\gram\per\metre\cubed}, ")
            if 'aeroextinc' in tp6data['Profiles'][0].keys():
                llines.append(f"extinction {tp6data['Profiles'][0]['aeroextinc']:.3e}~""\si[per-mode=reciprocal]{\centi\metre\\tothe{-1}}, ")
            if 'visibility' in tp6data['Profiles'][0].keys():
                llines.append(f"visibility {tp6data['Profiles'][0]['visibility']:.2f}~""\si{\kilo\metre}. ")

        if 'path radiance mode' in tp6data.keys():
            llines.append(f"Path radiance mode is {tp6data['path radiance mode']}"". ")
        if 'scatter mode' in tp6data.keys():
            llines.append(f"Scatter mode is {tp6data['scatter mode']}"". ")

        if 'SingleScat parameters' in tp6data.keys():
            if  tp6data['SingleScat parameters'] is not None:
                if 'LATITUDE AT H1ALT' in tp6data['SingleScat parameters'].keys():
                    llines.append(f"Location lat/long at H1 ")
                    llines.append(f"{tp6data['SingleScat parameters']['LATITUDE AT H1ALT']:.2f}""\si{\degree}N")
                    llines.append(f"/{tp6data['SingleScat parameters']['LATITUDE AT H1ALT']:.2f}""\si{\degree}W. ")
                if 'SUBSOLAR LATITUDE' in tp6data['SingleScat parameters'].keys():
                    llines.append(f"Subsolar lat/long ")
                    llines.append(f"{tp6data['SingleScat parameters']['SUBSOLAR LATITUDE']:.2f}""\si{\degree}N")
                    llines.append(f"/{tp6data['SingleScat parameters']['SUBSOLAR LONGITUDE']:.2f}""\si{\degree}W. ")
                llines.append(f"Day of the year {int(tp6data['SingleScat parameters']['DAY OF THE YEAR'])}, ")
        if 'phase function' in tp6data.keys():
            llines.append(f"Scattering phase function is {tp6data['phase function']}. ")
        if 'extra terrestial source' in tp6data.keys():
            llines.append(f"Extra terrestial source is the {tp6data['extra terrestial source']}. ")

        if 'Path' in tp6data.keys():
            if tp6data['Path'] is not None:
                llines.append(f"\n\nPath definition: ")
                if 'LENN' in  tp6data['Path']:
                    if tp6data['Path']['LENN'] < 0.5:
                        llines.append(f"short ")
                else:
                    llines.append(f"long ")
                if 'path type' in  tp6data['Path']:
                    llines.append(f"{tp6data['Path']['path type']}, ")
                if 'H1ALT' in  tp6data['Path']:
                    llines.append(f"H1 (start) altitude {tp6data['Path']['H1ALT']:.3f}~""\si{\kilo\metre}, ")
                if 'H2ALT' in  tp6data['Path']:
                    llines.append(f"H2 (final) altitude {tp6data['Path']['H2ALT']:.3f}~""\si{\kilo\metre}, ")
                if 'OBSZEN' in  tp6data['Path']:
                    llines.append(f"observer zenith angle {tp6data['Path']['OBSZEN']:.6f}""\si{\degree}, ")
                if 'HRANGE' in  tp6data['Path']:
                    llines.append(f"path range {tp6data['Path']['HRANGE']:.3f}~""\si{\kilo\metre}, ")
                if 'ECA' in  tp6data['Path']:
                    llines.append(f"subtended earth center angle {tp6data['Path']['ECA']:.6f}""\si{\degree}, ")
                if 'BCKZEN' in  tp6data['Path']:
                    llines.append(f"zenith angle from final altitude back to sensor {tp6data['Path']['BCKZEN']:.6f}""\si{\degree}, ")
                if 'HMIN' in  tp6data['Path']:
                    llines.append(f"minimum altitude {tp6data['Path']['HMIN']:.3f}~""\si{\kilo\metre}, ")
                if 'BENDING' in  tp6data['Path']:
                    llines.append(f"path bending {tp6data['Path']['BENDING']:.6f}""\si{\degree}, ")
                if 'CKRANG' in  tp6data['Path']:
                    llines.append(f"slant range for k-distribution output {tp6data['Path']['CKRANG']:.3f}~""\si{\kilo\metre}.")
        else:
            print('*** No path information in json file')


        if 'Frequency range' in tp6data.keys():
            if tp6data['Frequency range'] is not None:
                llines.append(f"\n\nFrequency range: ")
                llines.append(f"spectral range lower bound {tp6data['Frequency range']['IV1']:.0f}~""\si[per-mode=reciprocal]{\centi\metre\\tothe{-1}}, ")
                llines.append(f"spectral range upper bound {tp6data['Frequency range']['IV2']:.0f}~""\si[per-mode=reciprocal]{\centi\metre\\tothe{-1}}, ")
                llines.append(f"spectral increment {tp6data['Frequency range']['IDV']:.0f}~""\si[per-mode=reciprocal]{\centi\metre\\tothe{-1}}, ")
                llines.append(f"slit function full width at half maximum {tp6data['Frequency range']['IFWHM']:.0f}~""\si[per-mode=reciprocal]{\centi\metre\\tothe{-1}}.")
            else:
                print('*** No frequency information in json file')

        if 'albedo' in tp6data.keys():
            if tp6data['albedo'] is not None:
                llines.append(f"\n\nAlbedo and emissivity: ")
                if 'albedo model' in tp6data['albedo'].keys():
                    llines.append(f"{tp6data['albedo']['albedo model']}, ")
                if 'albedo name' in tp6data['albedo'].keys():
                    llines.append(f"albedo name \\verb+{tp6data['albedo']['albedo name']}+, ")
                if 'using surface' in tp6data['albedo'].keys():
                    llines.append(f"using surface \\verb+{tp6data['albedo']['using surface']}+, ")
                if 'from file' in tp6data['albedo'].keys():
                    llines.append(f"from file \\verb+{tp6data['albedo']['from file']}+, ")
                if 'area emissivity' in tp6data['albedo'].keys():
                    llines.append(f"area emissivity {tp6data['albedo']['area emissivity']}, ")
                if 'directional emissivity' in tp6data['albedo'].keys():
                    llines.append(f"directional emissivity {tp6data['albedo']['directional emissivity']}. ")


        with open(filel,'w') as fout:
            for lline in llines:
                fout.write(lline)


    def tp6tojsonlatex(self):
        tp6data = {}
        with open(self.tp6name,'r') as fin:
            lines = fin.readlines()
            _,tp6data = self.findPathRadianceMode(lines,tp6data)
            _,tp6data = self.findScatterMode(lines,tp6data)
            _,tp6data = self.findSMetRange(lines,tp6data)
            _,tp6data = self.findWindSpeed(lines,tp6data)
            _,tp6data = self.findPath(lines,tp6data)
            _,tp6data = self.findSingleScatter(lines,tp6data)
            _,tp6data = self.findExtraTerresSource(lines,tp6data)
            _,tp6data = self.findPhaseFunction(lines,tp6data)
            _,tp6data = self.findFreqRange(lines,tp6data)
            _,tp6data = self.findProfiles(lines,tp6data)
            _,tp6data = self.findAlbedo(lines,tp6data)

            filenamen = self.tp6name.replace('.tp6','.notes')
            if os.path.exists(filenamen):
                with open(filenamen,'r') as finn:
                    linesn = finn.readlines()
                    tp6data['notes'] = ' '.join(linesn)
            else:
                tp6data['notes'] = ''

        jsonfile = self.tp6name.replace('.tp6','tp6.json')
        with open(jsonfile, "w") as fout:
            json.dump(tp6data, fout, indent=4)

        latexfile = self.tp6name.replace('.tp6','.tex')
        self.writeLaTeX(tp6data,latexfile)

        return tp6data


##############################################################################
##http://stackoverflow.com/questions/1324067/how-do-i-get-str-translate-to-work-with-unicode-strings
def fixHeaders(instr):
    """
    Modifies the column header string to be compatible with numpy column lookup.

    Args:
        | list columns (string): column name.


    Returns:
        | list columns (string): fixed column name.

    Raises:
        | No exception is raised.
    """

    # intab = "+-[]@"
    # outtab = "pmbba"

    # if isinstance(instr, str):
    #     translate_table = dict((ord(char), str(outtab)) for char in intab)
    # else:
    #     assert isinstance(instr, str)
    #     translate_table = maketrans(intab, outtab)
    # print(instr, type(translate_table),translate_table )
    # return instr.translate(translate_table)

    if sys.version_info[0] > 2:
        intab = "+--[]@"
        outtab = "pmmbba"

        # print(instr)
        # if isinstance(instr, unicode):
        #     translate_table = dict((ord(char), unicode(outtab)) for char in intab)
        # else:
        #     assert isinstance(instr, str)
        translate_table = str.maketrans(intab, outtab)
    else:
        intab = u"+-[]@"
        outtab = u"pmbba"

        # print(instr)
        if isinstance(instr, unicode):
            translate_table = dict((ord(char), unicode(outtab)) for char in intab)
        else:
            assert isinstance(instr, str)
            translate_table = maketrans(intab, outtab)

    rtnstring = instr.translate(translate_table)
    return rtnstring




##############################################################################
##
def fixHeadersList(headcol):
    """
    Modifies the column headers to be compatible with numpy column lookup.

    Args:
        | list columns ([string]): list of column names.


    Returns:
        | list columns ([string]): fixed list of column names.

    Raises:
        | No exception is raised.
    """
    headcol = [fixHeaders(strn) for strn in headcol]
    return headcol

##############################################################################
##
def loadtape7(filename, colspec = []):

    """
    Read the Modtran tape7 file. This function was tested with Modtran5 files.

    Args:
        | filename (string): name of the input ASCII flatfile.
        | colspec ([string]): list of column names required in the output the spectral transmittance data.

    Returns:
        | np.array: an array with the selected columns. Col[0] is the wavenumber.

    Raises:
        | No exception is raised.



    This function reads in the tape7 file from MODerate spectral resolution
    atmospheric TRANsmission (MODTRAN) code, that is used to model the
    propagation of the electromagnetic radiation through the atmosphere. tape7
    is a primary file that contains all the spectral results of the MODTRAN
    run. The header information in the tape7 file contains portions of the
    tape5 information that will be deleted. The header section in tape7 is
    followed by a list of spectral points with corresponding transmissions.
    Each column has a different component of the transmission or radiance. 
    For more detail, see the modtran documentation.

    The user selects the appropriate columns by listing the column names, as
    listed below.

    The format of the tape7 file changes for different IEMSCT values. For
    the most part the differences are hidden in the details.  
    The various column headers used in the tape7 file are as follows:

    IEMSCT = 0 has two column header lines.  Different versions of modtran
    has different numbers of columns. In order to select the column, you
    must concatenate the two column headers with an underscore in between. All
    columns are available with the following column names: ['FREQ_CM-1',
    'COMBIN_TRANS', 'H2O_TRANS', 'UMIX_TRANS', 'O3_TRANS', 'TRACE_TRANS',
    'N2_CONT', 'H2O_CONT', 'MOLEC_SCAT', 'AER+CLD_TRANS', 'HNO3_TRANS',
    'AER+CLD_abTRNS', '-LOG_COMBIN', 'CO2_TRANS', 'CO_TRANS', 'CH4_TRANS',
    'N2O_TRANS', 'O2_TRANS', 'NH3_TRANS', 'NO_TRANS', 'NO2_TRANS',
    'SO2_TRANS', 'CLOUD_TRANS', 'CFC11_TRANS', 'CFC12_TRANS', 'CFC13_TRANS',
    'CFC14_TRANS', 'CFC22_TRANS', 'CFC113_TRANS', 'CFC114_TRANS',
    'CFC115_TRANS', 'CLONO2_TRANS', 'HNO4_TRANS', 'CHCL2F_TRANS',
    'CCL4_TRANS', 'N2O5_TRANS','H2-H2_TRANS','H2-HE_TRANS','H2-CH4_TRANS',
    'CH4-CH4_TRANS']

    IEMSCT = 1 has single line column headers. A number of columns have
    headers, but with no column numeric data.  In the following list the
    columns with header names ** are empty and hence not available: ['FREQ',
    'TOT_TRANS', 'PTH_THRML', 'THRML_SCT', 'SURF_EMIS', *SOL_SCAT*,
    *SING_SCAT*, 'GRND_RFLT', *DRCT_RFLT*, 'TOTAL_RAD', *REF_SOL*, *SOL@OBS*,
    'DEPTH', 'DIR_EM', *TOA_SUN*, 'BBODY_T[K]']. Hence, these columns do not
    have valid data: ['SOL_SCAT', 'SING_SCAT', 'DRCT_RFLT', 'REF_SOL',
    'SOL@OBS', 'TOA_SUN']

    IEMSCT = 2 has single line column headers. All the columns are available:
    ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'THRML_SCT', 'SURF_EMIS', 'SOL_SCAT',
    'SING_SCAT', 'GRND_RFLT', 'DRCT_RFLT', 'TOTAL_RAD', 'REF_SOL', 'SOL@OBS',
    'DEPTH', 'DIR_EM', 'TOA_SUN', 'BBODY_T[K]']

    IEMSCT = 3 has single line column headers.  One of these seems to be two
    words, which, in this code must be concatenated with an underscore. There
    is also  additional column (assumed to be depth in this code).  The
    columns available are ['FREQ', 'TRANS', 'SOL_TR', 'SOLAR', 'DEPTH']

    The tape7.scn file has missing columns, so this function does not work for
    tape7.scn files.  If you need a tape7.scn file with all the columns populated
    you would have to use the regular tape7 file and convolve this to lower resolution.

    UPDATE:
    Different versions of Modtran have different columns present in the tape7 file, 
    also not all the columns are necessarily filled, some have headings but no data.
    To further complicate matters, the headings are not always separated by spaces,
    sometime (not often) heading text runs into each other with no separator.

    It seems that the column headings are right aligned with the column data itself,
    so if we locate the right most position of the column data, we can locate the
    headings with valid data - even in cases with connected headings.  

    The algorithm used is as follows:

    1. step down to a data line beyond the heading lines (i.e., the data)
    2. locate the position of the last character of every discrete colummn
    3. From the last column of the previous col, find start of next col
    4. move up into the header again and isolate the header column text for each column

    In sptite of all the attempts to isolate special cases, reading of tape7 files
    remain a challange and may fail in newer versions of Modtran, until fixed.
    """

    infile = open(filename, 'r')
    idata = {}
    colHead = []
    lines = infile.readlines()#.strip()
    infile.close()
    if len(lines) < 10:
        print(f'Error reading file {filename}: too few lines!')
        return None

    #determine values for MODEL, ITYPE, IEMSCT, IMULT from card 1
    #tape5 input format (presumably also tape7, line 1 format?)
    #format Lowtran7  (13I5, F8.3, F7.0) = (MODEL, ITYPE, IEMSCT, IMULT)
    #format Modtran 4 (2A1, I3, 12I5, F8.3, F7.0) = (MODTRN, SPEED, MODEL, ITYPE, IEMSCT, IMULT)
    #format Modtran 5 (3A1, I2, 12I5, F8.0, A7) = (MODTRN, SPEED, BINARY, MODEL, ITYPE, IEMSCT, IMULT)
    #MODEL = int(lines[0][4])
    #ITYPE = int(lines[0][9])
    IEMSCT = int(lines[0][14])
    #IMULT = int(lines[0][19])
    #print('filename={0}, IEMSCT={1}'.format(filename,IEMSCT))

    #skip the first few rows that contains tape5 information and leave the
    #header for the different components of transimissions.
    #find the end of the header.
    headline = 0
    while lines[headline].find('FREQ') < 0:
        headline = headline + 1

    #some files has only a single text column head, while others have two
    # find out what the case is for this file and concatenate if necessary
    line1 = lines[headline] # alway header 1
    line2 = lines[headline+1] # maybe data, maybe header 2
    line3 = lines[headline+2] # definately data

    #see if there is a second header line
    p = re.compile('[a-df-zA-DF-Z]+')
    line2found = True if p.search(line2) is not None else False

    #modtran 4 does not use underscores
    line1 = line1.replace('TOT TRANS','TOT_TRANS')
    line1 = line1.replace('PTH THRML','PTH_THRML')
    line1 = line1.replace('THRML SCT','THRML_SCT')
    line1 = line1.replace('SURF EMIS','SURF_EMIS')
    line1 = line1.replace('SOL SCAT','SOL_SCAT')
    line1 = line1.replace('SING SCAT','SING_SCAT')
    line1 = line1.replace('GRND RFLT','GRND_RFLT')
    line1 = line1.replace('DRCT RFLT','DRCT_RFLT')
    line1 = line1.replace('TOTAL RAD','TOTAL_RAD')
    line1 = line1.replace('REF SOL','REF_SOL')

    colcount = 0
    colEnd = []
    #strip newline from the data line
    linet = line3.rstrip()

    idx = 0
    while idx < len(linet):
        while linet[idx].isspace():
            idx += 1
            if idx == len(linet):
                break
        while not linet[idx].isspace():
            idx += 1
            if idx == len(linet):
                break
        colEnd.append(idx+1)

    colSrt = [0] + [v-1 for v in colEnd[:-1]]
    if sys.version_info[0] > 2:
        # http://www.diveintopython3.net/porting-code-to-python-3-with-2to3.html
        # zip returns an iterator, not a list
        collim = list(zip(colSrt,colEnd))
    else:
        collim = zip(colSrt,colEnd)

    # iemsct=3 has a completely messed up header, replace with this
    if IEMSCT == 3:
        colHead1st = ' '.join(['FREQ', 'TRANS', 'SOL_TR', 'SOLAR', 'DEPTH'])
    else:
        # each entry in collim defines the slicing start and end for each col, including leading whitepace
        # missing columns may have headers that came through as single long string, 
        # now remove by splitting on space and taking the last one
        colHead1st = [line1[lim[0]:lim[1]-1].split()[-1].strip() for lim in collim]
    colHead2nd = [line2[lim[0]:lim[1]-1].split()[-1].strip() for lim in collim]

    # if colHead2nd[0].find('CM') >= 0:
    if line2found:
        colHead = [h1+'_'+h2 for (h1,h2) in zip(colHead1st,colHead2nd)]
        deltaHead = 1
    else:
        colHead = colHead1st
        deltaHead = 0

    # print(colHead)

    #different IEMSCT values have different column formats
    # some cols have headers and some are empty.
    # for IEMSCT of 0 and 2 the column headers are correct and should work as is.
    #for IEMSCT of 1 the following columns are empty and must be deleted from the header

    if IEMSCT == 1:
        removeIEMSCT1 = ['SOL_SCAT', 'SING_SCAT', 'DRCT_RFLT', 'REF_SOL', 'SOL@OBS', 'TOA_SUN']
        colHead = [x for x in colHead if x not in removeIEMSCT1]

    if IEMSCT == 3:
        colHead = ['FREQ', 'TRANS', 'SOL_TR', 'SOLAR', 'DEPTH']

    # build a new data set with appropriate column header and numeric data
    #change all - and +  to alpha to enable table lookup
    colHead = fixHeadersList(colHead)

    s = ' '.join(colHead) + '\n'
    # now append the numeric data, ignore the original header and last row in the file
    s = s + ''.join(lines[headline+1+deltaHead:-1])

    #read the string in from a StringIO in-memory file
    # lines = np.ndfromtxt(io.StringIO(s), dtype=None,  names=True)
    if sys.version_info[0] > 2:
        # lines = np.ndfromtxt(StringIO(s), dtype=None,  names=True)
        lines = np.ndfromtxt(BytesIO(s.encode('utf-8')), dtype=None,  names=True)
    else:
        lines = np.ndfromtxt(StringIO.StringIO(s), dtype=None,  names=True)

    # print('lines=',lines)

    #extract the wavenumber col as the first column in the new table
    # print(lines)
    coldata= lines[fixHeaders(colspec[0])].reshape(-1, 1)
    # then append the other required columns
    for colname in colspec[1:]:
        coldata = np.hstack((coldata, lines[fixHeaders(colname)].reshape(-1, 1)))

    return coldata



##############################################################################
##
def variationTape5File(scenario, variation, tape5,varFunc,fmtstr, srcstr,outfile='tape5' ):
    """Read tape5 in specified folder, modify one or more variable, write out new tape5 in new folder

    One tape5 is written with as many changes as are present in the four lists:
        <variation, varFunc,fmtstr, srcstr> with corresponding values.
    
    If only one value is to be changed, the four values may be scalar (not lists).
    If the values are scalar, lists are formed before processing.

    Replace a search string <srcstr> (must be unique in the file) with the value 
    of a variation <variation> formatted  by <fmtstr> as required by the tape5 format. 
    The variation input value <variation> is processed by the function <varFunc>.
    All occurrences of the search string <srcstr> are blindly replaced, 
    and must occur only once in the file and be of the same width as the field
    to be replaced.

    The variation folder name is constructed by joining with '-' the Python str() 
    representation of the variables.

    The variation variable must be of the form that can be used to create sub-folders in
    the master scenario folder. The variation can be processed (scaled, etc.) by <varFunc>
    before writing out to file. For example, variation can be an integer in metre, 
    then scaled to kilometer. If varFunc is None, the value is used as is.

    The variation tape5 is written to a new sub folder with the name variation, constructed
    in the scenario folder.  The template tape5 file must be in the scenario root file.

    

    dir structure:
    |dir root
        | file domodtran.py
        | dir scenario 1
            |dir variation 1
            |dir variation 2
            |dir variation 3
            |file tape5 template for scenario 1
        | dir scenario 2
            |dir variation 1
            |dir variation 2
            |dir variation 3
            |file tape5 template for scenario 2
    The scenario directories must exist, but the variation scenarios are created.

    Args:
        | scenario (string): path to folder with the scenario tape5
        | variation ([int/float]): number that defines the variation, will be modified by varFunc
        | tape5 (string): tape5 filename, must be in the scenario folder
        | varFunc ([fn]): Function to process the variation number before writing, can be None
        | fmtstr ([string]): print format string to be used when writing to tape5
        | srcstr ([string]): the string in the tape5 file to be replaced
        | outfile (]string]): the name to be used when writing the modified tape5

    Returns:
        | side effect creates new folders and files.

    Raises:
        | No exception is raised.
    """

    import os

    # if not lists, make lists
    if not isinstance(variation, list): 
        variation = [variation]
    if not isinstance(varFunc, list): 
        varFunc = [varFunc]
    if not isinstance(fmtstr, list): 
        fmtstr = [fmtstr]
    if not isinstance(srcstr, list): 
        srcstr = [srcstr]

    # check lists same length
    if len(variation)!= len(varFunc):
        print('len(variation)!= len(varFunc)')
        return None
    if len(variation)!= len(fmtstr):
        print('len(variation)!= len(fmtstr)')
        return None
    if len(variation)!= len(srcstr):
        print('len(variation)!= len(srcstr)')
        return None

    #read the tape5 base file
    tape5base = os.path.join(scenario,tape5)
    with open(tape5base) as fin:
        lines = fin.readlines()

    varcomb = '-'.join([str(i) for i in variation])
    # print(tape5base, varcomb, scenario)

    #change the srcstr to new value with specfied format for all variations
    outlines = []
    for line in lines:
        for i,_ in enumerate(variation):
            # use varFunc if not None
            newVal = varFunc[i](variation[i])if varFunc[i] else variation[i]
            line = line.replace(srcstr[i],f'{newVal:{fmtstr[i]}}')
        outlines.append(line)

    #create new scenario and write file to new dir
    dirname = os.path.join('.',scenario,'{}'.format(varcomb))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = os.path.join(dirname, outfile)
    with open(filename,'w') as fout:
        # print('writing',filename)
        fout.writelines(outlines)

    return varcomb


##############################################################################
##
def runModtranAndCopy(root, research, pathToModtranBin,execname,clean=False):
    """
    Look for input files in directories, run modtran and copy results to dir.

    Finds all files below the root directory that matches the 
    regex pattern in research. Then runs modtran on these files
    and write the tape5/6/7/8 back to where the input file was.

    Each input file must be in a separate directory, because the results are
    all written to files with the names 'tape5', etc.

    Args:
        | root ([string]): path to root dir containing the dirs with modtran input files.
        | research ([string]): regex to use when searching for input files ('.*.ltn' or 'tape5')
        | pathToModtranBin ([string]): path to modtran executable directory.
        | execname ([string]): modtran executable filename.

    Returns:
        | List of all the files processed.

    Raises:
        | No exception is raised.
    """
    import os
    import shutil
    import subprocess
    import time
    import pyradi.ryfiles as ryfiles

    filepaths = ryfiles.listFiles(root, patterns=research, recurse=1, return_folders=0, useRegex=True)

    # print('**********************',root,research,filepaths)
 
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        dirname = os.path.dirname(filepath)

        #get rid of clutter to make sure that our tape5 will be used
        for rfile in ['tape5','tape6','tape7','tape7.scn','tape8','modin','modin.ontplt','mod5root.in','.ezl20ck']:
            rpath = os.path.join(pathToModtranBin, rfile)
            if os.path.exists(rpath):
                os.remove(rpath)

        #copy our tape5 across
        tape5path = os.path.join(pathToModtranBin, 'tape5')
        shutil.copy2(filepath, tape5path)

        #run modtran on the tape5 file in its bin directory
        if os.path.exists(tape5path):
            if  os.sep=='/':
                p = subprocess.Popen(os.path.join(pathToModtranBin, execname))
            else:
                p = subprocess.Popen(os.path.join(pathToModtranBin, execname), 
                        shell=True, stdout=None, stderr=None, cwd=pathToModtranBin)
            while  p.poll() == None:
                time.sleep(0.5)

            # #copy the tape5/6/7 files back to appropriate directory
            for outname in ['tape5', 'tape6', 'tape7','tape8']:
                    outpath = os.path.join(pathToModtranBin,outname)
                    if os.path.exists(outpath):
                        shutil.copy2(outpath, dirname)

    # cleaning up, get rid of clutter 
    for rfile in ['tape5','tape6','tape7','tape7.scn','tape8','modin','modin.ontplt','mod5root.in','.ezl20ck']:
        rpath = os.path.join(pathToModtranBin, rfile)
        if os.path.exists(rpath):
            os.remove(rpath)




##############################################################################
##
def runModtranModrootIn(root, pathToModtranBin,execname,filerootname='atmo'):
    """
    Look for input files in directories, run modtran and copy results to dir.

    Finds all *.tp5 files below the root directory that matches the 
    regex pattern in research. Then runs modtran on these files
    using modroot.in

    Each input file must be in a separate directory, because the results are
    all written to files with the names 'tape5', etc.

    Args:
        | root ([string]): path to root dir containing the dirs with modtran input files.
        | pathToModtranBin ([string]): path to modtran executable directory.
        | execname ([string]): modtran executable filename.
        | filerootname (string): root filename, no extension

    Returns:
        | List of all the files processed.

    Raises:
        | No exception is raised.
    """
    import os
    import shutil
    import subprocess
    import time
    import pyradi.ryfiles as ryfiles

    filepaths = ryfiles.listFiles(root, patterns=f'{filerootname}.tp5', recurse=1, return_folders=0, useRegex=False)

    # print('**********************',root,research,filepaths)
 
    for filepath in filepaths:

        # write tape5 path to modroot.in
        with open('modroot.in','w') as fout:
            fout.writelines([filepath.replace('.tp5','\n')])
        time.sleep(0.1)

        #run modtran
        if os.path.exists(filepath):
            if  os.sep=='/':
                p = subprocess.Popen(os.path.join(pathToModtranBin, execname))
            else:
                p = subprocess.Popen(os.path.join(pathToModtranBin, execname), 
                        shell=True, stdout=None, stderr=None, cwd=pathToModtranBin)
            while  p.poll() == None:
                time.sleep(0.5)




################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    import math
    import sys
    import numpy as np

    import pyradi.ryplot as ryplot
    import pyradi.ryutils as ryutils


    doAll = True
    
    if doAll:
        pathToModtranBin = r'C:\PcModWin5\bin'
        research = '.*.ltn'
        root = r'D:\work\ISP\SWIR-tradeoff\data\atmo\TropicalRural\slant'
        execname = 'OntarMod5_3_2.exe'
        runModtranAndCopy(root=root, research=research, pathToModtranBin=pathToModtranBin, execname=execname)

    if doAll:
        figtype = ".png"  # eps, jpg, png
        #figtype = ".eps"  # eps, jpg, png

        ## ----------------------- -----------------------------------------
        tape7= loadtape7("data/tape7-01", ['FREQ_CM-1', 'COMBIN_TRANS', 'MOLEC_SCAT', 'AER+CLD_TRANS', 'AER+CLD_abTRNS'] )
        np.savetxt('tape7-01a.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-01b", ['FREQ_CM-1', 'COMBIN_TRANS', 'MOLEC_SCAT', 'AER+CLD_TRANS', 'AER+CLD_abTRNS'] )
        np.savetxt('tape7-01b.txt', tape7, fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-01", ['FREQ_CM-1', 'COMBIN_TRANS', 'H2O_TRANS', 'UMIX_TRANS', 'O3_TRANS', 'TRACE_TRANS', 'N2_CONT', 'H2O_CONT', 'MOLEC_SCAT', 'AER+CLD_TRANS', 'HNO3_TRANS', 'AER+CLD_abTRNS', '-LOG_COMBIN', 'CO2_TRANS', 'CO_TRANS', 'CH4_TRANS', 'N2O_TRANS', 'O2_TRANS', 'NH3_TRANS', 'NO_TRANS', 'NO2_TRANS', 'SO2_TRANS', 'CLOUD_TRANS', 'CFC11_TRANS', 'CFC12_TRANS', 'CFC13_TRANS', 'CFC14_TRANS', 'CFC22_TRANS', 'CFC113_TRANS', 'CFC114_TRANS', 'CFC115_TRANS', 'CLONO2_TRANS', 'HNO4_TRANS', 'CHCL2F_TRANS', 'CCL4_TRANS', 'N2O5_TRANS'] )
        np.savetxt('tape7-01.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-02", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'THRML_SCT', 'SURF_EMIS', 'GRND_RFLT', 'TOTAL_RAD', 'DEPTH', 'DIR_EM', 'BBODY_T[K]'] )
        np.savetxt('tape7-02.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-02b", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'SURF_EMIS', 'GRND_RFLT', 'TOTAL_RAD', 'DEPTH', 'DIR_EM', 'BBODY_T[K]'] )
        np.savetxt('tape7-02.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-02c", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'SURF_EMIS', 'GRND_RFLT', 'TOTAL_RAD', 'DEPTH', 'DIR_EM', 'BBODY_T[K]'] )
        np.savetxt('tape7-02.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-03", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'THRML_SCT', 'SURF_EMIS', 'SOL_SCAT', 'SING_SCAT', 'GRND_RFLT', 'DRCT_RFLT', 'TOTAL_RAD', 'REF_SOL', 'SOL@OBS', 'DEPTH', 'DIR_EM', 'TOA_SUN', 'BBODY_T[K]'] )
        np.savetxt('tape7-03.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-04", ['FREQ', 'TRANS', 'SOL_TR', 'SOLAR', 'DEPTH'] )
        np.savetxt('tape7-04.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-05", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'SURF_EMIS', 'TOTAL_RAD'] )
        np.savetxt('tape7-05.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-05b", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'SURF_EMIS', 'TOTAL_RAD'] )
        np.savetxt('tape7-05b.txt', tape7,fmt=str('%.6e'))


    if doAll:

        colSelect =  ['FREQ_CM-1', 'COMBIN_TRANS', 'H2O_TRANS', 'UMIX_TRANS', \
              'O3_TRANS', 'H2O_CONT', 'MOLEC_SCAT', 'AER+CLD_TRANS']
        tape7= loadtape7("data/tape7VISNIR5kmTrop23Vis", colSelect )
        wavelen = ryutils.convertSpectralDomain(tape7[:,0],  type='nl')
        mT = ryplot.Plotter(1, 1, 1,"Modtran Tropical, 23 km Visibility (Rural)"\
                           + ", 5 km Path Length",figsize=(12,6))
        mT.plot(1, wavelen, tape7[:,1:], "","Wavelength [$\mu$m]", "Transmittance",
               label=colSelect[1:],legendAlpha=0.5, pltaxis=[0.4,1, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.saveFig('ModtranPlot.png')
        #mT.saveFig('ModtranPlot.eps')

        # this example plots the individual transmittance components
        colSelect =  ['FREQ_CM-1', 'COMBIN_TRANS', 'MOLEC_SCAT', 'CO2_TRANS', 'H2O_TRANS', 'H2O_CONT', 'CH4_TRANS',\
           'O3_TRANS', 'O2_TRANS', 'N2O_TRANS', 'AER+CLD_TRANS', 'SO2_TRANS']
        tape7= loadtape7("data/horizon5kmtropical.fl7", colSelect )
        wavelen = ryutils.convertSpectralDomain(tape7[:,0],  type='nl')
        mT = ryplot.Plotter(1, 9, 1,"Modtran Tropical, 23 km Visibility (Rural)"\
                           + ", 5 km Path Length",figsize=(6,12))
        mT.semilogX(1, wavelen, tape7[:,1], '','', '',
               label=colSelect[1:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.semilogX(2, wavelen, tape7[:,2], '','', '',
               label=colSelect[2:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.semilogX(3, wavelen, tape7[:,10], '','', '',
               label=colSelect[10:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.semilogX(4, wavelen, tape7[:,4] , '','', '',
               label=colSelect[4:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])

        mT.semilogX(5, wavelen, tape7[:,5] , '','', '',
               label=colSelect[5:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])

        mT.semilogX(6, wavelen, tape7[:,3]  , '','', '',
               label=colSelect[3:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.semilogX(7, wavelen, tape7[:,6]  , '','', '',
               label=colSelect[6:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.semilogX(8, wavelen, tape7[:,7] * tape7[:,8] , '','', '',
               label=colSelect[7:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.semilogX(9, wavelen, tape7[:,9]  , '','', '',
               label=colSelect[9:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])


        mT.saveFig('ModtranSpec.png')
        mT.saveFig('ModtranSpec.eps')

        # calculate the total path radiance over a spectral band
        #read the tape7 file with path radiance components, plot and integrate.
        colSelect =  ['FREQ', 'PTH_THRML','SOL_SCAT','SING_SCAT', 'TOTAL_RAD']
        skyrad= loadtape7("data/NIRscat.fl7", colSelect )
        sr = ryplot.Plotter(1, 4,1,"Path Radiance in NIR, Path to Space from 3 km",figsize=(12,8))
        # plot the components separately
        for i in [1,2,3,4]:
          Lpath = 1.0e4 * skyrad[:,i]
          sr.plot(i,  skyrad[:,0], Lpath, "","Wavenumber [cm$^{-1}$]", "L [W/(m$^2$.sr.cm$^{-1}$)]",
                 label=[colSelect[i][:]],legendAlpha=0.5, #pltaxis=[0.4,1, 0, 1],
                 maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])

          #convert from /cm^2 to /m2 and integrate using the wavenumber vector
          #normally you would multiply with a sensor spectral response before integration
          #this calculation is over the whole band, equally weighted.
          totinband = np.trapz(Lpath.reshape(-1, 1),skyrad[:,0], axis=0)[0]
          print('{0} sum is {1} [W/(m^2.sr)]'.format(colSelect[i][:],totinband))
        sr.saveFig('NIRPathradiance.png')
        print('Note that multiple scatter contributes significantly to the total path radiance')

        #repeat the same calculation, but this time do in wavelength domain
        colSelect =  ['FREQ', 'PTH_THRML','SOL_SCAT','SING_SCAT', 'TOTAL_RAD']
        skyrad= loadtape7("data/NIRscat.fl7", colSelect )
        sr = ryplot.Plotter(1, 4,1,"Path Radiance in NIR, Path to Space from 3 km",
                            figsize=(12,8))
        # plot the components separately
        for i in [1,2,3,4]:
          wl, Lpath = ryutils.convertSpectralDensity(skyrad[:,0], skyrad[:,i],'nl')
          Lpath *= 1.0e4
          sr.plot(i,  wl, Lpath, "","Wavelength [$\mu$m]","L [W/(m$^2$.sr.$\mu$m)]",
                 label=[colSelect[i][:]],legendAlpha=0.5, #pltaxis=[0.4,1, 0, 1],
                 maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
          totinband = - np.trapz(Lpath.reshape(-1, 1),wl,axis=0)[0]
          print('{0} integral is {1:.6e} [W/(m^2.sr)]'.format(colSelect[i][:],totinband))

        sr.saveFig('NIRPathradiancewl.png')
        print('Note that multiple scatter contributes significantly to total path radiance')


    print('\n\nrymodtran is done!')
