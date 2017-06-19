# -*- coding: utf-8 -*-

################################################################
# The contents of this file are subject to the BSD 3Clause (New)cense
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

################################################################
"""


Download the latest version of Christoph Gohlke's transformation Python code:
(look for the latest version for your version of Python)
from http://www.lfd.uci.edu/~gohlke/pythonlibs/

Install the transformations wheel to your Python environment by doing a wheel install.

If you are running Python in an environment, first activate the environment then run the wheel install command. 

You can either download the latest version of Gholke's transformation library or use the one supplied in this directory.

Open a command window in the directory that contains the transformations wheel and execute the command (or whatever the whl filename is):::

    wheel install transformations-2015.7.18-cp27-none-win_amd64.whl  
    wheel install transformations-2017.2.17-cp35-cp35m-win_amd64.whl  
    wheel install transformations-2015.7.18-cp27-none-win_amd64.whl  


From transformations.py:

This module follows the "column vectors on the right" and "row major storage"
(C contiguous) conventions. The translation components are in the right column
of the transformation matrix, i.e. M[:3, 3].
The transpose of the transformation matrices may have to be used to interface
with other graphics systems, e.g. with OpenGL's glMultMatrixd().

Calculations are carried out with numpy.float64 precision.

Vector, point, quaternion, and matrix function arguments are expected to be
"array like", i.e. tuple, list, or numpy arrays.

Return types are numpy arrays unless specified otherwise.

Angles are in radians unless specified otherwise.

Quaternions w+ix+jy+kz are represented as [w, x, y, z].

See also http://matthew-brett.github.io/transforms3d/reference/transforms3d.euler.html#direction-of-rotation
which explains as follows:

You specify conventions for interpreting the sequence of Euler angles with a four character string.

The first character is ‘r’ (rotating == intrinsic), or ‘s’ (static == extrinsic).

The next three characters give the axis (‘x’, ‘y’ or ‘z’) about which to perform the rotation, 
in the order in which the rotations will be performed.

For example the string ‘szyx’ specifies that the angles should be interpreted relative to 
extrinsic (static) coordinate axes, and be performed in the order: rotation about z axis; 
rotation about y axis; rotation about x axis. This is a relatively common convention, 
with customized implementations in taitbryan in this package.

The string ‘rzxz’ specifies that the angles should be interpreted relative to 
intrinsic (rotating) coordinate axes, and be performed in the order: 
rotation about z axis; rotation about the rotated x axis; rotation about the rotated z axis. 
Wolfram Mathworld claim this is the most common convention : 
http://mathworld.wolfram.com/EulerAngles.html.


Provides a simple, order of magnitude estimate of the photon flux and 
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

__version__ = "$Revision$"
__author__ = 'pyradi team'
__all__ = ['PFlux','lllPhotonrates']


import numpy as np
import math
import sys
import collections
import itertools
import pandas as pd
import pyradi.ryutils as ryutils
import pyradi.ryplanck as ryplanck
from numbers import Number
import transformations as tr
# from transforms3d.euler import euler2quat
# from transforms3d.quaternions import qmult
# from transforms3d.quaternions import qinverse
# from transforms3d.euler import quat2euler


def df_euler2quat(row):
    """ Given yaw, pitch and roll calculate quaternion for seq ypr.

    The transforms documentation states that the angle order in function call must
    correspond with the order in the sequence spec. First to first, etc. 
    
    The quaternion_from_euler/euler2quat function returns a (4,) array, 
    which must be encapsulated in a list in order to be returned as a 
    single column in pandas, otherwise error is raised. 
    Don't remove the list!
    """

    q = [tr.quaternion_from_euler(row['yawRad'],row['pitchRad'],row['rollRad'],row['sequence'])]

    return q

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
class Scene:
    """
    """
    ############################################################
    ##
    def __init__(self):
        """Class constructor

        Right Hand Coordinate system, North (x), East (y) Down (z).
        Yaw/Azimuth positive from x to y, clockwise when looking down along +z.
        Roll positive from y to z, clockwise when looking along +x.
        Pitch/Elevation positive from z to x, clockwise when looking along +y.

        Euler angle sequence to orientate the camera axes into the world is 
            yaw (z), pitch (y), roll (x). Don't change this order!!



        """
        self.dfLocus = None
        self.sequence = 'rzyx' # don't change this, see df_euler2quat() above


    def readLocus(self,filename):
        """Read a locus.csv file into data frame.

        File must have no spaces and contain data in this format:
        (time in seconds, x,y,z in metre and azim/elev in degrees)::
 
            time,x,y,z,yawDeg,pitchDeg,rollDeg
            0,0,0,0,0,0,0
            1,0,0,0,30,0,0
            2,0,0,0,30,15,0
            3,0,0,0,30,15,20
            4,0,0,0,0,0,0


        The angular values are processed to obtain angles in radians and quaternions. 
            """
        print(filename)
        self.dfLocus = pd.read_csv(filename)
        self.dfLocus['yawRad'] = self.dfLocus['yawDeg'] * np.pi / 180.
        self.dfLocus['rollRad'] = self.dfLocus['rollDeg'] * np.pi / 180.
        self.dfLocus['pitchRad'] = self.dfLocus['pitchDeg'] * np.pi / 180.
        self.dfLocus['sequence'] = self.sequence
        self.dfLocus['quat'] = self.dfLocus.apply(df_euler2quat, axis=1)



    def interpLocus(self,ltime):
        """Given a time, interpolate the direction locus vector.         
        """
        # find the two rows spanning the current value for time
        self.dfLocus['uidx'] = np.where(np.sign(self.dfLocus['time'] - ltime).diff()>0,True,False)
        # uidx is the index of the upper bound, uidx-1 is the lower bound
        uidx = self.dfLocus[self.dfLocus['uidx']].index.tolist()[0]
        # get parametric value between lower and upper, based on time
        t = (ltime-self.dfLocus.ix[uidx-1]['time'])/(self.dfLocus.ix[uidx]['time']-self.dfLocus.ix[uidx-1]['time'])
        # get the two quats at lower and upper
        lq = self.dfLocus.ix[uidx-1]['quat']
        uq = self.dfLocus.ix[uidx  ]['quat']
        # and do slerp
        # http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/
        q = tr.quaternion_slerp(lq[0], uq[0], t) 
        yaw,pit,rol = tr.euler_from_quaternion(q,axes=self.sequence)

        return yaw,pit,rol,q


################################################################
################################################################
##
## evaluation functions not part of library

def unitxrotate(df):
    """rotate a vector along x-axis to other directions
    """

    # define quaternion to a point at (1,0,0)
    pt = np.asarray([0,1,0,0])
    print(pt)
    for idx,item in sc.dfLocus.iterrows():
        print(10*'-')
        rts = tr.quaternion_from_euler(item['yawRad'],item['pitchRad'],item['rollRad'],sc.sequence)#(seq='rzyx')
        print('rot quat={}'.format(rts))           
        
        pts = tr.quaternion_multiply(rts,tr.quaternion_multiply(pt,tr.quaternion_inverse(rts)))
        print('point={}'.format(pts))           
        print('yaw/pit/rol={}'.format(tr.euler_from_quaternion(rts,sc.sequence)))

    print('\n---------------------\n')


def applyInterpLocus(t):
    """Helper function to interpolate pd locus at time t, return Euler and quat.

    This function is meant to be used with Pandas apply, given t of a row.
    """
    yaw,pit,rol,q = sc.interpLocus(t)
    ps = pd.Series({'yaw':yaw, 'pit':pit, 'rol':rol, 'quat':[q]})
    return ps

def locusPlot(sc):
    dfi = pd.DataFrame()
    numSamples = 50
    dfi['time'] = np.linspace(np.min(sc.dfLocus['time']),np.max(sc.dfLocus['time']),numSamples )
    dfi = dfi.merge(dfi.time.apply(applyInterpLocus), left_index=True, right_index=True)    
    p = ryplot.Plotter(1,1,1,'Test slerp',figsize=(8,5))
    p.plot(1,dfi['time'],dfi['yaw'],label=['yaw'])
    p.plot(1,dfi['time'],dfi['pit'],label=['pit'])
    p.plot(1,dfi['time'],dfi['rol'],'','Time [s]','Angle [rad]',label=['rol'])
    yoffset = p.getYLim(1)[0] + 0.05 * (p.getYLim(1)[1]-p.getYLim(1)[0])
    xoffset = p.getXLim(1)[0] + 0.05 * (p.getXLim(1)[1]-p.getXLim(1)[0])
    pstr = sc.dfLocus[['time', 'yawRad','pitchRad', 'rollRad']].to_string()
    p.getSubPlot(1).text(xoffset,yoffset, pstr, horizontalalignment='left', 
                verticalalignment='bottom', family='monospace',fontsize=10)
    p.saveFig('ryscene-quatslerp.png')


################################################################
################################################################
##

if __name__ == '__main__':

    doAll = False

    import pyradi.ryplot as ryplot

    if True:
        # create a scene object and load locus from excel file
        sc = Scene()
        sc.readLocus('data/scene/sensorlocus.csv')
        print(sc.dfLocus)

        # rotate a vector along x-axis to other directions
        unitxrotate(sc)

        # interpolate and plot locus
        locusPlot(sc)

    print('module ryscene done!')
        