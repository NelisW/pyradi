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

# The Initial Developer of the Original Code is CJ Willers,
# Portions created by CJ Willers are Copyright (C) 2006-2012
# All Rights Reserved.

# Contributor(s): ______________________________________.
################################################################
"""
This module provides tools for creating and viewing spherical plots.

The spherical plotting tool, using Mayavi, requires two sets of data
in order to create the spherical plot: the vertex locations in (x,y,z)
and the spatial relationship between the vertices, i.e. triangulation of
nearest neighbours.  This spatial relationship is required to create
surface elements between the vertices. If the spatial relationship
is not known, the data is merely a cloud of points, with no surface
content.

The easiest way to create the spatial relationships between the vertices
was to use a complex hull polygon model of an object.  The polygon or
wireframe model has the requires vertices and spatial relationships.

In the original application of this tool, a series of spheres were created
using MeshLab.  The essential property of these spheres was that the vertices
on the surface of the spheres were spaced equidistant over the surface, i.e.
an optimal spatial sampling distribution.  The files were exported as OFF
files, and should be in the pyradi data/plotspherical directory.
There are 6 possible input files, each with different number of samples
on the unit sphere:  12, 42, 162, 642, 2562 or 10242.

Any object, with the required vertices and spatial relationships can
be used. it does not have to be equi-sampled spheres.

The data must be in the OFF wireframe format.

There are two possible trajectory file types:
 * Stationary sensor and object with the target rotating.
   In this case the trajectory file specifies the target trajectory.
 * stationary object with orbiting sensor.
   In this case the trajectory file specifies the sensor trajectory.

This tool was originally developed to create trajectory files for the
Denel/CSIR OSSIM simulation.  The code was restructured for greater
universal application, but the final example is still an OSSIM case.

See the __main__ function for examples of use.
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['makerotatetrajfromoff']

import os.path, fnmatch
import numpy
from scipy.interpolate import interp1d

##############################################################################
##
def readOffFile(filename):
    """Reads an OFF file and returns the vertices and triangles in numpy arrays.

    The OFF file is read and the data captured in the array structures.
    This is a fairly trivial reading task.

    Args:
        | filename (string): name of the OFF file

    Returns:
        | vertices(numpy.array([])): array of vertices as [x y z]
        | triangles(numpy.array([])): array of triangles as []

    Raises:
        | No exception is raised.
   """
    with open(filename) as f:
        #first line [0] has only the word OFF
        lines = f.readlines()
        if lines[0].find('OFF') < 0:
            print('not an OFF file')
            return (None,None)
        #second line [1] has counts for ....
        counts = lines[1].split()
        vertexCount = int(counts[0])
        faceCount = int(counts[1])
        edgeCount =  int(counts[2])
        print('vertexCount={0} faceCount={1} edgeCount={2}'.format(vertexCount, faceCount, edgeCount))
        # then follows vertices from lines[2] to lines[2+vertexCount]
        vertices = numpy.asarray([float(s) for s in lines[2].split()])
        for line in lines[3:2+vertexCount]:
            vertices = numpy.vstack((vertices,numpy.asarray([float(s) for s in line.split()])))
        # now extract the triangles lines[2+vertexCount] to lines(-1)
        triangles = numpy.asarray([int(s) for s in lines[2+vertexCount].split()[1:]])
        for line in lines[3+vertexCount:2+vertexCount+faceCount]:
            if len(line) > 0:
                triangles = numpy.vstack((triangles,numpy.asarray([int(s) for s in line.split()[1:]])))

        if triangles.shape[0] != faceCount or vertices.shape[0] != vertexCount:
            return (None,None)
        else:
            return (vertices, triangles)



##############################################################################
##
def getRotateFromOffFile(filename, xPos, yPos, zPos):
    """ Reads an OFF file and returns object attitude and position.

    Reads an OFF format file wireframe description, and calculate the pitch
    and yaw angles to point the object's X-axis towards the OFF file vertex
    directions.

    Euler order is yaw-pitch-roll, with roll equal to zero.
    Yaw is defined in xy plane.
    Pitch is defined in xz plane.
    Roll is defined in yz plane.

    The object is assumed to stationary at the position (xPos, yPos, zPos),
    the position arrays are the same length as the attitude angle arrays,
    but all values in each individual array are all the same.

    Args:
        | filename (string): OFF file filename
        | xPos (double): scale factor to be applied to x axis
        | yPos (double): scale factor to be applied to y axis
        | zPos (double): scale factor to be applied to z axis


    Returns:
        | x(numpy.array()): array of x values
        | y(numpy.array()): array of y values
        | z(numpy.array()): array of z values
        | roll(numpy.array()): array of roll values
        | pitch(numpy.array()): array of pitch values
        | yaw(numpy.array()): array of yaw values

    Raises:
        | No exception is raised.
    """

    (geodesic, triangles) = readOffFile(filename)

    if geodesic is not None:

        ysign = (1 * (geodesic[:,1] < 0) - 1 * (geodesic[:,1] >= 0)).reshape(-1, 1)

        xyradial = (numpy.sqrt((geodesic[:,0]) ** 2 + (geodesic[:,1]) ** 2)).reshape(-1, 1)

        deltaX = (-geodesic[:,0]).reshape(-1, 1)
        #the strange '+ (xyradial==0)' below is to prevent divide by zero
        cosyaw = ((deltaX/(xyradial + (xyradial==0))) * (xyradial!=0) + 0 * (xyradial==0))
        yaw = ysign * numpy.arccos(cosyaw)
        pitch = - numpy.arctan2((-geodesic[:,2]).reshape(-1, 1), xyradial).reshape(-1, 1)
        roll = numpy.zeros(yaw.shape).reshape(-1, 1)

        onesv = numpy.ones(yaw.shape).reshape(-1, 1)
        x = xPos * onesv
        y = yPos * onesv
        z = zPos * onesv

        return (x, y, z, roll, pitch, yaw)
    else:
        return (None, None, None, None, None, None)


##############################################################################
##
def writeRotatingTargetOssimTrajFile(filename, trajType, distance, xTargPos,
    yTargPos, zTargPos, xVel, yVel, zVel, engine, deltaTime ):
    """ Reads OFF file and create OSSIM trajectory file for target rotation
        according to OFF file directions.

    This function writes a file in the custom OSSIM trajectory file format.
    Use this function as an example on how to use the ryplotspherical
    functionality in your application.

    This function reads an OFF format file wireframe description, in order
    calculate pitch and yaw angles to orientate an object's x-axis along the
    vertices in the OFF file.

    The trajectory file is written with the assumption that the rotating test
    target is located at (rangeTarget,0,altitude) while the observer
    is stationary at (0,0,0), looking along the x-axis.

    The velocity and engine setting are constant for all views.

    Two additional files are also written to assist with the subsequent viewing.

    1)
    The directions file contains the normalised vectors to from where the
    observer viewed the target for each view.  Hence these vectors are the
    direction of sampled intensity values.

    2)
    The triangles file defines triangles that provides the spatial linking when
    plotting the data. In essence, do we plot the complex hull comprising the
    triangles, with vertices along the direction vectors.

    OSSIM Users: For an example of how to use these trajectory files, see
    test point tp01m. The scenario files are present in the the appropriate
    subtest directory (m) and the plotting routines are in the utils dicectory
    in tp01.

    Args:
        | filename (string): OFF file filename
        | trajType (string): type of trajectory: 'Rotate' or 'Orbit'
        | distance (double): distance from sensor ro object
        | xTargPos (double): object x position.
        | yTargPos (double): object y position.
        | zTargPos (double): object z position.
        | xVel (double): velocity in x direction
        | yVel (double): velocity in y direction
        | zVel (double): velocity in z direction
        | engine (double): engine settiing
        | deltaTime (double): sampling time increment in output file

    Returns:
        | nothing

    Raises:
        | No exception is raised.
    """

    if trajType == 'Rotate':
        (x, y, z, roll, pitch, yaw) = getRotateFromOffFile(filename, xTargPos, yTargPos, zTargPos)
    else:
        (x, y, z, roll, pitch, yaw) = getOrbitFromOffFile(filename, xTargPos, yTargPos, zTargPos, distance)

    (geodesic, triangles) = readOffFile(filename)

    zerov = numpy.zeros(yaw.shape).reshape(-1, 1)
    onesv = numpy.ones(yaw.shape).reshape(-1, 1)

    time = deltaTime * numpy.asarray([i for i in range(0,zerov.shape[0])]).reshape(-1, 1)

    outp = time
    outp = numpy.hstack((outp, x))
    outp = numpy.hstack((outp, y))
    outp = numpy.hstack((outp, z))
    outp = numpy.hstack((outp, roll))
    outp = numpy.hstack((outp, yaw))
    outp = numpy.hstack((outp, pitch))
    outp = numpy.hstack((outp, xVel * onesv)) # x-velocity
    outp = numpy.hstack((outp, yVel * onesv)) # y-velocity
    outp = numpy.hstack((outp, zVel * onesv)) # z-velocity
    outp = numpy.hstack((outp, engine * onesv)) # engine setting

    outfile = os.path.basename(filename)
    idx=outfile.find('.')
    outfile = outfile[:idx]

    fid = open('Trajectory{0}{1}.txt'.format(trajType,outfile), 'w' )
    fid.write( 'Time x y z rol yaw pit vx vy vz engine \n' )
    fid.write( '0.0 infty infty infty infty infty infty infty infty infty infty \n' )
    fid.write( '0.0 infty infty infty infty infty infty infty infty infty infty\n' )
    numpy.savetxt( fid , outp )
    fid.close()

    fid = open('Triangles{0}{1}.txt'.format(trajType,outfile), 'w' )
    numpy.savetxt( fid , triangles )
    fid.close()

    fid = open('Directions{0}{1}.txt'.format(trajType,outfile), 'w' )
    numpy.savetxt( fid , geodesic )
    fid.close()

##############################################################################
##
def getOrbitFromOffFile(filename, xTargPos, yTargPos, zTargPos, distance):
    """ Reads an OFF file and returns sensor attitude and position.

    Reads an OFF format file wireframe description, and calculate the sensor
    attitude and position such that the sensor always look at the object
    located at ( xTargPos, yTargPos, zTargPos), at a constant range.

    Euler order is yaw-pitch-roll, with roll equal to zero.
    Yaw is defined in xy plane.
    Pitch is defined in xz plane.
    Roll is defined in yz plane.

    The object is assumed to stationary at the position
    (xTargPos, yTargPos, zTargPos).

    Args:
        | filename (string): OFF file filename
        | xTargPos (double): x target object position (fixed)
        | yTargPos (double): y target object position (fixed)
        | zTargPos (double): z target object position (fixed)
        | distance (double): range at which sensor orbits the target



    Returns:
        | x(numpy.array()): array of x values
        | y(numpy.array()): array of y values
        | z(numpy.array()): array of z values
        | roll(numpy.array()): array of roll values
        | pitch(numpy.array()): array of pitch values
        | yaw(numpy.array()): array of yaw values

    Raises:
        | No exception is raised.
    """


    (geodesic, triangles) = readOffFile(filename)

    if geodesic is not None:

        targPosition = numpy.asarray([xTargPos, yTargPos, zTargPos])

        mislPosition = distance * geodesic
        mislPosition[:,0] = mislPosition[:,0] + xTargPos
        mislPosition[:,1] = mislPosition[:,1] + yTargPos
        mislPosition[:,2] = mislPosition[:,2] + zTargPos

        # rotate missile in Euler angles, order ypr
        # yaw defined in xy plane
        #rrange = numpy.sqrt((targPosition[0]-mislPosition[:,0]) ** 2 + (targPosition[1]-mislPosition[:,1]) ** 2 + (targPosition[2]-mislPosition[:,2]) ** 2)
        ysign = (1 * (mislPosition[:,1] < 0) - 1 * (mislPosition[:,1] >= 0)).reshape(-1, 1)
        xyradial = (numpy.sqrt((targPosition[0]-mislPosition[:,0]) ** 2 + (targPosition[1]-mislPosition[:,1]) ** 2)).reshape(-1, 1)
        deltaX = (targPosition[0]-mislPosition[:,0]).reshape(-1, 1)
        #the strange '+ (xyradial==0)' below is to prevent divide by zero
        cosyaw = ((deltaX/(xyradial + (xyradial==0))) * (xyradial!=0) + 0 * (xyradial==0))
        yaw = ysign * numpy.arccos(cosyaw)
        pitch = - numpy.arctan2((targPosition[2]-mislPosition[:,2]).reshape(-1, 1), xyradial).reshape(-1, 1)
        roll = numpy.zeros(yaw.shape).reshape(-1, 1)

        return (mislPosition[:,0].reshape(-1, 1), mislPosition[:,1].reshape(-1, 1), mislPosition[:,2].reshape(-1, 1), roll, pitch, yaw)
    else:
        return (None, None, None, None, None, None)

################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    import math
    import sys

    import pyradi.ryplanck as ryplanck
    import pyradi.ryplot as ryplot
    import pyradi.ryfiles as ryfiles

    #figtype = ".png"  # eps, jpg, png
    figtype = ".eps"  # eps, jpg, png

    #write the OSSIM files for a rotating target and stationary sensor/observer.
    #use the OFF file with 10242 vertices.
    #the time increment  is 0.01 for each new position, velocity is zero here and
    #engine setting is 1. In this context the distance is irrelevant.
    writeRotatingTargetOssimTrajFile('data/plotspherical/sphere_5_10242.off', 'Rotate',
        None, 1000, 0, -1500,0, 0, 0, 1, 0.01)

    writeRotatingTargetOssimTrajFile('data/plotspherical/sphere_5_10242.off', 'Orbit',
        1000, 0, 0, -1500,0, 0, 0, 0, 0.01)


