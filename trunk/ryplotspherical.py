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

Note that the spherical plot has no way to discriminate between negative
values and a pi phase shift: there is confusion between sign and
direction.  This is inherent in the conversion between cartesian
and spherical coordinates. The user has to make provision for this,
possibly by plotting only negative or only positive values.

The data must be in the OFF wireframe format.

There are two possible trajectory file types:
 * 'Rotate' Stationary sensor and object with the target rotating.
   In this case the trajectory file specifies the target trajectory.
 * 'Orbit' Stationary object with orbiting sensor.
   In this case the trajectory file specifies the sensor trajectory.

The sphere data available in pyradi/data/plotspherical are:

===============  ===========
Filename         Resolution
                  (degrees)
===============  ===========
sphere_0_12       56.9
sphere_1_42       28.5
sphere_2_162      14.2
sphere_3_642      7.1
sphere_4_2562     3.56
sphere_5_10242    1.78
===============  ===========

The workflow is as follows:
 #. Use writeRotatingTargetOssimTrajFile (or your own equivalent) to
    calculate the appropriate trajectory file.
    At the same time, there are two additional files created
    (vertices and triangles) - keep these safe.
 #. Create your data set (e.g. run simulation) using the trajectory
    file. Collect the simulation data in a format for plotting.
 #. Use the simulation data, together with the triangles and
    vertices file, to plot the data. The triangles and vertices
    are require to set up the plotting environment, consider this
    the three-dimensional 'grid', while the simulation data
    provides the data to be plotted in this grid.

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
__all__= ['readOffFile','getRotateFromOffFile','getOrbitFromOffFile',
        'writeRotatingTargetOssimTrajFile']

import os.path, fnmatch
import numpy
from scipy.interpolate import interp1d
from mayavi import mlab

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
        print('vertexCount={0} faceCount={1} edgeCount={2}'.format(vertexCount,
            faceCount, edgeCount))
        #calculate the approx sampling density as surface of sphere divided by
        #number of faces, then size of each face, then angle from size
        areaFace = 4 * (numpy.pi)**2 / faceCount
        sizeFace = numpy.sqrt(areaFace / 2)
        resAngle = sizeFace
        print('sampling density is approx {0:.2f} mrad or {1:.3f} degrees.'.format(
            1000 * resAngle,resAngle * 180 / numpy.pi))
        # then follows vertices from lines[2] to lines[2+vertexCount]
        vertices = numpy.asarray([float(s) for s in lines[2].split()])
        for line in lines[3:2+vertexCount]:
            vertices = numpy.vstack((vertices,numpy.asarray([float(s) \
                for s in line.split()])))
        # now extract the triangles lines[2+vertexCount] to lines(-1)
        triangles = numpy.asarray([int(s) for s in lines[2+vertexCount].split()[1:]])
        for line in lines[3+vertexCount:2+vertexCount+faceCount]:
            if len(line) > 0:
                triangles = numpy.vstack((triangles,numpy.asarray([int(s) \
                    for s in line.split()[1:]])))

        if triangles.shape[0] != faceCount or vertices.shape[0] != vertexCount:
            return (None,None)
        else:
            return (vertices, triangles)



##############################################################################
##
def getRotateFromOffFile(filename, xPos, yPos, zPos):
    """ Reads an OFF file and returns object attitude and position.

    Calculate the pitch and yaw angles to point the object's X-axis towards
    the OFF file vertex directions.

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
        | vertices(numpy.array([])): array of vertices as [x y z]
        | triangles(numpy.array([])): array of triangles as []

    Raises:
        | No exception is raised.
    """

    (vertices, triangles) = readOffFile(filename)

    if vertices is not None:

        ysign = (1 * (vertices[:,1] < 0) - 1 * (vertices[:,1] >= 0)).reshape(-1, 1)

        xyradial = (numpy.sqrt((vertices[:,0]) ** 2 + \
            (vertices[:,1]) ** 2)).reshape(-1, 1)

        deltaX = (-vertices[:,0]).reshape(-1, 1)
        #the strange '+ (xyradial==0)' below is to prevent divide by zero
        cosyaw = ((deltaX/(xyradial + (xyradial==0))) * (xyradial!=0) + 0 * (xyradial==0))
        yaw = ysign * numpy.arccos(cosyaw)
        pitch = - numpy.arctan2((-vertices[:,2]).reshape(-1, 1), xyradial).reshape(-1, 1)
        roll = numpy.zeros(yaw.shape).reshape(-1, 1)

        onesv = numpy.ones(yaw.shape).reshape(-1, 1)
        x = xPos * onesv
        y = yPos * onesv
        z = zPos * onesv

        return (x, y, z, roll, pitch, yaw, vertices, triangles)
    else:
        return (None, None, None, None, None, None, None, None)

##############################################################################
##
def getOrbitFromOffFile(filename, xTargPos, yTargPos, zTargPos, distance):
    """ Reads an OFF file and returns sensor attitude and position.

    Calculate the sensor attitude and position such that the sensor always
    look at the object located at ( xTargPos, yTargPos, zTargPos), at
    a constant distance.

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
        | vertices(numpy.array([])): array of vertices as [x y z]
        | triangles(numpy.array([])): array of triangles as []

    Raises:
        | No exception is raised.
    """


    (vertices, triangles) = readOffFile(filename)

    if vertices is not None:

        targPosition = numpy.asarray([xTargPos, yTargPos, zTargPos])

        sensorPos = distance * vertices
        sensorPos[:,0] = sensorPos[:,0] + xTargPos
        sensorPos[:,1] = sensorPos[:,1] + yTargPos
        sensorPos[:,2] = sensorPos[:,2] + zTargPos

        ysign = (1 * (sensorPos[:,1] < 0) - 1 * (sensorPos[:,1] >= 0)).reshape(-1, 1)
        xyradial = (numpy.sqrt((targPosition[0]-sensorPos[:,0]) ** 2 + \
            (targPosition[1]-sensorPos[:,1]) ** 2)).reshape(-1, 1)
        deltaX = (targPosition[0]-sensorPos[:,0]).reshape(-1, 1)
        #the strange '+ (xyradial==0)' below is to prevent divide by zero
        cosyaw = ((deltaX/(xyradial + (xyradial==0))) * (xyradial!=0) + 0 * (xyradial==0))
        yaw = ysign * numpy.arccos(cosyaw)
        pitch = - numpy.arctan2((targPosition[2]-sensorPos[:,2]).reshape(-1, 1),
            xyradial).reshape(-1, 1)
        roll = numpy.zeros(yaw.shape).reshape(-1, 1)

        return (sensorPos[:,0].reshape(-1, 1), sensorPos[:,1].reshape(-1, 1), \
            sensorPos[:,2].reshape(-1, 1), roll, pitch, yaw, vertices, triangles)
    else:
        return (None, None, None, None, None, None, None, None)


##############################################################################
##
def writeRotatingTargetOssimTrajFile(filename, trajType, distance, xTargPos,
    yTargPos, zTargPos, xVel, yVel, zVel, engine, deltaTime ):
    """ Reads OFF file and create OSSIM trajectory files for rotating object
    or orbiting sensor.

    This function writes a file in the custom OSSIM trajectory file format.
    Use this function as an example on how to use the ryplotspherical
    functionality in your application.

    Two different types of trajectory files are created:
     #. **trajType = 'Rotate'**
        Calculate attitude (pitch and yaw angles only, roll is zero) to
        orientate an object's x-axis along the vertices in the OFF file.
        The location of the object is fixed at (xTargPos,yTargPos,zTargPos).

     #. **trajType = 'Orbit'**
        Calculate location and attitude (pitch and yaw angles only, roll is
        zero) of an orbiting sensor looking a a fixed location
        (xTargPos, TargPos, zTargPos) from a given distance.

    The velocity and engine settings are constant for all views at the
    values specified.

    The deltaTime parameter is used to define the time increment to be
    used in the trajectory file.

    Two additional files are also written to assist with the subsequent
    viewing.

     #. The **directions** file contains the normalised direction vectors
        between the object and observer. Depending on the trajectory type
        (see above), the sensor and object switch locations for these
        vectors. These vectors are the directions of sampled intensity values.

     #. The **triangles** file defines triangles that provides the spatial
        linking between adjacent vectors, used when plotting the data.
        We plot the complex hull comprising these triangles, with vertices
        along the direction vectors, with length given by the simulated
        data set.

    OSSIM Users: For examples of how to use these trajectory files, see
    test points tp01l (that is lowercase L) and tp01m. The scenario files
    are present in the the appropriate test point directory (l and m) and
    the plotting routines are in the tp01 utils directory.

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
        | writes a trajectory file
        | writes a triangles file
        | writes a vertices file

    Raises:
        | No exception is raised.
    """

    if trajType == 'Rotate':
        (x, y, z, roll, pitch, yaw, vertices, triangles) = \
        getRotateFromOffFile(filename, xTargPos, yTargPos, zTargPos)
    elif trajType == 'Orbit':
        (x, y, z, roll, pitch, yaw,vertices, triangles) = \
        getOrbitFromOffFile(filename, xTargPos, yTargPos, zTargPos, distance)
    else:
        print('Unkown trajectory type')
        return

    zerov = numpy.zeros(yaw.shape).reshape(-1, 1)
    onesv = numpy.ones(yaw.shape).reshape(-1, 1)

    time = numpy.array([deltaTime * i for i in range(0,zerov.shape[0])]).reshape(-1, 1)
    #time = numpy.around(time,2) # rounding does not help. internal representation!!

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
    numpy.savetxt(fid , outp)
    fid.close()

    fid = open('triangles{0}.txt'.format(outfile), 'w' )
    numpy.savetxt( fid , triangles )
    fid.close()

    fid = open('vertex{0}.txt'.format(outfile), 'w' )
    numpy.savetxt( fid , vertices )
    fid.close()

    print('Set OSSIM clock to {0} increments and max time {1}\n'.\
        format(deltaTime, deltaTime * yaw.shape[0]))

################################################################
##
def plotSpherical(dataset, vertices, triangles, ptitle='', tsize=0.4, theight=0.95):
    """Plot the spherical data given a data set, triangle set and vertex set.

    The vertex set defines the direction cosines of the individual samples.
    The triangle set defines how the surfrace must be structured between the samples.
    The data set defines, for each direction cosine, the length of the vector.

    Args:
        | dataset(numpy.array(double)): array of data set values
        | vertices(numpy.array([])): array of direction cosine vertices as [x y z]
        | triangles(numpy.array([])): array of triangles as []
        | ptitle(string): title or header for this display
        | tsize(double): title size (units not quite clear)
        | theight(double): title height (y value) (units not quite clear)

    Returns:
        | provides and mlab figure.

    Raises:
        | No exception is raised.
"""

    #calculate a (x,y,z) data set from the direction vectors
    x =  dataset * vertices[:,0]
    y =  dataset * vertices[:,1]
    z =  dataset * vertices[:,2]

    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    # Visualize the points
    pts = mlab.triangular_mesh(x, y, z, triangles )# z, scale_mode='none', scale_factor=0.2)
    mlab.title(ptitle, size=tsize, height=theight)
    mlab.show()



################################################################
##
def plotOSSIMSpherical(nColours, plottitle, datafile, vertexfile, trianglefile):
    """Plot the spherical data given a data set, triangle set and vertex set.

    The vertex set defines the direction cosines of the individual samples.
    The triangle set defines how the surfrace must be structured between the samples.
    The data set defines, for each direction cosine, the length of the vector.

    There is no means to discriminate between negative and pi phase shift.
    In this function we plot colour ratio values initially in absolute form,
    then only positive and then only negative values. In between these two
    shells the values are going through zero.

    Args:
        | nColours ([int]): selection of colours to display
        | plottitle (string): plot title or header
        | datafile (string): dataset file filename
        | vertexfile (string): vertex file filename
        | trianglefile (string): triangles file filename

    Returns:
        | provides and mlab figure.

    Raises:
        | No exception is raised.
"""
    vertices = numpy.genfromtxt(vertexfile, autostrip=True,comments='%')
    triangles = numpy.genfromtxt(trianglefile, autostrip=True,comments='%')
    radianArray = numpy.loadtxt(datafile, skiprows=1, dtype = float)
    specBand = ['8-12 um', '3-5 um', '1-2 um', '1.5-2.5 um']
    for i in nColours:
        dataset = radianArray[:,5+i]
        ptitle = '{0} {1}'.format(plottitle,specBand[i])
        plotSpherical(dataset, vertices, triangles, ptitle)

    #calculate colour ratio
    #   log() to compress the scales
    #   abs() to not loose negative values
    colourratio = numpy.log(numpy.abs(radianArray[:,6]/radianArray[:,5]))
    ptitle = '{0} {1}'.format(plottitle,'log(abs(3-5 um/8-12 um))')
    plotSpherical(colourratio, vertices, triangles, ptitle)

    colourratio = numpy.log(numpy.abs(radianArray[:,6]/radianArray[:,7]))
    ptitle = '{0} {1}'.format(plottitle,'log(abs(3-5 um/1-2 um))')
    plotSpherical(colourratio, vertices, triangles, ptitle)

    colourratio = numpy.log(radianArray[:,7]/radianArray[:,6])
    ptitle = '{0} {1}'.format(plottitle,'log(+(1-2 um/3-5 um)')
    plotSpherical(colourratio, vertices, triangles, ptitle)

    colourratio = numpy.log(-radianArray[:,7]/radianArray[:,6])
    ptitle = '{0} {1}'.format(plottitle,'log(-(1-2 um/3-5 um))')
    plotSpherical(colourratio, vertices, triangles, ptitle)


    colourratio = numpy.log(numpy.abs(radianArray[:,8]/radianArray[:,6]))
    ptitle = '{0} {1}'.format(plottitle,'log(abs(1.5-2.5 um/3-5 um))')
    plotSpherical(colourratio, vertices, triangles, ptitle)


    colourratio = numpy.log(radianArray[:,8]/radianArray[:,6])
    ptitle = '{0} {1}'.format(plottitle,'log(+1.5-2.5 um/3-5 um)')
    plotSpherical(colourratio, vertices, triangles, ptitle)

    colourratio = numpy.log(-radianArray[:,8]/radianArray[:,6])
    ptitle = '{0} {1}'.format(plottitle,'log(-(1.5-2.5 um/3-5 um))')
    plotSpherical(colourratio, vertices, triangles, ptitle)



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
    writeRotatingTargetOssimTrajFile('data/plotspherical/sphere_4_2562.off', 'Rotate',
        None, 1000, 0, -1500,0, 0, 0, 1, 0.01)

    writeRotatingTargetOssimTrajFile('data/plotspherical/sphere_4_2562.off', 'Orbit',
        1000, 0, 0, -1500,0, 0, 0, 0, 0.01)

    #plot orbiting dataset - in this case a signature from a simple jet aircraft model.
    plotOSSIMSpherical([0,1,2,3],'Orbiting','data/plotspherical/orbitIntensity2562.txt',
        'data/plotspherical/vertexsphere_4_2562.txt',
        'data/plotspherical/trianglessphere_4_2562.txt')

    plotOSSIMSpherical([0,1,2,3],'Rotating','data/plotspherical/rotateIntensity2562.txt',
        'data/plotspherical/vertexsphere_4_2562.txt',
        'data/plotspherical/trianglessphere_4_2562.txt')


