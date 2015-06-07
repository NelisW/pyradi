#  $Id$
#  $HeadURL$

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

# The Initial Developer of the Original Code is CJ Willers,
# Portions created by CJ Willers are Copyright (C) 2006-2012
# All Rights Reserved.

# Contributor(s): ______________________________________.
################################################################
"""

Please note that all mayavi-based code has been commented out.
This is because Mayavi is (1) not available in Anaconda and (2) not
yet ported to Python 3.  This capability might be brought back if 
there is a need.

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

===============  =========== ============== ===============
Filename         Resolution    Number          Number
   .              (degrees)    points         triangles
===============  =========== ============== ===============
sphere_0_12       63.4             12              20
sphere_1_42       33.9             42              80
sphere_2_162      17.2            162             320
sphere_3_642      8.6             642            1280
sphere_4_2562     4.32           2562            5120
sphere_5_10242    2.16          10242           20480
sphere_6_40962    1.08          40962           81920
sphere_7_163842   0.54         163842          327680
===============  =========== ============== ===============


The workflow is as follows:
 #. Use writeOSSIMTrajOFFFile (or your own equivalent) to
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

This package was partly developed to provide additional material in support of students 
and readers of the book Electro-Optical System Analysis and Design: A Radiometry 
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""





__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['readOffFile', 'getRotateFromOffFile', 'getOrbitFromOffFile',
        'writeOSSIMTrajOFFFile', 
        'writeOSSIMTrajElevAzim', 'getOrbitFromElevAzim','getRotateFromElevAzim', 
        'plotSpherical', 'plotOSSIMSpherical', 
        'sphericalPlotElevAzim', 'polarPlotElevAzim',
        'globePlotElevAzim','plotVertexSphere',]

import sys
if sys.version_info[0] > 2:
    print("pyradi is not yet ported to Python 3, because imported modules are not yet ported")
    exit(-1)


import os.path, fnmatch
import numpy as np
from scipy.interpolate import interp1d
#mayavi commented out
#from mayavi import mlab

##############################################################################
##
def readOffFile(filename):
    """Reads an OFF file and returns the vertices and triangles in numpy arrays.

    The OFF file is read and the data captured in the array structures.
    This is a fairly trivial reading task.

    Args:
        | filename (string): name of the OFF file

    Returns:
        | vertices(np.array([])): array of vertices as [x y z]
        | triangles(np.array([])): array of triangles as []

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
        areaFace = 4 * (np.pi)**2 / faceCount
        sizeFace = np.sqrt(areaFace / 2)
        resAngle = sizeFace
        print('sampling density is approx {0:.2f} mrad or {1:.3f} degrees.'.format(
            1000 * resAngle,resAngle * 180 / np.pi))
        # then follows vertices from lines[2] to lines[2+vertexCount]
        vertices = np.asarray([float(s) for s in lines[2].split()])
        for line in lines[3:2+vertexCount]:
            vertices = np.vstack((vertices,np.asarray([float(s) \
                for s in line.split()])))
        # now extract the triangles lines[2+vertexCount] to lines(-1)
        triangles = np.asarray([int(s) for s in lines[2+vertexCount].split()[1:]],dtype='i4')
        for line in lines[3+vertexCount:2+vertexCount+faceCount]:
            if len(line) > 0:
                triangles = np.vstack((triangles,np.asarray([int(s) \
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
        | xPos (double): object position on x axis
        | yPos (double): object position on y axis
        | zPos (double): object position on z axis


    Returns:
        | x(np.array()): array of x object location values
        | y(np.array()): array of y object location values
        | z(np.array()): array of z object location values
        | roll(np.array()): array of object location roll values
        | pitch(np.array()): array of object location pitch values
        | yaw(np.array()): array of object location yaw values
        | vertices(np.array([])): array of vertices as [x y z]
        | triangles(np.array([])): array of triangles as []

    Raises:
        | No exception is raised.
    """

    (vertices, triangles) = readOffFile(filename)

    if vertices is not None:

        ysign = (1 * (vertices[:,1] < 0) - 1 * (vertices[:,1] >= 0)).reshape(-1, 1)

        xyradial = (np.sqrt((vertices[:,0]) ** 2 + \
            (vertices[:,1]) ** 2)).reshape(-1, 1)

        deltaX = (-vertices[:,0]).reshape(-1, 1)
        #the strange '+ (xyradial==0)' below is to prevent divide by zero
        cosyaw = ((deltaX/(xyradial + (xyradial==0))) * (xyradial!=0) + 0 * (xyradial==0))
        yaw = ysign * np.arccos(cosyaw)
        pitch = - np.arctan2((-vertices[:,2]).reshape(-1, 1), xyradial).reshape(-1, 1)
        roll = np.zeros(yaw.shape).reshape(-1, 1)

        onesv = np.ones(yaw.shape).reshape(-1, 1)
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
        | x(np.array()): array of x sensor position values
        | y(np.array()): array of y sensor position values
        | z(np.array()): array of z sensor position values
        | roll(np.array()): array of sensor roll values
        | pitch(np.array()): array of sensor pitch values
        | yaw(np.array()): array of sensor yaw values
        | vertices(np.array([])): array of vertices as [x y z]
        | triangles(np.array([])): array of triangles as []

    Raises:
        | No exception is raised.
    """


    (vertices, triangles) = readOffFile(filename)

    if vertices is not None:

        targPosition = np.asarray([xTargPos, yTargPos, zTargPos])

        sensorPos = distance * vertices
        sensorPos[:,0] = sensorPos[:,0] + xTargPos
        sensorPos[:,1] = sensorPos[:,1] + yTargPos
        sensorPos[:,2] = sensorPos[:,2] + zTargPos

        ysign = (1 * (sensorPos[:,1] < 0) - 1 * (sensorPos[:,1] >= 0)).reshape(-1, 1)
        xyradial = (np.sqrt((targPosition[0]-sensorPos[:,0]) ** 2 + \
            (targPosition[1]-sensorPos[:,1]) ** 2)).reshape(-1, 1)
        deltaX = (targPosition[0]-sensorPos[:,0]).reshape(-1, 1)
        #the strange '+ (xyradial==0)' below is to prevent divide by zero
        cosyaw = ((deltaX/(xyradial + (xyradial==0))) * (xyradial!=0) + 0 * (xyradial==0))
        yaw = ysign * np.arccos(cosyaw)
        pitch = - np.arctan2((targPosition[2]-sensorPos[:,2]).reshape(-1, 1),
            xyradial).reshape(-1, 1)
        roll = np.zeros(yaw.shape).reshape(-1, 1)

        return (sensorPos[:,0].reshape(-1, 1), sensorPos[:,1].reshape(-1, 1), \
            sensorPos[:,2].reshape(-1, 1), roll, pitch, yaw, vertices, triangles)
    else:
        return (None, None, None, None, None, None, None, None)


##############################################################################
##
def writeOSSIMTrajOFFFile(filename, trajType, distance, xTargPos,
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

     #. The **vertex** file contains the normalised direction vectors
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
        | distance (double): distance from sensor to object
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

    zerov = np.zeros(yaw.shape).reshape(-1, 1)
    onesv = np.ones(yaw.shape).reshape(-1, 1)

    time = np.array([deltaTime * i for i in range(0,zerov.shape[0])]).reshape(-1, 1)
    #time = np.around(time,2) # rounding does not help. internal representation!!

    outp = time
    outp = np.hstack((outp, x))
    outp = np.hstack((outp, y))
    outp = np.hstack((outp, z))
    outp = np.hstack((outp, roll))
    outp = np.hstack((outp, yaw))
    outp = np.hstack((outp, pitch))
    outp = np.hstack((outp, xVel * onesv)) # x-velocity
    outp = np.hstack((outp, yVel * onesv)) # y-velocity
    outp = np.hstack((outp, zVel * onesv)) # z-velocity
    outp = np.hstack((outp, engine * onesv)) # engine setting

    outfile = os.path.basename(filename)
    idx=outfile.find('.')
    outfile = outfile[:idx]

    # fid = open('Trajectory{0}{1}.txt'.format(trajType,outfile), 'w' )
    fid = open('Alt{0}Range{1}{2}-{3}.lut'.format(-zTargPos,distance,trajType,outfile), 'w' )
    fid.write( 'Time x y z rol yaw pit vx vy vz engine \n' )
    fid.write( '0.0 infty infty infty infty infty infty infty infty infty infty \n' )
    fid.write( '0.0 infty infty infty infty infty infty infty infty infty infty\n' )
    np.savetxt(fid , outp)
    fid.close()

    # fid = open('triangles{0}.txt'.format(outfile), 'w' )
    fid = open('Alt{0}Range{1}{2}-{3}.dat'.format(-zTargPos,distance,'Triangles',outfile), 'w' )
    for i in range(triangles.shape[0]):
        fid.write('{0:d} {1:d} {2:d}\n'.format(triangles[i][0],triangles[i][1],triangles[i][2] ))
    # np.savetxt( fid , triangles, fmt=r'%d' )
    fid.close()

    # fid = open('vertex{0}.txt'.format(outfile), 'w' )
    fid = open('Alt{0}Range{1}{2}-{3}.dat'.format(-zTargPos,distance,'Vertices',outfile), 'w' )
    np.savetxt( fid , vertices )
    fid.close()

    print('Set OSSIM clock to {0} increments and max time {1}\n'.\
        format(deltaTime, deltaTime * yaw.shape[0]))

##############################################################################
##
def writeOSSIMTrajElevAzim(numSamplesAz,filename, trajType, distance, xTargPos,
    yTargPos, zTargPos, xVel, yVel, zVel, engine, deltaTime ):
    """ Create OSSIM trajectory files for rotating object or orbiting sensor
    for constant increments in azimuth and elevation (yaw and pitch).

    This function writes a file in the custom OSSIM trajectory file format.
    Use this function as an example on how to use the ryplotspherical
    functionality in your application.

    Two different types of trajectory files are created:
     #. **trajType = 'Rotate'**
        Calculate attitude (pitch and yaw angles only, roll is zero) to
        orientate an object's x-axis along the elevation and azimuth vectors.
        The location of the object is fixed at (xTargPos,yTargPos,zTargPos).

     #. **trajType = 'Orbit'**
        Calculate location and attitude (pitch and yaw angles only, roll is
        zero) of an orbiting sensor looking a a fixed location
        (xTargPos, TargPos, zTargPos) from a given distance along the 
        elevation and azimuth vectors.

    The velocity and engine settings are constant for all views at the
    values specified.

    The deltaTime parameter is used to define the time increment to be
    used in the trajectory file.

    Two additional files are also written to assist with the subsequent
    viewing.

     #. The **azimuth** file contains the sample values around the equator. 
        The value ranges from zero to two*pi

     #. The **elevation** file contains the sample values from North pole to South pole. 
        The value ranges from -pi to +pi

    Args:
        | numSamplesAz (int): The number of samples along the equator
        | filename (string): output file filename
        | trajType (string): type of trajectory: 'Rotate' or 'Orbit'
        | distance (double): distance from sensor to object
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
        | writes a azimuth, elevation file

    Raises:
        | No exception is raised.
    """

    #obtain odd number of samples around equator 2*pi
    if numSamplesAz % 2 == 0:
        numSamplesAz += 1
    azimuth = np.linspace(0,2 * np.pi, numSamplesAz)
    #create twice to many elevation samples, then take every second
    elevation2 = np.linspace(np.pi/2., -np.pi/2., numSamplesAz)
    elevation = elevation2[::2]

    if trajType == 'Rotate':
        (x, y, z, roll, pitch, yaw, azel) = \
        getRotateFromElevAzim(azimuth, elevation, xTargPos, yTargPos, zTargPos)
    elif trajType == 'Orbit':
        (x, y, z, roll, pitch, yaw, azel) = \
        getOrbitFromElevAzim(azimuth, elevation, xTargPos, yTargPos, zTargPos, distance)
    else:
        print('Unkown trajectory type')
        return
    zerov = np.zeros(yaw.shape).reshape(-1, 1)
    onesv = np.ones(yaw.shape).reshape(-1, 1)

    time = np.array([deltaTime * i for i in range(0,zerov.shape[0])]).reshape(-1, 1)
    #time = np.around(time,2) # rounding does not help. internal representation!!

    outp = time
    outp = np.hstack((outp, x))
    outp = np.hstack((outp, y))
    outp = np.hstack((outp, z))
    outp = np.hstack((outp, roll))
    outp = np.hstack((outp, yaw))
    outp = np.hstack((outp, pitch))
    outp = np.hstack((outp, xVel * onesv)) # x-velocity
    outp = np.hstack((outp, yVel * onesv)) # y-velocity
    outp = np.hstack((outp, zVel * onesv)) # z-velocity
    outp = np.hstack((outp, engine * onesv)) # engine setting

    outfile = os.path.basename(filename)
    idx=outfile.find('.') 
    if not idx < 0:
        outfile = outfile[:idx]

    # fid = open('Trajectory{0}{1}.txt'.format(trajType,outfile), 'w' )
    fid = open('Alt{0}Range{1}{2}-{3}-traj.lut'.format(-zTargPos,distance,trajType,outfile), 'w' )
    fid.write( 'Time x y z rol yaw pit vx vy vz engine \n' )
    fid.write( '0.0 infty infty infty infty infty infty infty infty infty infty \n' )
    fid.write( '0.0 infty infty infty infty infty infty infty infty infty infty\n' )
    np.savetxt(fid , outp)
    fid.close()

    fid = open('Alt{0}Range{1}{2}-{3}-Azel.dat'.format(-zTargPos,distance,trajType,outfile), 'w' )
    fid.write( 'Azimuth Elevation \n' )
    np.savetxt( fid, azel )
 
    print('Set OSSIM clock to {0} increments and max time {1}\n'.\
        format(deltaTime, deltaTime * yaw.shape[0]))

##############################################################################
##
def getRotateFromElevAzim(azimuth, elevation,  xPos, yPos, zPos):
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
        | azimuth (np.array(N,)): azimuth values
        | elevation (np.array(N,)): azimuth values
        | xPos (double): object position on x axis
        | yPos (double): object position on y axis
        | zPos (double): object position on z axis


    Returns:
        | x(np.array()): array of x object location values
        | y(np.array()): array of y object location values
        | z(np.array()): array of z object location values
        | roll(np.array()): array of object location roll values
        | pitch(np.array()): array of object location pitch values
        | yaw(np.array()): array of object location yaw values
        | azel(np.array()): array of azimuth,elevation values for each sample

    Raises:
        | No exception is raised.
    """
    azimgrid, elevgrid = np.meshgrid(azimuth,elevation)
    
    yaw = azimgrid.reshape(-1,1)
    pitch = elevgrid.reshape(-1,1)
    roll = np.zeros(yaw.shape).reshape(-1, 1)

    onesv = np.ones(yaw.shape).reshape(-1, 1)
    x = xPos * onesv
    y = yPos * onesv
    z = zPos * onesv

    azel = azimgrid.reshape(-1, 1)
    azel = np.hstack((azel, azimgrid.reshape(-1, 1).reshape(-1, 1)))
    return (x, y, z, roll, pitch, yaw, azel)



##############################################################################
##
def getOrbitFromElevAzim(azimuth, elevation,  xTargPos, yTargPos, zTargPos, distance):
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
        | azimuth (np.array(N,)): azimuth values
        | elevation (np.array(N,)): azimuth values
        | filename (string): OFF file filename
        | xTargPos (double): x target object position (fixed)
        | yTargPos (double): y target object position (fixed)
        | zTargPos (double): z target object position (fixed)
        | distance (double): range at which sensor orbits the target

    Returns:
        | x(np.array()): array of x sensor position values
        | y(np.array()): array of y sensor position values
        | z(np.array()): array of z sensor position values
        | roll(np.array()): array of sensor roll values
        | pitch(np.array()): array of sensor pitch values
        | yaw(np.array()): array of sensor yaw values
        | azel(np.array()): array of azimuth,elevation values for each sample

    Raises:
        | No exception is raised.
    """

    targPosition = np.asarray([xTargPos, yTargPos, zTargPos])
    print('target position {}'.format(targPosition))

    #get the sensor position from the azimuth and elevation angles
    #there must be a better way....
    firstTime = True
    for elev in elevation:
        for azim in azimuth:
            x = np.cos(azim) * np.cos(elev)
            y = np.sin(azim) * np.cos(elev)
            z = - np.sin(elev)   # NED coordinate system
            vertex = np.asarray([x, y, z])
            azelelement = np.asarray([azim, elev])
            # print(np.linalg.norm(vertex))
            if firstTime:
                azel = azelelement
                vertices = vertex
                firstTime = False
            else:
                vertices = np.vstack((vertices, vertex))
                azel = np.vstack((azel, azelelement))

    sensorPos = distance * vertices
    sensorPos[:,0] = sensorPos[:,0] + xTargPos
    sensorPos[:,1] = sensorPos[:,1] + yTargPos
    sensorPos[:,2] = sensorPos[:,2] + zTargPos

    ysign = (1 * (sensorPos[:,1] < 0) - 1 * (sensorPos[:,1] >= 0)).reshape(-1, 1)
    xyradial = (np.sqrt((targPosition[0]-sensorPos[:,0]) ** 2 + \
        (targPosition[1]-sensorPos[:,1]) ** 2)).reshape(-1, 1)
    deltaX = (targPosition[0]-sensorPos[:,0]).reshape(-1, 1)
    #the strange '+ (xyradial==0)' below is to prevent divide by zero
    cosyaw = ((deltaX/(xyradial + (xyradial==0))) * (xyradial!=0) + 0 * (xyradial==0))
    yaw = ysign * np.arccos(cosyaw)
    pitch = - np.arctan2((targPosition[2]-sensorPos[:,2]).reshape(-1, 1),
        xyradial).reshape(-1, 1)
    roll = np.zeros(yaw.shape).reshape(-1, 1)

    return (sensorPos[:,0].reshape(-1, 1), sensorPos[:,1].reshape(-1, 1), \
        sensorPos[:,2].reshape(-1, 1), roll, pitch, yaw, azel)


# #mayavi commented out
# ################################################################
# ##
# def plotSpherical(figure, dataset, vertices, triangles, ptitle='', tsize=0.4, theight=1):
#     """Plot the spherical data given a data set, triangle set and vertex set.

#     The vertex set defines the direction cosines of the individual samples.
#     The triangle set defines how the surfrace must be structured between the samples.
#     The data set defines, for each direction cosine, the length of the vector.

#     Args:
#         | figure(int): mlab figure number
#         | dataset(np.array(double)): array of data set values
#         | vertices(np.array([])): array of direction cosine vertices as [x y z]
#         | triangles(np.array([])): array of triangles as []
#         | ptitle(string): title or header for this display
#         | tsize(double): title width in in normalised figure width
#         | theight(double): title top vertical location in normalised figure height

#     Returns:
#         | provides an mlab figure.

#     Raises:
#         | No exception is raised.
# """

#     #calculate a (x,y,z) data set from the direction vectors
#     x =  dataset * vertices[:,0]
#     y =  dataset * vertices[:,1]
#     z =  dataset * vertices[:,2]

#     mlab.figure(figure, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

#     # Visualize the points
#     pts = mlab.triangular_mesh(x, y, z, triangles )# z, scale_mode='none', scale_factor=0.2)
#     mlab.title(ptitle, size=tsize, height=theight)



# #mayavi commented out
# ################################################################
# ##
# def plotOSSIMSpherical(basefigure, nColours, plottitle, datafile, vertexfile, trianglefile):
#     """Plot the spherical data given a data set, triangle set and vertex set.

#     The vertex set defines the direction cosines of the individual samples.
#     The triangle set defines how the surfrace must be structured between the samples.
#     The data set defines, for each direction cosine, the length of the vector.

#     There is no means to discriminate between negative and pi phase shift.
#     In this function we plot colour ratio values initially in absolute form,
#     then only positive and then only negative values. In between these two
#     shells the values are going through zero.

#     Args:
#         | basefigure (int): value where figure count must start
#         | nColours ([int]): selection of colours to display
#         | plottitle (string): plot title or header
#         | datafile (string): dataset file filename
#         | vertexfile (string): vertex file filename
#         | trianglefile (string): triangles file filename

#     Returns:
#         | provides an mlab figure.

#     Raises:
#         | No exception is raised.
# """
#     vertices = np.genfromtxt(vertexfile, autostrip=True,comments='%')
#     triangles = np.genfromtxt(trianglefile, autostrip=True,comments='%')
#     radianArray = np.loadtxt(datafile, skiprows=1, dtype = float)
#     specBand = ['LWIR', 'MWIR', 'SWIR1', 'SWIR2']
#     for i in nColours:
#         dataset = radianArray[:,5+i]
#         ptitle = '{0} {1}'.format(plottitle,specBand[i])
#         plotSpherical(basefigure+10+i, dataset, vertices, triangles, ptitle)

#     #calculate colour ratio
#     #   log() to compress the scales
#     #   abs() to not loose negative values
#     colourratio = np.log(np.abs(radianArray[:,6]/radianArray[:,5]))
#     ptitle = '{0} {1}'.format(plottitle,'log(abs(MWIR/LWIR))')
#     plotSpherical(basefigure+2,colourratio, vertices, triangles, ptitle)

#     colourratio = np.log(np.abs(radianArray[:,6]/radianArray[:,7]))
#     ptitle = '{0} {1}'.format(plottitle,'log(abs(MWIR/SWIR1))')
#     plotSpherical(basefigure+3,colourratio, vertices, triangles, ptitle)

#     colourratio = np.log(radianArray[:,7]/radianArray[:,6])
#     ptitle = '{0} {1}'.format(plottitle,'log(Positive ratio: +(SWIR1/MWIR)')
#     plotSpherical(basefigure+4,colourratio, vertices, triangles, ptitle)

#     colourratio = np.log(-radianArray[:,7]/radianArray[:,6])
#     ptitle = '{0} {1}'.format(plottitle,'log(Negative ratio: -(1SWIR1/MWIR))')
#     plotSpherical(basefigure+5,colourratio, vertices, triangles, ptitle)


#     colourratio = np.log(np.abs(radianArray[:,8]/radianArray[:,6]))
#     ptitle = '{0} {1}'.format(plottitle,'log(abs(SWIR2/MWIR))')
#     plotSpherical(basefigure+6,colourratio, vertices, triangles, ptitle)


#     colourratio = np.log(radianArray[:,8]/radianArray[:,6])
#     ptitle = '{0} {1}'.format(plottitle,'log(Positive ratio: +SWIR2/MWIR)')
#     plotSpherical(basefigure+7,colourratio, vertices, triangles, ptitle)

#     colourratio = np.log(-radianArray[:,8]/radianArray[:,6])
#     ptitle = '{0} {1}'.format(plottitle,'log(Negative ratio: -(SWIR2/MWIR))')
#     plotSpherical(basefigure+8,colourratio, vertices, triangles, ptitle)



# #mayavi commented out
# ################################################################
# ##
# def sphericalPlotElevAzim(figure, azimuth, elevation, value, ptitle=None,\
#     colormap='Spectral', doWireFrame=False, line_width = 0.2):
#     """Plot spherical data (azimuth, elevation and value) given in spherical format

#     This function assumes that the polar data is defined in terms of elevation
#     angle with zero at the equator and azimuth measured around the equator,
#     from 0 to 2pi.  All angles are in degrees.

#     All axes, x, y, and z are scaled with the magnitude of value in the
#     (azim, elev) direction.

#     The colour on the surface represents the value at the given azim/elev location.

#     Args:
#         | figure (int): mlab figure number
#         | azimuth (numpy 1D array): vector of values
#         | elevation (numpy 1D array): vector of values
#         | value (numpy 2D array): array with values corresponding with azim/elevation
#         | ptitle (string): plot title
#         | colormap (string): defines the colour map to be used
#         | doWireFrame (bool): switches fireframe on or off
#         | line_width (double): wireframe line width

#     Returns:
#         | provides an mlab figure.

#     Raises:
#         | No exception is raised.
# """
#     phi, theta = np.meshgrid(elevation,azimuth)
#     x = value * np.sin(phi) * np.cos(theta)
#     y = value * np.sin(phi) * np.sin(theta)
#     z = - value * np.cos(phi)
#     mlab.figure(figure, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 600))
#     mlab.clf()
#     mlab.mesh(x, y, z, scalars = value, colormap=colormap)
#     if doWireFrame:
#         mlab.mesh(x, y, z, color=(0,0,0), representation='wireframe', line_width = line_width)
#     if ptitle:
#         mlab.title(ptitle, size=0.5, height=1)


# #mayavi commented out
# ################################################################
# ##
# def polarPlotElevAzim(figure, azimuth, elevation, value, ptitle=None,\
#     colormap='Spectral', doWireFrame=False, line_width = 0.2):
#     """Plot spherical data (azimuth, elevation and value) given in polar format.

#     This function assumes that the polar data is defined in terms of elevation
#     angle with zero at the equator and azimuth measured around the equator,
#     from 0 to 2pi.  All angles are in degrees.

#     The x and y axes are scaled with the maximum magnitude of value. The z axis is
#     scaled with the actual magnitude of value in the  (azim, elev) direction. 

#     The colour on the surface represents the value at the given azim/elev location.

#     Args:
#         | figure (int): mlab figure number
#         | azimuth (numpy 1D array): vector of values
#         | elevation (numpy 1D array): vector of values
#         | value (numpy 2D array): array with values corresponding with azim/elevation
#         | ptitle (string): plot title
#         | colormap (string): defines the colour map to be used
#         | doWireFrame (bool): switches fireframe on or off
#         | line_width (double): wireframe line width

#     Returns:
#         | provides an mlab figure.

#     Raises:
#         | No exception is raised.
# """
#     line_width = 0.5
#     rmax = np.amax(np.amax(value))
#     phi, theta = np.meshgrid(elevation,azimuth)
#     x = rmax * np.sin(phi) * np.cos(theta)
#     y = rmax * np.sin(phi) * np.sin(theta)
#     z = - value * np.cos(phi)
#     mlab.figure(figure, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 600))
#     mlab.clf()
#     mlab.mesh(x, y, z, scalars = value, colormap=colormap)
#     if doWireFrame:
#         mlab.mesh(x, y, z, color=(0,0,0), representation='wireframe', line_width = line_width)
#     if ptitle:
#         mlab.title(ptitle, size=0.5, height=1)



# #mayavi commented out
# ################################################################
# ##
# def globePlotElevAzim(figure, azimuth, elevation, value, ptitle=None,\
#     colormap='Spectral', doWireFrame=False, line_width = 0.2):
#     """Plot spherical data on a sphere.

#     This function assumes that the polar data is defined in terms of elevation
#     angle with zero at the equator and azimuth measured around the equator,
#     from 0 to 2pi.  All angles are in degrees.

#     The x, y, and z values defines vertices on a sphere. The colour on the
#     sphere represents the value at the given azim/elev location.

#     Args:
#         | figure (int): mlab figure number
#         | azimuth (numpy 1D array): vector of values
#         | elevation (numpy 1D array): vector of values
#         | value (numpy 2D array): array with values corresponding with azim/elevation
#         | ptitle (string): plot title
#         | colormap (string): defines the colour map to be used
#         | doWireFrame (bool): switches fireframe on or off
#         | line_width (double): wireframe line width

#     Returns:
#         | provides an mlab figure.

#     Raises:
#         | No exception is raised.
# """
#     rmax = np.amax(np.amax(value))
#     phi, theta = np.meshgrid(elevation,azimuth)
#     x = np.sin(phi) * np.cos(theta)
#     y = np.sin(phi) * np.sin(theta)
#     z = - np.cos(phi)
#     mlab.figure(figure, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 600))
#     mlab.clf()
#     mlab.mesh(x, y, z, scalars = value, colormap=colormap)
#     if doWireFrame:
#         mlab.mesh(x, y, z, color=(0,0,0), representation='wireframe', line_width = line_width)
#     if ptitle:
#         mlab.title(ptitle, size=0.5, height=1)



# #mayavi commented out
# ################################################################
# ##
# def plotVertexSphere(figure, filename):
#     """Plot spherical data on a sphere.

#     This function assumes that the polar data is defined in terms of elevation
#     angle with zero at the equator and azimuth measured around the equator,
#     from 0 to 2pi.  All angles are in degrees.

#     The x, y, and z values defines vertices on a sphere. The colour on the
#     sphere represents the value at the given azim/elev location.

#     Args:
#         | figure (int): mlab figure number
#         | filename (string): filename for data to be plotted

#     Returns:
#         | provides an mlab figure.

#     Raises:
#         | No exception is raised.
# """
#     #load the data
#     dataset = np.loadtxt(filename)

#     #prepare the vertex vector structures
#     x = np.zeros(2)
#     y = np.zeros(2)
#     z = np.zeros(2)
#     x[0] = 0
#     y[0] = 0
#     z[0] = 0

#     #plot the vertex vectors
#     for i in range(dataset.shape[0]):
#         x[1] = dataset[i][0]
#         y[1] = dataset[i][1]
#         z[1] = dataset[i][2]
#         mlab.plot3d(x, y, z, tube_radius=0.025, colormap='Spectral')
#     #plot the three planes
#     x,y = np.mgrid[-1:1:2j, -1:1:2j]
#     z = np.zeros(x.shape)
#     mlab.surf(x,y,z, opacity=0.2,warp_scale=1,color=(1,0,0))
#     x,z = np.mgrid[-1:1:2j, -1:1:2j]
#     y = np.zeros(x.shape)
#     mlab.surf(x,y,z, opacity=0.2,warp_scale=1,color=(0,1,0))
#     z,y = np.mgrid[-1:1:2j, -1:1:2j]
#     x = np.zeros(y.shape)
#     mlab.surf(x,y,z, opacity=0.2,warp_scale=1,color=(0,0,1))

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

    xpos = 0    # m
    ypos = 0    # m
    distance = 500 # m
    engineSetting = 1 

    # zpos = -500  ;    velocityX = 72 ;     #tp14a & tp14i
    zpos = -1000  ;    velocityX = 99.7 ;     #tp14b
    # zpos = -7500  ;    velocityX = 117.3 ;     #tp14c
    # zpos = -7500  ;    velocityX = 96.8 ;     #tp14d
    # zpos = -1000  ;    velocityX = 73.5 ;    #tp14e
    # zpos = -500  ;    velocityX = 72 ;    #tp14f


    if abs(distance) > abs(zpos):
        print('Please check the altitude and distance values')
        exit(-1)

    # writeOSSIMTrajElevAzim(10,'AzEl90', 'Rotate', distance, xpos, ypos, zpos,  
    #     velocityX, 0, 0, 1, 0.01 )
    writeOSSIMTrajElevAzim(10,'AzEl10', 'Orbit', distance, xpos, ypos, zpos,  
        velocityX, 0, 0, engineSetting, 0.01 )

    #########################################################################
    print('Demo the LUT plots')

# #mayavi commented out
#     filename = 'data/plotspherical/vertexsphere_0_12.txt'
#     plotVertexSphere(1,filename)
#     mlab.show()

#     with open('data/plotspherical/source-H10-C2.dat') as f:
#         lines = f.readlines()
#         xlabel, ylabel, ptitle = lines[0].split()
#     aArray = np.loadtxt('data/plotspherical/source-H10-C2.dat', skiprows=1, dtype = float)
#     azim1D = aArray[1:,0] * (np.pi/180)
#     elev1D = aArray[0,1:] * (np.pi/180) - np.pi
#     pRad = aArray[1:,1:]
#     polarPlotElevAzim(1, azim1D, elev1D, pRad, ptitle, doWireFrame=True)
#     sphericalPlotElevAzim(2, azim1D, elev1D, pRad, ptitle, doWireFrame=True)
#     globePlotElevAzim(3, azim1D, elev1D, pRad, ptitle, doWireFrame=False)
#     mlab.show()



    print('Demo the spherical plots')
    #write the OSSIM files for a rotating target and stationary sensor/observer.
    #use the OFF file with required number of vertices.
    offFile = 'data/plotspherical/sphere_4_2562.off'
    # offFile = 'data/plotspherical/sphere_0_12.off'
    # offFile = 'data/plotspherical/sphere_1_42.off'
    # offFile = 'data/plotspherical/sphere_2_162.off'


    #the time increment  is 0.01 for each new position, velocity is zero here and
    #engine setting is 1. In this context the distance is irrelevant.
    distance = 1000
    xpos = 0
    ypos = 0
    zpos = -1000

    writeOSSIMTrajOFFFile(offFile, 'Rotate',
        distance, xpos, ypos, zpos, 0, 0, 0, 1, 0.01)


    writeOSSIMTrajOFFFile(offFile, 'Orbit',
        distance, xpos, ypos, zpos, 0, 0, 0, 0, 0.01)


# #mayavi commented out
#     #plot orbiting dataset - in this case a signature from a simple jet aircraft model.
#     plotOSSIMSpherical(0,[0,1,2,3],'Orbiting','data/plotspherical/orbitIntensity2562.txt',
#         'data/plotspherical/vertexsphere_4_2562.txt',
#         'data/plotspherical/trianglessphere_4_2562.txt')

#     plotOSSIMSpherical(100,[0,1,2,3],'Rotating','data/plotspherical/rotateIntensity2562.txt',
#         'data/plotspherical/vertexsphere_4_2562.txt',
#         'data/plotspherical/trianglessphere_4_2562.txt')

#     mlab.show()



    print('module ryplotspherical done!')
