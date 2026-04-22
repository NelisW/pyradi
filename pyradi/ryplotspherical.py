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
Please note that all Mayavi-based plotting functions have been commented out.
Mayavi is not available for Python 3 in standard environments.  The commented
blocks are retained as documentation of the original capabilities and may be
restored if Mayavi support is re-introduced.

This module provides tools for creating and viewing spherical plots.

The spherical plotting tool requires two sets of data to create the spherical
plot: the vertex locations in (x, y, z) and the spatial relationship between
the vertices (triangulation of nearest neighbours).  This spatial relationship
is required to create surface elements between the vertices.

The easiest way to create the spatial relationships between the vertices is to
use a convex-hull polygon model of an object.  The polygon or wireframe model
has the required vertices and spatial relationships.

In the original application, a series of spheres were created using MeshLab
with vertices equidistant over the surface (optimal spatial sampling).
The files were exported as OFF files and should be in the pyradi
data/plotspherical directory.  There are 6 possible input files, each with a
different number of samples on the unit sphere: 12, 42, 162, 642, 2562, 10242.

Any object with the required vertices and spatial relationships can be used;
it does not have to be an equi-sampled sphere.

Note that the spherical plot has no way to discriminate between negative values
and a pi phase shift: there is confusion between sign and direction.  This is
inherent in the conversion between Cartesian and spherical coordinates.  The
user must make provision for this, possibly by plotting only negative or only
positive values.

The data must be in the OFF wireframe format.

There are two possible trajectory file types:

 * **'Rotate'** — Stationary sensor and object with the target rotating.
   The trajectory file specifies the target trajectory.
 * **'Orbit'** — Stationary object with an orbiting sensor.
   The trajectory file specifies the sensor trajectory.

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

 1. Use ``writeOSSIMTrajOFFFile`` (or your own equivalent) to calculate the
    appropriate trajectory file.  Two additional files (vertices and triangles)
    are also created — keep these safe.
 2. Create your data set (e.g. run a simulation) using the trajectory file.
    Collect the simulation data in a format suitable for plotting.
 3. Use the simulation data together with the triangles and vertices files to
    plot the data.  The triangles and vertices set up the three-dimensional
    grid; the simulation data provides the values to be plotted on that grid.

This tool was originally developed to create trajectory files for the
Denel/CSIR OSSIM simulation.  The code was restructured for greater universal
application, but the final example is still an OSSIM case.

See the ``__main__`` block for examples of use.

This package was partly developed to provide additional material in support of
students and readers of the book Electro-Optical System Analysis and Design: A
Radiometry Perspective, Cornelius J. Willers, ISBN 9780819495693, SPIE
Monograph Volume PM236, SPIE Press, 2013.
http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""

__version__ = '0.1.0'
__author__ = 'pyradi team'
__all__ = [
    'readOffFile',
    'getRotateFromOffFile',
    'getOrbitFromOffFile',
    'writeOSSIMTrajOFFFile',
    'writeOSSIMTrajElevAzim',
    'getOrbitFromElevAzim',
    'getRotateFromElevAzim',
]

import os

import numpy as np


##############################################################################
##
def readOffFile(filename):
    """Read an OFF file and return the vertices and triangles as numpy arrays.

    The OFF file is read and the data captured in the array structures.

    Args:
        | filename (str): path to the OFF file

    Returns:
        | vertices (np.array): array of vertices, shape (N, 3)
        | triangles (np.array): array of triangle index triples, shape (M, 3)
        | Returns ``(None, None)`` if the file is not a valid OFF file or if
          the declared vertex/face counts do not match the data.

    Raises:
        | No exception is raised.
    """
    with open(filename) as f:
        lines = f.readlines()
        if lines[0].find('OFF') < 0:
            print('not an OFF file')
            return (None, None)

        counts = lines[1].split()
        vertexCount = int(counts[0])
        faceCount = int(counts[1])
        edgeCount = int(counts[2])
        print('vertexCount={0} faceCount={1} edgeCount={2}'.format(
            vertexCount, faceCount, edgeCount))

        # calculate the approximate sampling density:
        # surface of sphere divided by number of faces → size of each face → angle
        areaFace = 4 * (np.pi) ** 2 / faceCount
        sizeFace = np.sqrt(areaFace / 2)
        resAngle = sizeFace
        print('sampling density is approx {0:.2f} mrad or {1:.3f} degrees.'.format(
            1000 * resAngle, resAngle * 180 / np.pi))

        # extract vertices from lines[2] to lines[2+vertexCount]
        vertices = np.asarray([float(s) for s in lines[2].split()])
        for line in lines[3:2 + vertexCount]:
            vertices = np.vstack((vertices, np.asarray([float(s) for s in line.split()])))

        # extract triangles from lines[2+vertexCount] to end
        triangles = np.asarray(
            [int(s) for s in lines[2 + vertexCount].split()[1:]], dtype='i4')
        for line in lines[3 + vertexCount:2 + vertexCount + faceCount]:
            if len(line) > 0:
                triangles = np.vstack(
                    (triangles, np.asarray([int(s) for s in line.split()[1:]])))

        if triangles.shape[0] != faceCount or vertices.shape[0] != vertexCount:
            return (None, None)
        else:
            return (vertices, triangles)


##############################################################################
##
def getRotateFromOffFile(filename, xPos, yPos, zPos):
    """Read an OFF file and return object attitude and position for a rotating target.

    Calculate the pitch and yaw angles to point the object's X-axis towards
    each OFF file vertex direction.

    Euler order is yaw-pitch-roll, with roll equal to zero.
    Yaw is defined in the xy plane.
    Pitch is defined in the xz plane.
    Roll is defined in the yz plane.

    The object is assumed stationary at position (xPos, yPos, zPos).
    The position arrays have the same length as the attitude angle arrays,
    but all values in each individual array are identical.

    Args:
        | filename (str): OFF file path
        | xPos (float): object position on the x axis
        | yPos (float): object position on the y axis
        | zPos (float): object position on the z axis

    Returns:
        | x (np.array): array of x object location values
        | y (np.array): array of y object location values
        | z (np.array): array of z object location values
        | roll (np.array): array of object roll values (all zero)
        | pitch (np.array): array of object pitch values
        | yaw (np.array): array of object yaw values
        | vertices (np.array): array of vertices as [x y z]
        | triangles (np.array): array of triangle index triples
        | Returns a tuple of eight ``None`` values if the OFF file is invalid.

    Raises:
        | No exception is raised.
    """
    (vertices, triangles) = readOffFile(filename)

    if vertices is not None:
        ysign = (1 * (vertices[:, 1] < 0) - 1 * (vertices[:, 1] >= 0)).reshape(-1, 1)

        xyradial = (np.sqrt((vertices[:, 0]) ** 2 + (vertices[:, 1]) ** 2)).reshape(-1, 1)

        deltaX = (-vertices[:, 0]).reshape(-1, 1)
        # '+ (xyradial==0)' prevents divide-by-zero at the poles
        cosyaw = ((deltaX / (xyradial + (xyradial == 0))) * (xyradial != 0)
                  + 0 * (xyradial == 0))
        yaw = ysign * np.arccos(cosyaw)
        pitch = -np.arctan2((-vertices[:, 2]).reshape(-1, 1), xyradial).reshape(-1, 1)
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
    """Read an OFF file and return sensor attitude and position for an orbiting sensor.

    Calculate the sensor attitude and position such that the sensor always
    looks at the object located at (xTargPos, yTargPos, zTargPos) from a
    constant distance.

    Euler order is yaw-pitch-roll, with roll equal to zero.
    Yaw is defined in the xy plane.
    Pitch is defined in the xz plane.
    Roll is defined in the yz plane.

    The object is assumed stationary at position (xTargPos, yTargPos, zTargPos).

    Args:
        | filename (str): OFF file path
        | xTargPos (float): x target object position (fixed)
        | yTargPos (float): y target object position (fixed)
        | zTargPos (float): z target object position (fixed)
        | distance (float): range at which the sensor orbits the target

    Returns:
        | x (np.array): array of x sensor position values
        | y (np.array): array of y sensor position values
        | z (np.array): array of z sensor position values
        | roll (np.array): array of sensor roll values (all zero)
        | pitch (np.array): array of sensor pitch values
        | yaw (np.array): array of sensor yaw values
        | vertices (np.array): array of vertices as [x y z]
        | triangles (np.array): array of triangle index triples
        | Returns a tuple of eight ``None`` values if the OFF file is invalid.

    Raises:
        | No exception is raised.
    """
    (vertices, triangles) = readOffFile(filename)

    if vertices is not None:
        targPosition = np.asarray([xTargPos, yTargPos, zTargPos])

        sensorPos = distance * vertices
        sensorPos[:, 0] = sensorPos[:, 0] + xTargPos
        sensorPos[:, 1] = sensorPos[:, 1] + yTargPos
        sensorPos[:, 2] = sensorPos[:, 2] + zTargPos

        ysign = (1 * (sensorPos[:, 1] < 0) - 1 * (sensorPos[:, 1] >= 0)).reshape(-1, 1)
        xyradial = (np.sqrt((targPosition[0] - sensorPos[:, 0]) ** 2
                            + (targPosition[1] - sensorPos[:, 1]) ** 2)).reshape(-1, 1)
        deltaX = (targPosition[0] - sensorPos[:, 0]).reshape(-1, 1)
        # '+ (xyradial==0)' prevents divide-by-zero at the poles
        cosyaw = ((deltaX / (xyradial + (xyradial == 0))) * (xyradial != 0)
                  + 0 * (xyradial == 0))
        yaw = ysign * np.arccos(cosyaw)
        pitch = -np.arctan2(
            (targPosition[2] - sensorPos[:, 2]).reshape(-1, 1), xyradial).reshape(-1, 1)
        roll = np.zeros(yaw.shape).reshape(-1, 1)

        return (sensorPos[:, 0].reshape(-1, 1), sensorPos[:, 1].reshape(-1, 1),
                sensorPos[:, 2].reshape(-1, 1), roll, pitch, yaw, vertices, triangles)
    else:
        return (None, None, None, None, None, None, None, None)


##############################################################################
##
def writeOSSIMTrajOFFFile(filename, trajType, distance, xTargPos,
                          yTargPos, zTargPos, xVel, yVel, zVel, engine, deltaTime):
    """Read an OFF file and write OSSIM trajectory files for a rotating object or orbiting sensor.

    This function writes a file in the custom OSSIM trajectory file format.
    Use this function as an example of how to use the ryplotspherical
    functionality in your application.

    Two different types of trajectory files are created:

     1. **trajType = 'Rotate'**
        Calculate attitude (pitch and yaw angles only, roll is zero) to
        orientate an object's x-axis along the vertices in the OFF file.
        The location of the object is fixed at (xTargPos, yTargPos, zTargPos).

     2. **trajType = 'Orbit'**
        Calculate location and attitude (pitch and yaw angles only, roll is
        zero) of an orbiting sensor looking at a fixed location
        (xTargPos, yTargPos, zTargPos) from a given distance.

    The velocity and engine settings are constant for all views.

    Two additional files are also written to assist with subsequent viewing:

     1. The **vertex** file contains the normalised direction vectors between
        the object and observer.
     2. The **triangles** file defines the spatial linking between adjacent
        vectors, used when plotting the data.

    Args:
        | filename (str): OFF file path
        | trajType (str): trajectory type — 'Rotate' or 'Orbit'
        | distance (float): distance from sensor to object
        | xTargPos (float): object x position
        | yTargPos (float): object y position
        | zTargPos (float): object z position
        | xVel (float): velocity in x direction
        | yVel (float): velocity in y direction
        | zVel (float): velocity in z direction
        | engine (float): engine setting
        | deltaTime (float): sampling time increment in output file

    Returns:
        | Writes a trajectory ``.lut`` file, a triangles ``.dat`` file,
          and a vertices ``.dat`` file to the current directory.
        | Returns ``None`` for an unknown ``trajType``.

    Raises:
        | No exception is raised.
    """
    if trajType == 'Rotate':
        (x, y, z, roll, pitch, yaw, vertices, triangles) = \
            getRotateFromOffFile(filename, xTargPos, yTargPos, zTargPos)
    elif trajType == 'Orbit':
        (x, y, z, roll, pitch, yaw, vertices, triangles) = \
            getOrbitFromOffFile(filename, xTargPos, yTargPos, zTargPos, distance)
    else:
        print('Unkown trajectory type')
        return

    zerov = np.zeros(yaw.shape).reshape(-1, 1)
    onesv = np.ones(yaw.shape).reshape(-1, 1)

    time = np.array([deltaTime * i for i in range(0, zerov.shape[0])]).reshape(-1, 1)

    outp = time
    outp = np.hstack((outp, x))
    outp = np.hstack((outp, y))
    outp = np.hstack((outp, z))
    outp = np.hstack((outp, roll))
    outp = np.hstack((outp, yaw))
    outp = np.hstack((outp, pitch))
    outp = np.hstack((outp, xVel * onesv))    # x-velocity
    outp = np.hstack((outp, yVel * onesv))    # y-velocity
    outp = np.hstack((outp, zVel * onesv))    # z-velocity
    outp = np.hstack((outp, engine * onesv))  # engine setting

    print(np.mean(x))
    print(np.mean(y))
    print(np.mean(z))

    outfile = os.path.basename(filename)
    idx = outfile.find('.')
    outfile = outfile[:idx]

    fid = open('Alt{0}Range{1}{2}-{3}.lut'.format(
        -zTargPos, distance, trajType, outfile), 'w')
    fid.write('Time x y z rol yaw pit vx vy vz engine \n')
    fid.write('0.0 infty infty infty infty infty infty infty infty infty infty \n')
    fid.write('0.0 infty infty infty infty infty infty infty infty infty infty\n')
    np.savetxt(fid, outp)
    fid.close()

    fid = open('Alt{0}Range{1}{2}-{3}.dat'.format(
        -zTargPos, distance, 'Triangles', outfile), 'w')
    for i in range(triangles.shape[0]):
        fid.write('{0:d} {1:d} {2:d}\n'.format(
            triangles[i][0], triangles[i][1], triangles[i][2]))
    fid.close()

    fid = open('Alt{0}Range{1}{2}-{3}.dat'.format(
        -zTargPos, distance, 'Vertices', outfile), 'w')
    np.savetxt(fid, vertices)
    fid.close()

    print('Set OSSIM clock to {0} increments and max time {1}\n'.format(
        deltaTime, deltaTime * yaw.shape[0]))


##############################################################################
##
def writeOSSIMTrajElevAzim(numSamplesAz, filename, trajType, distance, xTargPos,
                           yTargPos, zTargPos, xVel, yVel, zVel, engine, deltaTime):
    """Write OSSIM trajectory files for rotating object or orbiting sensor using elevation/azimuth sampling.

    This function writes a file in the custom OSSIM trajectory file format using
    constant increments in azimuth and elevation (yaw and pitch).

    Two different types of trajectory files are created:

     1. **trajType = 'Rotate'**
        Calculate attitude (pitch and yaw only, roll is zero) to orientate an
        object's x-axis along the elevation and azimuth vectors.
        The location of the object is fixed at (xTargPos, yTargPos, zTargPos).

     2. **trajType = 'Orbit'**
        Calculate location and attitude (pitch and yaw only, roll is zero) of
        an orbiting sensor looking at a fixed location from a given distance,
        along the elevation and azimuth vectors.

    The velocity and engine settings are constant for all views.

    Two additional files are also written:

     1. The **azimuth** file contains sample values around the equator (0 to 2π).
     2. The **elevation** file contains sample values from north to south pole (−π/2 to +π/2).

    Args:
        | numSamplesAz (int): number of samples along the equator
        | filename (str): output file base name
        | trajType (str): trajectory type — 'Rotate' or 'Orbit'
        | distance (float): distance from sensor to object
        | xTargPos (float): object x position
        | yTargPos (float): object y position
        | zTargPos (float): object z position
        | xVel (float): velocity in x direction
        | yVel (float): velocity in y direction
        | zVel (float): velocity in z direction
        | engine (float): engine setting
        | deltaTime (float): sampling time increment in output file

    Returns:
        | Writes a trajectory ``.lut`` file and an azimuth/elevation ``.dat``
          file to the current directory.
        | Returns ``None`` for an unknown ``trajType``.

    Raises:
        | No exception is raised.
    """
    # ensure an odd number of samples around the equator
    if numSamplesAz % 2 == 0:
        numSamplesAz += 1
    azimuth = np.linspace(0, 2 * np.pi, numSamplesAz)
    # create twice as many elevation samples, then take every second
    elevation2 = np.linspace(np.pi / 2., -np.pi / 2., numSamplesAz)
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

    time = np.array([deltaTime * i for i in range(0, zerov.shape[0])]).reshape(-1, 1)

    outp = time
    outp = np.hstack((outp, x))
    outp = np.hstack((outp, y))
    outp = np.hstack((outp, z))
    outp = np.hstack((outp, roll))
    outp = np.hstack((outp, yaw))
    outp = np.hstack((outp, pitch))
    outp = np.hstack((outp, xVel * onesv))    # x-velocity
    outp = np.hstack((outp, yVel * onesv))    # y-velocity
    outp = np.hstack((outp, zVel * onesv))    # z-velocity
    outp = np.hstack((outp, engine * onesv))  # engine setting

    outfile = os.path.basename(filename)
    idx = outfile.find('.')
    if not idx < 0:
        outfile = outfile[:idx]

    fid = open('Alt{0}Range{1}{2}-{3}-traj.lut'.format(
        -zTargPos, distance, trajType, outfile), 'w')
    fid.write('Time x y z rol yaw pit vx vy vz engine \n')
    fid.write('0.0 infty infty infty infty infty infty infty infty infty infty \n')
    fid.write('0.0 infty infty infty infty infty infty infty infty infty infty\n')
    np.savetxt(fid, outp)
    fid.close()

    fid = open('Alt{0}Range{1}{2}-{3}-Azel.dat'.format(
        -zTargPos, distance, trajType, outfile), 'w')
    fid.write('Azimuth Elevation \n')
    np.savetxt(fid, azel)
    fid.close()

    print('Set OSSIM clock to {0} increments and max time {1}\n'.format(
        deltaTime, deltaTime * yaw.shape[0]))


##############################################################################
##
def getRotateFromElevAzim(azimuth, elevation, xPos, yPos, zPos):
    """Return object attitude and position for a rotating target on an azimuth/elevation grid.

    Calculate the pitch and yaw angles to point the object's X-axis towards
    each (azimuth, elevation) direction.

    Euler order is yaw-pitch-roll, with roll equal to zero.
    Yaw is defined in the xy plane.
    Pitch is defined in the xz plane.
    Roll is defined in the yz plane.

    The object is assumed stationary at position (xPos, yPos, zPos).
    The position arrays have the same length as the attitude angle arrays,
    but all values in each individual array are identical.

    Args:
        | azimuth (np.array[N,]): azimuth values in radians
        | elevation (np.array[M,]): elevation values in radians
        | xPos (float): object position on the x axis
        | yPos (float): object position on the y axis
        | zPos (float): object position on the z axis

    Returns:
        | x (np.array): array of x object location values, shape (N*M, 1)
        | y (np.array): array of y object location values, shape (N*M, 1)
        | z (np.array): array of z object location values, shape (N*M, 1)
        | roll (np.array): array of object roll values (all zero), shape (N*M, 1)
        | pitch (np.array): array of object pitch values, shape (N*M, 1)
        | yaw (np.array): array of object yaw values, shape (N*M, 1)
        | azel (np.array): array of (azimuth, elevation) for each sample, shape (N*M, 2)

    Raises:
        | No exception is raised.
    """
    azimgrid, elevgrid = np.meshgrid(azimuth, elevation)

    yaw = azimgrid.reshape(-1, 1)
    pitch = elevgrid.reshape(-1, 1)
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
def getOrbitFromElevAzim(azimuth, elevation, xTargPos, yTargPos, zTargPos, distance):
    """Return sensor attitude and position for an orbiting sensor on an azimuth/elevation grid.

    Calculate the sensor attitude and position such that the sensor always
    looks at the object located at (xTargPos, yTargPos, zTargPos) from a
    constant distance, sampling at each (azimuth, elevation) grid point.

    Euler order is yaw-pitch-roll, with roll equal to zero.
    Yaw is defined in the xy plane.
    Pitch is defined in the xz plane.
    Roll is defined in the yz plane.

    The object is assumed stationary at position (xTargPos, yTargPos, zTargPos).

    Args:
        | azimuth (np.array[N,]): azimuth values in radians
        | elevation (np.array[M,]): elevation values in radians
        | xTargPos (float): x target object position (fixed)
        | yTargPos (float): y target object position (fixed)
        | zTargPos (float): z target object position (fixed)
        | distance (float): range at which the sensor orbits the target

    Returns:
        | x (np.array): array of x sensor position values, shape (N*M, 1)
        | y (np.array): array of y sensor position values, shape (N*M, 1)
        | z (np.array): array of z sensor position values, shape (N*M, 1)
        | roll (np.array): array of sensor roll values (all zero), shape (N*M, 1)
        | pitch (np.array): array of sensor pitch values, shape (N*M, 1)
        | yaw (np.array): array of sensor yaw values, shape (N*M, 1)
        | azel (np.array): array of (azimuth, elevation) for each sample, shape (N*M, 2)

    Raises:
        | No exception is raised.
    """
    targPosition = np.asarray([xTargPos, yTargPos, zTargPos])
    print('target position {}'.format(targPosition))

    # build sensor position from the azimuth/elevation angles
    firstTime = True
    for elev in elevation:
        for azim in azimuth:
            x = np.cos(azim) * np.cos(elev)
            y = np.sin(azim) * np.cos(elev)
            z = -np.sin(elev)   # NED coordinate system
            vertex = np.asarray([x, y, z])
            azelelement = np.asarray([azim, elev])
            if firstTime:
                azel = azelelement
                vertices = vertex
                firstTime = False
            else:
                vertices = np.vstack((vertices, vertex))
                azel = np.vstack((azel, azelelement))

    sensorPos = distance * vertices
    sensorPos[:, 0] = sensorPos[:, 0] + xTargPos
    sensorPos[:, 1] = sensorPos[:, 1] + yTargPos
    sensorPos[:, 2] = sensorPos[:, 2] + zTargPos

    ysign = (1 * (sensorPos[:, 1] < 0) - 1 * (sensorPos[:, 1] >= 0)).reshape(-1, 1)
    xyradial = (np.sqrt((targPosition[0] - sensorPos[:, 0]) ** 2
                        + (targPosition[1] - sensorPos[:, 1]) ** 2)).reshape(-1, 1)
    deltaX = (targPosition[0] - sensorPos[:, 0]).reshape(-1, 1)
    # '+ (xyradial==0)' prevents divide-by-zero at the poles
    cosyaw = ((deltaX / (xyradial + (xyradial == 0))) * (xyradial != 0)
              + 0 * (xyradial == 0))
    yaw = ysign * np.arccos(cosyaw)
    pitch = -np.arctan2(
        (targPosition[2] - sensorPos[:, 2]).reshape(-1, 1), xyradial).reshape(-1, 1)
    roll = np.zeros(yaw.shape).reshape(-1, 1)

    return (sensorPos[:, 0].reshape(-1, 1), sensorPos[:, 1].reshape(-1, 1),
            sensorPos[:, 2].reshape(-1, 1), roll, pitch, yaw, azel)


# #mayavi commented out
# ################################################################
# ##
# def plotSpherical(figure, dataset, vertices, triangles, ptitle='', tsize=0.4, theight=1):
#     """Plot the spherical data given a data set, triangle set and vertex set.
#
#     The vertex set defines the direction cosines of the individual samples.
#     The triangle set defines how the surface must be structured between the samples.
#     The data set defines, for each direction cosine, the length of the vector.
#
#     Args:
#         | figure(int): mlab figure number
#         | dataset(np.array(double)): array of data set values
#         | vertices(np.array([])): array of direction cosine vertices as [x y z]
#         | triangles(np.array([])): array of triangles as []
#         | ptitle(string): title or header for this display
#         | tsize(double): title width in normalised figure width
#         | theight(double): title top vertical location in normalised figure height
#
#     Returns:
#         | provides an mlab figure.
#
#     Raises:
#         | No exception is raised.
# """
#     x =  dataset * vertices[:,0]
#     y =  dataset * vertices[:,1]
#     z =  dataset * vertices[:,2]
#     mlab.figure(figure, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
#     pts = mlab.triangular_mesh(x, y, z, triangles)
#     mlab.title(ptitle, size=tsize, height=theight)


# #mayavi commented out
# ################################################################
# ##
# def plotOSSIMSpherical(basefigure, nColours, plottitle, datafile, vertexfile, trianglefile):
#     """Plot the spherical data given a data set, triangle set and vertex set.
#
#     Args:
#         | basefigure (int): value where figure count must start
#         | nColours ([int]): selection of colours to display
#         | plottitle (string): plot title or header
#         | datafile (string): dataset file filename
#         | vertexfile (string): vertex file filename
#         | trianglefile (string): triangles file filename
#
#     Returns:
#         | provides an mlab figure.
#
#     Raises:
#         | No exception is raised.
# """
#     vertices = np.genfromtxt(vertexfile, autostrip=True, comments='%')
#     triangles = np.genfromtxt(trianglefile, autostrip=True, comments='%')
#     radianArray = np.loadtxt(datafile, skiprows=1, dtype=float)
#     specBand = ['LWIR', 'MWIR', 'SWIR1', 'SWIR2']
#     for i in nColours:
#         dataset = radianArray[:, 5 + i]
#         ptitle = '{0} {1}'.format(plottitle, specBand[i])
#         plotSpherical(basefigure + 10 + i, dataset, vertices, triangles, ptitle)
#     colourratio = np.log(np.abs(radianArray[:, 6] / radianArray[:, 5]))
#     ptitle = '{0} {1}'.format(plottitle, 'log(abs(MWIR/LWIR))')
#     plotSpherical(basefigure + 2, colourratio, vertices, triangles, ptitle)
#     colourratio = np.log(np.abs(radianArray[:, 6] / radianArray[:, 7]))
#     ptitle = '{0} {1}'.format(plottitle, 'log(abs(MWIR/SWIR1))')
#     plotSpherical(basefigure + 3, colourratio, vertices, triangles, ptitle)


# #mayavi commented out
# ################################################################
# ##
# def sphericalPlotElevAzim(figure, azimuth, elevation, value, ptitle=None,
#                           colormap='Spectral', doWireFrame=False, line_width=0.2):
#     """Plot spherical data (azimuth, elevation, value) in spherical format.
#
#     Args:
#         | figure (int): mlab figure number
#         | azimuth (numpy 1D array): vector of azimuth values
#         | elevation (numpy 1D array): vector of elevation values
#         | value (numpy 2D array): values at each (azim, elev) location
#         | ptitle (string): plot title
#         | colormap (string): colour map
#         | doWireFrame (bool): enable wireframe overlay
#         | line_width (double): wireframe line width
#
#     Returns:
#         | provides an mlab figure.
# """
#     phi, theta = np.meshgrid(elevation, azimuth)
#     x = value * np.sin(phi) * np.cos(theta)
#     y = value * np.sin(phi) * np.sin(theta)
#     z = -value * np.cos(phi)
#     mlab.figure(figure, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 600))
#     mlab.clf()
#     mlab.mesh(x, y, z, scalars=value, colormap=colormap)
#     if doWireFrame:
#         mlab.mesh(x, y, z, color=(0, 0, 0), representation='wireframe', line_width=line_width)
#     if ptitle:
#         mlab.title(ptitle, size=0.5, height=1)


# #mayavi commented out
# ################################################################
# ##
# def polarPlotElevAzim(figure, azimuth, elevation, value, ptitle=None,
#                       colormap='Spectral', doWireFrame=False, line_width=0.2):
#     """Plot spherical data (azimuth, elevation, value) in polar format.
#
#     Args:
#         | figure (int): mlab figure number
#         | azimuth (numpy 1D array): vector of azimuth values
#         | elevation (numpy 1D array): vector of elevation values
#         | value (numpy 2D array): values at each (azim, elev) location
#         | ptitle (string): plot title
#         | colormap (string): colour map
#         | doWireFrame (bool): enable wireframe overlay
#         | line_width (double): wireframe line width
#
#     Returns:
#         | provides an mlab figure.
# """
#     rmax = np.amax(np.amax(value))
#     phi, theta = np.meshgrid(elevation, azimuth)
#     x = rmax * np.sin(phi) * np.cos(theta)
#     y = rmax * np.sin(phi) * np.sin(theta)
#     z = -value * np.cos(phi)
#     mlab.figure(figure, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 600))
#     mlab.clf()
#     mlab.mesh(x, y, z, scalars=value, colormap=colormap)
#     if doWireFrame:
#         mlab.mesh(x, y, z, color=(0, 0, 0), representation='wireframe', line_width=line_width)
#     if ptitle:
#         mlab.title(ptitle, size=0.5, height=1)


# #mayavi commented out
# ################################################################
# ##
# def globePlotElevAzim(figure, azimuth, elevation, value, ptitle=None,
#                       colormap='Spectral', doWireFrame=False, line_width=0.2):
#     """Plot spherical data on a globe.
#
#     Args:
#         | figure (int): mlab figure number
#         | azimuth (numpy 1D array): vector of azimuth values
#         | elevation (numpy 1D array): vector of elevation values
#         | value (numpy 2D array): values at each (azim, elev) location
#         | ptitle (string): plot title
#         | colormap (string): colour map
#         | doWireFrame (bool): enable wireframe overlay
#         | line_width (double): wireframe line width
#
#     Returns:
#         | provides an mlab figure.
# """
#     phi, theta = np.meshgrid(elevation, azimuth)
#     x = np.sin(phi) * np.cos(theta)
#     y = np.sin(phi) * np.sin(theta)
#     z = -np.cos(phi)
#     mlab.figure(figure, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 600))
#     mlab.clf()
#     mlab.mesh(x, y, z, scalars=value, colormap=colormap)
#     if doWireFrame:
#         mlab.mesh(x, y, z, color=(0, 0, 0), representation='wireframe', line_width=line_width)
#     if ptitle:
#         mlab.title(ptitle, size=0.5, height=1)


# #mayavi commented out
# ################################################################
# ##
# def plotVertexSphere(figure, filename):
#     """Plot vertex vectors on a unit sphere.
#
#     Args:
#         | figure (int): mlab figure number
#         | filename (string): filename for vertex data to be plotted
#
#     Returns:
#         | provides an mlab figure.
# """
#     dataset = np.loadtxt(filename)
#     x = np.zeros(2)
#     y = np.zeros(2)
#     z = np.zeros(2)
#     x[0] = 0
#     y[0] = 0
#     z[0] = 0
#     for i in range(dataset.shape[0]):
#         x[1] = dataset[i][0]
#         y[1] = dataset[i][1]
#         z[1] = dataset[i][2]
#         mlab.plot3d(x, y, z, tube_radius=0.025, colormap='Spectral')
#     x, y = np.mgrid[-1:1:2j, -1:1:2j]
#     z = np.zeros(x.shape)
#     mlab.surf(x, y, z, opacity=0.2, warp_scale=1, color=(1, 0, 0))
#     x, z = np.mgrid[-1:1:2j, -1:1:2j]
#     y = np.zeros(x.shape)
#     mlab.surf(x, y, z, opacity=0.2, warp_scale=1, color=(0, 1, 0))
#     z, y = np.mgrid[-1:1:2j, -1:1:2j]
#     x = np.zeros(y.shape)
#     mlab.surf(x, y, z, opacity=0.2, warp_scale=1, color=(0, 0, 1))

################################################################
##

if __name__ == '__main__':

    import math
    import sys
    from scipy.interpolate import interp1d

    import pyradi.ryplanck as ryplanck
    import pyradi.ryplot as ryplot
    import pyradi.ryfiles as ryfiles

    xpos = 0      # m
    ypos = 0      # m
    distance = 500  # m
    engineSetting = 1

    zpos = -1000  ;    velocityX = 99.7  # tp14b
    # zpos = -500  ;    velocityX = 72    # tp14a & tp14i
    # zpos = -7500  ;    velocityX = 117.3  # tp14c
    # zpos = -7500  ;    velocityX = 96.8   # tp14d
    # zpos = -1000  ;    velocityX = 73.5   # tp14e
    # zpos = -500  ;    velocityX = 72      # tp14f

    if abs(distance) > abs(zpos):
        print('Please check the altitude and distance values')
        exit(-1)

    writeOSSIMTrajElevAzim(10, 'AzEl10', 'Orbit', distance, xpos, ypos, zpos,
        velocityX, 0, 0, engineSetting, 0.01)

    #########################################################################
    print('Demo the LUT plots')

# #mayavi commented out
#     filename = 'data/plotspherical/vertexsphere_0_12.txt'
#     plotVertexSphere(1,filename)
#     mlab.show()

    print('Demo the spherical plots')
    # write the OSSIM files for a rotating target and stationary sensor/observer.
    # use the OFF file with the required number of vertices.
    offFile = 'data/plotspherical/sphere_4_2562.off'
    # offFile = 'data/plotspherical/sphere_0_12.off'
    # offFile = 'data/plotspherical/sphere_1_42.off'
    # offFile = 'data/plotspherical/sphere_2_162.off'

    # time increment is 0.01 for each new position; velocity and engine setting
    # are irrelevant for the trajectory geometry.
    distance = 1000
    xpos = 0
    ypos = 0
    zpos = -1000

    writeOSSIMTrajOFFFile(offFile, 'Rotate',
        distance, xpos, ypos, zpos, 0, 0, 0, 1, 0.01)

    writeOSSIMTrajOFFFile(offFile, 'Orbit',
        distance, xpos, ypos, zpos, 0, 0, 0, 0, 0.01)

# #mayavi commented out
#     plotOSSIMSpherical(0, [0,1,2,3], 'Orbiting',
#         'data/plotspherical/orbitIntensity2562.txt',
#         'data/plotspherical/vertexsphere_4_2562.txt',
#         'data/plotspherical/trianglessphere_4_2562.txt')
#     mlab.show()

    print('module ryplotspherical done!')
