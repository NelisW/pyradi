# -*- coding: utf-8 -*-

################################################################
# The contents of this file are subject to the BSD 3-Clause License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://directory.fsf.org/wiki/License:BSD_3Clause
#
# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
# License for the specific language governing rights and limitations
# under the License.
#
# The Original Code is part of the PyRadi toolkit.
#
# The Initial Developer of the Original Code is CJ Willers.
# Portions created by CJ Willers are Copyright (C) 2006-2012.
# All Rights Reserved.
################################################################
"""Scene orientation locus modelling using quaternion SLERP interpolation.

Provides the :class:`Scene` class for reading a time-stamped 6-DOF sensor
locus from a CSV file, converting Euler angles to quaternions (via the
``transformations`` library by Christoph Gohlke), and spherical-linear
(SLERP) interpolating orientation between time steps.

The ``transformations`` library follows the "column vectors on the right"
and "row major storage" (C contiguous) conventions.  Angles are in radians
unless stated otherwise.  Quaternions w+ix+jy+kz are represented as
``[w, x, y, z]``.

Euler angle sequence used throughout: yaw (z), pitch (y), roll (x) --
encoded as ``'rzyx'`` in the ``transformations`` API.  Do **not** change
this sequence without updating :attr:`Scene.sequence` and all callers.

See the ``__main__`` block for a worked usage example.

This package was partly developed to provide additional material in support
of students and readers of *Electro-Optical System Analysis and Design: A
Radiometry Perspective*, C. J. Willers, ISBN 9780819495693, SPIE Monograph
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423
"""

__version__ = '0.1.0'
__author__ = 'pyradi team'
__all__ = ['Scene', 'df_euler2quat']


import numpy as np
import pandas as pd
import transformations as tr


def df_euler2quat(row):
    """Convert a DataFrame row's Euler angles to a quaternion.

    Intended to be called via ``DataFrame.apply(df_euler2quat, axis=1)``.
    The result is wrapped in a list so that pandas stores the entire
    quaternion array as a single cell value rather than expanding it into
    separate columns.

    Args:
        row: A ``pandas.Series`` with at minimum the keys ``'yawRad'``,
            ``'pitchRad'``, ``'rollRad'``, and ``'sequence'``.

    Returns:
        A one-element list whose sole item is a ``(4,)`` numpy array
        ``[w, x, y, z]`` representing the quaternion.  The list wrapper
        is required by pandas -- do not remove it.
    """
    q = [tr.quaternion_from_euler(row['yawRad'], row['pitchRad'], row['rollRad'], row['sequence'])]
    return q


class Scene:
    """Model a sensor's 6-DOF orientation locus and interpolate between poses.

    Reads a time-stamped position and orientation CSV file, converts Euler
    angles to quaternions, and provides SLERP-based interpolation so that
    the sensor orientation can be queried at arbitrary times.

    Coordinate system: Right-Hand, North (x), East (y), Down (z).

    * **Yaw / Azimuth** -- positive from x to y; clockwise when viewed along +z.
    * **Roll** -- positive from y to z; clockwise when viewed along +x.
    * **Pitch / Elevation** -- positive from z to x; clockwise when viewed along +y.

    The Euler sequence used to map camera axes into the world frame is
    yaw (z) -> pitch (y) -> roll (x), encoded as ``'rzyx'``.

    Attributes:
        dfLocus: ``pandas.DataFrame`` loaded by :meth:`readLocus`, or
            ``None`` before the first call.
        sequence: Euler-angle sequence string used by the
            ``transformations`` library (always ``'rzyx'``).
    """

    def __init__(self):
        """Initialise an empty Scene with no locus data."""
        self.dfLocus = None
        self.sequence = 'rzyx'  # do not change -- see df_euler2quat()

    def readLocus(self, filename):
        """Read a locus CSV file and compute derived angle and quaternion columns.

        The CSV must have no spaces and use the following column layout
        (time in seconds, x/y/z in metres, angles in degrees)::

            time,x,y,z,yawDeg,pitchDeg,rollDeg
            0,0,0,0,0,0,0
            1,0,0,0,30,0,0
            2,0,0,0,30,15,0
            3,0,0,0,30,15,20
            4,0,0,0,0,0,0

        After loading, the DataFrame gains four extra columns:

        * ``yawRad``, ``pitchRad``, ``rollRad`` -- angles converted to radians.
        * ``sequence`` -- the Euler sequence string (``'rzyx'``) on every row.
        * ``quat`` -- a list containing the ``(4,)`` quaternion array for that row.

        Args:
            filename: Path to the locus CSV file.
        """
        print(filename)
        self.dfLocus = pd.read_csv(filename)
        self.dfLocus['yawRad'] = self.dfLocus['yawDeg'] * np.pi / 180.
        self.dfLocus['rollRad'] = self.dfLocus['rollDeg'] * np.pi / 180.
        self.dfLocus['pitchRad'] = self.dfLocus['pitchDeg'] * np.pi / 180.
        self.dfLocus['sequence'] = self.sequence
        self.dfLocus['quat'] = self.dfLocus.apply(df_euler2quat, axis=1)

    def interpLocus(self, ltime):
        """Interpolate the sensor orientation at an arbitrary time using SLERP.

        Finds the two locus rows that bracket ``ltime``, computes the
        parametric fraction *t* between them, and spherically interpolates
        the quaternions.

        Args:
            ltime: Query time in seconds.  Must lie strictly between the
                first and last times recorded in :attr:`dfLocus`.

        Returns:
            A 4-tuple ``(yaw, pitch, roll, q)`` where *yaw*, *pitch*, and
            *roll* are the interpolated Euler angles in radians and *q* is
            the interpolated ``(4,)`` quaternion array.
        """
        # find the two rows spanning the current value for time
        self.dfLocus['uidx'] = np.where(
            np.sign(self.dfLocus['time'] - ltime).diff() > 0, True, False
        )
        # uidx is the label index of the upper bound; uidx-1 is the lower bound
        uidx = self.dfLocus[self.dfLocus['uidx']].index.tolist()[0]
        # get parametric value between lower and upper, based on time
        t = (
            (ltime - self.dfLocus.loc[uidx - 1]['time'])
            / (self.dfLocus.loc[uidx]['time'] - self.dfLocus.loc[uidx - 1]['time'])
        )
        # get the two quats at lower and upper
        lq = self.dfLocus.loc[uidx - 1]['quat']
        uq = self.dfLocus.loc[uidx]['quat']
        # SLERP between the two quaternions
        # http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/
        q = tr.quaternion_slerp(lq[0], uq[0], t)
        yaw, pit, rol = tr.euler_from_quaternion(q, axes=self.sequence)
        return yaw, pit, rol, q


################################################################
# Evaluation helpers -- not part of the public library API.
# These functions are intended for use from the __main__ block.
################################################################

def unitxrotate(sc):
    """Rotate a unit vector along the x-axis through each locus pose.

    Prints the rotation quaternion and rotated point for every row in
    ``sc.dfLocus``.  This is a diagnostic/visualisation helper; it is
    not part of the public API.

    Args:
        sc: A :class:`Scene` instance with :attr:`~Scene.dfLocus` loaded.
    """
    # define a quaternion pointing at (1, 0, 0)
    pt = np.asarray([0, 1, 0, 0])
    print(pt)
    for idx, item in sc.dfLocus.iterrows():
        print(10 * '-')
        rts = tr.quaternion_from_euler(
            item['yawRad'], item['pitchRad'], item['rollRad'], sc.sequence
        )
        print(f'rot quat={rts}')
        pts = tr.quaternion_multiply(
            rts, tr.quaternion_multiply(pt, tr.quaternion_inverse(rts))
        )
        print(f'point={pts}')
        print(f'yaw/pit/rol={tr.euler_from_quaternion(rts, sc.sequence)}')
    print('\n---------------------\n')


def applyInterpLocus(sc, t):
    """Interpolate the locus at time *t* and return a labelled Series.

    Intended to be used with ``DataFrame.apply`` when the time column is
    passed as the argument, with ``sc`` bound via a lambda or partial::

        dfi['result'] = dfi['time'].apply(lambda t: applyInterpLocus(sc, t))

    Args:
        sc: A :class:`Scene` instance with :attr:`~Scene.dfLocus` loaded.
        t: Query time in seconds.

    Returns:
        A ``pandas.Series`` with keys ``'yaw'``, ``'pit'``, ``'rol'``,
        and ``'quat'``.
    """
    yaw, pit, rol, q = sc.interpLocus(t)
    return pd.Series({'yaw': yaw, 'pit': pit, 'rol': rol, 'quat': [q]})


def locusPlot(sc, ryplot):
    """Plot interpolated yaw, pitch, and roll over the full locus time span.

    Produces a ``ryscene-quatslerp.png`` figure showing the three Euler
    angles sampled at 50 uniformly-spaced times across the locus range,
    annotated with the raw locus table.

    Args:
        sc: A :class:`Scene` instance with :attr:`~Scene.dfLocus` loaded.
        ryplot: The ``pyradi.ryplot`` module (passed explicitly to avoid a
            module-level import in production code).
    """
    dfi = pd.DataFrame()
    numSamples = 50
    dfi['time'] = np.linspace(
        np.min(sc.dfLocus['time']), np.max(sc.dfLocus['time']), numSamples
    )
    dfi = dfi.merge(
        dfi['time'].apply(lambda t: applyInterpLocus(sc, t)),
        left_index=True, right_index=True,
    )
    p = ryplot.Plotter(1, 1, 1, 'Test slerp', figsize=(8, 5))
    p.plot(1, dfi['time'], dfi['yaw'], label=['yaw'])
    p.plot(1, dfi['time'], dfi['pit'], label=['pit'])
    p.plot(1, dfi['time'], dfi['rol'], '', 'Time [s]', 'Angle [rad]', label=['rol'])
    yoffset = p.getYLim(1)[0] + 0.05 * (p.getYLim(1)[1] - p.getYLim(1)[0])
    xoffset = p.getXLim(1)[0] + 0.05 * (p.getXLim(1)[1] - p.getXLim(1)[0])
    pstr = sc.dfLocus[['time', 'yawRad', 'pitchRad', 'rollRad']].to_string()
    p.getSubPlot(1).text(
        xoffset, yoffset, pstr,
        horizontalalignment='left',
        verticalalignment='bottom',
        family='monospace',
        fontsize=10,
    )
    p.saveFig('ryscene-quatslerp.png')


################################################################

if __name__ == '__main__':

    import pyradi.ryplot as ryplot

    # create a scene object and load locus from csv file
    sc = Scene()
    sc.readLocus('data/scene/sensorlocus.csv')
    print(sc.dfLocus)

    # rotate a unit vector along x-axis through each pose
    unitxrotate(sc)

    # interpolate and plot locus
    locusPlot(sc, ryplot)

    print('module ryscene done!')
