"""The goedesic datasets in this folder provide even coverage over a sphere.

Sometimes a more dense coverage is required in some part of the sphere.
In such a case a composite file with extracts from several of the
geodesic data files can be made.

For example, suppose more dense coverage is required in a cone around the 
x axis. Or more dense coverage is required near the three primary planes.

The sphere data available in pyradi/data/plotspherical are:

===============  =========== ============== ===============
Filename         Resolution    Number          Number
   .              (degrees)    points         triangles
===============  =========== ============== ===============
vertexsphere_0_12       63.4             12              20
vertexsphere_1_42       33.9             42              80
vertexsphere_2_162      17.2            162             320
vertexsphere_3_642      8.6             642            1280
vertexsphere_4_2562     4.32           2562            5120
vertexsphere_5_10242    2.16          10242           20480
vertexsphere_6_40962    1.08          40962           81920
vertexsphere_7_163842   0.54         163842          327680

The geodesic vertex vectors are assumed to have length of 1.
"""

import numpy as np

# use the low res set as the basic set
filelores = 'vertexsphere_2_162.txt'
# use the medium res set in the designated medium res areas
filemeres = 'vertexsphere_3_642.txt'
# use the hi res set in the designated hires areas
filehires = 'vertexsphere_4_2562.txt'

filedataset = 'compositesphere.txt'

xAxisConeApexDeg = 20
xPlaneHalfAngle = 1
yPlaneHalfAngle = 1
zPlaneHalfAngle = 1

dataset = np.loadtxt(filelores,delimiter=' ')
meres = np.loadtxt(filemeres,delimiter=' ')
hires = np.loadtxt(filehires,delimiter=' ')
print(f'now {dataset.shape} vertices')

# select hires vertices in the cone around x axis
xCone = np.cos(xAxisConeApexDeg*np.pi/180)
print(f'Selecting all vertices with x>{xCone}')
dataset = np.vstack((dataset, hires[np.all([ hires[:,0]>=xCone], axis=0)]))
print(f'now {dataset.shape} vertices')

#  select meres vertices near x==0 plane 
xPlane = np.tan(xPlaneHalfAngle*np.pi/180)
print(f'Selecting all vertices with x<{xPlane} and x>{-xPlane}')
dataset = np.vstack((dataset, meres[np.logical_and(
    np.all([ meres[:,0]<xPlane], axis=0),
    np.all([ meres[:,0]>-xPlane], axis=0)
    )]))
print(f'now {dataset.shape} vertices')

#  select meres vertices near y==0 plane 
yPlane = np.tan(yPlaneHalfAngle*np.pi/180)
print(f'Selecting all vertices with y<{yPlane} and y>{-yPlane}')
dataset = np.vstack((dataset, meres[np.logical_and(
    np.all([ meres[:,1]<yPlane], axis=0),
    np.all([ meres[:,1]>-yPlane], axis=0)
    )]))
print(f'now {dataset.shape} vertices')


#  select meres vertices near z==0 plane 
zPlane = np.tan(zPlaneHalfAngle*np.pi/180)
print(f'Selecting all vertices with z<{zPlane} and z>{-zPlane}')
dataset = np.vstack((dataset, meres[np.logical_and(
    np.all([ meres[:,2]<zPlane], axis=0),
    np.all([ meres[:,2]>-zPlane], axis=0)
    )]))
print(f'now {dataset.shape} vertices')

np.savetxt(filedataset, dataset, delimiter = ' ')




