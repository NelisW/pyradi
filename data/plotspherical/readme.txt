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
files.
There are different input files, each with different number of samples
on the unit sphere:  12, 42, 162, 642, 2562, 10242, 40962 or 163842.


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

     #. The **vertex** file contains the normalised direction vectors
        between the object and observer. Depending on the trajectory type
        (see above), the sensor and object switch locations for these
        vectors. These vectors are the directions of sampled intensity values.

     #. The **triangles** file defines triangles that provides the spatial
        linking between adjacent vectors, used when plotting the data.
        We plot the complex hull comprising these triangles, with vertices
        along the direction vectors, with length given by the simulated
        data set.

