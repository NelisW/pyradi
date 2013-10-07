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

===============  =========== ==============
Filename         Resolution    Number
   .              (degrees)    points
===============  =========== ==============
sphere_0_12       56.9             12
sphere_1_42       28.5             42
sphere_2_162      14.2            162
sphere_3_642      7.1             642
sphere_4_2562     3.56           2562
sphere_5_10242    1.78          10242
sphere_6_40962    0.889         40962
sphere_7_163842   0.445        163842 

     #. The **vertex** file contains the normalised direction vectors
        between the object and observer. Depending on the trajectory type
        (see above), the sensor and object switch locations for these
        vectors. These vectors are the directions of sampled intensity values.

     #. The **triangles** file defines triangles that provides the spatial
        linking between adjacent vectors, used when plotting the data.
        We plot the complex hull comprising these triangles, with vertices
        along the direction vectors, with length given by the simulated
        data set.

