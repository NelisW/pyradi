# HDF5 as Data Format in Python

##Overview

[HDF5](http://en.wikipedia.org/wiki/Hierarchical_Data_Format) is a data model, library, and file format for storing and managing data. It supports an unlimited variety of datatypes, and is designed for flexible and efficient I/O and for high volume and complex data. HDF5 is portable and is extensible, allowing applications to evolve in their use of HDF5. The HDF5 Technology suite includes tools and applications for managing, manipulating, viewing, and analyzing data in the HDF5 format [[from HDF5 web page](https://www.hdfgroup.org/HDF5/)].  The HDF5 format can be read and written in many computer languages, such as C and C++, but also from [Python](http://www.h5py.org/) and [Matlab](http://de.mathworks.com/help/matlab/hdf5-files.html).  There are also support in the form of [file viewing tools](https://www.hdfgroup.org/HDF5/Tutor/tools.html).

For more information see:  
<https://www.hdfgroup.org/HDF5/>  
<https://www.hdfgroup.org/HDF5/Tutor/>  
<http://docs.h5py.org/en/latest/>

From [someone's practical experience](http://stackoverflow.com/questions/4871670/experience-with-using-h5py-to-do-analytical-work-on-big-data-in-python):  
HDF5 advantages:

- data can be inspected conveniently using the h5view application, h5py/ipython and the h5* commandline tools
- APIs are available for different platforms and languages
- structure data using groups
- annotating data using attributes
- worry-free built-in data compression
- io on single datasets is fast

HDF5 pitfalls:

- Performance breaks down, if a h5 file contains too many datasets/groups (> 1000), because traversing them is very slow. On the other side, io is fast for a few big datasets.
- Advanced data queries (SQL like) are clumsy to implement and slow (consider SQLite in that case)
- HDF5 is not thread-safe in all cases: one has to ensure, that the library was compiled with the correct options
- changing h5 datasets (resize, delete etc.) blows up the file size (in the best case) or is impossible (in the worst case the whole hdf5 file has to be copied to flatten it again).

#HDF5 in Python

##Motivation for using HDF5 in pyradi.rystare

An HDF5 file is a convenient means to store data in a structure.  In pyradi.rystare it is used to store all input and output data in one collective location.  This is a convenient way to keep track of all the variables in one place.  

Using HDF5 for this purpose is definitely more verbose and requires careful coding to keep track of the `[...]` and `.value` constructs.


##Opening a file and creating variables

To open an HDF5 file. The data in this file will be written when the data is updated or the app is closed.

	import h5py
    f = h5py.File(filename)  

After use the file must be closed (but I found that this appears not to be always necessary):

	f.close()

The documentation states that a file can be opened in memory only, which means that it is never written to disk:

    f = h5py.File(filename, driver='core',backing_store=False)  


To create a value in the file

	def erase_create_HDF(filename):
	    """Create new HDS5 file with the given filename, erase the file if existing.
	    """
	    if os.path.isfile(filename):
	        os.remove(filename)
	    f = h5py.File(filename)
	    return f

    g = erase_create_HDF('test.hdf5')
    g['ccd/value/illumination/value/image_irrad_scale/value'] = 4

The last line in the example above **creates** a variable in a group.  If the variable is already existing, this line will result in an error (ellipsis must be used to assign to an existing variable, see below).

To read a value in the file, use the `.value` attribute to retrieve the value. Without the `.value` attribute an object of the HDF5 dataset is returned.  The following line prints a value from the file:

    print(g['ccd/value/illumination/value/image_irrad_scale/value'].value)

An HDF5 file is opened for reading and writing, but once a variable has been created/written in an existing file, you cannot *create* the variable again.  You can *assign* to the existing variable with a slightly different notation (adding the slicing ellipsis [...] to the end, see the reason below):

    g = erase_create_HDF('test.hdf5')
    g['rystare/value/illumination/value/image_irrad_scale/value'] = 4
    print(g['rystare/value/illumination/value/image_irrad_scale/value'].value)
    g['rystare/value/illumination/value/image_irrad_scale/value'] = 6 # ERROR!!!
    print(g['rystare/value/illumination/value/image_irrad_scale/value'].value)

    g = erase_create_HDF('test.hdf5')
    g['rystare/value/illumination/value/image_irrad_scale/value'] = 4
    print(g['rystare/value/illumination/value/image_irrad_scale/value'].value)
    g['rystare/value/illumination/value/image_irrad_scale/value'][...] = 6 # working now
    print(g['rystare/value/illumination/value/image_irrad_scale/value'].value)

The slicing ellipsis `...` is used to denote 'all remaining dimensions' when slicing an array, as in [see [here](http://nbviewer.ipython.org/github/NelisW/ComputationalRadiometry/blob/master/02-PythonWhirlwindCheatSheet.ipynb)]:

	a = np.zeros((2,3,4,5))
	print('a.shape = {}'.format(a.shape))
	print('a[...].shape = {}'.format(a[...].shape))
	print('a[1,...].shape = {}'.format(a[1,...].shape))
	print('a[1,...,1].shape = {}'.format(a[1,...,1].shape))
	print('a[...,1].shape = {}'.format(a[...,1].shape))

resulting in 

	a.shape = (2, 3, 4, 5)
	a[...].shape = (2, 3, 4, 5)
	a[1,...].shape = (3, 4, 5)
	a[1,...,1].shape = (3, 4)
	a[...,1].shape = (2, 3, 4)

The meaning of `[...]` therefore means 'all dimensions' in the array, because none are specified.  So when we assign in this statement

    g['rystare/value/illumination/value/image_irrad_scale/value'][...] = 6 # working now

it means that the existing variable is assigned along all the dimensions with the new value.

This works for scalars and arrays, but the ellipsis notation must still be used to assign to an existing variable:

    g = erase_create_HDF('test.hdf5')
    g['rystare/illumination/image'] = np.ones((10,10))
    print(g['rystare/illumination/image'].value)
    g['rystare/illumination/image'][...] = np.zeros((10,10))
    print(g['rystare/illumination/image'].value)

The above example implies that a variable size cannot be changed - you cannot replace a small array with a larger one (or vice versa).

You can test to see if a dataset is defined along a path with a simple test:

        if 'rystare/dark/responseNU/filter_params' in g:
            filter_params = g['rystare/dark/responseNU/filter_params'].value

You can overwrite an existing value in the HDF5 file as explained above.  However, the data type and shape must be the same as the currently existing value in the file.  *If you want to **change the type or shape, first delete the existing variable:***

	g = erase_create_HDF('test.hdf5')
	g['rystare/illumination/image'] = np.ones((3,3))
	print(g['rystare/illumination/image'].value)
	del g['rystare/illumination/image']
	g.create_dataset('ccd/illumination/image', data=np.ones((10,10)))
	print(g['rystare/illumination/image'].value)


HDF5 appears to be a **strongly typed** file format.  If a scalar value is first  created as an integer, a float assigned to the same value will be truncated to an integer, as shown here.

    g = erase_create_HDF('test.hdf5')
    g['rystare/illumination/image_irrad_scale'] = 4
    print(g['rystare/illumination/image_irrad_scale'].value)
    g['rystare/illumination/image_irrad_scale'][...] = 0.001
    print(g['rystare/illumination/image_irrad_scale'].value)
    g['rystare/illumination/image_irrad_scaleflt']= 0.001
    print(g['rystare/illumination/image_irrad_scaleflt'].value)

yields the following (note that the first float value is truncated):

	4
	0
	0.001

It appears possible but not advisable to delete arrays from the file.

So in summary the using variables in the HDF5 file is as follows:

- `hdf5file['group path']`  is used when creating and assigning a new variable.
- `hdf5file['group path'].value`  is used when reading the variable.
- `hdf5file['group path'][...]`  is used when assigning to an existing variable.


###Iterating through the file
To print a complete list of all the branches in the file  

    def printname(name):
        print(name)
    f.visit(printname)

To print a complete list of all dataset values at the end of group paths:  

    def printdatasetvalue(var, obj):
        if type(obj.file[var]) is h5py._hl.dataset.Dataset:
            print(var, obj.file[var].name)
    f.visititems(printdatasetvalue)    

To build a list of  all dataset values at the end of group paths:

    mylist = []
    hdf5File.visit(mylist.append)
    print(mylist)

##Flushing and closing file contents
The `flush()` function flushes all changed data to file and `close()` closes the file.

	g = erase_create_HDF('test.hdf5')
	g['rystare/illumination/image_irrad_scale'] = 4
	g.flush()
	g.close()


#Viewing HDF5 File Contents

The [HDFView tool](https://www.hdfgroup.org/products/java/hdfview/index.html) allows you to browse through the file and view its contents.  The viewer allows you to plot vector-style variables as graphs (up to ten lines are drawn).  Two-dimensional arrays van be plotted along rows or columns.  You can also view the properties and attributes of variables in the file.

