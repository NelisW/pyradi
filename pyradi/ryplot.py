#  $Id$
#  $HeadURL$
################################################################
# The contents of this file are subject to the BSD 3Clause (New)
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

# Contributor(s): MS Willers, PJ van der Merwe, A de Waal
################################################################
"""
This module provides functions for plotting cartesian and polar plots.
This class provides a basic plotting capability, with a minimum
number of lines. These are all wrapper functions,
based on existing functions in other Python classes.
Provision is made for combinations of linear and log scales, as well
as polar plots for two-dimensional graphs.
The Plotter class can save files to disk in a number of formats.

For more examples of use see:
https://github.com/NelisW/ComputationalRadiometry

See the __main__ function for examples of use.

This package was partly developed to provide additional material in support of students
and readers of the book Electro-Optical System Analysis and Design: A Radiometry
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""

__version__ = "$Revision$"
__author__ = 'pyradi team'
__all__ = ['Plotter','cubehelixcmap', 'FilledMarker', 'Markers','ProcessImage',
            'savePlot']

import numpy as np
import math
import sys
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

# following for the pie plots
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import MaxNLocator

####################################################################
##
class FilledMarker:
    """Filled marker user-settable values.

    This class encapsulates a few variables describing a Filled marker.
    Default values are provided that can be overridden in user plots.

    Values relevant to filled makers are as follows:
     | marker = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
     | fillstyle = ['full', 'left', 'right', 'bottom', 'top', 'none']
     | colour names =       http://www.w3schools.com/html/html_colornames.asp
    """

    def __init__(self, markerfacecolor=None, markerfacecoloralt=None,
                 markeredgecolor=None, marker=None, markersize=None,
                 fillstyle=None):
        """Define marker default values.

            Args:
                | markerfacecolor (colour): main colour for marker (optional)
                | markerfacecoloralt (colour): alterive colour for marker (optional)
                | markeredgecolor (colour): edge colour for marker (optional)
                | marker (string): string to specify the marker  (optional)
                | markersize (int)): size of the marker  (optional)
                | fillstyle (string): string to define fill style  (optional)

            Returns:
                | Nothing. Creates the figure for subequent use.

            Raises:
                | No exception is raised.

        """

        __all__ = ['__init__']

        if markerfacecolor is None:
            self.markerfacecolor = 'r'
        else:
            self.markerfacecolor = markerfacecolor

        if markerfacecoloralt is None:
            self.markerfacecoloralt = 'b'
        else:
            self.markerfacecoloralt = markerfacecoloralt

        if markeredgecolor is None:
            self.markeredgecolor = 'k'
        else:
            self.markeredgecolor = markeredgecolor

        if marker is None:
            self.marker = 'o'
        else:
            self.marker = marker

        if markersize is  None:
            self.markersize = 20
        else:
            self.markersize = markersize

        if fillstyle is None:
            self.fillstyle = 'full'
        else:
            self.fillstyle = fillstyle



###################################################################################
###################################################################################

class Markers:
    """Collect marker location and types and mark subplot.

    Build a list of markers at plot locations with the specified marker.
    """

####################################################################
##
    def __init__(self, markerfacecolor = None, markerfacecoloralt = None,
                markeredgecolor = None, marker = None, markersize = None,
                fillstyle = None):
        """Set default marker values for this collection

        Specify default marker properties to be used for all markers
        in this instance. If no marker properties are specified here,
        the  default FilledMarker marker properties will be used.

            Args:
                | markerfacecolor (colour): main colour for marker (optional)
                | markerfacecoloralt (colour): alterive colour for marker (optional)
                | markeredgecolor (colour): edge colour for marker (optional)
                | marker (string): string to specify the marker  (optional)
                | markersize (int)): size of the marker  (optional)
                | fillstyle (string): string to define fill style  (optional)

            Returns:
                | Nothing. Creates the figure for subequent use.

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', 'add',  'plot']

        if markerfacecolor is None:
            self.markerfacecolor = None
        else:
            self.markerfacecolor = markerfacecolor

        if markerfacecoloralt is  None:
            self.markerfacecoloralt = None
        else:
            self.markerfacecoloralt = markerfacecoloralt

        if markeredgecolor is  None:
            self.markeredgecolor = None
        else:
            self.markeredgecolor = markeredgecolor

        if marker is  None:
            self.marker = None
        else:
            self.marker = marker

        if markersize is   None:
            self.markersize = markersize
        else:
            self.markersize = markersize

        if fillstyle is  None:
            self.fillstyle = None
        else:
            self.fillstyle = fillstyle

        #list if markers to be drawn
        self.markers = []

####################################################################
##
    def add(self,x,y,markerfacecolor = None, markerfacecoloralt = None,
                markeredgecolor = None, marker = None, markersize = None,
                fillstyle = None):
        """Add a marker to the list, overridding properties if necessary.

        Specify location and any specific marker properties to be used.
        The location can be (xy,y) for cartesian plots or (theta,rad) for polars.

        If no marker properties are specified, the current marker class
        properties will be used.  If the current maker instance does not
        specify properties, the default marker properties will be used.

            Args:
                | x (float): the x/theta location for the marker
                | y (float): the y/radial location for the marker
                | markerfacecolor (colour): main colour for marker (optional)
                | markerfacecoloralt (colour): alterive colour for marker (optional)
                | markeredgecolor (colour): edge colour for marker (optional)
                | marker (string): string to specify the marker  (optional)
                | markersize (int)): size of the marker  (optional)
                | fillstyle (string): string to define fill style  (optional)

            Returns:
                | Nothing. Creates the figure for subequent use.

            Raises:
                | No exception is raised.
        """

        if markerfacecolor is None:
            if self.markerfacecolor is not None:
                markerfacecolor = self.markerfacecolor

        if markerfacecoloralt is None:
            if self.markerfacecoloralt is not None:
                markerfacecoloralt = self.markerfacecoloralt

        if markeredgecolor is None:
            if self.markeredgecolor is not None:
                markeredgecolor = self.markeredgecolor

        if marker is None:
            if self.marker is not None:
                marker = self.marker

        if markersize is None:
            if self.markersize is not None:
                markersize = self.markersize

        if fillstyle is None:
            if self.fillstyle is not None:
                fillstyle = self.fillstyle

        marker = FilledMarker(markerfacecolor, markerfacecoloralt ,
                        markeredgecolor , marker, markersize , fillstyle)
        self.markers.append((x,y,marker))


####################################################################
##
    def plot(self,ax):
        """Plot the current list of markers on the given axes.

        All the markers currently stored in the class will be
        drawn.

            Args:
                | ax (axes): an axes handle for the plot

            Returns:
                | Nothing. Creates the figure for subequent use.

            Raises:
                | No exception is raised.
        """
        usetex = plt.rcParams['text.usetex']
        plt.rcParams['text.usetex'] = False # otherwise, '^' will cause trouble

        for marker in self.markers:
            ax.plot(marker[0], marker[1],
                color = marker[2].markerfacecolor,
                markerfacecoloralt = marker[2].markerfacecoloralt,
                markeredgecolor = marker[2].markeredgecolor,
                marker = marker[2].marker,
                markersize = marker[2].markersize,
                fillstyle = marker[2].fillstyle)

        plt.rcParams['text.usetex'] = usetex


###################################################################################
###################################################################################

class ProcessImage:
    """This class provides a functions to assist in the optimal display of images.
    """

    #define the compression rule to be used in the equalisation function
    compressSet = [
               [lambda x : x , lambda x : x, 'Linear'],
               [np.log,  np.exp, 'Natural Log'],
               [np.sqrt,  np.square, 'Square Root']]

    ############################################################
    def __init__(self):
        """Class constructor

        Sets up some variables for use in this class

            Args:
                | None

            Returns:
                | Nothing

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', 'compressEqualizeImage',  'reprojectImageIntoPolar']


    ############################################################
    def compressEqualizeImage(self, image, selectCompressSet=2, numCbarlevels=20,
                    cbarformat='.3f'):
        """Compress an image (and then inversely expand the color bar values),
           prior to histogram equalisation to ensure that the two keep in step,
           we store the compression function names as pairs, and invoke the
           compression function as follows:  linear, log. sqrt.  Note that the
           image is histogram equalised in all cases.

           Args:
                | image (np.ndarray):  the image to be processed
                | selectCompressSet (int): compression selection [0,1,2] (optional)
                | numCbarlevels (int): number of labels in the colourbar (optional)
                | cbarformat (string): colourbar label format, e.g., '10.3f', '.5e' (optional)

            Returns:
                | imgHEQ  (np.ndarray): the equalised image array
                | customticksz (zip(float, string)): colourbar levels and associated levels

            Raises:
                | No exception is raised.

     """

        #compress the input image  - rescale color bar tick to match below
        #also collapse into single dimension
        imgFlat = self.compressSet[selectCompressSet][0](image.flatten())
        imgFlatSort = np.sort(imgFlat)
        #cumulative distribution
        cdf = imgFlatSort.cumsum()/imgFlatSort[-1]
        #remap image values to achieve histogram equalisation
        y=np.interp(imgFlat,imgFlatSort, cdf )
        #and reshape to original image shape
        imgHEQ = y.reshape(image.shape)

        #        #plot the histogram mapping
        #        minData = np.min(imgFlat)
        #        maxData = np.max(imgFlat)
        #        print('Image irradiance range minimum={0} maximum={1}'.format(minData, maxData))
        #        irradRange=np.linspace(minData, maxData, 100)
        #        normalRange = np.interp(irradRange,imgFlatSort, cdf )
        #        H = ryplot.Plotter(1, 1, 1,'Mapping Input Irradiance to Equalised Value',
        #           figsize=(10, 10))
        #        H.plot(1, "","Irradiance [W/(m$^2$)]", "Equalised value",irradRange,
        #           normalRange, powerLimits = [-4,  2,  -10,  2])
        #        #H.getPlot().show()
        #        H.saveFig('cumhist{0}.png'.format(entry), dpi=300)

        #prepare the color bar tick labels from image values (as plotted)
        imgLevels = np.linspace(np.min(imgHEQ), np.max(imgHEQ), numCbarlevels)
        #map back from image values to original values as read it (inverse to above)
        irrLevels=np.interp(imgLevels,cdf, imgFlatSort)
        #uncompress the tick labels  - match  with compression above

        fstr = '{0:' + cbarformat + '}'
        customticksz = list(zip(imgLevels, [fstr.format(self.compressSet[selectCompressSet][1](x)) for x in irrLevels]))

        return imgHEQ, customticksz

    ##############################################################################
    ##
    def reprojectImageIntoPolar(self, data, origin=None, framesFirst=True,cval=0.0):
        """Reprojects a 3D numpy array into a polar coordinate system, relative to some origin.

        This function reprojects an image from cartesian to polar coordinates.
        The origin of the new coordinate system  defaults to the center of the image,
        unless the user supplies a new origin.

        The data format can be data.shape = (rows, cols, frames) or
        data.shape = (frames, rows, cols), the format of which is indicated by the
        framesFirst parameter.

        The reprojectImageIntoPolar function maps radial to cartesian coords.
        The radial image is however presented in a cartesian grid, the corners have no meaning.
        The radial coordinates are mapped to the radius, not the corners.
        This means that in order to map corners, the frequency is scaled with sqrt(2),
        The corners are filled with the value specified in cval.

        Args:
            | data (np.array): 3-D array to which transformation must be applied.
            | origin ( (x-orig, y-orig) ): data-coordinates of where origin should be placed
            | framesFirst (bool): True if data.shape is (frames, rows, cols), False if
                data.shape is (rows, cols, frames)
            | cval (float): the fill value to be used in coords outside the mapped range(optional)

        Returns:
            | output (float np.array): transformed images/array data in the same sequence as input sequence.
            | r_i (np.array[N,]): radial values for returned image.
            | theta_i (np.array[M,]): angular values for returned image.

        Raises:
            | No exception is raised.

        original code by Joe Kington
        https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
        """

        import pyradi.ryutils as ryutils
        # import scipy as sp
        import scipy.ndimage as spndi

        if framesFirst:
            data = ryutils.framesLast(data)

        ny, nx = data.shape[:2]

        if origin is None:
            origin = (nx//2, ny//2)

        # Determine what the min and max r and theta coords will be
        x, y = ryutils.index_coords(data, origin=origin, framesFirst=framesFirst )

        r, theta = ryutils.cart2polar(x, y)

        # Make a regular (in polar space) grid based on the min and max r & theta
        r_i = np.linspace(r.min(), r.max(), nx)
        theta_i = np.linspace(theta.min(), theta.max(), ny)
        theta_grid, r_grid = np.meshgrid(theta_i, r_i)

        # Project the r and theta grid back into pixel coordinates
        xi, yi = ryutils.polar2cart(r_grid, theta_grid)
        xi += origin[0] # We need to shift the origin back to
        yi += origin[1] # back to the lower-left corner...
        xi, yi = xi.flatten(), yi.flatten()
        coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)

        # Reproject each band individually and the restack
        # (uses less memory than reprojection the 3-dimensional array in one step)
        bands = []
        for band in data.T:
            zi = spndi.map_coordinates(band, coords, order=1,cval=cval)
            bands.append(zi.reshape((nx, ny)))
        output = np.dstack(bands)
        if framesFirst:
            output = ryutils.framesFirst(output)
        return output, r_i, theta_i

###################################################################################
###################################################################################

class Plotter:
    """ Encapsulates a plotting environment, optimized for compact code.

    This class provides a wrapper around Matplotlib to provide a plotting
    environment specialised towards typical pyradi visualisation.
    These functions were developed to provide sophisticated plots by entering
    the various plot options on a few lines, instead of typing many commands.

    Provision is made for plots containing subplots (i.e., multiple plots on
    the same figure), linear scale and log scale plots, images, and cartesian,
    3-D and polar plots.
    """

    ############################################################
    ##
    def __init__(self,fignumber=0,subpltnrow=1,subpltncol=1,\
                 figuretitle=None, figsize=(9,9), titlefontsize=14):
        """Class constructor

        The constructor defines the number for this figure, allowing future reference
        to this figure. The number of subplot rows and columns allow the user to define
        the subplot configuration.  The user can also provide a title to be
        used for the figure (centred on top) and finally, the size of the figure in inches
        can be specified to scale the text relative to the figure.

            Args:
                | fignumber (int): the plt figure number, must be supplied
                | subpltnrow (int): subplot number of rows
                | subpltncol (int): subplot number of columns
                | figuretitle (string): the overall heading for the figure
                | figsize ((w,h)): the figure size in inches
                | titlefontsize (int): the figure title size in points

            Returns:
                | Nothing. Creates the figure for subequent use.

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', 'saveFig', 'getPlot', 'plot', 'logLog', 'semilogX',
                        'semilogY', 'polar', 'showImage', 'plot3d', 'buildPlotCol',
                        'getSubPlot', 'meshContour', 'nextPlotCol', 'plotArray',
                        'polarMesh', 'resetPlotCol', 'mesh3D', 'polar3d', 'labelSubplot',
                        'emptyPlot','setup_pie_axes','pie']

        version=mpl.__version__.split('.')
        vnum=float(version[0]+'.'+version[1])

        if vnum<1.1:
            print('Install Matplotlib 1.1 or later')
            print('current version is {0}'.format(vnum))
            sys.exit(-1)

        self.figurenumber = fignumber
        self.fig = plt.figure(self.figurenumber, frameon=False)
        self.fig.set_size_inches(figsize[0], figsize[1])
        self.fig.clear()
        self.figuretitle = figuretitle

        self.nrow=subpltnrow
        self.ncol=subpltncol

        # width reserved for space between subplots
        self.fig.subplots_adjust(wspace=0.25)
        #height reserved for space between subplots
        self.fig.subplots_adjust(hspace=0.4)
        #height reserved for top of the subplots of the figure
        self.fig.subplots_adjust(top=0.88)

        # define the default line colour and style
        self.buildPlotCol(plotCol=None, n=None)

        self.bbox_extra_artists = []
        self.subplots = {}
        self.gridSpecsOuter = {}
        self.arrayRows = {}
        self.gridSpecsInner = {}

        if figuretitle:
            self.figtitle=plt.gcf().text(.5,.95,figuretitle,\
                        horizontalalignment='center',\
                        fontproperties=FontProperties(size=titlefontsize))
            self.bbox_extra_artists.append(self.figtitle)



    ############################################################
    ##
    def buildPlotCol(self, plotCol=None, n=None):
        """Set a sequence of default colour styles of
           appropriate length.

           The constructor provides a sequence with length
           14 pre-defined plot styles.
           The user can define a new sequence if required.
           This function modulus-folds either sequence, in
           case longer sequences are required.

           Colours can be one of the basic colours:
           ['b', 'g', 'r', 'c', 'm', 'y', 'k']
           or it can be a gray shade float value between 0 and 1,
           such as '0.75', or it can be in hex format '#eeefff'
           or it can be one of the legal html colours.
           See http://html-color-codes.info/ and
           http://www.computerhope.com/htmcolor.htm.
           http://latexcolor.com/

            Args:
                | plotCol ([strings]): User-supplied list
                |    of plotting styles(can be empty []).
                | n (int): Length of required sequence.

            Returns:
                | A list with sequence of plot styles, of required length.

            Raises:
                | No exception is raised.
        """
        # assemble the list as requested, use default if not specified
        if plotCol is None:
            plotCol = ['b', 'g', 'r', 'c', 'm', 'y', 'k',
            '#5D8AA8','#E52B50','#FF7E00','#9966CC','#CD9575','#915C83',
            '#008000','#4B5320','#B2BEB5','#A1CAF1','#FE6F5E','#333399',
            '#DE5D83','#800020','#1E4D2B','#00BFFF','#007BA7','#FFBCD9']

        if n is None:
            n = len(plotCol)

        self.plotCol = [plotCol[i % len(plotCol)] for i in range(n)]

        # copy this to circular list as well
        self.plotColCirc = itertools.cycle(self.plotCol)

        return self.plotCol



    ############################################################
    ##
    def nextPlotCol(self):
        """Returns the next entry in a sequence of default
           plot line colour styles in circular list.
           One day I want to do this with a generator....

            Args:
                | None

            Returns:
                | The next plot colour in the sequence.

            Raises:
                | No exception is raised.
        """

        col = next(self.plotColCirc)
        return col

    ############################################################
    ##
    def resetPlotCol(self):
        """Resets the plot colours to start at the beginning of
        the cycle.

             Args:
                | None

            Returns:
                | None.

            Raises:
                | No exception is raised.
        """

        self.plotColCirc = itertools.cycle(self.plotCol)


    ############################################################
    ##
    def saveFig(self, filename='mpl.png',dpi=300,bbox_inches='tight',\
                pad_inches=0.1, useTrueType = True):
        """Save the plot to a disk file, using filename, dpi specification and bounding box limits.

        One of matplotlib's design choices is a bounding box strategy  which may result in a bounding box
        that is smaller than the size of all the objects on the page.  It took a while to figure this out,
        but the current default values for bbox_inches and pad_inches seem to create meaningful
        bounding boxes. These are however larger than the true bounding box. You still need a
        tool such as epstools or Adobe Acrobat to trim eps files to the true bounding box.

        The type of file written is picked up in the filename.
        Most backends support png, pdf, ps, eps and svg.

            Args:
                | filename (string): output filename to write plot, file ext
                | dpi (int): the resolution of the graph in dots per inch
                | bbox_inches: see matplotlib docs for more detail.
                | pad_inches: see matplotlib docs for more detail.
                | useTrueType: if True, truetype fonts are used in eps/pdf files, otherwise Type3


            Returns:
                | Nothing. Saves a file to disk.

            Raises:
                | No exception is raised.
        """

        # http://matplotlib.1069221.n5.nabble.com/TrueType-font-embedding-in-eps-problem-td12691.html
        # http://stackoverflow.com/questions/5956182/cannot-edit-text-in-chart-exported-by-matplotlib-and-opened-in-illustrator
        # http://newsgroups.derkeiler.com/Archive/Comp/comp.soft-sys.matlab/2008-07/msg02038.html


        if useTrueType:
            mpl.rcParams['pdf.fonttype'] = 42
            mpl.rcParams['ps.fonttype'] = 42

        #http://stackoverflow.com/questions/15341757/how-to-check-that-pylab-backend-of-matplotlib-runs-inline/17826459#17826459
        # print(mpl.get_backend())
        if 'inline' in mpl.get_backend():
            print('****  To use saveFig inside IPython please comment out the line "#%matplotlib inline" ')
            return

        if len(filename)>0:
            if self.bbox_extra_artists:
                self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                            pad_inches=pad_inches,\
                            bbox_extra_artists= self.bbox_extra_artists)
            else:
                self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                            pad_inches=pad_inches)


    ############################################################
    ##
    def getPlot(self):
      """Returns a handle to the current figure

            Args:
                | None

            Returns:
                | A handle to the current figure.

            Raises:
                | No exception is raised.
      """
      return self.fig


    ############################################################
    ##
    def labelSubplot(self, spax, ptitle=None, xlabel=None, ylabel=None, zlabel=None,
      titlefsize=10, labelfsize=10, ):
      """Set the sub-figure title and axes labels (cartesian plots only).

            Args:
                | spax (handle): subplot axis handle where labels must be drawn
                | ptitle (string): plot title (optional)
                | xlabel (string): x-axis label (optional)
                | ylabel (string): y-axis label (optional)
                | zlabel (string): z axis label (optional)
                | titlefsize (float): title  fontsize (optional)
                | labelfsize (float): x,y,z label fontsize (optional)

            Returns:
                | None.

            Raises:
                | No exception is raised.
      """
      if xlabel is not None:
        spax.set_xlabel(xlabel,fontsize=labelfsize)
      if ylabel is not None:
        spax.set_ylabel(ylabel,fontsize=labelfsize)
      if zlabel is not None:
        spax.set_ylabel(zlabel,fontsize=labelfsize)
      if ptitle is not None:
        spax.set_title(ptitle,fontsize=titlefsize)


    ############################################################
    ##
    def getSubPlot(self, subplotNum = 1):
        """Returns a handle to the subplot, as requested per subplot number.
        Subplot numbers range from 1 upwards.

            Args:
                | subplotNumer (int) : number of the subplot

            Returns:
                | A handle to the requested subplot or None if not found.

            Raises:
                | No exception is raised.
        """
        if (self.nrow,self.ncol, subplotNum) in list(self.subplots.keys()):
            return self.subplots[(self.nrow,self.ncol, subplotNum)]
        else:
            return None

   ############################################################
    ##
    def getXLim(self, subplotNum = 1):
        """Returns the x limits of the current subplot.
        Subplot numbers range from 1 upwards.

            Args:
                | subplotNumer (int) : number of the subplot

            Returns:
                | An array with the two limits

            Raises:
                | No exception is raised.
        """
        if (self.nrow,self.ncol, subplotNum) in list(self.subplots.keys()):
            return np.asarray(self.subplots[(self.nrow,self.ncol, subplotNum)].get_xlim())
        else:
            return None


   ############################################################
    ##
    def getYLim(self, subplotNum = 1):
        """Returns the y limits of the current subplot.
        Subplot numbers range from 1 upwards.

            Args:
                | subplotNumer (int) : number of the subplot

            Returns:
                | An array with the two limits

            Raises:
                | No exception is raised.
        """
        if (self.nrow,self.ncol, subplotNum) in list(self.subplots.keys()):
            return np.asarray(self.subplots[(self.nrow,self.ncol, subplotNum)].get_ylim())
        else:
            return None

   ############################################################
    ##
    def verticalLineCoords(self,subplotNum=1,x=0):
        """Returns two arrays for vertical line at x in the specific subplot.

        The line is drawn at specified x, with current y limits in subplot.
        Subplot numbers range from 1 upwards.

        Use as follows to draw a vertical line in plot:
            p.plot(1,*p.verticalLineCoords(subplotNum=1,x=freq),plotCol=['k'])

            Args:
                | subplotNumer (int) : number of the subplot
                | x (double): horizontal value used for line

            Returns:
                | A tuple with two arrays for line (x-coords,y-coords)

            Raises:
                | No exception is raised.
        """
        if (self.nrow,self.ncol, subplotNum) in list(self.subplots.keys()):
            handle = self.subplots[(self.nrow,self.ncol, subplotNum)]

            x = np.asarray((x,x))
            y = self.getYLim(subplotNum)
            return x,y
        else:
            return None

   ############################################################
    ##
    def horizontalLineCoords(self,subplotNum=1,y=0):
        """Returns two arrays for horizontal line at y in the specific subplot.

        The line is drawn at specified y, with current x limits in subplot.
        Subplot numbers range from 1 upwards.

        Use as follows to draw a horizontal line in plot:
            p.plot(1,*p.verticalLineCoords(subplotNum=1,x=freq),plotCol=['k'])

            Args:
                | subplotNumer (int) : number of the subplot
                | y (double): horizontal value used for line

            Returns:
                | A tuple with two arrays for line (x-coords,y-coords)

            Raises:
                | No exception is raised.
        """
        if (self.nrow,self.ncol, subplotNum) in list(self.subplots.keys()):
            handle = self.subplots[(self.nrow,self.ncol, subplotNum)]

            y = np.asarray((y,y))
            x = self.getXLim(subplotNum)
            return x,y
        else:
            return None

    ############################################################
    ##
    def plot(self, plotnum, x, y, ptitle=None, xlabel=None, ylabel=None,
                    plotCol=[], linewidths=None, label=[], legendAlpha=0.0,
                    legendLoc='best',
                    pltaxis=None, maxNX=10, maxNY=10, linestyle=None,
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12,
                    xylabelfsize = 12,  xytickfsize = 10, labelfsize=10,
                    xScientific=False, yScientific=False,
                    yInvert=False, xInvert=False, drawGrid=True,xIsDate=False,
                    xTicks=None, xtickRotation=0, markers=[],
                    markevery=None, zorders=None, clip_on=True ):
      """Cartesian plot on linear scales for abscissa and ordinates.

        Given an existing figure, this function plots in a specified subplot position.
        The function arguments are described below in some detail. Note that the y-values
        or ordinates can be more than one column, each column representing a different
        line in the plot. This is convenient if large arrays of data must be plotted. If more
        than one column is present, the label argument can contain the legend labels for
        each of the columns/lines.  The pltaxis argument defines the min/max scale values
        for the x and y axes.

            Args:
                | plotnum (int): subplot number, 1-based index
                | x (np.array[N,] or [N,1]): abscissa
                | y (np.array[N,] or [N,M]): ordinates - could be M columns
                | ptitle (string): plot title (optional)
                | xlabel (string): x-axis label (optional)
                | ylabel (string): y-axis label (optional)
                | plotCol ([strings]): plot colour and line style, list with M entries, use default if [] (optional)
                | linewidths ([float]): plot line width in points, list with M entries, use default if None  (optional)
                | label  ([strings]): legend label for ordinate, list with M entries (optional)
                | legendAlpha (float): transparency for legend box (optional)
                | legendLoc (string): location for legend box (optional)
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. Let Matplotlib decide if None. (optional)
                | maxNX (int): draw maxNX+1 tick labels on x axis (optional)
                | maxNY (int): draw maxNY+1 tick labels on y axis (optional)
                | linestyle (string): linestyle for this plot (optional)
                | powerLimits[float]:  scientific tick label power limits [x-low, x-high, y-low, y-high] (optional)
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x-axis, y-axis label font size, default 12pt (optional)
                | xytickfsize (int): x-axis, y-axis tick font size, default 10pt (optional)
                | labelfsize (int): label/legend font size, default 10pt (optional)
                | xScientific (bool): use scientific notation on x axis (optional)
                | yScientific (bool): use scientific notation on y axis (optional)
                | drawGrid (bool): draw the grid on the plot (optional)
                | yInvert (bool): invert the y-axis (optional)
                | xInvert (bool): invert the x-axis (optional)
                | xIsDate (bool): convert the datetime x-values to dates (optional)
                | xTicks ({tick:label}): dict of x-axis tick locations and associated labels (optional)
                | xtickRotation (float) x-axis tick label rotation angle (optional)
                | markers ([string]) markers to be used for plotting data points (optional)
                | markevery (int | (startind, stride)) subsample when using markers (optional)
                | zorders ([int]) list of zorder for drawing sequence, highest is last (optional)
                | clip_on (bool) clips objects to drawing axes (optional)

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
       """
      ## see self.MyPlot for parameter details.
      pkey = (self.nrow, self.ncol, plotnum)
      if pkey not in list(self.subplots.keys()):
            self.subplots[pkey] = \
                         self.fig.add_subplot(self.nrow,self.ncol, plotnum)
      ax = self.subplots[pkey]

      self.myPlot(ax.plot, plotnum, x, y, ptitle, xlabel, ylabel,
                    plotCol, linewidths, label,legendAlpha, legendLoc,
                    pltaxis, maxNX, maxNY, linestyle,
                    powerLimits,titlefsize,
                    xylabelfsize, xytickfsize,
                    labelfsize, drawGrid,
                    xScientific, yScientific,
                    yInvert, xInvert, xIsDate,
                    xTicks, xtickRotation, markers, markevery, zorders, clip_on)
      return ax

    ############################################################
    ##
    def logLog(self, plotnum, x, y, ptitle=None, xlabel=None, ylabel=None,
                    plotCol=[], linewidths=None, label=[],legendAlpha=0.0,
                    legendLoc='best',
                    pltaxis=None, maxNX=10, maxNY=10, linestyle=None,
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12,
                    xylabelfsize = 12, xytickfsize = 10,labelfsize=10,
                    xScientific=False, yScientific=False,
                    yInvert=False, xInvert=False, drawGrid=True,xIsDate=False,
                    xTicks=None, xtickRotation=0, markers=[],
                    markevery=None, zorders=None, clip_on=True   ):
      """Plot data on logarithmic scales for abscissa and ordinates.

        Given an existing figure, this function plots in a specified subplot position.
        The function arguments are described below in some detail. Note that the y-values
        or ordinates can be more than one column, each column representing a different
        line in the plot. This is convenient if large arrays of data must be plotted. If more
        than one column is present, the label argument can contain the legend labels for
        each of the columns/lines.  The pltaxis argument defines the min/max scale values
        for the x and y axes.

            Args:
                | plotnum (int): subplot number, 1-based index
                | x (np.array[N,] or [N,1]): abscissa
                | y (np.array[N,] or [N,M]): ordinates - could be M columns
                | ptitle (string): plot title (optional)
                | xlabel (string): x-axis label (optional)
                | ylabel (string): y-axis label (optional)
                | plotCol ([strings]): plot colour and line style, list with M entries, use default if [] (optional)
                | linewidths ([float]): plot line width in points, list with M entries, use default if None  (optional)
                | label  ([strings]): legend label for ordinate, list with M entries (optional)
                | legendAlpha (float): transparency for legend box (optional)
                | legendLoc (string): location for legend box (optional)
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. Let Matplotlib decide if None. (optional)
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. Let Matplotlib decide if None. (optional)
                | maxNX (int): draw maxNX+1 tick labels on x axis (optional)
                | maxNY (int): draw maxNY+1 tick labels on y axis (optional)
                | linestyle (string): linestyle for this plot (optional)
                | powerLimits[float]:  scientific tick label power limits [x-low, x-high, y-low, y-high] (optional) (optional)
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x-axis, y-axis label font size, default 12pt (optional)
                | xytickfsize (int): x-axis, y-axis tick font size, default 10pt (optional)
                | labelfsize (int): label/legend font size, default 10pt (optional)
                | xScientific (bool): use scientific notation on x axis (optional)
                | yScientific (bool): use scientific notation on y axis (optional)
                | drawGrid (bool): draw the grid on the plot (optional)
                | yInvert (bool): invert the y-axis (optional)
                | xInvert (bool): invert the x-axis (optional)
                | xIsDate (bool): convert the datetime x-values to dates (optional)
                | xTicks ({tick:label}): dict of x-axis tick locations and associated labels (optional)
                | xtickRotation (float) x-axis tick label rotation angle (optional)
                | markers ([string]) markers to be used for plotting data points (optional)
                | markevery (int | (startind, stride)) subsample when using markers (optional)
                | zorders ([int]) list of zorder for drawing sequence, highest is last (optional)
                | clip_on (bool) clips objects to drawing axes (optional)

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
      """
      ## see self.MyPlot for parameter details.
      pkey = (self.nrow, self.ncol, plotnum)
      if pkey not in list(self.subplots.keys()):
            self.subplots[pkey] = \
                         self.fig.add_subplot(self.nrow,self.ncol, plotnum)
      ax = self.subplots[pkey]

        # self.myPlot(ax.loglog, plotnum, x, y, ptitle, xlabel,ylabel,\
        #             plotCol, label,legendAlpha, pltaxis, \
        #             maxNX, maxNY, linestyle, powerLimits,titlefsize,xylabelfsize,
        #             xytickfsize,labelfsize, drawGrid
        #             xTicks, xtickRotation,
        #             markers=markers)

      self.myPlot(ax.loglog, plotnum, x, y, ptitle, xlabel, ylabel,
                    plotCol, linewidths, label,legendAlpha, legendLoc,
                    pltaxis, maxNX, maxNY, linestyle,
                    powerLimits,titlefsize,
                    xylabelfsize, xytickfsize,
                    labelfsize, drawGrid,
                    xScientific, yScientific,
                    yInvert, xInvert, xIsDate,
                    xTicks, xtickRotation, markers, markevery, zorders, clip_on)

      return ax


    ############################################################
    ##
    def semilogX(self, plotnum, x, y, ptitle=None, xlabel=None, ylabel=None,
                    plotCol=[], linewidths=None, label=[],legendAlpha=0.0,
                    legendLoc='best',
                    pltaxis=None, maxNX=10, maxNY=10, linestyle=None,
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12,
                    xylabelfsize = 12, xytickfsize = 10,labelfsize=10,
                    xScientific=False, yScientific=False,
                    yInvert=False, xInvert=False, drawGrid=True,xIsDate=False,
                    xTicks=None, xtickRotation=0, markers=[],
                    markevery=None, zorders=None, clip_on=True   ):

      """Plot data on logarithmic scales for abscissa and linear scale for ordinates.

        Given an existing figure, this function plots in a specified subplot position.
        The function arguments are described below in some detail. Note that the y-values
        or ordinates can be more than one column, each column representing a different
        line in the plot. This is convenient if large arrays of data must be plotted. If more
        than one column is present, the label argument can contain the legend labels for
        each of the columns/lines.  The pltaxis argument defines the min/max scale values
        for the x and y axes.

            Args:
                | plotnum (int): subplot number, 1-based index
                | x (np.array[N,] or [N,1]): abscissa
                | y (np.array[N,] or [N,M]): ordinates - could be M columns
                | ptitle (string): plot title (optional)
                | xlabel (string): x-axis label (optional)
                | ylabel (string): y-axis label (optional)
                | plotCol ([strings]): plot colour and line style, list with M entries, use default if [] (optional)
                | linewidths ([float]): plot line width in points, list with M entries, use default if None  (optional)
                | label  ([strings]): legend label for ordinate, list with M entries (optional)
                | legendAlpha (float): transparency for legend box (optional)
                | legendLoc (string): location for legend box (optional)
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. Let Matplotlib decide if None. (optional)
                | maxNX (int): draw maxNX+1 tick labels on x axis (optional)
                | maxNY (int): draw maxNY+1 tick labels on y axis (optional)
                | linestyle (string): linestyle for this plot (optional)
                | powerLimits[float]: scientific tick label notation power limits [x-low, x-high, y-low, y-high] (optional) (optional)
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x-axis, y-axis label font size, default 12pt (optional)
                | xytickfsize (int): x-axis, y-axis tick font size, default 10pt (optional)
                | labelfsize (int): label/legend font size, default 10pt (optional)
                | xScientific (bool): use scientific notation on x axis (optional)
                | yScientific (bool): use scientific notation on y axis (optional)
                | drawGrid (bool): draw the grid on the plot (optional)
                | yInvert (bool): invert the y-axis (optional)
                | xInvert (bool): invert the x-axis (optional)
                | xIsDate (bool): convert the datetime x-values to dates (optional)
                | xTicks ({tick:label}): dict of x-axis tick locations and associated labels (optional)
                | xtickRotation (float) x-axis tick label rotation angle (optional)
                | markers ([string]) markers to be used for plotting data points (optional)
                | markevery (int | (startind, stride)) subsample when using markers (optional)
                | zorders ([int]) list of zorder for drawing sequence, highest is last (optional)
                | clip_on (bool) clips objects to drawing axes (optional)

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
       """
      ## see self.MyPlot for parameter details.
      pkey = (self.nrow, self.ncol, plotnum)
      if pkey not in list(self.subplots.keys()):
            self.subplots[pkey] = \
                         self.fig.add_subplot(self.nrow,self.ncol, plotnum)
      ax = self.subplots[pkey]

      self.myPlot(ax.semilogx, plotnum, x, y, ptitle, xlabel, ylabel,\
                    plotCol, linewidths, label,legendAlpha, legendLoc,
                    pltaxis, maxNX, maxNY, linestyle,
                    powerLimits, titlefsize,
                    xylabelfsize, xytickfsize,
                    labelfsize, drawGrid,
                    xScientific, yScientific,
                    yInvert, xInvert, xIsDate,
                    xTicks, xtickRotation, markers, markevery, zorders, clip_on)

      return ax

    ############################################################
    ##
    def semilogY(self, plotnum, x, y, ptitle=None, xlabel=None, ylabel=None,
                    plotCol=[], linewidths=None, label=[],legendAlpha=0.0,
                    legendLoc='best',
                    pltaxis=None, maxNX=10, maxNY=10, linestyle=None,
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12,
                    xylabelfsize = 12, xytickfsize = 10, labelfsize=10,
                    xScientific=False, yScientific=False,
                    yInvert=False, xInvert=False, drawGrid=True,xIsDate=False,
                     xTicks=None, xtickRotation=0, markers=[],
                    markevery=None, zorders=None, clip_on=True  ):
      """Plot data on linear scales for abscissa and logarithmic scale for ordinates.

        Given an existing figure, this function plots in a specified subplot position.
        The function arguments are described below in some detail. Note that the y-values
        or ordinates can be more than one column, each column representing a different
        line in the plot. This is convenient if large arrays of data must be plotted. If more
        than one column is present, the label argument can contain the legend labels for
        each of the columns/lines.  The pltaxis argument defines the min/max scale values
        for the x and y axes.

            Args:
                | plotnum (int): subplot number, 1-based index
                | x (np.array[N,] or [N,1]): abscissa
                | y (np.array[N,] or [N,M]): ordinates - could be M columns
                | ptitle (string): plot title (optional)
                | xlabel (string): x-axis label (optional)
                | ylabel (string): y-axis label (optional)
                | plotCol ([strings]): plot colour and line style, list with M entries, use default if [] (optional)
                | linewidths ([float]): plot line width in points, list with M entries, use default if None  (optional)
                | label  ([strings]): legend label for ordinate, list withM entries (optional)
                | legendAlpha (float): transparency for legend box (optional)
                | legendLoc (string): location for legend box (optional)
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. Let Matplotlib decide if None. (optional)
                | maxNX (int): draw maxNX+1 tick labels on x axis (optional)
                | maxNY (int): draw maxNY+1 tick labels on y axis (optional)
                | linestyle (string): linestyle for this plot (optional)
                | powerLimits[float]:  scientific tick label power limits [x-low, x-high, y-low, y-high] (optional) (optional)
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x-axis, y-axis label font size, default 12pt (optional)
                | xytickfsize (int): x-axis, y-axis tick font size, default 10pt (optional)
                | labelfsize (int): label/legend font size, default 10pt (optional)
                | xScientific (bool): use scientific notation on x axis (optional)
                | yScientific (bool): use scientific notation on y axis (optional)
                | drawGrid (bool): draw the grid on the plot (optional)
                | yInvert (bool): invert the y-axis (optional)
                | xInvert (bool): invert the x-axis (optional)
                | xIsDate (bool): convert the datetime x-values to dates (optional)
                | xTicks ({tick:label}): dict of x-axis tick locations and associated labels (optional)
                | xtickRotation (float) x-axis tick label rotation angle (optional)
                | markers ([string]) markers to be used for plotting data points (optional)
                | markevery (int | (startind, stride)) subsample when using markers (optional)
                | zorders ([int]) list of zorder for drawing sequence, highest is last (optional)
                | clip_on (bool) clips objects to drawing axes (optional)

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
      """
      ## see self.MyPlot for parameter details.
      pkey = (self.nrow, self.ncol, plotnum)
      if pkey not in list(self.subplots.keys()):
            self.subplots[pkey] = \
                         self.fig.add_subplot(self.nrow,self.ncol, plotnum)
      ax = self.subplots[pkey]

      self.myPlot(ax.semilogy, plotnum, x, y, ptitle,xlabel,ylabel,
                    plotCol, linewidths, label,legendAlpha, legendLoc,
                    pltaxis, maxNX, maxNY, linestyle,
                    powerLimits, titlefsize,
                    xylabelfsize, xytickfsize,
                    labelfsize, drawGrid,
                    xScientific, yScientific,
                    yInvert, xInvert, xIsDate,
                    xTicks, xtickRotation, markers, markevery, zorders, clip_on)

      return ax

    ############################################################
    ##
    def stackplot(self, plotnum, x, y, ptitle=None, xlabel=None, ylabel=None,
                    plotCol=[], linewidths=None, label=[],legendAlpha=0.0,
                    legendLoc='best',
                    pltaxis=None, maxNX=10, maxNY=10, linestyle=None,
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12,
                    xylabelfsize = 12, xytickfsize = 10, labelfsize=10,
                    xScientific=False, yScientific=False,
                    yInvert=False, xInvert=False, drawGrid=True,xIsDate=False,
                     xTicks=None, xtickRotation=0, markers=[],
                    markevery=None, zorders=None, clip_on=True  ):
      """Plot stacked data on linear scales for abscissa and ordinates.

        Given an existing figure, this function plots in a specified subplot position.
        The function arguments are described below in some detail. Note that the y-values
        or ordinates can be more than one column, each column representing a different
        line in the plot.  If more
        than one column is present, the label argument can contain the legend labels for
        each of the columns/lines.  The pltaxis argument defines the min/max scale values
        for the x and y axes.

            Args:
                | plotnum (int): subplot number, 1-based index
                | x (np.array[N,] or [N,1]): abscissa
                | y (np.array[N,] or [N,M]): ordinates - could be M columns
                | ptitle (string): plot title (optional)
                | xlabel (string): x-axis label (optional)
                | ylabel (string): y-axis label (optional)
                | plotCol ([strings]): plot colour and line style, list with M entries, use default if [] (optional)
                | linewidths ([float]): plot line width in points, list with M entries, use default if None  (optional)
                | label  ([strings]): legend label for ordinate, list withM entries (optional)
                | legendAlpha (float): transparency for legend box (optional)
                | legendLoc (string): location for legend box (optional)
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. Let Matplotlib decide if None. (optional)
                | maxNX (int): draw maxNX+1 tick labels on x axis (optional)
                | maxNY (int): draw maxNY+1 tick labels on y axis (optional)
                | linestyle (string): linestyle for this plot (optional)
                | powerLimits[float]:  scientific tick label power limits [x-low, x-high, y-low, y-high] (optional) (optional)
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x-axis, y-axis label font size, default 12pt (optional)
                | xytickfsize (int): x-axis, y-axis tick font size, default 10pt (optional)
                | labelfsize (int): label/legend font size, default 10pt (optional)
                | xScientific (bool): use scientific notation on x axis (optional)
                | yScientific (bool): use scientific notation on y axis (optional)
                | drawGrid (bool): draw the grid on the plot (optional)
                | yInvert (bool): invert the y-axis (optional)
                | xInvert (bool): invert the x-axis (optional)
                | xIsDate (bool): convert the datetime x-values to dates (optional)
                | xTicks ({tick:label}): dict of x-axis tick locations and associated labels (optional)
                | xtickRotation (float) x-axis tick label rotation angle (optional)
                | markers ([string]) markers to be used for plotting data points (optional)
                | markevery (int | (startind, stride)) subsample when using markers (optional)
                | zorders ([int]) list of zorder for drawing sequence, highest is last (optional)
                | clip_on (bool) clips objects to drawing axes (optional)

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
      """
      ## see self.MyPlot for parameter details.
      pkey = (self.nrow, self.ncol, plotnum)
      if pkey not in list(self.subplots.keys()):
            self.subplots[pkey] = \
                         self.fig.add_subplot(self.nrow,self.ncol, plotnum)
      ax = self.subplots[pkey]

      self.myPlot(ax.stackplot, plotnum, x, y.T, ptitle,xlabel,ylabel,
                    plotCol, linewidths, label,legendAlpha, legendLoc,
                    pltaxis, maxNX, maxNY, linestyle,
                    powerLimits, titlefsize,
                    xylabelfsize, xytickfsize,
                    labelfsize, drawGrid,
                    xScientific, yScientific,
                    yInvert, xInvert, xIsDate,
                    xTicks, xtickRotation, markers, markevery, zorders, clip_on)

      return ax



    ############################################################
    ##
    def myPlot(self, plotcommand,plotnum, x, y, ptitle=None,xlabel=None, ylabel=None,
                    plotCol=[], linewidths=None, label=[], legendAlpha=0.0,
                    legendLoc='best',
                    pltaxis=None, maxNX=0, maxNY=0, linestyle=None,
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12,
                    xylabelfsize = 12, xytickfsize = 10,
                    labelfsize=10, drawGrid=True,
                    xScientific=False, yScientific=False,
                    yInvert=False, xInvert=False, xIsDate=False,
                    xTicks=None, xtickRotation=0, markers=[] ,
                    markevery=None, zorders=None, clip_on=True  ):
      """Low level helper function to create a subplot and plot the data as required.

      This function does the actual plotting, labelling etc. It uses the plotting
      function provided by its user functions.

      lineStyles = {
      '': '_draw_nothing',
      ' ': '_draw_nothing',
      'None': '_draw_nothing',
      '--': '_draw_dashed',
      '-.': '_draw_dash_dot',
      '-': '_draw_solid',
      ':': '_draw_dotted'}

          Args:
              | plotcommand: name of a MatplotLib plotting function
              | plotnum (int): subplot number, 1-based index
              | ptitle (string): plot title
              | xlabel (string): x axis label
              | ylabel (string): y axis label
              | x (np.array[N,] or [N,1]): abscissa
              | y (np.array[N,] or [N,M]): ordinates - could be M columns
              | plotCol ([strings]): plot colour and line style, list with M entries, use default if []
              | linewidths ([float]): plot line width in points, list with M entries, use default if None  (optional)
              | label  ([strings]): legend label for ordinate, list with M entries
              | legendAlpha (float): transparency for legend box
              | legendLoc (string): location for legend box (optional)
              | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. Let Matplotlib decide if None.
              | maxNX (int): draw maxNX+1 tick labels on x axis
              | maxNY (int): draw maxNY+1 tick labels on y axis
              | linestyle (string): linestyle for this plot (optional)
              | powerLimits[float]:  scientific tick label power limits [x-low, x-high, y-low, y-high] (optional)
              | titlefsize (int): title font size, default 12pt (optional)
              | xylabelfsize (int): x-axis, y-axis label font size, default 12pt (optional)
              | xytickfsize (int): x-axis, y-axis tick font size, default 10pt (optional)
              | labelfsize (int): label/legend font size, default 10pt (optional)
              | drawGrid (bool): draw the grid on the plot (optional)
              | xScientific (bool): use scientific notation on x axis (optional)
              | yScientific (bool): use scientific notation on y axis (optional)
              | yInvert (bool): invert the y-axis (optional)
              | xInvert (bool): invert the x-axis (optional)
              | xIsDate (bool): convert the datetime x-values to dates (optional)
              | xTicks ({tick:label}): dict of x-axis tick locations and associated labels (optional)
              | xtickRotation (float) x-axis tick label rotation angle (optional)
              | markers ([string]) markers to be used for plotting data points (optional)
              | markevery (int | (startind, stride)) subsample when using markers (optional)
              | zorders ([int]) list of zorder for drawing sequence, highest is last (optional)
              | clip_on (bool) clips objects to drawing axes (optional)

          Returns:
              | the axis object for the plot

          Raises:
              | No exception is raised.
      """

      if x.ndim>1:
          xx=x
      else:
          xx=x.reshape(-1, 1)

      if y.ndim>1:
          yy=y
      else:
          yy=y.reshape(-1, 1)

      # plotCol = self.buildPlotCol(plotCol, yy.shape[1])

      pkey = (self.nrow, self.ncol, plotnum)
      ax = self.subplots[pkey]

      if drawGrid:
          ax.grid(True)
      else:
          ax.grid(False)

      # use scientific format on axes
      #yfm = sbp.yaxis.get_major_formatter()
      #yfm.set_powerlimits([ -3, 3])

      if xlabel is not None:
          ax.set_xlabel(xlabel, fontsize=xylabelfsize)

      if ylabel is not None:
          ax.set_ylabel(ylabel, fontsize=xylabelfsize)

      if xIsDate:
          ax.xaxis_date()
          ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
          ax.xaxis.set_major_locator(mdates.DayLocator())

      if maxNX >0:
          ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(maxNX))
      if maxNY >0:
          ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(maxNY))

      if xScientific:
          # formx = plt.FormatStrFormatter('%.3e')
          formx = plt.ScalarFormatter()
          formx.set_powerlimits([powerLimits[0], powerLimits[1]])
          formx.set_scientific(True)
          ax.xaxis.set_major_formatter(formx)

          # http://matplotlib.1069221.n5.nabble.com/ScalarFormatter-td28042.html
          # http://matplotlib.org/api/ticker_api.html
          # http://matplotlib.org/examples/pylab_examples/newscalarformatter_demo.html
          # ax.xaxis.set_major_formatter( plt.FormatStrFormatter('%d'))
          # http://matplotlib.org/1.3.1/api/axes_api.html#matplotlib.axes.Axes.ticklabel_format
          # plt.ticklabel_format(style='sci', axis='x',
          #      scilimits=(powerLimits[0], powerLimits[1]))

      if yScientific:
          formy = plt.ScalarFormatter()
          formy.set_powerlimits([powerLimits[2], powerLimits[3]])
          formy.set_scientific(True)
          ax.yaxis.set_major_formatter(formy)

      ###############################stacked plot #######################
      if plotcommand==ax.stackplot:
        if not plotCol:
            plotCol = [self.nextPlotCol() for col in range(0,yy.shape[0])]

        ax.stackplot(xx.reshape(-1), yy, colors=plotCol)
        ax.margins(0, 0) # Set margins to avoid "whitespace"

        # creating the legend manually
        ax.legend([mpl.patches.Patch(color=col) for col in plotCol], label,
            loc=legendLoc, framealpha=legendAlpha)

      ###############################line plot #######################
      else: # not a stacked plot
        for i in range(yy.shape[1]):
            #set up the line style, either given or next in sequence
            mmrk = ''
            if markers:
                if i >= len(markers):
                    mmrk = markers[-1]
                else:
                    mmrk = markers[i]

            if plotCol:
                if i >= len(plotCol):
                    col = plotCol[-1]
                else:
                    col = plotCol[i]
            else:
                col = self.nextPlotCol()

            if linestyle is None:
                linestyleL = '-'
            else:
                if type(linestyle) == type([1]):
                    linestyleL = linestyle[i]
                else:
                    linestyleL = linestyle

            if zorders:
              if len(zorders) > 1:
                zorder = zorders[i]
              else:
                zorder = zorders[0]
            else:
              zorder = 2

            if not label:
                if linewidths is not None:
                  plotcommand(xx, yy[:, i], col, label=None, linestyle=linestyleL,
                         marker=mmrk, markevery=markevery, linewidth=linewidths[i],
                         clip_on=clip_on, zorder=zorder)
                else:
                  plotcommand(xx, yy[:, i], col, label=None, linestyle=linestyleL,
                         marker=mmrk, markevery=markevery,
                         clip_on=clip_on, zorder=zorder)
            else:
                if linewidths is not None:
                  # print('***************',linewidths)
                  line, = plotcommand(xx,yy[:,i],col,#label=label[i],
                        linestyle=linestyleL,
                        marker=mmrk, markevery=markevery, linewidth=linewidths[i],
                        clip_on=clip_on, zorder=zorder)
                else:
                  line, = plotcommand(xx,yy[:,i],col,#label=label[i],
                        linestyle=linestyleL,
                        marker=mmrk, markevery=markevery,
                        clip_on=clip_on, zorder=zorder)
                line.set_label(label[i])
                leg = ax.legend( loc=legendLoc, fancybox=True,fontsize=labelfsize)
                leg.get_frame().set_alpha(legendAlpha)
                # ax.legend()
                self.bbox_extra_artists.append(leg)

      if xIsDate:
          plt.gcf().autofmt_xdate()

      #scale the axes
      if pltaxis is not None:
          # ax.axis(pltaxis)
          if not xIsDate:
              ax.set_xlim(pltaxis[0],pltaxis[1])
          ax.set_ylim(pltaxis[2],pltaxis[3])

      if xTicks is not None:
          ticks = ax.set_xticks(list(xTicks.keys()))
          ax.set_xticklabels([xTicks[key] for key in xTicks],
              rotation=xtickRotation, fontsize=xytickfsize)

      if  xTicks is None and xtickRotation is not None:
          ticks = ax.get_xticks()
          if xIsDate:
              from datetime import date
              ticks = [date.fromordinal(int(tick)).strftime('%Y-%m-%d') for tick in ticks]
          ax.set_xticklabels(ticks,
              rotation=xtickRotation, fontsize=xytickfsize)


      if(ptitle is not None):
          ax.set_title(ptitle, fontsize=titlefsize)

      # minor ticks are two points smaller than major
      ax.tick_params(axis='both', which='major', labelsize=xytickfsize)
      ax.tick_params(axis='both', which='minor', labelsize=xytickfsize-2)

      if yInvert:
          ax.set_ylim(ax.get_ylim()[::-1])
      if xInvert:
          ax.set_xlim(ax.get_xlim()[::-1])

      return ax


    ############################################################
    ##
    def emptyPlot(self,plotnum,projection='rectilinear'):
      """Returns a handler to an empty plot.

      This function does not do any plotting, the use must add plots using
      the standard MatPlotLib means.

          Args:
              | plotnum (int): subplot number, 1-based index
              | rectilinear (str): type of axes projection, from  
                ['aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear.].

          Returns:
              | the axis object for the plot

          Raises:
              | No exception is raised.
      """

      pkey = (self.nrow, self.ncol, plotnum)
      if pkey not in list(self.subplots.keys()):
            self.subplots[pkey] = \
                         self.fig.add_subplot(self.nrow,self.ncol, plotnum,projection=projection)
      ax = self.subplots[pkey]

      return ax


    ############################################################
    ##
    def meshContour(self, plotnum, xvals, yvals, zvals, levels=10,
                  ptitle=None, xlabel=None, ylabel=None, shading='flat',
                  plotCol=[], pltaxis=None, maxNX=0, maxNY=0,
                  xScientific=False, yScientific=False,
                  powerLimits=[-4,  2,  -4,  2], titlefsize=12,
                  xylabelfsize=12, xytickfsize=10,
                  meshCmap=cm.rainbow, cbarshow=False, cbarorientation='vertical',
                  cbarcustomticks=[], cbarfontsize=12,
                  drawGrid=False, yInvert=False, xInvert=False,
                  contourFill=True, contourLine=True, logScale=False,
                  negativeSolid=False, zeroContourLine=None,
                  contLabel=False, contFmt='%.2f', contCol='k', contFonSz=8, contLinWid=0.5,
                  zorders=None, clip_on=True ):
      """XY colour mesh  countour plot for (xvals, yvals, zvals) input sets.

        The data values must be given on a fixed mesh grid of three-dimensional
        $(x,y,z)$ array input sets. The mesh grid is defined in $(x,y)$, while the height
        of the mesh is the $z$ value.

        Given an existing figure, this function plots in a specified subplot position.
        Only one contour plot is drawn at a time.  Future contours in the same subplot
        will cover any previous contours.

        The data set must have three two dimensional arrays, each for x, y, and z.
        The data in x, y, and z arrays must have matching data points.  The x and y arrays
        each define the grid in terms of x and y values, i.e., the x array contains the
        x values for the data set, while the y array contains the y values.  The z array
        contains the z values for the corresponding x and y values in the contour mesh.

        Z-values can be plotted on a log scale, in which case the colourbar is adjusted
        to show true values, but on the nonlinear scale.

        The current version only saves png files, since there appears to be a problem
        saving eps files.

        The xvals and yvals vectors may have non-constant grid-intervals, i.e., they do not
        have to be on regular intervals.

            Args:
                | plotnum (int): subplot number, 1-based index
                | xvals (np.array[N,M]): array of x values
                | yvals (np.array[N,M]): array of y values
                | zvals (np.array[N,M]): values on a (x,y) grid
                | levels (int or [float]): number of contour levels or a list of levels (optional)
                | ptitle (string): plot title (optional)
                | xlabel (string): x axis label (optional)
                | ylabel (string): y axis label (optional)
                | shading (string):  not used currently (optional)
                | plotCol ([strings]): plot colour and line style, list with M entries, use default if [] (optional)
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. Let Matplotlib decide if None. (optional)
                | maxNX (int): draw maxNX+1 tick labels on x axis (optional)
                | maxNY (int): draw maxNY+1 tick labels on y axis (optional)
                | xScientific (bool): use scientific notation on x axis (optional)
                | yScientific (bool): use scientific notation on y axis (optional)
                | powerLimits[float]:  scientific tick label power limits [x-low, x-high, y-low, y-high] (optional)
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x-axis, y-axis label font size, default 12pt (optional)
                | xytickfsize (int): x-axis, y-axis tick font size, default 10pt (optional)
                | meshCmap (cm): colour map for the mesh (optional)
                | cbarshow (bool): if true, the show a colour bar (optional)
                | cbarorientation (string): 'vertical' (right) or 'horizontal' (below) (optional)
                | cbarcustomticks zip([z values/float],[tick labels/string])`  define custom colourbar ticks locations for given z values(optional)
                | cbarfontsize (int): font size for colour bar (optional)
                | drawGrid (bool): draw the grid on the plot (optional)
                | yInvert (bool): invert the y-axis. Flip the y-axis up-down (optional)
                | xInvert (bool): invert the x-axis. Flip the x-axis left-right (optional)
                | contourFill (bool): fill contours with colour (optional)
                | contourLine (bool): draw a series of contour lines (optional)
                | logScale (bool): do Z values on log scale, recompute colourbar values (optional)
                | negativeSolid (bool): draw negative contours in solid lines, dashed otherwise (optional)
                | zeroContourLine (double): draw a single contour at given value (optional)
                | contLabel (bool): label the contours with values (optional)
                | contFmt (string): contour label c-printf format (optional)
                | contCol (string): contour label colour, e.g., 'k' (optional)
                | contFonSz (float): contour label fontsize (optional)
                | contLinWid (float): contour line width in points (optional)
                | zorder ([int]) list of zorder for drawing sequence, highest is last (optional)
                | clip_on (bool) clips objects to drawing axes (optional)

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
      """

      #to rank 2
      xx=xvals.reshape(-1, 1)
      yy=yvals.reshape(-1, 1)

      #if this is a log scale plot
      if logScale is True:
          zvals = np.log10(zvals)

      contour_negative_linestyle = plt.rcParams['contour.negative_linestyle']
      if contourLine:
          if negativeSolid:
              plt.rcParams['contour.negative_linestyle'] = 'solid'
          else:
              plt.rcParams['contour.negative_linestyle'] = 'dashed'

      #create subplot if not existing
      if (self.nrow,self.ncol, plotnum) not in list(self.subplots.keys()):
          self.subplots[(self.nrow,self.ncol, plotnum)] = \
               self.fig.add_subplot(self.nrow,self.ncol, plotnum)

      #get axis
      ax = self.subplots[(self.nrow,self.ncol, plotnum)]

      if drawGrid:
          ax.grid(True)
      else:
          ax.grid(False)

      if xlabel is not None:
          ax.set_xlabel(xlabel, fontsize=xylabelfsize)
          if xScientific:
              formx = plt.ScalarFormatter()
              formx.set_scientific(True)
              formx.set_powerlimits([powerLimits[0], powerLimits[1]])
              ax.xaxis.set_major_formatter(formx)
      if ylabel is not None:
          ax.set_ylabel(ylabel, fontsize=xylabelfsize)
          if yScientific:
              formy = plt.ScalarFormatter()
              formy.set_powerlimits([powerLimits[2], powerLimits[3]])
              formy.set_scientific(True)
              ax.yaxis.set_major_formatter(formy)

      if maxNX >0:
          ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(maxNX))
      if maxNY >0:
          ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(maxNY))

      if plotCol:
          col = plotCol[0]
      else:
          col = self.nextPlotCol()

      if zorders:
        if len(zorders) > 1:
          zorder = zorders[i]
        else:
          zorder = zorders[0]
      else:
        zorder = 2


      #do the plot
      if contourFill:
          pmplotcf = ax.contourf(xvals, yvals, zvals, levels,
              cmap=meshCmap, zorder=zorder, clip_on=clip_on)

      if contourLine:
          pmplot = ax.contour(xvals, yvals, zvals, levels, cmap=None, linewidths=contLinWid,
               colors=col, zorder=zorder, clip_on=clip_on)

      if zeroContourLine:
          pmplot = ax.contour(xvals, yvals, zvals, (zeroContourLine,), cmap=None, linewidths=contLinWid,
               colors=col, zorder=zorder, clip_on=clip_on)


      if contLabel: # and not contourFill:
        plt.clabel(pmplot, fmt = contFmt, colors = contCol, fontsize=contFonSz, zorder=zorder, clip_on=clip_on)

      if cbarshow and (contourFill):
          #http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes
          divider = make_axes_locatable(ax)
          if cbarorientation == 'vertical':
            cax = divider.append_axes("right", size="5%", pad=0.05)
          else:
            cax = divider.append_axes("bottom", size="5%", pad=0.1)

          if not cbarcustomticks:
              # cbar = self.fig.colorbar(pmplotcf,orientation=cbarorientation)
              cbar = self.fig.colorbar(pmplotcf,cax=cax)
              if logScale:
                  cbartickvals = cbar.ax.yaxis.get_ticklabels()
                  tickVals = []
                  # need this roundabout trick to handle minus sign in unicode
                  for item in cbartickvals:
                      valstr = float(item.get_text().replace(u'\N{MINUS SIGN}', '-').replace('$',''))
                      # valstr = item.get_text().replace('\u2212', '-').replace('$','')
                      val = 10**float(valstr)
                      if abs(val) < 1000:
                          str = '{0:f}'.format(val)
                      else:
                          str = '{0:e}'.format(val)
                      tickVals.append(str)
                  cbartickvals = cbar.ax.yaxis.set_ticklabels(tickVals)
          else:
              ticks,  ticklabels = list(zip(*cbarcustomticks))
              # cbar = self.fig.colorbar(pmplotcf,ticks=ticks, orientation=cbarorientation)
              cbar = self.fig.colorbar(pmplotcf,ticks=ticks, cax=cax)
              if cbarorientation == 'vertical':
                  cbar.ax.set_yticklabels(ticklabels)
              else:
                  cbar.ax.set_xticklabels(ticklabels)

          if cbarorientation == 'vertical':
              for t in cbar.ax.get_yticklabels():
                   t.set_fontsize(cbarfontsize)
          else:
              for t in cbar.ax.get_xticklabels():
                   t.set_fontsize(cbarfontsize)

      #scale the axes
      if pltaxis is not None:
          ax.axis(pltaxis)

      if(ptitle is not None):
          ax.set_title(ptitle, fontsize=titlefsize)

      # minor ticks are two points smaller than major
      ax.tick_params(axis='both', which='major', labelsize=xytickfsize)
      ax.tick_params(axis='both', which='minor', labelsize=xytickfsize-2)

      if yInvert:
          ax.set_ylim(ax.get_ylim()[::-1])
      if xInvert:
          ax.set_xlim(ax.get_xlim()[::-1])

      plt.rcParams['contour.negative_linestyle'] = contour_negative_linestyle

      return ax

    ############################################################
    ##
    def mesh3D(self, plotnum, xvals, yvals, zvals,
                  ptitle=None, xlabel=None, ylabel=None, zlabel=None,
                  rstride=1, cstride=1, linewidth=0,
                  plotCol=None, edgeCol=None, pltaxis=None, maxNX=0, maxNY=0, maxNZ=0,
                  xScientific=False, yScientific=False, zScientific=False,
                  powerLimits=[-4,  2,  -4,  2, -2, 2], titlefsize=12,
                  xylabelfsize=12, xytickfsize=10, wireframe=False, surface=True,
                  cmap=cm.rainbow, cbarshow=False,
                  cbarorientation = 'vertical', cbarcustomticks=[], cbarfontsize = 12,
                  drawGrid=True, xInvert=False, yInvert=False, zInvert=False,
                  logScale=False, alpha=1, alphawire=1,
                  azim=45, elev=30, distance=10, zorders=None, clip_on=True
                   ):
      """XY colour mesh plot for (xvals, yvals, zvals) input sets.

        Given an existing figure, this function plots in a specified subplot position.
        Only one mesh is drawn at a time.  Future meshes in the same subplot
        will cover any previous meshes.

        The mesh grid is defined in (x,y), while the height of the mesh is the z value.

        The data set must have three two dimensional arrays, each for x, y, and z.
        The data in x, y, and z arrays must have matching data points.
        The x and y arrays each define the grid in terms of x and y values, i.e.,
        the x array contains the x values for the data set, while the y array
        contains the y values.  The z array contains the z values for the
        corresponding x and y values in the mesh.

        Use wireframe=True to obtain a wireframe plot.

        Use surface=True to obtain a surface plot with fill colours.

        Z-values can be plotted on a log scale, in which case the colourbar is adjusted
        to show true values, but on the nonlinear scale.

        The xvals and yvals vectors may have non-constant grid-intervals, i.e.,
        they do not have to be on regular intervals, but z array must correspond
        to the (x,y) grid.

            Args:
                | plotnum (int): subplot number, 1-based index
                | xvals (np.array[N,M]): array of x values, corresponding to (x,y) grid
                | yvals (np.array[N,M]): array of y values, corresponding to (x,y) grid
                | zvals (np.array[N,M]): array of z values, corresponding to (x,y) grid
                | ptitle (string): plot title (optional)
                | xlabel (string): x axis label (optional)
                | ylabel (string): y axis label (optional)
                | zlabel (string): z axis label (optional)
                | rstride (int): mesh line row (y axis) stride, every rstride value along y axis (optional)
                | cstride (int): mesh line column (x axis)  stride, every cstride value along x axis (optional)
                | linewidth (float): mesh line width in points (optional)
                | plotCol ([strings]): fill colour, list with M=1 entries, use default if None (optional)
                | edgeCol ([strings]): mesh line colour , list with M=1 entries, use default if None (optional)
                | pltaxis ([xmin, xmax, ymin, ymax]): scale for x,y axes. z scale is not settable.  Let Matplotlib decide if None (optional)
                | maxNX (int): draw maxNX+1 tick labels on x axis (optional)
                | maxNY (int): draw maxNY+1 tick labels on y axis (optional)
                | maxNZ (int): draw maxNY+1 tick labels on z axis (optional)
                | xScientific (bool): use scientific notation on x axis (optional)
                | yScientific (bool): use scientific notation on y axis (optional)
                | zScientific (bool): use scientific notation on z-axis (optional)
                | powerLimits[float]:  scientific tick label power limits [x-neg, x-pos, y-neg, y-pos, z-neg, z-pos]  (optional)
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x-axis, y-axis, z-axis label font size, default 12pt (optional)
                | xytickfsize (int): x-axis, y-axis, z-axis tick font size, default 10pt (optional)
                | wireframe (bool): If True, do a wireframe plot,  (optional)
                | surface (bool): If True, do a surface plot,  (optional)
                | cmap (cm): color map for the mesh (optional)
                | cbarshow (bool): if true, the show a color bar (optional)
                | cbarorientation (string): 'vertical' (right) or 'horizontal' (below) (optional)
                | cbarcustomticks zip([z values/float],[tick labels/string]):  define custom colourbar ticks locations for given z values(optional)
                | cbarfontsize (int): font size for color bar (optional)
                | drawGrid (bool): draw the grid on the plot (optional)
                | xInvert (bool): invert the x-axis. Flip the x-axis left-right (optional)
                | yInvert (bool): invert the y-axis. Flip the y-axis left-right (optional)
                | zInvert (bool): invert the z-axis. Flip the z-axis up-down (optional)
                | logScale (bool): do Z values on log scale, recompute colourbar vals (optional)
                | alpha (float): surface transparency (optional)
                | alphawire (float): mesh transparency (optional)
                | azim (float): graph view azimuth angle  [degrees] (optional)
                | elev (float): graph view evelation angle  [degrees] (optional)
                | distance (float): distance between viewer and plot (optional)
                | zorder ([int]) list of zorder for drawing sequence, highest is last (optional)
                | clip_on (bool) clips objects to drawing axes (optional)

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
      """

      from mpl_toolkits.mplot3d.axes3d import Axes3D

      #if this is a log scale plot
      if logScale is True:
          zvals = np.log10(zvals)

      #create subplot if not existing
      if (self.nrow,self.ncol, plotnum) not in list(self.subplots.keys()):
          self.subplots[(self.nrow,self.ncol, plotnum)] = \
               self.fig.add_subplot(self.nrow,self.ncol, plotnum, projection='3d')

      #get axis
      ax = self.subplots[(self.nrow,self.ncol, plotnum)]

      if drawGrid:
          ax.grid(True)
      else:
          ax.grid(False)

      if xlabel is not None:
          ax.set_xlabel(xlabel, fontsize=xylabelfsize)
          if xScientific:
              formx = plt.ScalarFormatter()
              formx.set_scientific(True)
              formx.set_powerlimits([powerLimits[0], powerLimits[1]])
              ax.xaxis.set_major_formatter(formx)
      if ylabel is not None:
          ax.set_ylabel(ylabel, fontsize=xylabelfsize)
          if yScientific:
              formy = plt.ScalarFormatter()
              formy.set_powerlimits([powerLimits[2], powerLimits[3]])
              formy.set_scientific(True)
              ax.yaxis.set_major_formatter(formy)
      if zlabel is not None:
          ax.set_zlabel(zlabel, fontsize=xylabelfsize)
          if zScientific:
              formz = plt.ScalarFormatter()
              formz.set_powerlimits([powerLimits[4], powerLimits[5]])
              formz.set_scientific(True)
              ax.zaxis.set_major_formatter(formz)

      if maxNX >0:
          ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(maxNX))
      if maxNY >0:
          ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(maxNY))
      if maxNZ >0:
          ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(maxNZ))

      if plotCol:
          col = plotCol[0]
      else:
          col = self.nextPlotCol()

      if edgeCol:
          edcol = edgeCol[0]
      else:
          edcol = self.nextPlotCol()

      if zorders:
        if len(zorders) > 1:
          zorder = zorders[i]
        else:
          zorder = zorders[0]
      else:
        zorder = 1

      #do the plot

      if surface:
        pmplot = ax.plot_surface(xvals, yvals, zvals, rstride=rstride, cstride=cstride,
              edgecolor=edcol, cmap=cmap, linewidth=linewidth, alpha=alpha,
              zorder=zorder, clip_on=clip_on)

      if wireframe:
        pmplot = ax.plot_wireframe(xvals, yvals, zvals, rstride=rstride, cstride=cstride,
              color=col, edgecolor=edcol, linewidth=linewidth, alpha=alphawire,
              zorder=zorder, clip_on=clip_on)


      ax.view_init(azim=azim, elev=elev)
      ax.dist = distance

      if cbarshow is True and cmap is not None:
        #http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes
        # divider = make_axes_locatable(ax)
        # if cbarorientation == 'vertical':
        #   cax = divider.append_axes("right", size="5%", pad=0.05)
        # else:
        #   cax = divider.append_axes("bottom", size="5%", pad=0.1)

        if not cbarcustomticks:
              cbar = self.fig.colorbar(pmplot,orientation=cbarorientation)
              # cbar = self.fig.colorbar(pmplot,cax=cax)
              if logScale:
                  cbartickvals = cbar.ax.yaxis.get_ticklabels()
                  tickVals = []
                  # need this roundabout trick to handle minus sign in unicode
                  for item in cbartickvals:
                      valstr = item.get_text().replace('\u2212', '-').replace('$','')
                      val = 10**float(valstr)
                      if abs(val) < 1000:
                          str = '{0:f}'.format(val)
                      else:
                          str = '{0:e}'.format(val)
                      tickVals.append(str)
                  cbartickvals = cbar.ax.yaxis.set_ticklabels(tickVals)
        else:
              ticks,  ticklabels = list(zip(*cbarcustomticks))
              cbar = self.fig.colorbar(pmplot,ticks=ticks, orientation=cbarorientation)
              # cbar = self.fig.colorbar(pmplot,ticks=ticks, cax=cax)
              if cbarorientation == 'vertical':
                  cbar.ax.set_yticklabels(ticklabels)
              else:
                  cbar.ax.set_xticklabels(ticklabels)

        if cbarorientation == 'vertical':
              for t in cbar.ax.get_yticklabels():
                   t.set_fontsize(cbarfontsize)
        else:
              for t in cbar.ax.get_xticklabels():
                   t.set_fontsize(cbarfontsize)

      if(ptitle is not None):
          plt.title(ptitle, fontsize=titlefsize)

      #scale the axes
      if pltaxis is not None:
          # ax.axis(pltaxis)
          ax.set_xlim(pltaxis[0], pltaxis[1])
          ax.set_ylim(pltaxis[2], pltaxis[3])
          ax.set_zlim(pltaxis[4], pltaxis[5])

      if(ptitle is not None):
          ax.set_title(ptitle, fontsize=titlefsize)

      # minor ticks are two points smaller than major
      ax.tick_params(axis='both', which='major', labelsize=xytickfsize)
      ax.tick_params(axis='both', which='minor', labelsize=xytickfsize-2)

      if xInvert:
          ax.set_xlim(ax.get_xlim()[::-1])
      if yInvert:
          ax.set_ylim(ax.get_ylim()[::-1])
      if zInvert:
          ax.set_zlim(ax.get_zlim()[::-1])


      return ax

    ############################################################
    ##
    def polar(self, plotnum, theta, r, ptitle=None, \
                    plotCol=None, label=[],labelLocation=[-0.1, 0.1], \
                    highlightNegative=True, highlightCol='#ffff00', highlightWidth=4,\
                    legendAlpha=0.0, \
                    rscale=None, rgrid=[0,5], thetagrid=[30], \
                    direction='counterclockwise', zerooffset=0, titlefsize=12, drawGrid=True,
                    zorders=None, clip_on=True, markers=[], markevery=None,
):
      """Create a subplot and plot the data in polar coordinates (linear radial orginates only).

        Given an existing figure, this function plots in a specified subplot position.
        The function arguments are described below in some detail. Note that the radial values
        or ordinates can be more than one column, each column representing a different
        line in the plot. This is convenient if large arrays of data must be plotted. If more
        than one column is present, the label argument can contain the legend labels for
        each of the columns/lines.  The scale for the radial ordinates can be set with rscale.
        The number of radial grid circles can be set with rgrid - this provides a somewhat
        better control over the built-in radial grid in matplotlib. thetagrids defines the angular
        grid interval.  The angular rotation direction can be set to be clockwise or
        counterclockwise. Likewise, the rotation offset where the plot zero angle must be,
        is set with `zerooffset`.

        For some obscure reason Matplitlib version 1.13 does not plot negative values on the
        polar plot.  We therefore force the plot by making the values positive and then highlight it as negative.

            Args:
                | plotnum (int): subplot number, 1-based index
                | theta (np.array[N,] or [N,1]): angular abscissa in radians
                | r (np.array[N,] or [N,M]): radial ordinates - could be M columns
                | ptitle (string): plot title (optional)
                | plotCol ([strings]): plot colour and line style, list with M entries, use default if None (optional)
                | label  ([strings]): legend label, list with M entries (optional)
                | labelLocation ([x,y]): where the legend should located (optional)
                | highlightNegative (bool): indicate if negative data must be highlighted (optional)
                | highlightCol (string): negative highlight colour string (optional)
                | highlightWidth (int): negative highlight line width(optional)
                | legendAlpha (float): transparency for legend box (optional)
                | rscale ([rmin, rmax]): radial plotting limits. use default setting if None.
                  If rmin is negative the zero is a circle and rmin is at the centre of the graph (optional)
                | rgrid ([rinc, numinc]): radial grid, use default is [0,5].
                  If rgrid is None don't show. If rinc=0 then numinc is number of intervals.
                  If rinc is not zero then rinc is the increment and numinc is ignored (optional)
                | thetagrids (float): theta grid interval [degrees], if None don't show (optional)
                | direction (string): direction in increasing angle, 'counterclockwise' or 'clockwise' (optional)
                | zerooffset (float):  rotation offset where zero should be [rad]. Positive
                  zero-offset rotation is counterclockwise from 3'o'clock (optional)
                | titlefsize (int): title font size, default 12pt (optional)
                | drawGrid (bool): draw a grid on the graph (optional)
                | zorder ([int]) list of zorder for drawing sequence, highest is last (optional)
                | clip_on (bool) clips objects to drawing axes (optional)
                | markers ([string]) markers to be used for plotting data points (optional)
                | markevery (int | (startind, stride)) subsample when using markers (optional)

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
      """

      if theta.ndim>1:
          tt=theta
      else:
          tt=theta.reshape(-1, 1)

      if r.ndim>1:
          rr=r
      else:
          rr=r.reshape(-1, 1)


      MakeAbs = True
      if rscale is not None:
          if rscale[0] < 0:
              MakeAbs = False
          else:
              highlightNegative=True #override the function value
      else:
          highlightNegative=True #override the function value

      #plotCol = self.buildPlotCol(plotCol, rr.shape[1])

      ax = None
      pkey = (self.nrow, self.ncol, plotnum)
      if pkey not in list(self.subplots.keys()):
          self.subplots[pkey] = \
                       self.fig.add_subplot(self.nrow,self.ncol, plotnum, polar=True)

      ax = self.subplots[pkey]

      ax.grid(drawGrid)

      rmax=0

      for i in range(rr.shape[1]):
          # negative val :forcing positive and phase shifting
          # if forceAbsolute:
          #     ttt = tt + np.pi*(rr[:, i] < 0).reshape(-1, 1)
          #     rrr = np.abs(rr[:, i])
          # else:
          ttt = tt.reshape(-1,)
          rrr = rr[:, i].reshape(-1,)

          #print(rrr)

          if highlightNegative:
              #find zero crossings in data
              zero_crossings = np.where(np.diff(np.sign(rr),axis=0))[0] + 1
              #split the input into different subarrays according to crossings
              negrrr = np.split(rr,zero_crossings)
              negttt = np.split(tt,zero_crossings)

              # print('zero crossing',zero_crossings)
              # print(len(negrrr))
              # print(negrrr)

          mmrk = ''
          if markers:
              if i >= len(markers):
                  mmrk = markers[-1]
              else:
                  mmrk = markers[i]


          #set up the line style, either given or next in sequence
          if plotCol:
              col = plotCol[i]
          else:
              col = self.nextPlotCol()

          # print('p',ttt.shape)
          # print('p',rrr.shape)
          if zorders:
            if len(zorders) > 1:
              zorder = zorders[i]
            else:
              zorder = zorders[0]
          else:
            zorder = 2

          if not label:
              if highlightNegative:
                  lines = ax.plot(ttt, rrr, col, clip_on=clip_on, zorder=zorder,marker=mmrk, markevery=markevery)
                  neglinewith = highlightWidth*plt.getp(lines[0],'linewidth')
                  for ii in range(0,len(negrrr)):
                      if len(negrrr[ii]) > 0:
                          if negrrr[ii][0] < 0:
                              if MakeAbs:
                                  ax.plot(negttt[ii], np.abs(negrrr[ii]), highlightCol,
                                    linewidth=neglinewith, clip_on=clip_on, zorder=zorder,
                                    marker=mmrk, markevery=markevery,)
                              else:
                                  ax.plot(negttt[ii], negrrr[ii], highlightCol,
                                    linewidth=neglinewith, clip_on=clip_on, zorder=zorder,
                                    marker=mmrk, markevery=markevery)
              ax.plot(ttt, rrr, col, clip_on=clip_on, zorder=zorder,marker=mmrk, markevery=markevery)
              rmax = np.maximum(np.abs(rrr).max(), rmax)
              rmin = 0
          else:
              if highlightNegative:
                  lines = ax.plot(ttt, rrr, col, clip_on=clip_on, zorder=zorder,marker=mmrk, markevery=markevery)
                  neglinewith = highlightWidth*plt.getp(lines[0],'linewidth')
                  for ii in range(0,len(negrrr)):
                      if len(negrrr[ii]) > 0:
                          # print(len(negrrr[ii]))
                          # if negrrr[ii][0] < 0:
                          if negrrr[ii][0][0] < 0:
                              if MakeAbs:
                                  ax.plot(negttt[ii], np.abs(negrrr[ii]), highlightCol,
                                    linewidth=neglinewith, clip_on=clip_on, zorder=zorder,
                                    marker=mmrk, markevery=markevery)
                              else:
                                  ax.plot(negttt[ii], negrrr[ii], highlightCol,
                                    linewidth=neglinewith, clip_on=clip_on, zorder=zorder,
                                    marker=mmrk, markevery=markevery)
              ax.plot(ttt, rrr, col,label=label[i], clip_on=clip_on, zorder=zorder,marker=mmrk, markevery=markevery)
              rmax=np.maximum(np.abs(rrr).max(), rmax)
              rmin = 0

          if MakeAbs:
              ax.plot(ttt, np.abs(rrr), col, clip_on=clip_on, zorder=zorder,marker=mmrk, markevery=markevery)
          else:
              ax.plot(ttt, rrr, col, clip_on=clip_on, zorder=zorder,marker=mmrk, markevery=markevery)

      if label:
          fontP = mpl.font_manager.FontProperties()
          fontP.set_size('small')
          leg = ax.legend(loc='upper left',
                  bbox_to_anchor=(labelLocation[0], labelLocation[1]),
                  prop = fontP, fancybox=True)
          leg.get_frame().set_alpha(legendAlpha)
          self.bbox_extra_artists.append(leg)


      ax.set_theta_direction(direction)
      ax.set_theta_offset(zerooffset)


      #set up the grids
      if thetagrid is None:
        ax.set_xticklabels([])
      else:
        plt.thetagrids(list(range(0, 360, thetagrid[0])))

      #Set increment and maximum radial limits
      if rscale is None:
        rscale = [rmin, rmax]

      if rgrid is None:
        ax.set_yticklabels([])
      else:
        if rgrid[0] is 0:
          ax.set_yticks(np.linspace(rscale[0],rscale[1],rgrid[1]))
        if rgrid[0] is not 0:
          numrgrid = (rscale[1] - rscale[0] ) / rgrid[0]
          ax.set_yticks(np.linspace(rscale[0],rscale[1],numrgrid++1.000001))

      ax.set_ylim(rscale[0],rscale[1])

      if(ptitle is not None):
          ax.set_title(ptitle, fontsize=titlefsize, \
              verticalalignment ='bottom', horizontalalignment='center')

      return ax

    ############################################################
    ##
    def showImage(self, plotnum, img,  ptitle=None, xlabel=None, ylabel=None,
                  cmap=plt.cm.gray, titlefsize=12, cbarshow=False,
                  cbarorientation = 'vertical', cbarcustomticks=[], cbarfontsize = 12,
                  labelfsize=10, xylabelfsize = 12,interpolation=None):
      """Creates a subplot and show the image using the colormap provided.

            Args:
                | plotnum (int): subplot number, 1-based index
                | img (np.ndarray): numpy 2d array containing the image
                | ptitle (string): plot title (optional)
                | xlabel (string): x axis label (optional)
                | ylabel (string): y axis label (optional)
                | cmap: matplotlib colormap, default gray (optional)
                | fsize (int): title font size, default 12pt (optional)
                | cbarshow (bool): if true, the show a colour bar (optional)
                | cbarorientation (string): 'vertical' (right) or 'horizontal' (below)  (optional)
                | cbarcustomticks zip([tick locations/float],[tick labels/string]): locations in image grey levels  (optional)
                | cbarfontsize (int): font size for colour bar  (optional)
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x-axis, y-axis label font size, default 12pt (optional)
                | interpolation (str):   'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'(optional, see pyplot.imshow)

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
      """
    #http://matplotlib.sourceforge.net/examples/pylab_examples/colorbar_tick_labelling_demo.html
    #http://matplotlib.1069221.n5.nabble.com/Colorbar-Ticks-td21289.html

      pkey = (self.nrow, self.ncol, plotnum)
      if pkey not in list(self.subplots.keys()):
          self.subplots[pkey] = \
                       self.fig.add_subplot(self.nrow,self.ncol, plotnum)

      ax = self.subplots[pkey]

      cimage = ax.imshow(img, cmap,interpolation=interpolation)

      if xlabel is not None:
          ax.set_xlabel(xlabel, fontsize=xylabelfsize)

      if ylabel is not None:
          ax.set_ylabel(ylabel, fontsize=xylabelfsize)

      ax.axis('off')
      if cbarshow is True:
          #http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes
          divider = make_axes_locatable(ax)
          if cbarorientation == 'vertical':
            cax = divider.append_axes("right", size="5%", pad=0.05)
          # else:
          #   cay = divider.append_axes("bottom", size="5%", pad=0.1)

          if not cbarcustomticks:
            if cbarorientation == 'vertical':
              cbar = self.fig.colorbar(cimage,cax=cax)
            else:
              cbar = self.fig.colorbar(cimage,orientation=cbarorientation)

          else:
              ticks,  ticklabels = list(zip(*cbarcustomticks))

              if cbarorientation == 'vertical':
                cbar = self.fig.colorbar(cimage,ticks=ticks, cax=cax)
              else:
                cbar = self.fig.colorbar(cimage,ticks=ticks, orientation=cbarorientation)

              if cbarorientation == 'vertical':
                  cbar.ax.set_yticklabels(ticklabels)
              else:
                  cbar.ax.set_xticklabels(ticklabels)


          if cbarorientation == 'vertical':
              for t in cbar.ax.get_yticklabels():
                   t.set_fontsize(cbarfontsize)
          else:
            for t in cbar.ax.get_xticklabels():
                t.set_fontsize(cbarfontsize)

      if(ptitle is not None):
          ax.set_title(ptitle, fontsize=titlefsize)

      return ax


    ############################################################
    ##
    def plot3d(self, plotnum, x, y, z, ptitle=None, xlabel=None, ylabel=None, zlabel=None,
               plotCol=[], linewidths=None, pltaxis=None, label=None, legendAlpha=0.0, titlefsize=12,
               xylabelfsize = 12, xInvert=False, yInvert=False, zInvert=False,scatter=False,
               markers=None, markevery=None, azim=45, elev=30, zorders=None, clip_on=True, edgeCol=None):
        """3D plot on linear scales for x y z input sets.

        Given an existing figure, this function plots in a specified subplot position.
        The function arguments are described below in some detail.

        Note that multiple 3D data sets can be plotted simultaneously by adding additional
        columns to the input coordinates of  the (x,y,z) arrays, each set of columns representing
        a different line in the plot. This is convenient if large arrays of data must
        be plotted. If more than one column is present, the label argument can contain the
        legend labels for each of the columns/lines.

            Args:
                | plotnum (int): subplot number, 1-based index
                | x (np.array[N,] or [N,M]) x coordinates of each line.
                | y (np.array[N,] or [N,M]) y coordinates of each line.
                | z (np.array[N,] or [N,M]) z coordinates of each line.
                | ptitle (string): plot title (optional)
                | xlabel (string): x-axis label (optional)
                | ylabel (string): y-axis label (optional)
                | zlabel (string): z axis label (optional)
                | plotCol ([strings]): plot colour and line style, list with M entries, use default if None (optional)
                | linewidths ([float]): plot line width in points, list with M entries, use default if None  (optional)
                | pltaxis ([xmin, xmax, ymin, ymax, zmin, zmax])  scale for x,y,z axes.  Let Matplotlib decide if None. (optional)
                | label  ([strings]): legend label for ordinate, list with M entries (optional)
                | legendAlpha (float): transparency for legend box (optional)
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x, y, z label font size, default 12pt (optional)
                | xInvert (bool): invert the x-axis (optional)
                | yInvert (bool): invert the y-axis (optional)
                | zInvert (bool): invert the z-axis (optional)
                | scatter (bool): draw only the points, no lines (optional)
                | markers ([string]): markers to be used for plotting data points (optional)
                | markevery (int | (startind, stride)): subsample when using markers (optional)
                | azim (float): graph view azimuth angle  [degrees] (optional)
                | elev (float): graph view evelation angle  [degrees] (optional)
                | zorder ([int]): list of zorder for drawing sequence, highest is last (optional)
                | clip_on (bool): clips objects to drawing axes (optional)
                | edgeCol ([int]): list of colour specs, value at [0] used for edge colour (optional).
            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
        """

        # if required convert 1D arrays into 2D arrays
        if x.ndim < 2:
            x = x.reshape(-1,1)
        if y.ndim < 2:
            y = y.reshape(-1,1)
        if z.ndim < 2:
            z = z.reshape(-1,1)

        # if not plotCol:
        #     plotCol = self.nextPlotCol()
        # else:
        # plotCol = self.buildPlotCol(plotCol, x.shape[-1])



        if (self.nrow,self.ncol, plotnum) not in list(self.subplots.keys()):
            self.subplots[(self.nrow,self.ncol, plotnum)] = \
               self.fig.add_subplot(self.nrow,self.ncol, plotnum, projection='3d')

        ax = self.subplots[(self.nrow,self.ncol, plotnum)]

        # print(x.shape[-1])

        for i in range(x.shape[-1]):
            if plotCol:
                if i >= len(plotCol):
                    col = plotCol[-1]
                else:
                    col = plotCol[i]
            else:
                col = self.nextPlotCol()

            if markers:
                marker = markers[i]
            else:
                marker = None

            if zorders:
                if len(zorders) > 1:
                  zorder = zorders[i]
                else:
                  zorder = zorders[0]
            else:
                zorder = 2

            if linewidths is not None:
                if scatter:
                    ax.scatter(x[:,i], y[:,i], z[:,i], c=col, linewidth=linewidths[i],
                        marker=marker, zorder=zorder, clip_on=clip_on)
                else:
                    ax.plot(x[:,i], y[:,i], z[:,i], c=col, linewidth=linewidths[i],
                        marker=marker,markevery=markevery, zorder=zorder, clip_on=clip_on)
            else:
                if scatter:
                    ax.scatter(x[:,i], y[:,i], z[:,i], c=col, marker=marker,
                        zorder=zorder, clip_on=clip_on)
                else:
                    ax.plot(x[:,i], y[:,i], z[:,i], c=col, marker=marker,
                        markevery=markevery, zorder=zorder, clip_on=clip_on)

        if edgeCol:
          edcol = edgeCol
        else:
          edcol = self.nextPlotCol()


        #scale the axes
        if pltaxis is not None:
            # ax.axis(pltaxis)
            # if not xIsDate:
            ax.set_xlim(pltaxis[0],pltaxis[1])
            ax.set_ylim(pltaxis[2],pltaxis[3])
            ax.set_zlim(pltaxis[4],pltaxis[5])

        ax.view_init(azim=azim, elev=elev)


        if xInvert:
            ax.set_xlim(ax.get_xlim()[::-1])
        if yInvert:
            ax.set_ylim(ax.get_ylim()[::-1])
        if zInvert:
            ax.set_zlim(ax.get_zlim()[::-1])

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize = xylabelfsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize = xylabelfsize)
        if zlabel is not None:
            ax.set_zlabel(zlabel, fontsize = xylabelfsize)

        if label is not None:
            leg = plt.legend(label, loc='best', fancybox=True)
            leg.get_frame().set_alpha(legendAlpha)
            self.bbox_extra_artists.append(leg)

        if(ptitle is not None):
           plt.title(ptitle, fontsize=titlefsize)

        return ax

    ############################################################
    ##
    def polar3d(self, plotnum, theta, radial, zvals, ptitle=None,
                xlabel=None, ylabel=None, zlabel=None, zscale=None,
               titlefsize=12, xylabelfsize = 12,
               thetaStride=1, radialstride=1, meshCmap = cm.rainbow,
               linewidth=0.1, azim=45, elev=30, zorders=None, clip_on=True,
               facecolors=None, alpha=1, edgeCol=None):
      """3D polar surface/mesh plot for (r, theta, zvals) input sets.

        Given an existing figure, this function plots in a specified subplot position.

        Only one mesh is drawn at a time.  Future meshes in the same subplot
        will cover any previous meshes.

        The data in zvals must be on a grid where the theta vector correspond to
        the number of rows in zvals and the radial vector corresponds to the
        number of columns in zvals.

        The r and p vectors may have non-constant grid-intervals, i.e., they do not
        have to be on regular intervals.

            Args:
                | plotnum (int): subplot number, 1-based index
                | theta (np.array[N,M]): array of angular values [0..2pi] corresponding to (theta,rho) grid.
                | radial (np.array[N,M]): array of radial values  corresponding to (theta,rho) grid.
                | zvals (np.array[N,M]): array of z values  corresponding to (theta,rho) grid.
                | ptitle (string): plot title (optional)
                | xlabel (string): x-axis label (optional)
                | ylabel (string): y-axis label (optional)
                | zlabel (string): z-axis label (optional)
                | zscale ([float]): z axis [min, max] in the plot.
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x, y, z label font size, default 12pt (optional)
                | thetaStride (int): theta stride in input data (optional)
                | radialstride (int): radial stride in input data  (optional)
                | meshCmap (cm): color map for the mesh (optional)
                | linewidth (float): width of the mesh lines
                | azim (float): graph view azimuth angle  [degrees] (optional)
                | elev (float): graph view evelation angle  [degrees] (optional)
                | zorder ([int]) list of zorder for drawing sequence, highest is last (optional)
                | clip_on (bool) clips objects to drawing axes (optional)
                | facecolors ((np.array[N,M]): array of z value facecolours, corresponding to (theta,rho) grid.
                | alpha (float): facecolour surface transparency (optional)
                | edgeCol ([int]): list of colour specs, value at [0] used for edge colour (optional).


            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
      """

      # transform to cartesian system, using meshgrid
      Radial,Theta = np.meshgrid(radial,theta)
      X,Y = Radial*np.cos(Theta),Radial*np.sin(Theta)

      #create subplot if not existing
      if (self.nrow,self.ncol, plotnum) not in list(self.subplots.keys()):
          self.subplots[(self.nrow,self.ncol, plotnum)] = \
               self.fig.add_subplot(self.nrow,self.ncol, plotnum, projection='3d')
      #get axis
      ax = self.subplots[(self.nrow,self.ncol, plotnum)]

      if zorders:
        if len(zorders) > 1:
          zorder = zorders[i]
        else:
          zorder = zorders[0]
      else:
        zorder = 2

      if edgeCol:
        edcol = edgeCol
      else:
        edcol = self.nextPlotCol()

      #do the plot
      if facecolors is not None:
        ax.plot_surface(X, Y, zvals, rstride=thetaStride, cstride=radialstride,
          linewidth=linewidth, cmap=meshCmap, zorder=zorder, clip_on=clip_on,
          facecolors=facecolors, edgecolors=edcol, alpha=alpha)
      else:
        ax.plot_surface(X, Y, zvals, rstride=thetaStride, cstride=radialstride,
          linewidth=linewidth, cmap=meshCmap, zorder=zorder, clip_on=clip_on,
          alpha=alpha, edgecolors=edcol)

      ax.view_init(azim=azim, elev=elev)


      #label and clean up
      if zscale==None:
          ax.set_zlim3d(np.min(zvals), np.max(zvals))
      else:
          ax.set_zlim3d(zscale[0], zscale[1])

      if xlabel is not None:
          ax.set_xlabel(xlabel, fontsize = xylabelfsize)
      if ylabel is not None:
          ax.set_ylabel(ylabel, fontsize = xylabelfsize)
      if zlabel is not None:
          ax.set_zlabel(zlabel, fontsize = xylabelfsize)

      if(ptitle is not None):
          plt.title(ptitle, fontsize=titlefsize)

      return ax

    ############################################################
    ##
    def polarMesh(self, plotnum, theta, radial, zvals, ptitle=None, shading='flat',
                radscale=None, titlefsize=12,  meshCmap=cm.rainbow, cbarshow=False,
                  cbarorientation='vertical', cbarcustomticks=[], cbarfontsize=12,
                  rgrid=[0,5], thetagrid=[30], drawGrid=False,
                  thetagridfontsize=12, radialgridfontsize=12,
                  direction='counterclockwise', zerooffset=0, logScale=False,
                  plotCol=[], levels=10, contourFill=True, contourLine=True,
                  zeroContourLine=None, negativeSolid=False,
                  contLabel=False, contFmt='%.2f', contCol='k', contFonSz=8, contLinWid=0.5,
                  zorders=None, clip_on=True):
      """Polar colour contour and filled contour plot for (theta, r, zvals) input sets.

        The data values must be given on a fixed mesh grid of three-dimensional (theta,rho,z)
        array input sets (theta is angle,  and rho is radial distance). The mesh grid is
        defined in (theta,rho), while the height of the mesh is the z value. The
        (theta,rho) arrays may have non-constant grid-intervals, i.e., they do not
        have to be on regular intervals.

        Given an existing figure, this function plots in a specified subplot position.
        Only one contour plot is drawn at a time.  Future contours in the same subplot
        will cover any previous contours.

        The data set must have three two dimensional arrays, each for theta, rho, and z.
        The data in theta, rho, and z arrays must have matching data points.
        The theta and rho arrays each define the grid in terms of theta and rho values,
        i.e., the theta array contains the angular values for the data set, while the
        rho array contains the radial values.  The z array contains the z values for the
        corresponding theta and rho values in the contour mesh.

        Z-values can be plotted on a log scale, in which case the colourbar is adjusted
        to show true values, but on the nonlinear scale.

        The current version only saves png files, since there appears to be a problem
        saving eps files.

            Args:
                | plotnum (int): subplot number, 1-based index
                | theta (np.array[N,M]) array of angular values [0..2pi] corresponding to (theta,rho) grid.
                | radial (np.array[N,M]) array of radial values  corresponding to (theta,rho) grid.
                | zvals (np.array[N,M]) array of z values  corresponding to (theta,rho) grid.
                | ptitle (string): plot title (optional)
                | shading (string): 'flat' | 'gouraud'  (optional)
                | radscale ([float]): inner and outer radial scale max in the plot.
                | titlefsize (int): title font size, default 12pt (optional)
                | meshCmap (cm): color map for the mesh (optional)
                | cbarshow (bool): if true, the show a color bar
                | cbarorientation (string): 'vertical' (right) or 'horizontal' (below)
                | cbarcustomticks zip([tick locations/float],[tick labels/string]): locations in image grey levels
                | cbarfontsize (int): font size for color bar
                | rgrid ([float]): radial grid - None, [number], [inc,max]
                | thetagrid ([float]): angular grid - None, [inc]
                | drawGrid (bool): draw the grid on the plot (optional)
                | thetagridfontsize (float): font size for the angular grid
                | radialgridfontsize (float): font size for the radial grid
                | direction (string)= 'counterclockwise' or 'clockwise' (optional)
                | zerooffset (float) = rotation offset where zero should be [rad] (optional)
                | logScale (bool): do Z values on log scale, recompute colourbar vals
                | plotCol ([strings]): plot colour and line style, list with M entries, use default if []
                | levels (int or [float]): number of contour levels or a list of levels (optional)
                | contourFill (bool): fill contours with colour (optional)
                | contourLine (bool): draw a series of contour lines
                | zeroContourLine (double): draw a contour at the stated value (optional)
                | negativeSolid (bool): draw negative contours in solid lines, dashed otherwise (optional)
                | contLabel (bool): label the contours with values (optional)
                | contFmt (string): contour label c-printf format (optional)
                | contCol (string): contour label colour, e.g., 'k' (optional)
                | contFonSz (float): contour label fontsize (optional)
                | contLinWid (float): contour line width in points (optional)
                | zorder ([int]) list of zorder for drawing sequence, highest is last (optional)
                | clip_on (bool) clips objects to drawing axes (optional)

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.

      """

      # # transform to cartesian system, using meshgrid
      # Radial,Theta = np.meshgrid(radial,theta)
      # X,Y = Radial*np.cos(Theta),Radial*np.sin(Theta)

      #if this is a log scale plot
      if logScale is True:
          zvals = np.log10(zvals)

      contour_negative_linestyle = plt.rcParams['contour.negative_linestyle']
      if contourLine:
          if negativeSolid:
              plt.rcParams['contour.negative_linestyle'] = 'solid'
          else:
              plt.rcParams['contour.negative_linestyle'] = 'dashed'


      #create subplot if not existing
      if (self.nrow,self.ncol, plotnum) not in list(self.subplots.keys()):
          self.subplots[(self.nrow,self.ncol, plotnum)] = \
               self.fig.add_subplot(self.nrow,self.ncol, plotnum, projection='polar')
      #get axis
      ax = self.subplots[(self.nrow,self.ncol, plotnum)]

      if plotCol:
          col = plotCol[0]
      else:
          col = self.nextPlotCol()

      if zorders:
        if len(zorders) > 1:
          zorder = zorders[i]
        else:
          zorder = zorders[0]
      else:
        zorder = 2

      #do the plot
      if contourLine:
          pmplot = ax.contour(theta, radial, zvals, levels, cmap=None, linewidths=contLinWid,
               colors=col, zorder=zorder, clip_on=clip_on)

      if zeroContourLine:
          pmplot = ax.contour(theta, radial, zvals, (zeroContourLine,), cmap=None, linewidths=contLinWid,
               colors=col, zorder=zorder, clip_on=clip_on)

      if contourFill:
        pmplot = ax.pcolormesh(theta, radial, zvals, shading=shading, cmap=meshCmap,
            zorder=zorder, clip_on=clip_on)

      if contLabel:
        plt.clabel(pmplot, fmt = contFmt, colors = contCol, fontsize=contFonSz)

      ax.grid(drawGrid)


      if(ptitle is not None):
          plt.title(ptitle, fontsize=titlefsize)

      #set up the grids
      # add own labels: http://astrometry.net/svn/trunk/projects/masers/py/poster/plot_data.py
      #                 http://matplotlib.org/devel/add_new_projection.html
      if thetagrid is None:
          plt.thetagrids([])
      else:
          plt.thetagrids(list(range(0, 360, thetagrid[0])))
      plt.tick_params(axis='x', which='major', labelsize=thetagridfontsize)

          # plt.thetagrids(radscale[0], radscale[1],5)

      if radscale==None:
        rscale = [np.min(radial), np.max(radial)]
      else:
        rscale = radscale

      ax.set_ylim(rscale[0],rscale[1])

      if rgrid is None:
        ax.set_yticklabels([])
      else :
        #set the number of intervals
        if rgrid[0] is 0:
          ax.set_yticks(np.linspace(rscale[0],rscale[1],rgrid[1]))
        #set the interval incremental size
        if rgrid[0] is not 0:
          numrgrid = (rscale[1] - rscale[0] ) / (rgrid[0])
          ax.set_yticks(np.linspace(rscale[0],rscale[1],numrgrid+1.000001))

      plt.tick_params(axis='y', which='major', labelsize=radialgridfontsize)

      ax.set_theta_direction(direction)
      ax.set_theta_offset(zerooffset)


      if cbarshow is True:
          #http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes
          # this does not work with the polar projection, use gridspec to do this.
          # divider = make_axes_locatable(ax)
          # if cbarorientation == 'vertical':
          #   cax = divider.append_axes("right", size="5%", pad=0.05)
          # else:
          #   cax = divider.append_axes("bottom", size="5%", pad=0.1)

          if not cbarcustomticks:
              cbar = self.fig.colorbar(pmplot,orientation=cbarorientation)
              # cbar = self.fig.colorbar(pmplot,cax=cax)
              if logScale:
                  cbartickvals = cbar.ax.yaxis.get_ticklabels()
                  tickVals = []
                  # need this roundabout trick to handle minus sign in unicode
                  for item in cbartickvals:
                      valstr = item.get_text().replace('\u2212', '-').replace('$','')
                      val = 10**float(valstr)
                      if abs(val) < 1000:
                          str = '{0:f}'.format(val)
                      else:
                          str = '{0:e}'.format(val)
                      tickVals.append(str)
                  cbartickvals = cbar.ax.yaxis.set_ticklabels(tickVals)
          else:
              ticks,  ticklabels = list(zip(*cbarcustomticks))
              cbar = self.fig.colorbar(pmplot,ticks=ticks, orientation=cbarorientation)
              # cbar = self.fig.colorbar(pmplot,ticks=ticks, cax=cax)
              if cbarorientation == 'vertical':
                  cbar.ax.set_yticklabels(ticklabels)
              else:
                  cbar.ax.set_xticklabels(ticklabels)

          if cbarorientation == 'vertical':
              for t in cbar.ax.get_yticklabels():
                   t.set_fontsize(cbarfontsize)
          else:
              for t in cbar.ax.get_xticklabels():
                   t.set_fontsize(cbarfontsize)

      plt.rcParams['contour.negative_linestyle'] = contour_negative_linestyle

      return ax


    ############################################################
    ##
    def plotArray(self, plotnum, inarray, slicedim = 0, labels = None,
                        maxNX=0, maxNY=0, titlefsize = 8, xylabelfsize = 8,
                        xytickfsize = 8, selectCols=None, sepSpace=0.2,
                        allPlotCol='r'  ):
        """Creates a plot from an input array.

        Given an input array with m x n dimensions, this function creates a subplot for vectors
        [1-n]. Vector 0 serves as the x-axis for each subplot. The slice dimension can be in
        columns (0) or rows (1).


            Args:
                | plotnum (int):  The subplot number, 1-based index, according to Matplotlib conventions.
                  This value must always be given, even if only a single 1,1 subplot is used.
                | inarray (np.array): data series to be plotted.  Data direction can be cols or rows.
                  The abscissa (x axis) values must be the first col/row, with ordinates in following cols/rows.
                | slicedim (int): slice along columns (0) or rows (1) (optional).
                | labels (list):  a list of strings as labels for each subplot.
                  x=labels[0], y=labels[1:] (optional).
                | maxNX (int): draw maxNX+1 tick labels on x axis (optional).
                | maxNY (int): draw maxNY+1 tick labels on y axis (optional).
                | titlefsize (int): title font size, default 12pt (optional).
                | xylabelfsize (int): x-axis, y-axis label font size, default 12pt (optional).
                | xytickfsize (int): x-axis, y-axis tick font size, default 10pt (optional).
                | selectCols ([int]): select columns for plot. Col 0 corresponds to col 1 in input data
                  (because col 0 is abscissa),plot  all if not given (optional).
                | sepSpace (float): vertical spacing between sub-plots in inches (optional).
                | allPlotCol (str): make all plot lines this colour (optional).

            Returns:
                | Nothing

            Raises:
                | No exception is raised.
        """

        #prepare the data
        #if slicedim = 0, slice across columns
        if slicedim == 0:
            pass
        elif slicedim == 1:
            inarray = inarray.T

        x = inarray[:,0]
        yAll = inarray[:,1:].transpose()
        nestnrow = inarray.shape[1]-1
        nestncol = 1

        xlabel = labels[0]
        ylabels = labels[1:]

        ## keep track of whether the outer grid was already defined.
        #use current subplot number as outer grid reference
        ogkey = (self.nrow, self.ncol)
        if ogkey not in list(self.gridSpecsOuter.keys()):
            self.gridSpecsOuter[ogkey] = \
                         gridspec.GridSpec(self.nrow,self.ncol, wspace=0, hspace=0)
        outer_grid = self.gridSpecsOuter[ogkey]

        ## keep track of whether the inner grid was already defined.
        #inner_grid (nested):
        igkey = (self.nrow, self.ncol, plotnum)
        if igkey not in list(self.gridSpecsInner.keys()):
            self.gridSpecsInner[igkey] = \
                         gridspec.GridSpecFromSubplotSpec(nestnrow,nestncol,
                      subplot_spec=outer_grid[plotnum-1],wspace=0, hspace=sepSpace)
        inner_grid = self.gridSpecsInner[igkey]

        #set up list of all cols if required
        if not selectCols:
            selectCols = list(range(yAll.shape[0]))

        nestplotnum = 0
        #create subplot for each y-axis vector
        for index,y in enumerate(yAll):
            if index in selectCols:

                ## if this row of array plot in key, else create
                rkey = (self.nrow, self.ncol, plotnum, nestplotnum)
                if rkey not in list(self.arrayRows.keys()):
                    self.arrayRows[rkey] = \
                                 plt.Subplot(self.fig, inner_grid[nestplotnum])
                    self.fig.add_subplot(self.arrayRows[rkey])

                nestplotnum = nestplotnum + 1
                ax = self.arrayRows[rkey]

                # plot the data
                ax.plot(x,y,allPlotCol)
                if ylabels is not None:
                    ax.set_ylabel(ylabels[index], fontsize=xylabelfsize)
                    #align ylabels
                    ax.yaxis.set_label_coords(-0.05, 0.5)

                #tick label fonts
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(xytickfsize)
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(xytickfsize)

                if maxNX > 0:
                    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(maxNX))
                if maxNY > 0:
                    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(maxNY))

                #share x ticklabels and label to avoid clutter and overlapping
                plt.setp([a.get_xticklabels() for a in self.fig.axes[:-1]], visible=False)
                if xlabel is not None:
                    self.fig.axes[-1].set_xlabel(xlabel, fontsize=xylabelfsize)

            # minor ticks are two points smaller than major
            # ax.tick_params(axis='both', which='major', labelsize=xytickfsize)
            # ax.tick_params(axis='both', which='minor', labelsize=xytickfsize-2)



    ############################################################
    ##
    def setup_pie_axes(self,fig, rect, thetaAxis, radiusAxis,radLabel='',angLabel='',numAngGrid=5, 
        numRadGrid=10,drawGrid=True, degreeformatter="%d$^\circ$"):
        """Sets up the axes_grid for the pie plot, not using regulat Matplotlib axes.

        http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html
        http://matplotlib.org/mpl_toolkits/axes_grid/api/axis_artist_api.html
        http://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html
        http://matplotlib.org/examples/axes_grid/demo_floating_axes.html
        https://fossies.org/dox/matplotlib-1.5.3/classmpl__toolkits_1_1axisartist_1_1angle__helper_1_1FormatterDMS.html

            Args:
              | fig (matplotlib figure):  which figure to use
              | rect (matplotlib subaxis): which subplot to use
              | thetaAxis ([float]): [min,max] for angular scale
              | radiusAxis ([float]): [min,max] for radial scale
              | radLabel (str):  radial label
              | angLabel (str):  angular label
              | numAngGrid (int): number of ticks on angular grid
              | numRadGrid (int): number of ticks on radial grid
              | drawGrid (bool):  must grid be drawn?
              | degreeformatter (str): format string for angular tick labels
 
             Returns:
              | the axes and parasitic axes object for the plot

            Raises:
              | No exception is raised.
        """

        # PolarAxes.PolarTransform takes radian. However, we want our coordinate
        # system in degree
        tr = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform()

        # Find grid values appropriate for the coordinate (degree).
        # The argument is an approximate number of grids.
        grid_locator1 = angle_helper.LocatorD(numAngGrid)

        # And also use an appropriate formatter:
        tick_formatter1 = angle_helper.FormatterDMS()
        tick_formatter1.fmt_d = degreeformatter

        # set up number of ticks for the r-axis
        grid_locator2 = MaxNLocator(numRadGrid)

        # the extremes are passed to the function
        grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                    extremes=(thetaAxis[0], thetaAxis[1], radiusAxis[0], radiusAxis[1]),
                                    grid_locator1=grid_locator1,
                                    grid_locator2=grid_locator2,
                                    tick_formatter1=tick_formatter1,
                                    tick_formatter2=None,
                                    )

        ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
        fig.add_subplot(ax1)

        # create a parasite axes
        aux_ax = ax1.get_aux_axes(tr)

        aux_ax.patch = ax1.patch # for aux_ax to have a clip path as in ax
        ax1.patch.zorder=0.9 # but this has a side effect that the patch is
                             # drawn twice, and possibly over some other
                             # artists. So, we decrease the zorder a bit to
                             # prevent this.

        return ax1, aux_ax


    ############################################################
    ##
    def pie(self, plotnum,theta,radius, ptitle=None,angLabel='',radLabel='',
                    thetaAxis=[0,360.],radiusAxis=[0,1],plotCol=[], 
                    linewidths=None, label=[], legendAlpha=0.0,
                    legendLoc='best',
                    linestyle=None,
                    titlefsize = 12,
                    numAngGrid=5, numRadGrid=10,
                    labelfsize=10, drawGrid=True,
                    markers=[],markevery=None,
                    radangfsize = 12, 
                    xytickfsize = 10,
                    zorders=None, clip_on=True,
                    degreeformatter="%d$^\circ$"  ):
        """Plots data in pie section on a polar grid.

            Args:
              | plotnum (int): subplot number, 1-based index
              | theta (np.array[N,] or [N,M]): angular data set in degrees - could be M columns
              | radius (np.array[N,] or [N,M]): radial data set - could be M columns
              | ptitle (string): plot title (optional)
              | angLabel (string): angular axis label (optional)
              | radLabel (string): radial axis label (optional)
              | thetaAxis ([minAngle, maxAnlge]): the angular extent to be displayed, degrees (optional)
              | radiusAxis ([minRad, maxRad]): the radial extent to be displayed, degrees (optional)
              | plotCol ([strings]): plot colour and line style, list with M entries, use default if [] (optional)
              | linewidths ([float]): plot line width in points, list with M entries, use default if None  (optional)
              | label  ([strings]): legend label for ordinate, list with M entries
              | legendAlpha (float): transparency for legend box
              | legendLoc (string): location for legend box (optional)
              | linestyle (string): linestyle for this plot (optional)
              | titlefsize (int): title font size, default 12pt (optional)
              | numAngGrid (int): number of grid or tick marks along angular extent
              | numRadGrid (int): number of grid or tick marks along angular extent
              | labelfsize (int): label/legend font size, default 10pt (optional)
              | drawGrid (bool): draw the grid on the plot (optional)
              | markers ([string]) markers to be used for plotting data points (optional)
              | markevery (int | (startind, stride)) subsample when using markers (optional)
              | radangfsize (int): x-axis, y-axis label font size, default 12pt (optional)
              | xytickfsize (int): x-axis, y-axis tick font size, default 10pt (optional)
              | zorders ([int]) list of zorder for drawing sequence, highest is last (optional)
              | clip_on (bool) clips objects to drawing axes (optional)
              | degreeformatter (str) format string to defie the angular tick labels (optional)

             Returns:
              | the axis object for the plot

            Raises:
              | No exception is raised.
        """

        pkey = (self.nrow, self.ncol, plotnum)
        if pkey not in list(self.subplots.keys()):
            ax, aux_ax1 = self.setup_pie_axes(self.fig, '{}{}{}'.format(*pkey), 
                thetaAxis, radiusAxis, 
                radLabel=radLabel,angLabel=angLabel,numAngGrid=numAngGrid, 
                numRadGrid=numRadGrid,drawGrid=drawGrid,degreeformatter=degreeformatter)
            self.subplots[pkey] = (ax,aux_ax1)
        else:
            (ax,aux_ax1) = self.subplots[pkey]

        # reshape input dataset into rank 2
        xx = theta if theta.ndim>1 else theta.reshape(-1, 1)
        yy = radius if radius.ndim>1 else radius.reshape(-1, 1)

        ax.grid(drawGrid)

        for i in range(yy.shape[1]):
            #set up the line style, either given or next in sequence
            mmrk = ''
            if markers:
                mmrk = markers[-1] if i >= len(markers) else markers[i]

            if plotCol:
                if i >= len(plotCol):
                    col = plotCol[-1]
                else:
                    col = plotCol[i]
            else:
                col = self.nextPlotCol()

            if linestyle is None:
                linestyleL = '-'
            else:
                if type(linestyle) == type([1]):
                    linestyleL = linestyle[i]
                else:
                    linestyleL = linestyle

            if zorders:
                if len(zorders) > 1:
                    zorder = zorders[i]
                else:
                    zorder = zorders[0]
            else:
                zorder = 2

            if not label:
                if linewidths is not None:
                    line = aux_ax1.plot(xx[:, i], yy[:, i], col, label=None, linestyle=linestyleL,
                         marker=mmrk, markevery=markevery, linewidth=linewidths[i],
                         clip_on=clip_on, zorder=zorder)
                else:
                    line = aux_ax1.plot(xx[:, i], yy[:, i], col, label=None, linestyle=linestyleL,
                         marker=mmrk, markevery=markevery,
                         clip_on=clip_on, zorder=zorder)
                line = line[0]
            else:
                if linewidths is not None:
                  line = aux_ax1.plot(xx[:, i],yy[:,i],col,#label=label[i],
                        linestyle=linestyleL,
                        marker=mmrk, markevery=markevery, linewidth=linewidths[i],
                        clip_on=clip_on, zorder=zorder)
                else:
                  line = aux_ax1.plot(xx[:, i],yy[:,i],col,#label=label[i],
                        linestyle=linestyleL,
                        marker=mmrk, markevery=markevery,
                        clip_on=clip_on, zorder=zorder)
                line = line[0]
                line.set_label(label[i])
                leg = aux_ax1.legend( loc=legendLoc, fancybox=True,fontsize=labelfsize)
                leg.get_frame().set_alpha(legendAlpha)
                # aux_ax1.legend()
                self.bbox_extra_artists.append(leg)


        if(ptitle is not None):
            ax.set_title(ptitle, fontsize=titlefsize)

        # adjust axis
        # the axis artist lets you call axis with
        # "bottom", "top", "left", "right"


        # radial axis scale are the left/right of the graph

        # draw labels outside the grapgh
        if thetaAxis[0] > 90  and thetaAxis[0] < 270:
            ax.axis["left"].set_visible(False)
            ax.axis["right"].set_visible(True)
            ax.axis["right"].set_axis_direction("top")
            ax.axis["right"].toggle(ticklabels=True, label=True)
            ax.axis["right"].major_ticklabels.set_axis_direction("bottom")
            ax.axis["right"].label.set_axis_direction("bottom")
            #set radial label
            ax.axis["right"].label.set_text(radLabel)
            ax.axis["right"].label.set_size(radangfsize)
            aux_ax1.plot(np.array([thetaAxis[0],thetaAxis[0]]),
                np.array([radiusAxis[0],radiusAxis[1]]),'k',linewidth=2.5)
            ax.axis["right"].major_ticklabels.set_size(xytickfsize)
            ax.axis["right"].minor_ticklabels.set_size(xytickfsize-2)
        else:
            # ax.axis["right"].set_visible(False)
            ax.axis["left"].set_axis_direction("bottom")
            # ax.axis["right"].set_axis_direction("top")
            #set radial label
            ax.axis["left"].label.set_text(radLabel)
            ax.axis["left"].label.set_size(radangfsize)

            ax.axis["left"].major_ticklabels.set_size(xytickfsize)
            ax.axis["left"].minor_ticklabels.set_size(xytickfsize-2)

        # angular axis scale are top / bottom
        ax.axis["bottom"].set_visible(False)
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        #set angular label
        ax.axis["top"].label.set_text(angLabel)
        ax.axis["top"].label.set_size(radangfsize)


        ax.axis["top"].major_ticklabels.set_size(xytickfsize)
        ax.axis["top"].minor_ticklabels.set_size(xytickfsize-2)

        #draw the inner grif boundary,somehow not done by matplotlib
        # if radiusAxis[0] > 0.:
        numqi = 20
        thqi = np.linspace(thetaAxis[0], thetaAxis[1],numqi)
        raqi = np.linspace(radiusAxis[0],radiusAxis[0],numqi)
        aux_ax1.plot(thqi,raqi,'k',linewidth=2.5)

        return aux_ax1




################################################################
################################################################
##
## plot graphs and confirm the correctness of the functions
from contextlib import contextmanager

@contextmanager
def savePlot(fignumber=0,subpltnrow=1,subpltncol=1,
                 figuretitle=None, figsize=(9,9), saveName=None):
    """Uses 'with' statement to create a plot and save to file on exit.

    Use as follows::

        x=np.linspace(-3,3,20)
        with savePlot(1,saveName=['testwith.png','testwith.eps']) as p:
            p.plot(1,x,x*x)

    where the savePlot parameters are exactly the same as ``Plotter``,
    except that a new named parameter ``saveName`` is now present.
    If ``saveName`` is not ``None``, the list of filenames is used to
    save files of the plot (any number of names/types)

    Args:
        | fignumber (int): the plt figure number, must be supplied
        | subpltnrow (int): subplot number of rows
        | subpltncol (int): subplot number of columns
        | figuretitle (string): the overall heading for the figure
        | figsize ((w,h)): the figure size in inches
        | saveName str or [str]: string or list of save filenames

    Returns:
        | The plotting object, used to populate the plot (see example)

    Raises:
        | No exception is raised.

    """
    p = Plotter(fignumber,subpltnrow,subpltncol, figuretitle, figsize)
    try:
        yield p
    finally:
        if saveName is not None:
            if isinstance(saveName, str):
                p.saveFig(filename=saveName)
            else:
                for fname in saveName:
                    p.saveFig(filename=fname)


################################################################
##
def cubehelixcmap(start=0.5, rot=-1.5, gamma=1.0, hue=1.2, reverse=False, nlev=256.):
    """
    A full implementation of Dave Green's "cubehelix" for Matplotlib.
    Based on the FORTRAN 77 code provided in
    D.A. Green, 2011, BASI, 39, 289.

    http://adsabs.harvard.edu/abs/2011arXiv1108.5083G
    http://www.astron-soc.in/bulletin/11June/289392011.pdf

    User can adjust all parameters of the cubehelix algorithm.
    This enables much greater flexibility in choosing color maps, while
    always ensuring the color map scales in intensity from black
    to white. A few simple examples:

    Default color map settings produce the standard "cubehelix".

    Create color map in only blues by setting rot=0 and start=0.

    Create reverse (white to black) backwards through the rainbow once
    by setting rot=1 and reverse=True.


    Args:
        | start : scalar, optional
        |    Sets the starting position in the color space. 0=blue, 1=red,
        |    2=green. Defaults to 0.5.
        | rot : scalar, optional
        |   The number of rotations through the rainbow. Can be positive
        |    or negative, indicating direction of rainbow. Negative values
        |    correspond to Blue->Red direction. Defaults to -1.5
        | gamma : scalar, optional
        |    The gamma correction for intensity. Defaults to 1.0
        | hue : scalar, optional
        |    The hue intensity factor. Defaults to 1.2
        | reverse : boolean, optional
        |    Set to True to reverse the color map. Will go from black to
        |    white. Good for density plots where shade~density. Defaults to False
        | nevl : scalar, optional
        |    Defines the number of discrete levels to render colors at.
        |    Defaults to 256.

    Returns:
        |   matplotlib.colors.LinearSegmentedColormap object

    Example:
    >>> import cubehelix
    >>> cx = cubehelix.cmap(start=0., rot=-0.5)
    >>> plot(x,cmap=cx)

    Revisions
    2014-04 (@jradavenport) Ported from IDL version

    source
    https://github.com/jradavenport/cubehelix

    Licence
    Copyright (c) 2014, James R. A. Davenport and contributors All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of
    conditions and the following disclaimer.

    Redistributions in binary form must reproduce the above copyright notice, this
    list of conditions and the following disclaimer in the documentation and/or
    other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
    EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    from matplotlib.colors import LinearSegmentedColormap as LSC

 #-- set up the parameters
    fract = np.arange(nlev)/(nlev-1.)
    angle = 2.0*np.pi * (start/3.0 + 1.0 + rot*fract)
    fract = fract**gamma
    amp   = hue*fract*(1.-fract)/2.

#-- compute the RGB vectors according to main equations
    red   = fract+amp*(-0.14861*np.cos(angle)+1.78277*np.sin(angle))
    grn   = fract+amp*(-0.29227*np.cos(angle)-0.90649*np.sin(angle))
    blu   = fract+amp*(1.97294*np.cos(angle))

#-- find where RBB are outside the range [0,1], clip
    red[np.where((red > 1.))] = 1.
    grn[np.where((grn > 1.))] = 1.
    blu[np.where((blu > 1.))] = 1.

    red[np.where((red < 0.))] = 0.
    grn[np.where((grn < 0.))] = 0.
    blu[np.where((blu < 0.))] = 0.

#-- optional color reverse
    if reverse==True:
        red = red[::-1]
        blu = blu[::-1]
        grn = grn[::-1]

#-- put in to tuple & dictionary structures needed
    rr = []
    bb = []
    gg = []
    for k in range(0,int(nlev)):
        rr.append((float(k)/(nlev-1.), red[k], red[k]))
        bb.append((float(k)/(nlev-1.), blu[k], blu[k]))
        gg.append((float(k)/(nlev-1.), grn[k], grn[k]))

    cdict = {'red':rr, 'blue':bb, 'green':gg}
    return LSC('cubehelix_map',cdict)



################################################################
################################################################
##
## plot graphs and confirm the correctness of the functions

if __name__ == '__main__':

    import datetime as dt
    import ryutils
    rit = ryutils.intify_tuple

    doAll = True

    if doAll:
        p = Plotter(1,2,3,figsize=(12,12))
        theta = np.linspace(-10,10,20) + np.random.random(20) # in degrees
        radius = np.linspace(.5, 1., 20) + np.random.random(20) /50.
        thetax2 = np.hstack((theta.reshape(-1,1), -4 + theta.reshape(-1,1)))
        radiusx2 = np.hstack((radius.reshape(-1,1), 0.1+radius.reshape(-1,1)))

        # plot one data set
        p.pie(1,theta,radius,ptitle='test 1',radLabel='Distance m',angLabel='OTA deg',thetaAxis=[-20,20], radiusAxis=[0.5,1],
            numAngGrid=3, numRadGrid=5,linewidths=[5],linestyle=[''],markers=['x'],label=['dada'],legendAlpha=0.7,
            labelfsize=14,titlefsize=18)

        p.pie(2,theta,radius,ptitle='test 2',radLabel='Distance m',angLabel='OTA deg',thetaAxis=[-20,20], radiusAxis=[0.,1],
            numAngGrid=3, numRadGrid=5,linestyle=['--'])

        # plot two datasets in one np.array
        p.pie(3,thetax2,radiusx2,ptitle='test 3',radLabel='Distance m',angLabel='OTA deg',thetaAxis=[-20,20], radiusAxis=[0.5,1],
            numAngGrid=3, numRadGrid=5,linewidths=[2,1],linestyle=['-',':'],markers=['v','o'],drawGrid=False,
        label=['dada','dodo'],clip_on=False)

        p.pie(4,theta+180.,radius,ptitle='',radLabel='Distance m',angLabel='OTA deg',thetaAxis=[90,270], radiusAxis=[0.,1],
            numAngGrid=10, numRadGrid=5,linestyle=['--'],degreeformatter="%d")

        p.pie(5,theta+180.,radius,ptitle='',radLabel='Distance m',angLabel='OTA deg',thetaAxis=[91,270], radiusAxis=[0.,1],
            numAngGrid=10, numRadGrid=5,linestyle=['--'],degreeformatter="%d")

        # use the same subplot more than once
        p.pie(6,theta+180,radius,ptitle='test 6',radLabel='Distance m',angLabel='OTA deg',
            thetaAxis=[135,270], radiusAxis=[0,1],xytickfsize=8,numAngGrid=3, numRadGrid=5,
            linewidths=[5],linestyle=[''],markers=['x'],label=['dada'],radangfsize=8)

        p.pie(6,theta+185,radius,ptitle='test 6',radLabel='Distance m',angLabel='OTA deg',
            thetaAxis=[135,271], radiusAxis=[0,1],xytickfsize=8,numAngGrid=3, numRadGrid=5,
            linewidths=[2],linestyle=['-'],markers=['o'],label=['dodo'],markevery=4,radangfsize=8)

        p.saveFig('piepol.png')


    if doAll:  # stacked plot
        np.random.seed(1)
        fnx = lambda : np.random.randint(5, 50, 10)
        y = np.row_stack((fnx(), fnx(), fnx()))
        x = np.arange(10)

        # Make new array consisting of fractions of column-totals,
        # using .astype(float) to avoid integer division
        percent = y /  y.sum(axis=0).astype(float) * 100
        #data must vary along rows for single column  (row-major)
        percent = percent.T
        # print(rit(percent.shape))
        sp = Plotter(1,1,1,figsize=(16,8))
        sp.stackplot(1,x,percent,'Stack plot','X-axis label','Y-axis label',
            plotCol=['crimson','teal','#553300'], label=['aaa','bbb','cccc'],legendAlpha=0.5)
        sp.saveFig('stackplot.png')

    if doAll:    #next line include both 0 and 360 degrees, i.e., overlap on edge
        angled = np.linspace(0.,360.,25)
        angler = np.pi * angled / 180.
        grange = np.linspace(500.,4000.,8)
        #create a 2-D meshgrid.
        grangeg, anglerg= np.meshgrid(grange,angler + np.pi * 7.5 / 180)

        height  = 2000.
        launch = (1 + np.cos(anglerg) ) ** .1 * (1 - np.exp(-( 500 + grangeg) / 2000.) )
        launch *=  np.exp(-( 500 + grangeg) / (6000. -  height))
        launch = np.where(launch<0.2, 0.2, launch)
        #normalise
        launch -= np.min(launch)
        launch /= np.max(launch)

        pm = Plotter(1,1,2,figsize=(16,8))
        pm.polarMesh(1,angler+np.pi, grange, launch.T,
            ptitle='Probability of launch for height {:.0f} [m]'.format(height),
            radscale=[0, 4000], cbarshow=True,
            cbarorientation='vertical', cbarcustomticks=[], cbarfontsize=12,
            rgrid=[500], thetagrid=[45], drawGrid=True,
            direction='clockwise', zerooffset=np.pi/2, )
        pm.polar3d(2, angler, grange, launch, zlabel='zlabel',
            linewidth=1, zscale=[0, 1], azim=135, elev=60, alpha=0.5,edgeCol=['k'])
        pm.saveFig('3Dlaunch.png')

    if doAll:
        ############################################################################
        #create the wireframe for the sphere
        u = np.linspace(0, np.pi, 100)
        v = np.linspace(0, 2 * np.pi, 100)
        x = np.outer(np.sin(u), np.sin(v))
        y = np.outer(np.sin(u), np.cos(v))
        z = np.outer(np.cos(u), np.ones_like(v))

        #create the random point samples on the sphere
        samples = 500
        np.random.seed(1)
        np.random.RandomState(200)
        theta = 2 * np.pi * np.random.uniform(0, 1, size=samples)
        #biased sampling with higher density towards the poles
        phib = np.pi * (2 * np.random.uniform(0, 1, size=samples) -1 ) / 2
        #uniform sampling corrected for polar bias
        phiu = np.arccos(2 * np.random.uniform(0, 1, size=samples) -1 ) - np.pi/2

        #create normal vectors using the pairs of random angles in a transformation
        xsb = np.cos(phib) * np.cos(theta)
        ysb = np.cos(phib) * np.sin(theta)
        zsb = np.sin(phib)
        xsu = np.cos(phiu) * np.cos(theta)
        ysu = np.cos(phiu) * np.sin(theta)
        zsu = np.sin(phiu)

        azim = 45 # view angle
        elev = 45 # view angle
        sph = Plotter(1,1,2, figsize=(20,10))
        sph.mesh3D(1,x,y,z,'','x','y','z',alpha=0.1, wireframe=False, surface=True,linewidth=0, drawGrid=False)
        sph.mesh3D(1,x,y,z,'','x','y','z', alphawire=0.4, wireframe=True, surface=False,
            edgeCol=['b'],plotCol=['b'],linewidth=0.4,rstride=2,cstride=2, drawGrid=False)
        sph.plot3d(1, xsb, ysb, zsb, ptitle='', scatter=True,markers=['o' for i in range(len(xsb))],
                   azim=azim, elev=elev)

        sph.mesh3D(2,x,y,z,'','x','y','z',alpha=0.1, wireframe=False, surface=True,linewidth=0, drawGrid=False)
        sph.mesh3D(2,x,y,z,'','x','y','z', alphawire=0.4, wireframe=True, surface=False,
            edgeCol=['b'],plotCol=['b'],linewidth=0.4,rstride=2,cstride=2, drawGrid=False)
        sph.plot3d(2, xsu, ysu, zsu, ptitle='', scatter=True,markers=['o' for i in range(len(xsu))],
                   azim=azim, elev=elev)
        sph.saveFig('3dsphere.png')

        ############################################################################
        #demonstrate the use of a polar 3d plot
        #create the radial and angular vectors
        r = np.linspace(0,1.25,25)
        p = np.linspace(0,2*np.pi,50)
        #the r and p vectors may have non-constant grid-intervals
        # r = np.logspace(np.log10(0.001),np.log10(1.25),50)
        # p = np.logspace(np.log10(0.001),np.log10(2*np.pi),100)
        #build a meshgrid (2-D array of values)
        R,P = np.meshgrid(r,p)
        #calculate the z values on the cartesian grid
        # value = (np.tan(P**3)*np.cos(P**2)*(R**2 - 1)**2)
        value = ((R**2 - 1)**2)
        p3D = Plotter(1, 1, 1,'Polar plot in 3-D',figsize=(12,8))
        p3D.polar3d(1, p, r, value, ptitle='3-D Polar Plot',
            xlabel='xlabel', ylabel='ylabel', zlabel='zlabel')#,zscale=[-2,1])
        p3D.saveFig('p3D.png')
        #p3D.saveFig('p3D.eps')

        with open('./data/Intensity-max.dat', 'rt') as fin:
            aArray = np.loadtxt( fin, skiprows=1 , dtype=float )
            azim = aArray[1:,0] + np.pi   # to positive angles
            elev = aArray[0,1:] + np.pi/2 # get out of negative data on polar
            intensity = aArray[1:,1:]

            p3D = Plotter(1, 2, 2,'Polar plot in 3-D',figsize=(12,8))
            elev1 = elev
            p3D.polar3d(1, azim, elev1, intensity, zlabel='zlabel',zscale=[0, 600], azim=45, elev=30)
            p3D.polar3d(2, azim, elev, intensity, zlabel='zlabel',zscale=[0, 2000], azim=-45, elev=30)
            elev3 = elev + np.pi/2 # get hole in centre
            p3D.polar3d(3, azim, elev3, intensity, zlabel='zlabel',zscale=[0, 2000], azim=60, elev=60)
            p3D.polar3d(4, azim, elev, intensity, zlabel='zlabel',zscale=[0, 1000], azim=110, elev=-30)

            p3D.saveFig('p3D2.png')
            #p3D.saveFig('p3D2.eps')

        ############################################################################
        xv,yv = np.mgrid[-2:2:21j, -2:2:21j]
        z = np.exp(np.exp(-(xv**2 + yv**2)))
        I = Plotter(4, 1, 2,'High dynamic range image', figsize=(8, 4))
        I.showImage(1, z, ptitle='xv**2 + yv**2', titlefsize=10,  cbarshow=True, cbarorientation = 'vertical', cbarfontsize = 7)
        ip = ProcessImage()
        zz, customticksz = ip.compressEqualizeImage(z, 2, 10)
        I.showImage(2, zz, ptitle='Equalized xv**2 + yv**2', titlefsize=10,  cbarshow=True, cbarorientation = 'vertical', cbarcustomticks=customticksz, cbarfontsize = 7)
        I.saveFig('HistoEq.png')
        #    I.saveFig('HistoEq.eps')

        ############################################################################
        # demonstrate dates on the x-axis
        dates = ['01/02/1991','01/03/1991','01/04/1991']
        x = np.asarray([dt.datetime.strptime(d,'%m/%d/%Y').date() for d in dates])
        y = np.asarray(list(range(len(x))))
        pd = Plotter(1)
        pd.plot(1,x,y,xIsDate=True,pltaxis=[x[0],x[-1],-1,4],xtickRotation=30)
        pd.saveFig('plotdateX.png')

        #demonstrate the use of arbitrary x-axis tick marks
        x = np.asarray([1, 2, 3, 4])
        y = np.asarray([1, 2, 3, 4])
        px = Plotter(2)
        px.plot(1,x,y,xTicks={1:'One', 2:'Two', 3:'Three', 4:'Four'}, xtickRotation=90)
        px.saveFig('plotxTick.png')

    ############################################################################
    #demonstrate the use of a polar mesh plot radial scales
    #create the radial and angular vectors

    if doAll:
        r = np.linspace(0,1.25,100)
        p = np.linspace(0,2*np.pi,100)
        P,R = np.meshgrid(p,r)
        # value = ((np.sin(P))**2 + np.cos(P)*(R**2 - 1)**2)
        value = ((R**2 - 1)**2) * np.sin(2 * P)
        pmesh = Plotter(1, 2, 2,'Polar plot in mesh',figsize=(12,8))
        pmesh.polarMesh(1, p, r, value, meshCmap = cm.jet_r, cbarshow=True,\
                      drawGrid=False, rgrid=None, thetagrid=None,\
                      thetagridfontsize=10, radialgridfontsize=8,
                      direction='counterclockwise', zerooffset=0)
        pmesh.polarMesh(2, p, r, value, meshCmap = cm.gray, cbarshow=True,\
                      drawGrid=True, rgrid=[3],\
                      thetagridfontsize=10, radialgridfontsize=8,
                      direction='clockwise', zerooffset=np.pi/2)
        pmesh.polarMesh(3, p, r, value, meshCmap = cm.hot, cbarshow=True,\
                      drawGrid=True,thetagrid=[45], rgrid=[.25,1.25],\
                      thetagridfontsize=10, radialgridfontsize=8,
                      direction='counterclockwise', zerooffset=0)
        pmesh.polarMesh(4, p, r, value, meshCmap = cm.jet, cbarshow=True,\
                      drawGrid=True, thetagrid=[15], rgrid=[0.2, 1.],\
                      thetagridfontsize=10, radialgridfontsize=8,
                      direction='clockwise', zerooffset=-np.pi/2)#, radscale=[0.5,1.25])
        pmesh.saveFig('pmeshrad.png')
        #pmesh.saveFig('pmeshrad.eps')

        #3D plot example
        def parametricCurve(z, param1 = 2, param2 = 1):
            r = z**param1 + param2
            theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
            return (r * np.sin(theta), r * np.cos(theta))

        P3D = Plotter(5, 1, 1,'Plot 3D Single', figsize=(12,8))
        z = np.linspace(-2, 2, 100)
        x, y = parametricCurve(z)
        P3D.plot3d(1, x.T, y.T, z.T, 'Parametric Curve', 'X', 'Y', 'Z',zInvert=True)
        P3D.saveFig('3D.png')

        P3D = Plotter(6, 1, 1,'Plot 3D Single', figsize=(12,8))
        plabel = ['parametric curve 1', 'parametric curve 2', 'parametric curve 3']
        P3D.plot3d(1, x.T, y.T, z.T, 'Parametric Curve', 'X', 'Y', 'Z', legendAlpha=0.5)
        P3D.saveFig('3DwithLabel.png')
        P3D.plot3d(1, 1.3*x.T, 0.8*y.T, 0.7*z.T, 'Parametric Curve', 'X', 'Y', 'Z', legendAlpha=0.5)
        P3D.plot3d(1, 0.8*x.T, 0.9*y.T, 1.2*z.T, 'Parametric Curve', 'X', 'Y', 'Z', label=plabel, legendAlpha=0.5)
        P3D.saveFig('3DwithLabelRepeat.png')

        P3D = Plotter(7, 2, 2,'Plot 3D Aspects', figsize=(12,8))
        P3D.plot(1, x.T, y.T, 'Top View', 'X', 'Y')
        P3D.plot(2, x.T, z.T, 'Side View Along Y Axis', 'X', 'Z')
        P3D.plot(3, y.T, z.T, 'Side View Along X Axis', 'Y', 'Z')
        P3D.plot3d(4, x.T, y.T, z.T, '3D View', 'X', 'Y', 'Z')
        P3D.saveFig('S3D.png')

        P3D = Plotter(8, 1, 1,'Plot 3D Multiple', figsize=(12,8))
        label = ['Param1={} Param2={}'.format(2,1)]
        for i in range(2):
            param1 = 2-i
            param2 = i
            label.append('Param1={} Param2={}'.format(param1, param2))
            x1, y1 = parametricCurve(z, param1, param2)
            x = np.vstack((x,x1))
            y = np.vstack((y,y1))

        z = np.vstack((z,z,z))

        P3D.plot3d(1, x.T, y.T, z.T, 'Parametric Curve', 'X', 'Y', 'Z', label=label,
            legendAlpha=0.5, markers=['o','v','^','<'], markevery=4)
        P3D.saveFig('M3D.png')

    ############################################################################
    # demonstrate the use of the contextmanager and with statement
    if doAll:
        x=np.linspace(-3,3,20)
        with savePlot(1,saveName=['testwith.png','testwith.eps']) as p:
            p.plot(1,x,x*x)
        with savePlot(1,saveName='testwith.svg') as p:
            p.plot(1,x,x*x)

    ############################################################################
    import matplotlib.mlab as mlab
    if doAll:
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-2.0, 2.0, delta)
        X, Y = np.meshgrid(x, y)
        Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
        Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
        # difference of Gaussians
        Z = 10.0 * (Z2 - Z1)
        pmc = Plotter(1)
        pmc.meshContour(1, X, Y, Z, levels=15,
                    ptitle='meshContour', shading='gouraud',plotCol=['k'],
                    titlefsize=12,  meshCmap=cm.rainbow, cbarshow=True,
                    cbarorientation='vertical', cbarfontsize=12,
                    xlabel='X-value', ylabel='Y-value',
                    drawGrid=True, yInvert=True, negativeSolid=False,
                    contourFill=True, contourLine=True, logScale=False )
        #the current version uses pngs, since there appears to be a
        #problem with eps files.
        pmc.saveFig('meshContour.png', dpi=300)



    ############################################################################
    #demonstrate the use of 3D mesh plots
    def myFunc(x,y):
      scale = np.sqrt(np.exp(-(x**2 +y**2)))
      return np.sin(2 * x) * np.cos(4 * y) * scale

    if doAll:
        x = np.linspace(-2, 2, 101)
        y = np.linspace(-2, 2, 101)
        varx, vary = np.meshgrid(x, y)
        zdata = myFunc(varx.flatten(), vary.flatten()).reshape(varx.shape)

        # print(zdata.shape)

        p = Plotter(1,2,2,figsize=(18,14))
        p.mesh3D(1, varx, vary, zdata, ptitle='Title', xlabel='x', ylabel='y', zlabel='z',
          rstride=3, cstride=3, linewidth= 1, maxNX=5, maxNY=5, maxNZ=0,
          drawGrid=True, cbarshow=True, cmap=None)
        p.mesh3D(2, varx, vary, zdata, ptitle='Title', xlabel='x', ylabel='y', zlabel='z',
          rstride=3, cstride=3, linewidth= 0.3, maxNX=5, maxNY=5, maxNZ=0,
          drawGrid=True, cbarshow=True, alpha=0.2)
        p.mesh3D(3, varx, vary, zdata, ptitle='Title', xlabel='x', ylabel='y', zlabel='z',
          rstride=3, cstride=3, linewidth= 0.2, maxNX=5, maxNY=5, maxNZ=0,
          drawGrid=True, cmap=cm.jet,  cbarshow=True, elev=70, azim=15)
        p.mesh3D(4, varx, vary, zdata, ptitle='Title', xlabel='x', ylabel='y', zlabel='z',
          rstride=3, cstride=3, linewidth= 0, maxNX=5, maxNY=5, maxNZ=0, drawGrid=True,
          cmap=cm.brg, cbarshow=True)

        p.saveFig('mesh3d01.png')



    ############################################################################
    #demonstrate the use of plotArray
    #import array from example data file
    if doAll:
        filename = "data/arrayplotdemo.txt"
        f = open(filename)
        lines = f.readlines()
        #the labels are in the first line (row). Skip the '%' symbol
        labels = lines[0].split()[1:]
        #the array is the rest of the file
        arrDummyDemo = np.genfromtxt(filename,skip_header=1)
        #the figure title is the filename
        maintitle = filename.split('/')[-1]

        Ar = Plotter(9, 1, 1,maintitle)
        Ar.plotArray(1,arrDummyDemo, 0, labels=labels, titlefsize = 12, maxNX = 5, maxNY=3,
            sepSpace=0.05)
        Ar.saveFig('ArrayPlot01.png')

        t = np.linspace(0, 4 * np.pi, 100).reshape(-1,1)
        arr = np.hstack((t,   np.sin(t * 2 * np.pi / 4.0)))
        arr = np.hstack((arr, np.cos(t * 2 * np.pi / 4.0)))
        labels = ['time','a','b']
        for i in range(7):
            arr = np.hstack((arr, np.sin(t * 2 * np.pi / (i+1))))
            arr = np.hstack((arr, np.cos(t * 2 * np.pi / (i+1))))
            labels.append(' s{} '.format(i))
            labels.append(' c{} '.format(i))

        Ar2 = Plotter(10, 1, 1,maintitle)
        Ar2.plotArray(1,arr, 0, labels=labels, titlefsize = 12, maxNX = 5, maxNY=3,
            sepSpace=0.05,selectCols=[0,2,4,6,8,10,12,14],allPlotCol='b')
        Ar2.plotArray(1,arr, 0, labels=labels, titlefsize = 12, maxNX = 5, maxNY=3,
            sepSpace=0.05,selectCols=[1,3,5,7,9,11,13,15], allPlotCol='r')
        Ar2.saveFig('ArrayPlot02.png')

        Ar3 = Plotter(11, 1, 1,maintitle)
        Ar3.plotArray(1,arr.T, 1, labels=labels, titlefsize = 12, maxNX = 5, maxNY=3,
            sepSpace=0.05,selectCols=[0,2,4,6,8,10,12,14],allPlotCol='b')
        Ar3.plotArray(1,arr.T, 1, labels=labels, titlefsize = 12, maxNX = 5, maxNY=3,
            sepSpace=0.05,selectCols=[1,3,5,7,9,11,13,15], allPlotCol='r')
        Ar3.saveFig('ArrayPlot03.png')


    ############################################################################
    #demonstrate the use of a polar mesh plot and markers
    #create the radial and angular vectors
    if doAll:

        r = np.linspace(0,1.25,100)
        p = np.linspace(0,2*np.pi,100)
        P, R = np.meshgrid(p, r)
        value =  ((R**2 - 1)**2) * np.sin(2 * P)
        pmesh = Plotter(1, 1, 1,'Polar plot in mesh',figsize=(12,8))
        pmesh.polarMesh(1, p, r, value, cbarshow=True,
                      cbarorientation = 'vertical', cbarfontsize = 10)

        # add filled markers
        markers = Markers(markerfacecolor='y', marker='*')
        markers.add(0*np.pi/6,1)
        markers.add(1*np.pi/6,0.9,markerfacecolor='k', marker='^',fillstyle='top')
        markers.add(2*np.pi/6,0.8,fillstyle='top',markeredgecolor='g')
        markers.add(3*np.pi/6,0.7,marker='v',markerfacecolor='r')
        markers.add(4*np.pi/6,0.6,marker='p',fillstyle='top')
        markers.add(5*np.pi/6,0.5,markerfacecolor='r',marker='H',fillstyle='bottom',markerfacecoloralt='PaleGreen')
        markers.add(6*np.pi/6,0.4,marker='D',fillstyle='left',markerfacecoloralt='Sienna',markersize=10)
        markers.plot(pmesh.getSubPlot(1))

        pmesh.saveFig('pmesh.png')
        #pmesh.saveFig('pmesh.eps')


    ############################################################################
    ##create some data
    if doAll:
        xLinS=np.linspace(0, 10, 50).reshape(-1, 1)
        np.random.seed(1)

        yLinS=1.0e3 * np.random.random(xLinS.shape[0]).reshape(-1, 1)
        yLinSS=1.0e3 * np.random.random(xLinS.shape[0]).reshape(-1, 1)

        yLinA=yLinS
        yLinA = np.hstack((yLinA, \
                1.0e7 * np.random.random(xLinS.shape[0]).reshape(-1, 1)))
        yLinA = np.hstack((yLinA, \
                1.0e7 * np.random.random(xLinS.shape[0]).reshape(-1, 1)))

        A = Plotter(1, 2, 2,'Array Plots',figsize=(12,8))
        A.plot(1, xLinS, yLinA, "Array Linear","X", "Y",
                plotCol=['c'],
               label=['A1', 'A2', 'A3'],legendAlpha=0.5,
               pltaxis=[0, 10, 0, 2000],
               maxNX=10, maxNY=2,
               powerLimits = [-4,  2, -5, 5])
        A.logLog(2, xLinS, yLinA, "Array LogLog","X", "Y",\
                 label=['A1', 'A2', 'A3'],legendAlpha=0.5)
        A.semilogX(3, xLinS, yLinA, "Array SemilogX","X", "Y",\
                   label=['A1', 'A2', 'A3'],legendAlpha=0.5)
        A.semilogY(4, xLinS, yLinA, "Array SemilogY","X", "Y",\
                   label=['A1', 'A2', 'A3'],legendAlpha=0.5)
        A.saveFig('A.png')
        #A.saveFig('A.eps')

        AA = Plotter(1, 1, 1,'Demonstrate late labels',figsize=(12,8))
        AA.plot(1, xLinS, yLinA, plotCol=['b'],
               label=['A1', 'A2', 'A3'],legendAlpha=0.5,
               pltaxis=[0, 10, 0, 2000],
               maxNX=10, maxNY=2,
               powerLimits = [-4,  2, -5, 5])
        currentP = AA.getSubPlot(1)
        currentP.set_xlabel('X Label')
        currentP.set_ylabel('Y Label')
        currentP.set_title('The figure title')
        currentP.annotate('axes center', xy=(.5, .5),  xycoords='axes fraction',
                    horizontalalignment='center', verticalalignment='center')
        currentP.text(0.5 * 10, 1300,
             r"$\int_a^b f(x)\mathrm{d}x$", horizontalalignment='center',
             fontsize=20)
        for xmaj in currentP.xaxis.get_majorticklocs():
            currentP.axvline(x=xmaj,ls='-')
        for xmin in currentP.xaxis.get_minorticklocs():
            currentP.axvline(x=xmin,ls='--')
        for ymaj in currentP.yaxis.get_majorticklocs():
            currentP.axhline(y=ymaj,ls='--')
        for ymin in currentP.yaxis.get_minorticklocs():
            currentP.axhline(y=ymin,ls='--')

        AA.saveFig('AA.png')
        # AA.saveFig('AA.eps')

        S = Plotter(2, 2, 2,'Single Plots',figsize=(12,8))
        S.plot(1, xLinS, yLinS, "Single Linear","X", "Y",\
               label=['Original'],legendAlpha=0.5)
        S.logLog(2, xLinS, yLinS, "Single LogLog","X", "Y",\
                 label=['Original'],legendAlpha=0.5)
        S.semilogX(3, xLinS, yLinS, "Single SemilogX","X", "Y",\
                   label=['Original'],legendAlpha=0.5)
        S.semilogY(4, xLinS, yLinS, "Single SemilogY","X", "Y",\
                   label=['Original'],legendAlpha=0.5)
        S.saveFig('S.png', dpi=300)
        #S.saveFig('S.eps')
        #plot again on top of the existing graphs
        S.plot(1, xLinS, yLinSS, "Single Linear","X", "Y",\
                   plotCol='r',label=['Repeat on top'],legendAlpha=0.5)
        S.logLog(2, xLinS, 1.3*yLinSS, "Single LogLog","X", "Y",\
                  plotCol='g',label=['Repeat on top'],legendAlpha=0.5)
        S.semilogX(3, xLinS, 0.5*yLinSS, "Single SemilogX","X", "Y",\
                   plotCol='k',label=['Repeat on top'],legendAlpha=0.5)
        S.semilogY(4, xLinS, 0.85*yLinSS, "Single SemilogY","X", "Y",\
                   plotCol='y',label=['Repeat on top'],legendAlpha=0.5)
        S.saveFig('SS.png', dpi=300)
        #S.saveFig('SS.eps')

        r = np.arange(0, 3.01, 0.01).reshape(-1, 1)
        theta = 2*np.pi*r
        r2 = np.hstack((r,r**2))
        P = Plotter(3, 2, 2,'Polar Plots', figsize=(12,8))
        P.polar(1,theta, r, "Single Polar",\
               label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0.5,3])
        P.polar(2,theta, r2, "Array Polar",\
               label=['A', 'B'],legendAlpha=0.5,rscale=[2,6],rgrid=[2,6],\
               thetagrid=[45], direction='clockwise', zerooffset=0)
        P.polar(3,theta, r, "Single Polar",\
               label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0,3], \
               direction='clockwise', zerooffset=np.pi/2)
        P.polar(4,theta, r2, "Array Polar",\
               label=['A', 'B'],legendAlpha=0.5,rscale=[0,9],rgrid=[0,6],\
               thetagrid=[45], direction='counterclockwise', zerooffset=-np.pi/2)
        P.saveFig('P.png')
        #P.saveFig('P.eps')
        #plot again on top of existing graphs
        rr = np.arange(0.1, 3.11, 0.01).reshape(-1, 1)
        thetar = 2*np.pi*rr
        P.polar(1,thetar, 0.5 * rr, "Single Polar",\
               plotCol='r',label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0.5,3])
        P.polar(2,thetar, 0.75 * rr, "Array Polar",\
               plotCol='g',label=['A', 'B'],legendAlpha=0.5,rscale=[0,6],rgrid=[2,6],\
               thetagrid=[45], direction='clockwise', zerooffset=0)
        P.polar(3,thetar, 1.2 * rr, "Single Polar",\
               plotCol='k',label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0,3], \
               direction='clockwise', zerooffset=np.pi/2)
        P.polar(4,thetar, 1.5 * rr, "Array Polar",\
               plotCol='y',label=['A', 'B'],legendAlpha=0.5,rscale=[0,9],rgrid=[0,6],\
               thetagrid=[45], direction='counterclockwise', zerooffset=-np.pi/2)
        P.saveFig('PP.png')
        #P.saveFig('PP.eps')

        #polar with negative values
        theta=np.linspace(0,2.0*np.pi,600)
        r = np.sin(3.4*theta)
        PN = Plotter(3, 2, 2,'Negative Polar Plots', figsize=(12,8))
        PN.polar(1,theta, r, "sin(3.3x)",\
               legendAlpha=0.5,rscale=[0,1.5],rgrid=[0.5,1.5],highlightNegative=True)
        tt = np.linspace(0,24*np.pi,3000)
        rr = np.exp(np.cos(tt)) - 2 * np.cos(4 * tt) + (np.sin(tt / 12))**5
        PN.polar(2,tt, rr, "Math function",\
               legendAlpha=0.5,rscale=[0,5],rgrid=[0.5,5.0],highlightNegative=True,
               highlightCol='r',highlightWidth=4)
        PN.polar(3,theta, r, "sin(3.3x)", \
               legendAlpha=0.5,rscale=[-1.5,1.5],rgrid=[0.5,1.5],highlightNegative=True)
        tt = np.linspace(0,2 * np.pi,360)
        rr = 1 + 3 * np.sin(tt)
        PN.polar(4,tt,rr, "1 + 3sin(x)", \
               legendAlpha=0.5,rgrid=[1,5],highlightNegative=True,direction='clockwise',
               zerooffset=np.pi/2,highlightCol='r',highlightWidth=2)
        PN.saveFig('PN.png')
        #PN.saveFig('PN.eps')

        #test/demo to show that multiple plots can be done in the same subplot, on top of older plots
        xLinS=np.linspace(0, 10, 50).reshape(-1, 1)
        M= Plotter(1, 1, 1,'Multi-plots',figsize=(12,8))
        #it seems that all attempts to plot in same subplot space must use same ptitle.
        np.random.seed(1)

        yLinS=np.random.random(xLinS.shape[0]).reshape(-1, 1)
        M.plot(1, xLinS, yLinS, None,"X", "Y",plotCol=['b'], label=['A1'])
        yLinS=np.random.random(xLinS.shape[0]).reshape(-1, 1)
        M.plot(1, xLinS, yLinS, None,"X", "Y",plotCol=['g'], label=['A2'])
        yLinS=np.random.random(xLinS.shape[0]).reshape(-1, 1)
        M.plot(1, xLinS, yLinS, None,"X", "Y",plotCol=['r'], label=['A3'])
        yLinS=np.random.random(xLinS.shape[0]).reshape(-1, 1)
        M.plot(1, xLinS, yLinS, None,"X", "Y",plotCol=['c'], \
               label=['A4'],legendAlpha=0.5, maxNX=10, maxNY=2)
        M.saveFig('M.png')
        #M.saveFig('M.eps')


        xv,yv = np.mgrid[-5:5:21j, -5:5:21j]
        z = np.sin(np.sqrt(xv**2 + yv**2))
        I = Plotter(4, 2, 2,'Images & Array Linear', figsize=(12, 8))
        I.showImage(1, z, ptitle='winter colormap, font 10pt', cmap=plt.cm.winter, titlefsize=10,  cbarshow=True, cbarorientation = 'horizontal', cbarfontsize = 7)
        barticks = list(zip([-1, 0, 1], ['low', 'med', 'high']))
        I.showImage(2, z, ptitle='prism colormap, default font ', cmap=plt.cm.prism, cbarshow=True, cbarcustomticks=barticks)
        I.showImage(3, z, ptitle='default gray colormap, font 8pt', cbarshow=True, titlefsize=8)
        I.plot(4, xv[:, 1],  z, "Array Linear","x", "z")
        I.saveFig('I.png')
    #    I.saveFig('I.eps')
        #plot on existing
        I.showImage(1, z, ptitle='winter colormap, font 10pt', cmap=plt.cm. winter, titlefsize=10,  cbarshow=True, cbarorientation = 'horizontal', cbarfontsize = 7)
        barticks = list(zip([-1, 0, 1], ['low', 'med', 'high']))
        I.showImage(2, z, ptitle='prism colormap, default font ', cmap=plt.cm. prism, cbarshow=True, cbarcustomticks=barticks)
        I.showImage(3, z, ptitle='default gray colormap, font 8pt', titlefsize=8)
        I.plot(4, xv[:, 1],  z, "Array Linear","x", "z")
        I.saveFig('II.png')
    #    I.saveFig('II.eps')

        I = Plotter(5, 2, 2,'Images & Array Linear', figsize=(12, 8))
        I.showImage(1, z, ptitle='winter colormap, font 10pt', cmap=plt.cm. winter, titlefsize=10,  cbarshow=True, cbarorientation = 'horizontal', cbarfontsize = 7)
        barticks = list(zip([-1, 0, 1], ['low', 'med', 'high']))
        I.showImage(2, z, ptitle='cubehelix colormap, default font ',cmap=cubehelixcmap(), cbarshow=True, cbarcustomticks=barticks)
        I.showImage(3, z, ptitle='default gray colormap, font 8pt', cbarshow=True, titlefsize=8)
        I.plot(4, xv[:, 1],  z, "Array Linear","x", "z")
        I.saveFig('cmaps.png')

        #demonstrate setting axis values
        x=np.linspace(-3,3,20)
        p = Plotter(1)
        p.plot(1,x,x,pltaxis=[-2,1,-3,2])
        p.saveFig('testaxis.png')


        #test the ability to return to existing plots and add new lines
        x = np.linspace(0,10,10)
        a = Plotter(1)
        b = Plotter(2)
        c = Plotter(3)
        for i in [1,2]:
            a.plot(1,x,x ** i, str(i))
            b.plot(1,x,(-x) ** i,str(i))
            c.plot(1,x,(5-x) ** i,str(i))
        a.saveFig('ma.png')
        b.saveFig('mb.png')
        c.saveFig('mc.png')

    ############################################################################
    #demonstrate multipage pdf output
    #reference for the multipage pdf code: http://blog.marmakoide.org/?p=94
    if doAll:

        x=np.linspace(0, 2*np.pi, 50).reshape(-1, 1)
        np.random.seed(1)
        y=1 + np.random.random(x.shape[0]).reshape(-1, 1)

        #create the pdf document
        pdf_pages = PdfPages('multipagepdf.pdf')

        # create the first page
        A = Plotter(1, 2, 1,figsize=(12,8))
        A.plot(1, x, y, "Array Linear","X", "Y")
        A.logLog(2, x, y, "Array LogLog","X", "Y")
        # A.getPlot().tight_layout()
        pdf_pages.savefig(A.getPlot())

        #create the second page
        B = Plotter(1, 1, 1,figsize=(12,8))
        B.polar(1, x, y, "Polar")
        # B.getPlot().tight_layout()
        pdf_pages.savefig(B.getPlot())

        # Write the PDF document to the disk
        pdf_pages.close()

    if doAll:

        x = np.linspace(-30,40,200)

        q = Plotter(1,1,1,figsize=(6,2))
        q.buildPlotCol(['#ff5577','#FFFF31'])
        q.plot(1,x,x,pltaxis=[-2,1,-3,2],drawGrid=False)
        q.plot(1,x,x**2,pltaxis=[-2,1,-3,2],drawGrid=False)
        q.saveFig('userplotcol.png')

        x = np.linspace(0,1,200).reshape(-1,1)
        t = Plotter(1,1,1,figsize=(6,2))
        y = x
        for i in range(len(t.plotCol)):
            y = np.hstack((y,x+i*0.02))
        t.plot(1,x,y,'Display default plot colours',pltaxis=[0,1,0,1],drawGrid=False)
        t.saveFig('userplotcol02.png')


    print('module ryplot done!')
