#  $Id$
#  $HeadURL$
################################################################
# The contents of this file are subject to the Mozilla Publiccense
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


See the __main__ function for examples of use.

This package was partly developed to provide additional material in support of students 
and readers of the book Electro-Optical System Analysis and Design: A Radiometry 
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals


__version__ = "$Revision$"
__author__ = 'pyradi team'
__all__ = ['Plotter','cubehelixcmap', 'FilledMarker', 'Markers','ProcessImage',
            'savePlot']

import sys
if sys.version_info[0] > 2:
    print("pyradi is not yet ported to Python 3, because imported modules are not yet ported")
    exit(-1)

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
    def __init__(self, markerfacecolor = None, markerfacecoloralt = None,\
                markeredgecolor = None, marker = None, markersize = None, \
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

        marker = FilledMarker(markerfacecolor, markerfacecoloralt ,\
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
                c = marker[2].markerfacecolor,
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

   

    ############################################################
    def compressEqualizeImage(self, image, selectCompressSet=2, numCbarlevels=20):
        """Compress an image (and then inversely expand the color bar values), 
           prior to histogram equalisation to ensure that the two keep in step, 
           we store the function names as pairs, and below invoke the pair elements
           cases are as follows:  linear, log. sqrt.  Note that the image is 
           histogram equalised in all cases.    
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
        customticksz = zip(imgLevels, ['{0:10.3e}'.format(self.compressSet[selectCompressSet][1](x)) for x in irrLevels])

        return imgHEQ, customticksz

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
                 figuretitle=None, figsize=(9,9)):
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

            Returns:
                | Nothing. Creates the figure for subequent use.

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', 'saveFig', 'getPlot', 'plot', 'logLog', 'semilogX',
                        'semilogY', 'polar', 'showImage', 'plot3d', 'buildPlotCol', 
                        'getSubPlot', 'meshContour', 'nextPlotCol', 'plotArray',
                        'polarMesh', 'resetPlotCol', 'mesh3D', 'polar3d', 'labelSubplot',
                        ]

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

        self.bbox_extra_artists=[]
        self.subplots={}

        if figuretitle:
            self.figtitle=plt.gcf().text(.5,.95,figuretitle,\
                        horizontalalignment='center',\
                        fontproperties=FontProperties(size=16))
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
            'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 'k--',
            'b-.', 'g-.', 'r-.', 'c-.', 'm-.', 'y-.', 'k-.' ]

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

        col = self.plotColCirc.next()
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
        if (self.nrow,self.ncol, subplotNum) in self.subplots.keys():
            return self.subplots[(self.nrow,self.ncol, subplotNum)]
        else:
            return None


    ############################################################
    ##
    def plot(self, plotnum, x, y, ptitle=None, xlabel=None, ylabel=None,
                    plotCol=[], label=[],legendAlpha=0.0,
                    pltaxis=None, maxNX=10, maxNY=10, linestyle=None,
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12,
                    xylabelfsize = 12,  xytickfsize = 10, labelfsize=10,
                    xScientific=False, yScientific=False,
                    yInvert=False, xInvert=False, drawGrid=True,xIsDate=False,
                    xTicks=None, xtickRotation=0, markers=[],
                    markevery=None ):
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
                | label  ([strings]): legend label for ordinate, list with M entries (optional)
                | legendAlpha (float): transparancy for legend box (optional)
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

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
       """
      ## see self.MyPlot for parameter details.
      pkey = (self.nrow, self.ncol, plotnum)
      if pkey not in self.subplots.keys():
            self.subplots[pkey] = \
                         self.fig.add_subplot(self.nrow,self.ncol, plotnum)
      ax = self.subplots[pkey]

      self.myPlot(ax.plot, plotnum, x, y, ptitle, xlabel, ylabel,
                    plotCol, label,legendAlpha, 
                    pltaxis, maxNX, maxNY, linestyle,
                    powerLimits,titlefsize, 
                    xylabelfsize, xytickfsize,
                    labelfsize, drawGrid, 
                    xScientific, yScientific,
                    yInvert, xInvert, xIsDate,
                    xTicks, xtickRotation, markers, markevery)
      return ax

    ############################################################
    ##
    def logLog(self, plotnum, x, y, ptitle=None, xlabel=None, ylabel=None,
                    plotCol=[], label=[],legendAlpha=0.0,
                    pltaxis=None, maxNX=10, maxNY=10, linestyle=None,
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12,
                    xylabelfsize = 12, xytickfsize = 10,labelfsize=10,
                    xScientific=False, yScientific=False,
                    yInvert=False, xInvert=False, drawGrid=True,xIsDate=False,
                    xTicks=None, xtickRotation=0, markers=[],
                    markevery=None  ):
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
                | label  ([strings]): legend label for ordinate, list with M entries (optional)
                | legendAlpha (float): transparancy for legend box (optional)
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

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
      """
      ## see self.MyPlot for parameter details.
      pkey = (self.nrow, self.ncol, plotnum)
      if pkey not in self.subplots.keys():
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
                    plotCol, label,legendAlpha, 
                    pltaxis, maxNX, maxNY, linestyle,
                    powerLimits,titlefsize, 
                    xylabelfsize, xytickfsize,
                    labelfsize, drawGrid, 
                    xScientific, yScientific,
                    yInvert, xInvert, xIsDate,
                    xTicks, xtickRotation, markers, markevery)

      return ax


    ############################################################
    ##
    def semilogX(self, plotnum, x, y, ptitle=None, xlabel=None, ylabel=None,
                    plotCol=[], label=[],legendAlpha=0.0,
                    pltaxis=None, maxNX=10, maxNY=10, linestyle=None,
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12,
                    xylabelfsize = 12, xytickfsize = 10,labelfsize=10,
                    xScientific=False, yScientific=False,
                    yInvert=False, xInvert=False, drawGrid=True,xIsDate=False,
                    xTicks=None, xtickRotation=0, markers=[],
                    markevery=None  ):

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
                | label  ([strings]): legend label for ordinate, list with M entries (optional)
                | legendAlpha (float): transparancy for legend box (optional)
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

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
       """
      ## see self.MyPlot for parameter details.
      pkey = (self.nrow, self.ncol, plotnum)
      if pkey not in self.subplots.keys():
            self.subplots[pkey] = \
                         self.fig.add_subplot(self.nrow,self.ncol, plotnum)
      ax = self.subplots[pkey]

      self.myPlot(ax.semilogx, plotnum, x, y, ptitle, xlabel, ylabel,\
                    plotCol, label,legendAlpha, 
                    pltaxis, maxNX, maxNY, linestyle, 
                    powerLimits, titlefsize,
                    xylabelfsize, xytickfsize,
                    labelfsize, drawGrid, 
                    xScientific, yScientific,
                    yInvert, xInvert, xIsDate,
                    xTicks, xtickRotation, markers, markevery)

      return ax

    ############################################################
    ##
    def semilogY(self, plotnum, x, y, ptitle=None, xlabel=None, ylabel=None,
                    plotCol=[], label=[],legendAlpha=0.0,
                    pltaxis=None, maxNX=10, maxNY=10, linestyle=None,
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12,
                    xylabelfsize = 12, xytickfsize = 10, labelfsize=10,
                    xScientific=False, yScientific=False,
                    yInvert=False, xInvert=False, drawGrid=True,xIsDate=False,
                     xTicks=None, xtickRotation=0, markers=[],
                    markevery=None ):
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
                | label  ([strings]): legend label for ordinate, list withM entries (optional)
                | legendAlpha (float): transparancy for legend box (optional)
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

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
      """
      ## see self.MyPlot for parameter details.
      pkey = (self.nrow, self.ncol, plotnum)
      if pkey not in self.subplots.keys():
            self.subplots[pkey] = \
                         self.fig.add_subplot(self.nrow,self.ncol, plotnum)
      ax = self.subplots[pkey]

      self.myPlot(ax.semilogy, plotnum, x, y, ptitle,xlabel,ylabel,
                    plotCol, label,legendAlpha, 
                    pltaxis, maxNX, maxNY, linestyle, 
                    powerLimits, titlefsize,
                    xylabelfsize, xytickfsize, 
                    labelfsize, drawGrid, 
                    xScientific, yScientific,
                    yInvert, xInvert, xIsDate,
                    xTicks, xtickRotation, markers, markevery)

      return ax

    ############################################################
    ##
    def myPlot(self, plotcommand,plotnum, x, y, ptitle=None,xlabel=None, ylabel=None,
                     plotCol=[],label=[],legendAlpha=0.0,
                    pltaxis=None, maxNX=0, maxNY=0, linestyle=None,
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12,
                    xylabelfsize = 12, xytickfsize = 10, 
                    labelfsize=10, drawGrid=True,
                    xScientific=False, yScientific=False,
                    yInvert=False, xInvert=False, xIsDate=False,
                    xTicks=None, xtickRotation=0, markers=[] ,
                    markevery=None ):
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
              | label  ([strings]): legend label for ordinate, list with M entries
              | legendAlpha (float): transparancy for legend box
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
              if len(col)>1:
                  linestyleL = col[1:]
              else:
                  linestyleL = '-' 
          else:
              if type(linestyle) == type([1]):
                  linestyleL = linestyle[i] 
              else:
                  linestyleL = linestyle


          if not label:
              plotcommand(xx, yy[:, i], col ,label=None, linestyle=linestyleL, 
                marker=mmrk, markevery=markevery)
          else:
              plotcommand(xx,yy[:,i],col,label=label[i], linestyle=linestyleL, 
                marker=mmrk, markevery=markevery)

              leg = ax.legend(loc='best', fancybox=True,fontsize=labelfsize)
              leg.get_frame().set_alpha(legendAlpha)
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
          ticks = ax.set_xticks(xTicks.keys())
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
    def meshContour(self, plotnum, xvals, yvals, zvals, numLevels=10,
                  ptitle=None,
                  xlabel=None, ylabel=None, shading = 'flat',\
                  plotCol=[], pltaxis=None, maxNX=0, maxNY=0,
                  xScientific=False, yScientific=False,  \
                  powerLimits = [-4,  2,  -4,  2], titlefsize = 12,
                  xylabelfsize = 12, xytickfsize = 10,
                  meshCmap = cm.rainbow, cbarshow=False, \
                  cbarorientation = 'vertical', cbarcustomticks=[], cbarfontsize = 12,\
                  drawGrid = False, yInvert=False, xInvert=False,
                  contourFill=True, contourLine=True, logScale=False,
                  negativeSolid=False, zeroContourLine=False ):
      """XY colour mesh  countour plot for (xvals, yvals, zvals) input sets.

        Given an existing figure, this function plots in a specified subplot position.

        Only one mesh is drawn at a time.  Future meshes in the same subplot
        will cover any previous meshes.

        The data in zvals must be on a grid where the xvals vector correspond to
        the number of rows in zvals and the yvals vector corresponds to the
        number of columns in zvals.

        Z-values can be plotted on a log scale, in which case the colourbar is adjusted
        to show true values, but on the nonlinear scale.

        The xvals and yvals vectors may have non-constant grid-intervals, i.e., they do not
        have to be on regular intervals.

            Args:
                | plotnum (int): subplot number, 1-based index
                | xvals (np.array[N,] or [N,1]): vector of x values
                | yvals (np.array[M,] or [M,1]): vector of y values
                | zvals (np.array[N,M]): values on a (x,y) grid
                | numLevels (int): values of contour levels
                | ptitle (string): plot title (optional)
                | xlabel (string): x axis label
                | ylabel (string): y axis label
                | shading (string): 'flat' | 'gouraud'  (optional)
                | plotCol ([strings]): plot colour and line style, list with M entries, use default if []
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. Let Matplotlib decide if None.
                | maxNX (int): draw maxNX+1 tick labels on x axis
                | maxNY (int): draw maxNY+1 tick labels on y axis
                | xScientific (bool): use scientific notation on x axis (optional)
                | yScientific (bool): use scientific notation on y axis (optional)
                | powerLimits[float]:  scientific tick label power limits [x-low, x-high, y-low, y-high] (optional)
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x-axis, y-axis label font size, default 12pt (optional)
                | xytickfsize (int): x-axis, y-axis tick font size, default 10pt (optional)
                | meshCmap (cm): color map for the mesh (optional)
                | cbarshow (bool): if true, the show a color bar
                | cbarorientation (string): 'vertical' (right) or 'horizontal' (below)
                | cbarcustomticks zip([tick locations/float],[tick labels/string]): locations in image grey levels
                | cbarfontsize (int): font size for color bar
                | drawGrid (bool): draw the grid on the plot (optional)
                | invertY (bool): invert the y-axis (optional)
                | invertX (bool): invert the x-axis (optional)
                | contourFill (bool): fill contours with colour
                | contourLine (bool): draw a series of contour lines
                | logScale (bool): do Z values on log scale, recompute colourbar vals
                | negativeSolid (bool): draw negative contours in solid lines, dashed otherwise
                | zeroContourLine (bool): draw the contour at zero

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

      if contourLine:
          if negativeSolid:
              plt.rcParams['contour.negative_linestyle'] = 'solid'
          else:
              plt.rcParams['contour.negative_linestyle'] = 'dashed'

      #create subplot if not existing
      if (self.nrow,self.ncol, plotnum) not in self.subplots.keys():
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

      #do the plot
      if contourLine:
          pmplot = ax.contour(xvals, yvals, zvals, numLevels, cmap=None,
               colors=col)

      if zeroContourLine:
          pmplot = ax.contour(xvals, yvals, zvals, (0,), cmap=None, linewidths = 0.5,
               colors=col)

      if contourFill:
          pmplot = ax.contourf(xvals, yvals, zvals, numLevels, shading=shading,
              cmap=meshCmap)

      if cbarshow is True:
          if not cbarcustomticks:
              cbar = self.fig.colorbar(pmplot,orientation=cbarorientation)
              if logScale:
                  cbartickvals = cbar.ax.yaxis.get_ticklabels()
                  tickVals = []
                  # need this roundabout trick to handle minus sign in unicode
                  for item in cbartickvals:
                      valstr = item.get_text().replace(u'\u2212', '-').replace('$','')
                      val = 10**float(valstr)
                      if abs(val) < 1000:
                          str = '{0:f}'.format(val)
                      else:
                          str = '{0:e}'.format(val)
                      tickVals.append(str)
                  cbartickvals = cbar.ax.yaxis.set_ticklabels(tickVals)
          else:
              ticks,  ticklabels = zip(*cbarcustomticks)
              cbar = self.fig.colorbar(pmplot,ticks=ticks, orientation=cbarorientation)
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

      return ax

    ############################################################
    ##
    def mesh3D(self, plotnum, xvals, yvals, zvals, 
                  ptitle=None, xlabel=None, ylabel=None, zlabel=None, 
                  rstride=1, cstride=1, linewidth= 0, 
                  plotCol=[], pltaxis=None, maxNX=0, maxNY=0, maxNZ=0,
                  xScientific=False, yScientific=False, zScientific=False,  
                  powerLimits = [-4,  2,  -4,  2, -2, 2], titlefsize = 12,
                  xylabelfsize = 12, xytickfsize = 10, wireframe=False,
                  cmap = cm.rainbow, cbarshow=False, 
                  cbarorientation = 'vertical', cbarcustomticks=[], cbarfontsize = 12,
                  drawGrid = True, xInvert=False, yInvert=False, zInvert=False,
                  logScale=False, elevation=30, azimuth=45, alpha=1, 
                   ):
      """XY colour mesh plot for (xvals, yvals, zvals) input sets.

        Given an existing figure, this function plots in a specified subplot position.

        Only one mesh is drawn at a time.  Future meshes in the same subplot
        will cover any previous meshes.

        The data in xvals, yvals, and zvals must have matching points, all in 
        two-dimensional arrays.

        Use cmap=None to obtain a wireframe plot.

        Z-values can be plotted on a log scale, in which case the colourbar is adjusted
        to show true values, but on the nonlinear scale.

        The xvals and yvals vectors may have non-constant grid-intervals, i.e., they do not
        have to be on regular intervals.

            Args:
                | plotnum (int): subplot number, 1-based index
                | xvals (np.array[N,] or [N,1]): vector of x values
                | yvals (np.array[M,] or [M,1]): vector of y values
                | zvals (np.array[N,M]): values on a (x,y) grid
                | ptitle (string): plot title (optional)
                | xlabel (string): x axis label
                | ylabel (string): y axis label
                | zlabel (string): z axis label
                | rstride (int): row stride
                | cstride (int): column stride
                | linewidth (float): line width
                | plotCol ([strings]): plot colour and line style, list with M entries, use default if []
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. Let Matplotlib decide if None.
                | maxNX (int): draw maxNX+1 tick labels on x axis
                | maxNY (int): draw maxNY+1 tick labels on y axis
                | maxNZ (int): draw maxNY+1 tick labels on z axis
                | xScientific (bool): use scientific notation on x axis (optional)
                | yScientific (bool): use scientific notation on y axis (optional)
                | zScientific (bool): use scientific notation on z-axis (optional)
                | powerLimits[float]:  scientific tick label power limits [x-neg, x-pos, y-neg, y-pos, z-neg, z-pos]
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x-axis, y-axis label font size, default 12pt (optional)
                | xytickfsize (int): x-axis, y-axis tick font size, default 10pt (optional)
                | wireframe (bool): If True, do a wireframe plot
                | cmap (cm): color map for the mesh (optional)
                | cbarshow (bool): if true, the show a color bar
                | cbarorientation (string): 'vertical' (right) or 'horizontal' (below)
                | cbarcustomticks zip([tick locations/float],[tick labels/string]): locations in image grey levels
                | cbarfontsize (int): font size for color bar
                | drawGrid (bool): draw the grid on the plot (optional)
                | invertX (bool): invert the x-axis (optional)
                | invertY (bool): invert the y-axis (optional)
                | invertZ (bool): invert the z-axis
                | logScale (bool): do Z values on log scale, recompute colourbar vals
                | elevation (float): view elevation in degrees
                | azimuth (float): view azimuth in degrees
                | alpha (float): surface transparency

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
      if (self.nrow,self.ncol, plotnum) not in self.subplots.keys():
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

      #do the plot
      if not wireframe:
          pmplot = ax.plot_surface(xvals, yvals, zvals, rstride=rstride, cstride=cstride,
              color=col[0], cmap=cmap, linewidth=linewidth, alpha=alpha)
      else:
          pmplot = ax.plot_wireframe(xvals, yvals, zvals, rstride=rstride, cstride=cstride, 
              color=col[0], linewidth=linewidth, alpha=alpha)

      if cbarshow is True and cmap is not None:
          if not cbarcustomticks:
              cbar = self.fig.colorbar(pmplot,orientation=cbarorientation)
              if logScale:
                  cbartickvals = cbar.ax.yaxis.get_ticklabels()
                  tickVals = []
                  # need this roundabout trick to handle minus sign in unicode
                  for item in cbartickvals:
                      valstr = item.get_text().replace(u'\u2212', '-').replace('$','')
                      val = 10**float(valstr)
                      if abs(val) < 1000:
                          str = '{0:f}'.format(val)
                      else:
                          str = '{0:e}'.format(val)
                      tickVals.append(str)
                  cbartickvals = cbar.ax.yaxis.set_ticklabels(tickVals)
          else:
              ticks,  ticklabels = zip(*cbarcustomticks)
              cbar = self.fig.colorbar(pmplot,ticks=ticks, orientation=cbarorientation)
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
          ax.axis(pltaxis)

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

      #set the view direction
      ax.view_init(elevation, azimuth)

      return ax

    ############################################################
    ##
    def polar(self, plotnum, theta, r, ptitle=None, \
                    plotCol=None, label=[],labelLocation=[-0.1, 0.1], \
                    highlightNegative=True, highlightCol='#ffff00', highlightWidth=4,\
                    legendAlpha=0.0, \
                    rscale=None, rgrid=None, thetagrid=[30], \
                    direction='counterclockwise', zerooffset=0, titlefsize=12):
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
                | legendAlpha (float): transparancy for legend box (optional)
                | rscale ([rmin, rmax]): radial plotting limits. use default setting if None.  
                  If rmin is negative the zero is a circle and rmin is at the centre of the graph (optional)
                | rgrid ([rinc, numinc]): radial grid, use default if None. if rinc=0 then numinc is number 
                  of intervals.  If rinc is not zero then rinc is the increment and numinc is ignored (optional)
                | thetagrids (float): theta grid interval [degrees] (optional)
                | direction (string): direction in increasing angle, 'counterclockwise' or 'clockwise' (optional)
                | zerooffset (float):  rotation offset where zero should be [rad]. Positive 
                  zero-offset rotation is counterclockwise from 3'o'clock (optional)
                | titlefsize (int): title font size, default 12pt (optional)

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
      if pkey not in self.subplots.keys():
          self.subplots[pkey] = \
                       self.fig.add_subplot(self.nrow,self.ncol, plotnum, polar=True)

      ax = self.subplots[pkey]

      ax.grid(True)

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

          #set up the line style, either given or next in sequence
          if plotCol:
              col = plotCol[i]
          else:
              col = self.nextPlotCol()

          # print('p',ttt.shape)
          # print('p',rrr.shape)

          if not label:
              if highlightNegative:
                  lines = ax.plot(ttt, rrr, col)
                  neglinewith = highlightWidth*plt.getp(lines[0],'linewidth')
                  for ii in range(0,len(negrrr)):
                      if len(negrrr[ii]) > 0:
                          if negrrr[ii][0] < 0:
                              if MakeAbs:
                                  ax.plot(negttt[ii], np.abs(negrrr[ii]), highlightCol,linewidth=neglinewith)
                              else:
                                  ax.plot(negttt[ii], negrrr[ii], highlightCol,linewidth=neglinewith)
              ax.plot(ttt, rrr, col)
              rmax=np.maximum(np.abs(rrr).max(), rmax)
          else:
              if highlightNegative:
                  lines = ax.plot(ttt, rrr, col)
                  neglinewith = highlightWidth*plt.getp(lines[0],'linewidth')
                  for ii in range(0,len(negrrr)):
                      if len(negrrr[ii]) > 0:
                          # print(len(negrrr[ii]))
                          # if negrrr[ii][0] < 0:
                          if negrrr[ii][0][0] < 0:
                              if MakeAbs:
                                  ax.plot(negttt[ii], np.abs(negrrr[ii]), highlightCol,linewidth=neglinewith)
                              else:
                                  ax.plot(negttt[ii], negrrr[ii], highlightCol,linewidth=neglinewith)
              ax.plot(ttt, rrr, col,label=label[i])
              rmax=np.maximum(np.abs(rrr).max(), rmax)

          if MakeAbs:
              ax.plot(ttt, np.abs(rrr), col)
          else:
              ax.plot(ttt, rrr, col)

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
      plt.thetagrids(range(0, 360, thetagrid[0]))

      #Set increment and maximum radial limits
      if rscale is None:
          ax.set_ylim(0,rmax)
      else:
        if rgrid is None:
          ax.set_ylim(rscale[0],rscale[1])
          ax.set_yticks(np.linspace(rscale[0],rscale[1],5))
        else:
          if rgrid[0] is 0:
            ax.set_yticks(np.linspace(rscale[0],rscale[1],rgrid[1]))
          if rgrid[0] is not 0:
            numrgrid = (rscale[1] - rscale[0] ) / rgrid[0]
            ax.set_yticks(np.linspace(rscale[0],rscale[1],numrgrid+1))

      if(ptitle is not None):
          ax.set_title(ptitle, fontsize=titlefsize, \
              verticalalignment ='bottom', horizontalalignment='center')

      return ax

    ############################################################
    ##
    def showImage(self, plotnum, img,  ptitle=None, cmap=plt.cm.gray, titlefsize=12, cbarshow=False, \
                  cbarorientation = 'vertical', cbarcustomticks=[], cbarfontsize = 12):
      """Creates a subplot and show the image using the colormap provided.

            Args:
                | plotnum (int): subplot number, 1-based index
                | img (np.ndarray): numpy 2d array
                | ptitle (string): plot title (optional)
                | cmap: matplotlib colormap, default gray (optional)
                | fsize (int): title font size, default 12pt (optional)
                | cbarshow (bool): if true, the show a color bar
                | cbarorientation (string): 'vertical' (right) or 'horizontal' (below)
                | cbarcustomticks zip([tick locations/float],[tick labels/string]): locations in image grey levels
                | cbarfontsize (int): font size for color bar

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
      """
    #http://matplotlib.sourceforge.net/examples/pylab_examples/colorbar_tick_labelling_demo.html
    #http://matplotlib.1069221.n5.nabble.com/Colorbar-Ticks-td21289.html

      pkey = (self.nrow, self.ncol, plotnum)
      if pkey not in self.subplots.keys():
          self.subplots[pkey] = \
                       self.fig.add_subplot(self.nrow,self.ncol, plotnum)

      ax = self.subplots[pkey]

      cimage = ax.imshow(img, cmap)
      ax.axis('off')
      if cbarshow is True:
          if not cbarcustomticks:
              cbar = self.fig.colorbar(cimage,orientation=cbarorientation)
          else:
              ticks,  ticklabels = zip(*cbarcustomticks)
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
    def plot3d(self, plotnum, x, y, z, ptitle=None, xlabel=None, ylabel=None, zlabel=None, \
               plotCol=[], label=None, legendAlpha=0.0, titlefsize=12,
               xylabelfsize = 12 , xInvert=False, yInvert=False, zInvert=False):
        """3D plot on linear scales for x y z input sets.

        Given an existing figure, this function plots in a specified subplot position.
        The function arguments are described below in some detail.

        Note that multiple 3D data sets can be plotted simultaneously by adding additional columns to
        the input coordinates of vertices, each column representing a different function in the plot.
        This is convenient if large arrays of data must be plotted. If more than one column is present,
        the label argument can contain the legend labels for each of the columns/lines.

            Args:
                | plotnum (int): subplot number, 1-based index
                | x (np.array[N,] or [N,M]): x coordinates of vertices
                | y (np.array[N,] or [N,M]): y coordinates of vertices
                | z (np.array[N,] or [N,M]): z coordinates of vertices
                | ptitle (string): plot title (optional)
                | xlabel (string): x-axis label (optional)
                | ylabel (string): y-axis label (optional)
                | zlabel (string): z axis label (optional)
                | plotCol ([strings]): plot colour and line style, list with M entries, use default if [] (optional)
                | label  ([strings]): legend label for ordinate, list with M entries (optional)
                | legendAlpha (float): transparancy for legend box (optional)
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x, y, z label font size, default 12pt (optional)
                | xInvert (bool): invert the x-axis (optional)
                | yInvert (bool): invert the y-axis (optional)
                | zInvert (bool): invert the z-axis

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



        if (self.nrow,self.ncol, plotnum) not in self.subplots.keys():
            self.subplots[(self.nrow,self.ncol, plotnum)] = \
               self.fig.add_subplot(self.nrow,self.ncol, plotnum, projection='3d')

        ax = self.subplots[(self.nrow,self.ncol, plotnum)]

        for i in range(x.shape[-1]):
            if plotCol:
                if i >= len(plotCol):
                    col = plotCol[-1] 
                else:
                    col = plotCol[i]
            else:
                col = self.nextPlotCol()
            ax.plot(x[:,i], y[:,i], z[:,i], col)

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
    def polar3d(self, plotnum, theta, radial, zvals, ptitle=None, \
                xlabel=None, ylabel=None, zlabel=None, zscale=None,  \
               titlefsize=12, xylabelfsize = 12,
               thetaStride=1, radialstride=1, meshCmap = cm.rainbow,
               linewidth=0.1, azim=45, elev=30):
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
                | theta (np.array[N,] or [N,1]): vector of angular values (rad)
                | radial (np.array[M,] or [M,1]): vector if radial values
                | zvals (np.array[N,M]): values on a (theta,radial) grid
                | ptitle (string): plot title (optional)
                | xlabel (string): x-axis label (optional)
                | ylabel (string): y-axis label (optional)
                | zlabel (string): z axis label (optional)
                | zscale ([float]): z axis [min, max] in the plot.
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x, y, z label font size, default 12pt (optional)
                | thetaStride (int): theta stride in input data (optional)
                | radialstride (int): radial stride in input data  (optional)
                | meshCmap (cm): color map for the mesh (optional)
                | linewidth (float): width of the mesh lines
                | azim (float): camera azimuth angle viewing the graph [degrees]
                | elev (float): camera evelation angle viewing the graph [degrees]

            Returns:
                | the axis object for the plot

            Raises:
                | No exception is raised.
      """

      # transform to cartesian system, using meshgrid
      Radial,Theta = np.meshgrid(radial,theta)
      X,Y = Radial*np.cos(Theta),Radial*np.sin(Theta)

      #create subplot if not existing
      if (self.nrow,self.ncol, plotnum) not in self.subplots.keys():
          self.subplots[(self.nrow,self.ncol, plotnum)] = \
               self.fig.add_subplot(self.nrow,self.ncol, plotnum, projection='3d')
      #get axis
      ax = self.subplots[(self.nrow,self.ncol, plotnum)]
      #do the plot
      ax.plot_surface(X, Y, zvals, rstride=thetaStride, cstride=radialstride, 
          linewidth=linewidth, cmap=meshCmap)
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
    def polarMesh(self, plotnum, theta, radial, zvals, ptitle=None, shading = 'flat',\
                radscale=None, titlefsize=12,  meshCmap = cm.rainbow, cbarshow=False, \
                  cbarorientation = 'vertical', cbarcustomticks=[], cbarfontsize = 12,\
                  rgrid=[5], thetagrid=[30], drawGrid = False,\
                  thetagridfontsize = 12, radialgridfontsize=12,\
                  direction='counterclockwise', zerooffset=0, logScale=False,
                  plotCol=[], numLevels=10, contourLine=True, zeroContourLine=False):
      """Polar colour mesh plot for (r, theta, zvals) input sets.

        Given an existing figure, this function plots in a specified subplot position.

        Only one mesh is drawn at a time.  Future meshes in the same subplot
        will cover any previous meshes.

        The data in zvals must be on a grid where the theta vector correspond to
        the number of rows in zvals and the radial vector corresponds to the
        number of columns in zvals.

        Z-values can be plotted on a log scale, in which case the colourbar is adjusted
        to show true values, but on the nonlinear scale.

        The r and p vectors may have non-constant grid-intervals, i.e., they do not
        have to be on regular intervals.

            Args:
                | plotnum (int): subplot number, 1-based index
                | theta (np.array[N,] or [N,1]): vector of angular values [0..2pi]
                | radial (np.array[M,] or [M,1]): vector of radial values
                | zvals (np.array[N,M]): values on a (theta,radial) grid
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
                | numLevels (int): values of contour levels
                | contourLine (bool): draw a series of contour lines
                | zeroContourLine (bool): draw the contour at zero

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

      #create subplot if not existing
      if (self.nrow,self.ncol, plotnum) not in self.subplots.keys():
          self.subplots[(self.nrow,self.ncol, plotnum)] = \
               self.fig.add_subplot(self.nrow,self.ncol, plotnum, projection='polar')
      #get axis
      ax = self.subplots[(self.nrow,self.ncol, plotnum)]

      if plotCol:
          col = plotCol[0]
      else:
          col = self.nextPlotCol()

      #do the plot
      if contourLine:
          pmplot = ax.contour(theta, radial, zvals, numLevels, cmap=None,
               colors=col)

      if zeroContourLine:
          pmplot = ax.contour(theta, radial, zvals, (0,), cmap=None, linewidths = 0.5,
               colors=col)

      pmplot = ax.pcolormesh(theta, radial, zvals, shading=shading, cmap=meshCmap)

      #label and clean up
      if radscale==None:
          ax.set_ylim(np.min(radial), np.max(radial))
      else:
          ax.set_ylim(radscale[0], radscale[1])

      ax.grid(drawGrid)

      if cbarshow is True:
          if not cbarcustomticks:
              cbar = self.fig.colorbar(pmplot,orientation=cbarorientation)
              if logScale:
                  cbartickvals = cbar.ax.yaxis.get_ticklabels()
                  tickVals = []
                  # need this roundabout trick to handle minus sign in unicode
                  for item in cbartickvals:
                      valstr = item.get_text().replace(u'\u2212', '-').replace('$','')
                      val = 10**float(valstr)
                      if abs(val) < 1000:
                          str = '{0:f}'.format(val)
                      else:
                          str = '{0:e}'.format(val)
                      tickVals.append(str)
                  cbartickvals = cbar.ax.yaxis.set_ticklabels(tickVals)
          else:
              ticks,  ticklabels = zip(*cbarcustomticks)
              cbar = self.fig.colorbar(pmplot,ticks=ticks, orientation=cbarorientation)
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

      #set up the grids
      if thetagrid is None:
          plt.thetagrids([])
      else:
          plt.thetagrids(range(0, 360, thetagrid[0]))
      plt.tick_params(axis='x', which='major', labelsize=thetagridfontsize)

      if rgrid is None:
          ax.set_yticks([])
      elif len(rgrid) is 1:
          ax.set_yticks(np.linspace(0,np.max(radial),rgrid[0]))
      elif len(rgrid) is 2:
          plt.rgrids(np.arange(rgrid[0], rgrid[1], rgrid[0]))
      else:
          pass
      plt.tick_params(axis='y', which='major', labelsize=radialgridfontsize)

      ax.set_theta_direction(direction)
      ax.set_theta_offset(zerooffset)

         # rmax = np.max(radial)
          # if rmax>0:
          #     #round and increase the max value for nice numbers
          #     lrmax=round(math.floor(math.log10(rmax/rgrid[1])))
          #     frmax=rmax/(rgrid[1]*10**lrmax)
          #     rinc=10**lrmax*math.ceil(frmax)
          #     plt.rgrids(np.arange(rinc, rinc*rgrid[1], rinc))
          # else:
          #     ax.set_yticks(np.linspace(0,np.max(radial),5))

      return ax


    
    ############################################################
    ##
    def plotArray(self, plotnum, inarray, slicedim = 0, subtitles = None, xlabel=None, 
                        maxNX=0, maxNY=0, titlefsize = 8, xylabelfsize = 8,
                        xytickfsize = 8  ):
        """Creates a plot from an input array.

        Given an input array with m x n dimensions, this function creates a subplot for vectors
        [1-n]. Vector 0 serves as the x-axis for each subplot. The slice dimension can be in
        columns (0) or rows (1).


            Args:
                | plotnum (int): subplot number, 1-based index
                | inarray (array): np.array
                | slicedim (int): slice along columns (0) or rows (1)
                | subtitles (list): a list of strings as subtitles for each subplot
                | xlabel (string): x-axis label (optional)
                | maxNX (int): draw maxNX+1 tick labels on x axis (optional)
                | maxNY (int): draw maxNY+1 tick labels on y axis (optional)
                | titlefsize (int): title font size, default 12pt (optional)
                | xylabelfsize (int): x-axis, y-axis label font size, default 12pt (optional)
                | xytickfsize (int): x-axis, y-axis tick font size, default 10pt (optional)

            Returns:
                | Nothing

            Raises:
                | No exception is raised.
        """

        fig = self.fig
        #use current subplot number as outer grid reference
        outer_grid = gridspec.GridSpec(self.nrow,self.ncol, wspace=0, hspace=0)

        #if slicedim = 0, slice across columns
        if slicedim == 0:
            #x-axis is first vector
            x = inarray[:,0]
            #xlabel is time
            xlabel = subtitles[0]

            yAll = inarray[:,1:].transpose()

            nestnrow = inarray.shape[1]-1
            nestncol = 1
            plottitles = subtitles[1:]

        #if slicedim = 1, slice across rows
        elif slicedim == 1:
            x = range(0,inarray.shape[1]-1)
            #for slicedim = 1, the tick labels are in the x title
            xlabel = subtitles[1:inarray.shape[1]]

            yAll = inarray[:,1:]
            nestnrow = inarray.shape[0]
            nestncol = 1
            plottitles = inarray[:,0]


        #inner_grid (nested):
        inner_grid = gridspec.GridSpecFromSubplotSpec(nestnrow,nestncol, 
                        subplot_spec=outer_grid[0],wspace=0, hspace=0.2)

        #create subplot for each y-axis vector
        nestplotnum = 0

        for y in yAll:

            ax = plt.Subplot(fig, inner_grid[nestplotnum])
            ax.plot(x,y)
            if subtitles is not None:
                ax.set_ylabel(plottitles[nestplotnum], fontsize=xylabelfsize)
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
            
            nestplotnum = nestplotnum + 1
            fig.add_subplot(ax)

            #share x ticklabels and label to avoid clutter and overlapping
            plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
            if xlabel is not None:
                fig.axes[-1].set_xlabel(xlabel, fontsize=xylabelfsize)

            # minor ticks are two points smaller than major
            # ax.tick_params(axis='both', which='major', labelsize=xytickfsize)
            # ax.tick_params(axis='both', which='minor', labelsize=xytickfsize-2)

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
            if isinstance(saveName, basestring):
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
    

    Parameters
    ----------
    start : scalar, optional
        Sets the starting position in the color space. 0=blue, 1=red, 
        2=green. Defaults to 0.5.
    rot : scalar, optional
        The number of rotations through the rainbow. Can be positive 
        or negative, indicating direction of rainbow. Negative values
        correspond to Blue->Red direction. Defaults to -1.5
    gamma : scalar, optional
        The gamma correction for intensity. Defaults to 1.0        
    hue : scalar, optional
        The hue intensity factor. Defaults to 1.2
    reverse : boolean, optional
        Set to True to reverse the color map. Will go from black to
        white. Good for density plots where shade~density. Defaults to False
    nevl : scalar, optional
        Defines the number of discrete levels to render colors at.
        Defaults to 256.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap object

    Example
    -------
    >>> import cubehelix
    >>> cx = cubehelix.cmap(start=0., rot=-0.5)
    >>> plot(x,cmap=cx)

    Revisions
    ---------
    2014-04 (@jradavenport) Ported from IDL version

    source
    ------
    https://github.com/jradavenport/cubehelix

    Licence
    -------
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
    y = np.asarray(range(len(x)))
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

    P3D.plot3d(1, x.T, y.T, z.T, 'Parametric Curve', 'X', 'Y', 'Z', label=label, legendAlpha=0.5)
    P3D.saveFig('M3D.png')

    ############################################################################
    # demonstrate the use of the contextmanager and with statement
    x=np.linspace(-3,3,20)
    with savePlot(1,saveName=['testwith.png','testwith.eps']) as p:
        p.plot(1,x,x*x)
    with savePlot(1,saveName='testwith.svg') as p:
        p.plot(1,x,x*x)

    ############################################################################
    import matplotlib.mlab as mlab
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    # difference of Gaussians
    Z = 10.0 * (Z2 - Z1)
    pmc = Plotter(1)
    pmc.meshContour(1, X, Y, Z, numLevels=15,
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
      drawGrid=True, cmap=cm.jet,  cbarshow=True, elevation=70, azimuth=15)
    p.mesh3D(4, varx, vary, zdata, ptitle='Title', xlabel='x', ylabel='y', zlabel='z',
      rstride=3, cstride=3, linewidth= 0, maxNX=5, maxNY=5, maxNZ=0, drawGrid=True,
      cmap=cm.brg, cbarshow=True)

    p.saveFig('mesh3d01.png')



    ############################################################################
    #demonstrate the use of plotArray
    #import array from example data file
    filename = "data/arrayplotdemo.txt"
    f = open(filename)
    lines = f.readlines()
    #the titles are in the first line (row). Skip the '%' symbol
    titles = lines[0].split()[1:]
    #the array is the rest of the file
    arrDummyDemo = np.genfromtxt(filename,skip_header=1)
    #the figure title is the filename
    maintitle = filename.split('/')[-1]

    Ar = Plotter(9, 1, 1,maintitle)
    Ar.plotArray(1,arrDummyDemo, 0, titles, titlefsize = 12, maxNX = 5, maxNY=3)
    # Ar.saveFig('ArrayPlot.eps')
    Ar.saveFig('ArrayPlot.png')


    ############################################################################
    #demonstrate the use of a polar mesh plot and markers
    #create the radial and angular vectors

    r = np.linspace(0,1.25,100)
    p = np.linspace(0,2*np.pi,100)
    P, R = np.meshgrid(p, r)
    value =  ((R**2 - 1)**2) * np.sin(2 * P)
    pmesh = Plotter(1, 1, 1,'Polar plot in mesh',figsize=(12,8))
    pmesh.polarMesh(1, p, r, value, cbarshow=True, \
                  cbarorientation = 'vertical', cbarfontsize = 10,  \
                  )#, radscale=[0.5,1.25])

    # add filled markers
    markers = Markers(markerfacecolor='y', marker='*')
    markers.add(0*np.pi/6,1)
    markers.add(1*np.pi/6,0.9,markerfacecolor='k', marker='v',fillstyle='top')
    markers.add(2*np.pi/6,0.8,fillstyle='top',markeredgecolor='g')
    markers.add(3*np.pi/6,0.7,marker='v')
    markers.add(4*np.pi/6,0.6,marker='p',fillstyle='top')
    markers.add(5*np.pi/6,0.5,markerfacecolor='r',marker='H',fillstyle='bottom',markerfacecoloralt='PaleGreen')
    markers.add(6*np.pi/6,0.4,marker='D',fillstyle='left',markerfacecoloralt='Sienna',markersize=10)
    markers.plot(pmesh.getSubPlot(1))

    pmesh.saveFig('pmesh.png')
    #pmesh.saveFig('pmesh.eps')


    ############################################################################
    ##create some data
    xLinS=np.linspace(0, 10, 50).reshape(-1, 1)
    yLinS=1.0e3 * np.random.random(xLinS.shape[0]).reshape(-1, 1)
    yLinSS=1.0e3 * np.random.random(xLinS.shape[0]).reshape(-1, 1)

    yLinA=yLinS
    yLinA = np.hstack((yLinA, \
            1.0e7 * np.random.random(xLinS.shape[0]).reshape(-1, 1)))
    yLinA = np.hstack((yLinA, \
            1.0e7 * np.random.random(xLinS.shape[0]).reshape(-1, 1)))

    A = Plotter(1, 2, 2,'Array Plots',figsize=(12,8))
    A.plot(1, xLinS, yLinA, "Array Linear","X", "Y",
            plotCol=['c--'],
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
    AA.plot(1, xLinS, yLinA, plotCol=['b--'],
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
           thetagrid=[45], direction=u'clockwise', zerooffset=0)
    P.polar(3,theta, r, "Single Polar",\
           label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0,3], \
           direction=u'clockwise', zerooffset=np.pi/2)
    P.polar(4,theta, r2, "Array Polar",\
           label=['A', 'B'],legendAlpha=0.5,rscale=[0,9],rgrid=[0,6],\
           thetagrid=[45], direction=u'counterclockwise', zerooffset=-np.pi/2)
    P.saveFig('P.png')
    #P.saveFig('P.eps')
    #plot again on top of existing graphs
    rr = np.arange(0.1, 3.11, 0.01).reshape(-1, 1)
    thetar = 2*np.pi*rr
    P.polar(1,thetar, 0.5 * rr, "Single Polar",\
           plotCol='r',label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0.5,3])
    P.polar(2,thetar, 0.75 * rr, "Array Polar",\
           plotCol='g',label=['A', 'B'],legendAlpha=0.5,rscale=[0,6],rgrid=[2,6],\
           thetagrid=[45], direction=u'clockwise', zerooffset=0)
    P.polar(3,thetar, 1.2 * rr, "Single Polar",\
           plotCol='k',label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0,3], \
           direction=u'clockwise', zerooffset=np.pi/2)
    P.polar(4,thetar, 1.5 * rr, "Array Polar",\
           plotCol='y',label=['A', 'B'],legendAlpha=0.5,rscale=[0,9],rgrid=[0,6],\
           thetagrid=[45], direction=u'counterclockwise', zerooffset=-np.pi/2)
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
           legendAlpha=0.5,rgrid=[1,5],highlightNegative=True,direction=u'clockwise',
           zerooffset=np.pi/2,highlightCol='r',highlightWidth=2)
    PN.saveFig('PN.png')
    #PN.saveFig('PN.eps')

    #test/demo to show that multiple plots can be done in the same subplot, on top of older plots
    xLinS=np.linspace(0, 10, 50).reshape(-1, 1)
    M= Plotter(1, 1, 1,'Multi-plots',figsize=(12,8))
    #it seems that all attempts to plot in same subplot space must use same ptitle.
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
    barticks = zip([-1, 0, 1], ['low', 'med', 'high'])
    I.showImage(2, z, ptitle='prism colormap, default font ', cmap=plt.cm.prism, cbarshow=True, cbarcustomticks=barticks)
    I.showImage(3, z, ptitle='default gray colormap, font 8pt', cbarshow=True, titlefsize=8)
    I.plot(4, xv[:, 1],  z, "Array Linear","x", "z")
    I.saveFig('I.png')
#    I.saveFig('I.eps')
    #plot on existing
    I.showImage(1, z, ptitle='winter colormap, font 10pt', cmap=plt.cm. winter, titlefsize=10,  cbarshow=True, cbarorientation = 'horizontal', cbarfontsize = 7)
    barticks = zip([-1, 0, 1], ['low', 'med', 'high'])
    I.showImage(2, z, ptitle='prism colormap, default font ', cmap=plt.cm. prism, cbarshow=True, cbarcustomticks=barticks)
    I.showImage(3, z, ptitle='default gray colormap, font 8pt', titlefsize=8)
    I.plot(4, xv[:, 1],  z, "Array Linear","x", "z")
    I.saveFig('II.png')
#    I.saveFig('II.eps')

    I = Plotter(5, 2, 2,'Images & Array Linear', figsize=(12, 8))
    I.showImage(1, z, ptitle='winter colormap, font 10pt', cmap=plt.cm. winter, titlefsize=10,  cbarshow=True, cbarorientation = 'horizontal', cbarfontsize = 7)
    barticks = zip([-1, 0, 1], ['low', 'med', 'high'])
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

    x=np.linspace(0, 2*np.pi, 50).reshape(-1, 1)
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

    print('module ryplot done!')
