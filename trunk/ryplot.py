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

# Contributor(s): MS Willers, PJ van der Merwe.
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
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__ = "$Revision$"
__author__ = 'pyradi team'
__all__ = ['Plotter']

import numpy
import math
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

class Plotter:
    """ Encapsulates a plotting environment, optimized for
    radiometry plots.

    This class provides a wrapper around Matplotlib to provide a plotting
    environment specialised towards radiometry results.  These functions
    were developed to provide well labelled plots by entering only one or two lines.

    Provision is made for plots containing subplots (i.e. multiple plots on the same figure),
    linear scale and log scale plots, and cartesian and polar plots.
    Simple 3D line plots can also be made.
    """

    ############################################################
    ##
    def __init__(self,fignumber=0,subpltnrow=1,subpltncol=1,\
                 figuretitle=None,
                 figsize=(9,9)):
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
                        'semilogY', 'polar', 'showImage', 'plot3d']

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

        self.plotCol=['b', 'g', 'r', 'c', 'm', 'y', 'k', \
            'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 'k--']

        self.bbox_extra_artists=[]
        self.subplots={}

        if figuretitle:
            self.figtitle=plt.gcf().text(.5,.95,figuretitle,\
                        horizontalalignment='center',\
                        fontproperties=FontProperties(size=16))
            self.bbox_extra_artists.append(self.figtitle)



    ############################################################
    ##
    def buildPlotCol(self, plotCol, n):
        """Returns a sequence of default colour styles of
           appropriate length.

           The constructor provides a sequence with length 14 pre-defined plot styles.
           The user can define a new sequence if required.
           This function modulus-folds either sequence, in case longer sequences are required.

            Args:
                | plotCol ([strings]): User-supplied list  of plotting styles(can be empty []).
                | n (int): Length of required sequence.

            Returns:
                | A list with sequence of plot styles, of required length.

            Raises:
                | No exception is raised.
        """
        if not plotCol:
            return [self.plotCol[i % len(self.plotCol)] \
                                         for i in range(n)]
        else:
            return [plotCol[i % len(plotCol)] \
                                         for i in range(n)]

    ############################################################
    ##
    def saveFig(self, filename='mpl.png',dpi=100,bbox_inches='tight',\
                pad_inches=0.1):
        """Save the plot to a disk file, using filename, dpi specification and bounding box limits.

        One of matplotlib's design choices is a bounding box strategy  which may result in a bounding box
        that is smaller than the size of all the objects on the page.  It took a while to figure this out,
        but the current default values for bbox_inches and pad_inches seem to create meaningful
        bounding boxes. These are however larger than the true bounding box. You still need a
        tool such as epstools or Adobe Acrobat to trim eps files to the true bounding box.

            Args:
                | filename (string): output filename to write plot, file ext
                | dpi (int): the resolution of the graph in dots per inch
                | bbox_inches: see matplotlib docs for more detail.
                | pad_inches: see matplotlib docs for more detail.


            Returns:
                | Nothing. Saves a file to disk.

            Raises:
                | No exception is raised.
        """
        if len(filename)>0:
            if self.bbox_extra_artists:
                plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                            pad_inches=pad_inches,\
                            bbox_extra_artists= self.bbox_extra_artists)
            else:
                plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                            pad_inches=pad_inches)


    ############################################################
    ##
    def getPlot(self):
        """Returns a handle to the current plot

            Args:
                | None

            Returns:
                | A handle to the current plot.

            Raises:
                | No exception is raised.
        """
        return self.fig


    ############################################################
    ##
    def getSubPlot(self, subplotNum = 1):
        """Returns a handle the subplot, as requested per subplot number.
        Subplot numbers range from 1 upwards.

            Args:
                | subplotNumer (int) : number of the subplot

            Returns:
                | A handle to the requested subplot.

            Raises:
                | No exception is raised.
        """
        if (self.nrow,self.ncol, subplotNum) in self.subplots.keys():
            return self.subplots[(self.nrow,self.ncol, subplotNum)]
        else:
            return None


    ############################################################
    ##
    def plot(self, plotnum, x, y, ptitle=None, xlabel=None, ylabel=None, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], maxNX=10, maxNY=10, \
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12  ):
        """Cartesian plot on linear scales for abscissa and ordinates.

        Given an existing figure, this function plots in a specified subplot position.
        The function arguments are described below in some detail. Note that the y-values
        or ordinates can be more than one column, each column representing a different
        line in the plot. This is convenient if large arrays of data must be plotted. If more
        than one column is present, the label argument can contain the legend labels for
        each of the columns/lines.  The pltaxis argument defines the min/max scale values
        for the x and y axes.

            Args:
                | plotnum (int): subplot number
                | x (np.array[N,] or [N,1]): abscissa
                | y (np.array[N,] or [N,M]): ordinates - could be M columns
                | ptitle (string): plot title (optional)
                | xlabel (string): x axis label (optional)
                | ylabel (string): y axis label (optional)
                | plotCol ([strings]): plot line style, list with M entries, use default if [] (optional)
                | label  ([strings]): legend label for ordinate, list with M entries (optional)
                | legendAlpha (float): transparancy for legend (optional)
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. default if all zeros. (optional)
                | maxNX (int): draw maxNX+1 tick labels on x axis (optional)
                | maxNY (int): draw maxNY+1 tick labels on y axis (optional)
                | powerLimits[float]: axis notation power limits [x-neg, x-pos, y-neg, y-pos]
                | titlefsize (int): title font size, default 12pt (optional)

            Returns:
                | Nothing

            Raises:
                | No exception is raised.
       """
        ## see self.MyPlot for parameter details.
        self.myPlot(plt.plot, plotnum, x, y, ptitle, xlabel, ylabel, \
                    plotCol, label,legendAlpha, pltaxis, \
                    maxNX, maxNY, powerLimits)

    ############################################################
    ##
    def logLog(self, plotnum, x, y, ptitle=None, xlabel=None, ylabel=None, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], maxNX=10, maxNY=10, \
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12  ):
        """Plot data on logarithmic scales for abscissa and ordinates.

        Given an existing figure, this function plots in a specified subplot position.
        The function arguments are described below in some detail. Note that the y-values
        or ordinates can be more than one column, each column representing a different
        line in the plot. This is convenient if large arrays of data must be plotted. If more
        than one column is present, the label argument can contain the legend labels for
        each of the columns/lines.  The pltaxis argument defines the min/max scale values
        for the x and y axes.

            Args:
                | plotnum (int): subplot number
                | x (np.array[N,] or [N,1]): abscissa
                | y (np.array[N,] or [N,M]): ordinates - could be M columns
                | ptitle (string): plot title (optional)
                | xlabel (string): x axis label (optional)
                | ylabel (string): y axis label (optional)
                | plotCol ([strings]): plot line style, list with M entries, use default if [] (optional)
                | label  ([strings]): legend label for ordinate, list with M entries (optional)
                | legendAlpha (float): transparancy for legend (optional)
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. default if all zeros. (optional)
                | maxNX (int): draw maxNX+1 tick labels on x axis (optional)
                | maxNY (int): draw maxNY+1 tick labels on y axis (optional)
                | powerLimits[float]: axis notation power limits [x-neg, x-pos, y-neg, y-pos] (optional)
                | titlefsize (int): title font size, default 12pt (optional)

            Returns:
                | Nothing

            Raises:
                | No exception is raised.
       """
        ## see self.MyPlot for parameter details.
        self.myPlot(plt.loglog, plotnum, x, y, ptitle, xlabel,ylabel,\
                    plotCol, label,legendAlpha, pltaxis, \
                    maxNX, maxNY, powerLimits)

    ############################################################
    ##
    def semilogX(self, plotnum, x, y, ptitle=None, xlabel=None, ylabel=None, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], maxNX=10, maxNY=10, \
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12  ):
        """Plot data on logarithmic scales for abscissa and linear scale for ordinates.

        Given an existing figure, this function plots in a specified subplot position.
        The function arguments are described below in some detail. Note that the y-values
        or ordinates can be more than one column, each column representing a different
        line in the plot. This is convenient if large arrays of data must be plotted. If more
        than one column is present, the label argument can contain the legend labels for
        each of the columns/lines.  The pltaxis argument defines the min/max scale values
        for the x and y axes.

            Args:
                | plotnum (int): subplot number
                | x (np.array[N,] or [N,1]): abscissa
                | y (np.array[N,] or [N,M]): ordinates - could be M columns
                | ptitle (string): plot title (optional)
                | xlabel (string): x axis label (optional)
                | ylabel (string): y axis label (optional)
                | plotCol ([strings]): plot line style, list with M entries, use default if [] (optional)
                | label  ([strings]): legend label for ordinate, list with M entries (optional)
                | legendAlpha (float): transparancy for legend (optional)
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. default if all zeros. (optional)
                | maxNX (int): draw maxNX+1 tick labels on x axis (optional)
                | maxNY (int): draw maxNY+1 tick labels on y axis (optional)
                | powerLimits[float]: axis notation power limits [x-neg, x-pos, y-neg, y-pos] (optional)
                | titlefsize (int): title font size, default 12pt (optional)

            Returns:
                | Nothing

            Raises:
                | No exception is raised.
       """
        ## see self.MyPlot for parameter details.
        self.myPlot(plt.semilogx, plotnum, x, y, ptitle, xlabel, ylabel,\
                    plotCol, label,legendAlpha, pltaxis, \
                    maxNX, maxNY, powerLimits)

    ############################################################
    ##
    def semilogY(self, plotnum, x, y, ptitle=None, xlabel=None, ylabel=None, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], maxNX=10, maxNY=10, \
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12  ):
        """Plot data on linear scales for abscissa and logarithmic scale for ordinates.

        Given an existing figure, this function plots in a specified subplot position.
        The function arguments are described below in some detail. Note that the y-values
        or ordinates can be more than one column, each column representing a different
        line in the plot. This is convenient if large arrays of data must be plotted. If more
        than one column is present, the label argument can contain the legend labels for
        each of the columns/lines.  The pltaxis argument defines the min/max scale values
        for the x and y axes.

            Args:
                | plotnum (int): subplot number
                | x (np.array[N,] or [N,1]): abscissa
                | y (np.array[N,] or [N,M]): ordinates - could be M columns
                | ptitle (string): plot title (optional)
                | xlabel (string): x axis label (optional)
                | ylabel (string): y axis label (optional)
                | plotCol ([strings]): plot line style, list with M entries, use default if [] (optional)
                | label  ([strings]): legend label for ordinate, list withM entries (optional)
                | legendAlpha (float): transparancy for legend (optional)
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. default if all zeros. (optional)
                | maxNX (int): draw maxNX+1 tick labels on x axis (optional)
                | maxNY (int): draw maxNY+1 tick labels on y axis (optional)
                | powerLimits[float]: axis notation power limits [x-neg, x-pos, y-neg, y-pos] (optional)
                | titlefsize (int): title font size, default 12pt (optional)

            Returns:
                | Nothing

            Raises:
                | No exception is raised.
       """
        ## see self.MyPlot for parameter details.
        self.myPlot(plt.semilogy, plotnum, x, y, ptitle,xlabel,ylabel,\
                    plotCol, label,legendAlpha, pltaxis, \
                    maxNX, maxNY, powerLimits)

    ############################################################
    ##
    def myPlot(self, plotcommand,plotnum, x, y, ptitle=None,xlabel=None,ylabel=None,
                     plotCol=[],label=[],legendAlpha=0.0,\
                    pltaxis=[0, 0, 0, 0], maxNX=0, maxNY=0,  \
                    powerLimits = [-4,  2,  -4,  2], titlefsize = 12 ):
        """Low level helper function to create a subplot and plot the data as required.

        This function does the actual plotting, labelling etc. It uses the plotting
        function provided by its user functions.

            Args:
                | plotcommand: name of a MatplotLib plotting function
                | plotnum (int): subplot number
                | ptitle (string): plot title
                | xlabel (string): x axis label
                | ylabel (string): y axis label
                | x (np.array[N,] or [N,1]): abscissa
                | y (np.array[N,] or [N,M]): ordinates - could be M columns
                | plotCol ([strings]): plot line style, list with M entries, use default if []
                | label  ([strings]): legend label for ordinate, list with M entries
                | legendAlpha (float): transparancy for legend
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. default if all zeros.
                | maxNX (int): draw maxNX+1 tick labels on x axis
                | maxNY (int): draw maxNY+1 tick labels on y axis
                | powerLimits[float]: axis notation power limits [x-neg, x-pos, y-neg, y-pos]
                | titlefsize (int): title font size, default 12pt (optional)

            Returns:
                | Nothing

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

        plotCol = self.buildPlotCol(plotCol, yy.shape[1])

        ax = None
        pkey = (self.nrow, self.ncol, plotnum)
        if pkey not in self.subplots.keys():
            self.subplots[pkey] = \
                         self.fig.add_subplot(self.nrow,self.ncol, plotnum)

        ax = self.subplots[pkey]

        ax.grid(True)

        # use scientific format on axes
        #yfm = sbp.yaxis.get_major_formatter()
        #yfm.set_powerlimits([ -3, 3])

        if xlabel is not None:
            ax.set_xlabel(xlabel)
            formx = plt.ScalarFormatter()
            formx.set_scientific(True)
            formx.set_powerlimits([powerLimits[0], powerLimits[1]])
            ax.xaxis.set_major_formatter(formx)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
            formy = plt.ScalarFormatter()
            formy.set_powerlimits([powerLimits[2], powerLimits[3]])
            formy.set_scientific(True)
            ax.yaxis.set_major_formatter(formy)

        if maxNX >0:
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(maxNX))
        if maxNY >0:
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(maxNY))
        if not label:
            for i in range(yy.shape[1]):
                #plotcommand(xx, yy[:, i], plotCol[i],label=None)
                ax.plot(xx, yy[:, i], plotCol[i],label=None)
        else:
            for i in range(yy.shape[1]):
                #plotcommand(xx,yy[:,i],plotCol[i],label=label[i])
                ax.plot(xx,yy[:,i],plotCol[i],label=label[i])
            leg = ax.legend(loc='best', fancybox=True)
            leg.get_frame().set_alpha(legendAlpha)
            self.bbox_extra_artists.append(leg)

        #scale the axes
        if sum(pltaxis)!=0:
            ax.axis(pltaxis)

        if(ptitle is not None):
            ax.set_title(ptitle, fontsize=titlefsize)

    ############################################################
    ##
    def polar(self, plotnum, theta, r, ptitle=None, \
                    plotCol=[], label=[],labelLocation=[-0.1, 0.1], \
                    highlightNegative=False, highlightCol='#ffff00', highlightWidth=4,\
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
        counterclockwise. Likewise the rotation offset where the plot zero angle must be,
        is set with zerooffset.

            Args:
                | plotnum (int): subplot number
                | theta (np.array[N,] or [N,1]): angular abscissa
                | r (np.array[N,] or [N,M]): radial ordinates - could be M columns
                | ptitle (string): plot title (optional)
                | plotCol ([strings]): plot line style, list with M entries, use default if [] (optional)
                | label  ([strings]): legend label, list with M entries (optional)
                | labelLocation ([x,y]): where the legend should located (optional)
                | highlightNegative (bool): indicate if negative data be highlighted (optional)
                | highlightCol (string): highlighted colour string (optional)
                | highlightWidth (int): highlighted line width(optional)
                | legendAlpha (float): transparancy for legend (optional)
                | rscale ([rmin, rmax]): plotting limits. default if all 0 (optional)
                | rgrid ([rinc, rmax]): radial grid default if all 0. if rinc=0 then rmax is number of ntervals. (optional)
                | thetagrids (float): theta grid interval [degrees] (optional)
                | direction (string)= 'counterclockwise' or 'clockwise' (optional)
                | zerooffset (float) = rotation offset where zero should be [rad] (optional)
                | titlefsize (int): title font size, default 12pt (optional)

            Returns:
                | Nothing

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

        plotCol = self.buildPlotCol(plotCol, rr.shape[1])

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
            #     ttt = tt + numpy.pi*(rr[:, i] < 0).reshape(-1, 1)
            #     rrr = numpy.abs(rr[:, i])
            # else:
            ttt = tt
            rrr = rr[:, i]

            #print(rrr)

            if highlightNegative:
                #find zero crossings in data
                zero_crossings = numpy.where(numpy.diff(numpy.sign(rr),axis=0))[0]
                #split the input into different subarrays according to crossings
                negrrr = numpy.split(rr,zero_crossings)
                negttt = numpy.split(tt,zero_crossings)
                #print(zero_crossings)
                #print(negrrr)

            if not label:
                if highlightNegative:
                    lines = ax.plot(ttt, rrr, plotCol[i])
                    neglinewith = highlightWidth*plt.getp(lines[0],'linewidth')
                    for ii in range(0,len(negrrr)):
                        if len(negrrr[ii]) > 0:
                            if negrrr[ii][1] < 0:
                                ax.plot(negttt[ii], negrrr[ii], highlightCol,linewidth=neglinewith)
                ax.plot(ttt, rrr, plotCol[i])
                rmax=numpy.maximum(numpy.abs(rrr).max(), rmax)
            else:
                if highlightNegative:
                    lines = ax.plot(ttt, rrr, plotCol[i])
                    neglinewith = highlightWidth*plt.getp(lines[0],'linewidth')
                    for ii in range(0,len(negrrr)):
                        if len(negrrr[ii]) > 0:
                            if negrrr[ii][1] < 0:
                                ax.plot(negttt[ii], negrrr[ii], highlightCol,linewidth=neglinewith)
                ax.plot(ttt, rrr, plotCol[i],label=label[i])
                rmax=numpy.maximum(numpy.abs(rrr).max(), rmax)

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

        if rgrid is None:
            ax.set_yticks(numpy.linspace(0,rmax,5))
        else:
            if rgrid[0]==0:
                if rmax>0:
                    #round and increase the max value for nice numbers
                    lrmax=round(math.floor(math.log10(rmax/rgrid[1])))
                    frmax=rmax/(rgrid[1]*10**lrmax)
                    rinc=10**lrmax*math.ceil(frmax)
                    plt.rgrids(numpy.arange(rinc, rinc*rgrid[1], rinc))
            else:
                plt.rgrids(numpy.arange(rgrid[0], rgrid[1], rgrid[0]))



        #Set increment and maximum radial limits
        if rscale is not None:
            ax.set_ylim(rscale[0],rscale[1])
            ax.set_yticks(numpy.linspace(rscale[0],rscale[1],5))
        else:
            ax.set_ylim(0,rmax)


        if(ptitle is not None):
            ax.set_title(ptitle, fontsize=titlefsize, \
                verticalalignment ='bottom', horizontalalignment='center')


    ############################################################
    ##
    def showImage(self, plotnum, img,  ptitle=None, cmap=plt.cm.gray, titlefsize=12, cbarshow=False, \
                  cbarorientation = 'vertical', cbarcustomticks=[], cbarfontsize = 12):
        """Creates a subplot and show the image using the colormap provided.

            Args:
                | plotnum (int): subplot number
                | img (np.ndarray): numpy 2d array
                | ptitle (string): plot title (optional)
                | cmap: matplotlib colormap, default gray (optional)
                | fsize (int): title font size, default 12pt (optional)
                | cbarshow (bool): if true, the show a color bar
                | cbarorientation (string): 'vertical' (right) or 'horizontal' (below)
                | cbarcustomticks zip([tick locations/float],[tick labels/string]): locations in image grey levels
                | cbarfontsize (int): font size for color bar

            Returns:
                | Nothing

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





    ############################################################
    ##
    def plot3d(self, plotnum, x, y, z, ptitle=None, xlabel=None, ylabel=None, zlabel=None, \
               plotCol=[], label=None, legendAlpha=0.0, titlefsize=12):
        """3D plot on linear scales for x y z input sets.

        Given an existing figure, this function plots in a specified subplot position.
        The function arguments are described below in some detail.

        Note that multiple 3D data sets can be plotted simultaneously by adding additional columns to
        the input coordinates of vertices, each column representing a different function in the plot.
        This is convenient if large arrays of data must be plotted. If more than one column is present,
        the label argument can contain the legend labels for each of the columns/lines.

            Args:
                | plotnum (int): subplot number
                | x (np.array[N,] or [N,M]): x coordinates of vertices
                | y (np.array[N,] or [N,M]): y coordinates of vertices
                | z (np.array[N,] or [N,M]): z coordinates of vertices
                | ptitle (string): plot title (optional)
                | xlabel (string): x axis label (optional)
                | ylabel (string): y axis label (optional)
                | plotCol ([strings]): plot line style, list with M entries, use default if [] (optional)
                | label  ([strings]): legend label for ordinate, list with M entries (optional)
                | legendAlpha (float): transparancy for legend (optional)
                | titlefsize (int): title font size, default 12pt (optional)

            Returns:
                | Nothing

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

        plotCol = self.buildPlotCol(plotCol, x.shape[-1])

        if (self.nrow,self.ncol, plotnum) not in self.subplots.keys():
            self.subplots[(self.nrow,self.ncol, plotnum)] = \
                 self.fig.add_subplot(self.nrow,self.ncol, plotnum, projection='3d')

        ax = self.subplots[(self.nrow,self.ncol, plotnum)]

        for i in range(x.shape[-1]):
            ax.plot(x[:,i], y[:,i], z[:,i], plotCol[i])

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if zlabel is not None:
            ax.set_zlabel(zlabel)

        if label is not None:
            leg = plt.legend(label, loc='best', fancybox=True)
            leg.get_frame().set_alpha(legendAlpha)
            self.bbox_extra_artists.append(leg)

        if(ptitle is not None):
            plt.title(ptitle, fontsize=titlefsize)



################################################################
################################################################
##
## plot graphs and confirm the correctness of the functions

if __name__ == '__main__':



    # ##create some data
    # xLinS=numpy.linspace(0, 10, 50).reshape(-1, 1)
    # yLinS=1.0e3 * numpy.random.random(xLinS.shape[0]).reshape(-1, 1)
    # yLinSS=1.0e3 * numpy.random.random(xLinS.shape[0]).reshape(-1, 1)

    # yLinA=yLinS
    # yLinA = numpy.hstack((yLinA, \
    #         1.0e7 * numpy.random.random(xLinS.shape[0]).reshape(-1, 1)))
    # yLinA = numpy.hstack((yLinA, \
    #         1.0e7 * numpy.random.random(xLinS.shape[0]).reshape(-1, 1)))

    # A = Plotter(1, 2, 2,'Array Plots',figsize=(12,8))
    # A.plot(1, xLinS, yLinA, "Array Linear","X", "Y",
    #         plotCol=['c--'],
    #        label=['A1', 'A2', 'A3'],legendAlpha=0.5,
    #        pltaxis=[0, 10, 0, 2000],
    #        maxNX=10, maxNY=2,
    #        powerLimits = [-4,  2, -5, 5])
    # A.logLog(2, xLinS, yLinA, "Array LogLog","X", "Y",\
    #          label=['A1', 'A2', 'A3'],legendAlpha=0.5)
    # A.semilogX(3, xLinS, yLinA, "Array SemilogX","X", "Y",\
    #            label=['A1', 'A2', 'A3'],legendAlpha=0.5)
    # A.semilogY(4, xLinS, yLinA, "Array SemilogY","X", "Y",\
    #            label=['A1', 'A2', 'A3'],legendAlpha=0.5)
    # A.saveFig('A.png')
    # #A.saveFig('A.eps')

    # AA = Plotter(1, 1, 1,'Demonstrate late labels',figsize=(12,8))
    # AA.plot(1, xLinS, yLinA, plotCol=['b--'],
    #        label=['A1', 'A2', 'A3'],legendAlpha=0.5,
    #        pltaxis=[0, 10, 0, 2000],
    #        maxNX=10, maxNY=2,
    #        powerLimits = [-4,  2, -5, 5])
    # currentP = AA.getSubPlot(1)
    # currentP.set_xlabel('X Label')
    # currentP.set_ylabel('Y Label')
    # currentP.set_title('The figure title')
    # currentP.annotate('axes center', xy=(.5, .5),  xycoords='axes fraction',
    #             horizontalalignment='center', verticalalignment='center')
    # currentP.text(0.5 * 10, 1300,
    #      r"$\int_a^b f(x)\mathrm{d}x$", horizontalalignment='center',
    #      fontsize=20)
    # AA.saveFig('AA.png')
    # #AA.saveFig('AA.eps')

    # S = Plotter(2, 2, 2,'Single Plots',figsize=(12,8))
    # S.plot(1, xLinS, yLinS, "Single Linear","X", "Y",\
    #        label=['Original'],legendAlpha=0.5)
    # S.logLog(2, xLinS, yLinS, "Single LogLog","X", "Y",\
    #          label=['Original'],legendAlpha=0.5)
    # S.semilogX(3, xLinS, yLinS, "Single SemilogX","X", "Y",\
    #            label=['Original'],legendAlpha=0.5)
    # S.semilogY(4, xLinS, yLinS, "Single SemilogY","X", "Y",\
    #            label=['Original'],legendAlpha=0.5)
    # S.saveFig('S.png', dpi=300)
    # #S.saveFig('S.eps')
    # #plot again on top of the existing graphs
    # S.plot(1, xLinS, yLinSS, "Single Linear","X", "Y",\
    #            plotCol='r',label=['Repeat on top'],legendAlpha=0.5)
    # S.logLog(2, xLinS, 1.3*yLinSS, "Single LogLog","X", "Y",\
    #           plotCol='g',label=['Repeat on top'],legendAlpha=0.5)
    # S.semilogX(3, xLinS, 0.5*yLinSS, "Single SemilogX","X", "Y",\
    #            plotCol='k',label=['Repeat on top'],legendAlpha=0.5)
    # S.semilogY(4, xLinS, 0.85*yLinSS, "Single SemilogY","X", "Y",\
    #            plotCol='y',label=['Repeat on top'],legendAlpha=0.5)
    # S.saveFig('SS.png', dpi=300)
    # #S.saveFig('SS.eps')

    r = numpy.arange(0, 3.01, 0.01).reshape(-1, 1)
    theta = 2*numpy.pi*r
    r2 = numpy.hstack((r,r**2))
    P = Plotter(3, 2, 2,'Polar Plots', figsize=(12,8))
    P.polar(1,theta, r, "Single Polar",\
           label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0.5,3])
    P.polar(2,theta, r2, "Array Polar",\
           label=['A', 'B'],legendAlpha=0.5,rscale=[2,6],rgrid=[2,6],\
           thetagrid=[45], direction=u'clockwise', zerooffset=0)
    P.polar(3,theta, r, "Single Polar",\
           label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0,3], \
           direction=u'clockwise', zerooffset=numpy.pi/2)
    P.polar(4,theta, r2, "Array Polar",\
           label=['A', 'B'],legendAlpha=0.5,rscale=[0,9],rgrid=[0,6],\
           thetagrid=[45], direction=u'counterclockwise', zerooffset=-numpy.pi/2)
    P.saveFig('P.png')
    #P.saveFig('P.eps')
    #plot again on top of existing graphs
    rr = numpy.arange(0.1, 3.11, 0.01).reshape(-1, 1)
    thetar = 2*numpy.pi*rr
    P.polar(1,thetar, 0.5 * rr, "Single Polar",\
           plotCol='r',label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0.5,3])
    P.polar(2,thetar, 0.75 * rr, "Array Polar",\
           plotCol='g',label=['A', 'B'],legendAlpha=0.5,rscale=[0,6],rgrid=[2,6],\
           thetagrid=[45], direction=u'clockwise', zerooffset=0)
    P.polar(3,thetar, 1.2 * rr, "Single Polar",\
           plotCol='k',label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0,3], \
           direction=u'clockwise', zerooffset=numpy.pi/2)
    P.polar(4,thetar, 1.5 * rr, "Array Polar",\
           plotCol='y',label=['A', 'B'],legendAlpha=0.5,rscale=[0,9],rgrid=[0,6],\
           thetagrid=[45], direction=u'counterclockwise', zerooffset=-numpy.pi/2)
    P.saveFig('PP.png')
    #P.saveFig('PP.eps')

    #polar with negative values
    theta=numpy.linspace(0,2.0*numpy.pi,600)
    r = numpy.sin(3.4*theta)
    PN = Plotter(3, 2, 2,'Negative Polar Plots', figsize=(12,8))
    PN.polar(1,theta, r, "sin(3.3x)",\
           legendAlpha=0.5,rscale=[0,1.5],rgrid=[0.5,1.5],highlightNegative=True)
    PN.polar(2,theta, r, "sin(3.3x)",\
           legendAlpha=0.5,rscale=[0,1.5],rgrid=[0.5,1.5],highlightNegative=True,
           highlightCol='r',highlightWidth=4)
    PN.polar(3,theta, r, "sin(3.3x)", \
           legendAlpha=0.5,rscale=[-1.5,1.5],rgrid=[0.5,1.5],highlightNegative=True)
    PN.polar(4,theta, numpy.abs(r), "abs(sin(3.3x))", \
           legendAlpha=0.5,rscale=[-1.5,1.5],rgrid=[0.5,1.5],highlightNegative=True)
    PN.saveFig('PN.png')
    #PN.saveFig('PN.eps')

    exit()

    #test/demo to show that multiple plots can be done in the same subplot, on top of older plots
    xLinS=numpy.linspace(0, 10, 50).reshape(-1, 1)
    M= Plotter(1, 1, 1,'Multi-plots',figsize=(12,8))
    #it seems that all attempts to plot in same subplot space must use same ptitle.
    yLinS=numpy.random.random(xLinS.shape[0]).reshape(-1, 1)
    M.plot(1, xLinS, yLinS, None,"X", "Y",plotCol=['b'], label=['A1'])
    yLinS=numpy.random.random(xLinS.shape[0]).reshape(-1, 1)
    M.plot(1, xLinS, yLinS, None,"X", "Y",plotCol=['g'], label=['A2'])
    yLinS=numpy.random.random(xLinS.shape[0]).reshape(-1, 1)
    M.plot(1, xLinS, yLinS, None,"X", "Y",plotCol=['r'], label=['A3'])
    yLinS=numpy.random.random(xLinS.shape[0]).reshape(-1, 1)
    M.plot(1, xLinS, yLinS, None,"X", "Y",plotCol=['c'], \
           label=['A4'],legendAlpha=0.5, maxNX=10, maxNY=2)
    M.saveFig('M.png')
    #M.saveFig('M.eps')


    xv,yv = numpy.mgrid[-5:5:21j, -5:5:21j]
    z = numpy.sin(numpy.sqrt(xv**2 + yv**2))
    I = Plotter(4, 2, 2,'Images & Array Linear', figsize=(12, 8))
    I.showImage(1, z, ptitle='winter colormap, font 10pt', cmap=plt.cm. winter, titlefsize=10,  cbarshow=True, cbarorientation = 'horizontal', cbarfontsize = 7)
    barticks = zip([-1, 0, 1], ['low', 'med', 'high'])
    I.showImage(2, z, ptitle='prism colormap, default font ', cmap=plt.cm. prism, cbarshow=True, cbarcustomticks=barticks)
    I.showImage(3, z, ptitle='default gray colormap, font 8pt', titlefsize=8)
    I.plot(4, xv[:, 1],  z, "Array Linear","x", "z")
    I.saveFig('I.png')
#    I.saveFig('I.eps')
    #do new plots on top of existing images/plots
    I.showImage(1, z, ptitle='winter colormap, font 10pt', cmap=plt.cm. winter, titlefsize=10,  cbarshow=True, cbarorientation = 'horizontal', cbarfontsize = 7)
    barticks = zip([-1, 0, 1], ['low', 'med', 'high'])
    I.showImage(2, z, ptitle='prism colormap, default font ', cmap=plt.cm. prism, cbarshow=True, cbarcustomticks=barticks)
    I.showImage(3, z, ptitle='default gray colormap, font 8pt', titlefsize=8)
    I.plot(4, xv[:, 1],  z, "Array Linear","x", "z")
    I.saveFig('II.png')
#    I.saveFig('II.eps')


    #3D plot example
    def parametricCurve(z, param1 = 2, param2 = 1):
        r = z**param1 + param2
        theta = numpy.linspace(-4 * numpy.pi, 4 * numpy.pi, 100)
        return (r * numpy.sin(theta), r * numpy.cos(theta))

    P3D = Plotter(5, 1, 1,'Plot 3D Single', figsize=(12,8))
    z = numpy.linspace(-2, 2, 100)
    x, y = parametricCurve(z)

    P3D.plot3d(1, x.T, y.T, z.T, 'Parametric Curve', 'X', 'Y', 'Z')
    P3D.saveFig('3D.png')

    P3D = Plotter(6, 1, 1,'Plot 3D Single', figsize=(12,8))
    P3D.plot3d(1, x.T, y.T, z.T, 'Parametric Curve', 'X', 'Y', 'Z', plotCol='r', label=['parametric curve'], legendAlpha=0.5)
    P3D.saveFig('3DwithLabel.png')
    P3D.plot3d(1, 1.3*x.T, 0.8*y.T, 0.7*z.T, 'Parametric Curve', 'X', 'Y', 'Z', plotCol='b', label=['parametric curve'], legendAlpha=0.5)
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
        x = numpy.vstack((x,x1))
        y = numpy.vstack((y,y1))

    z = numpy.vstack((z,z,z))

    P3D.plot3d(1, x.T, y.T, z.T, 'Parametric Curve', 'X', 'Y', 'Z', label=label, legendAlpha=0.5)
    P3D.saveFig('M3D.png')

    print('module ryplot done!')
