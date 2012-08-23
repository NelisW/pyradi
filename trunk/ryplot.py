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

# Contributor(s): MS Willers.
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

__version__= "$Revision$"
__author__='pyradi team'
__all__=['Plotter']

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
    
        __all__ = ['__init__', 'BuildPlotCol', 'SaveFig', 'GetPlot', 'Plot', 
                   'LogLog', 'SemilogX', 'SemilogY', 'MyPlot', 'Polar']
        
        version=mpl.__version__.split('.')
        vnum=float(version[0]+'.'+version[1])
        
        if vnum<1.1:
            print('Install Matplotlib 1.1 or later')
            print('current version is {0}'.format(vnum))
            sys.exit(-1)
    
        self.figurenumber = fignumber
        self.nrow=subpltnrow
        self.ncol=subpltncol
        self.figuretitle = figuretitle

        self.fig = plt.figure(self.figurenumber, frameon=False)
        self.fig.set_size_inches(figsize[0], figsize[1]) 
        self.fig.clear()

        # width reserved for space between subplots
        self.fig.subplots_adjust(wspace=0.25)
        #height reserved for space between subplots
        self.fig.subplots_adjust(hspace=0.4)
        #height reserved for top of the subplots of the figure
        self.fig.subplots_adjust(top=0.88)
        
        self.plotCol=['b', 'g', 'r', 'c', 'm', 'y', 'k', \
            'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 'k--']
        
        self.bbox_extra_artists=[]

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
        """Returns the current plot
        
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
    def plot(self, plotnum, ptitle, xlabel, ylabel, x, y, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], maxNX=10, maxNY=10):
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
                
            Returns:
                | Nothing
        
            Raises:
                | No exception is raised.
       """
        ## see self.MyPlot for parameter details.
        self.myPlot(plt.plot, plotnum, ptitle, xlabel, ylabel, \
                    x, y,plotCol, label,legendAlpha, pltaxis, \
                    maxNX, maxNY)

    ############################################################
    ##
    def logLog(self, plotnum, ptitle, xlabel, ylabel, x, y, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], maxNX=10, maxNY=10):
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
                
            Returns:
                | Nothing
        
            Raises:
                | No exception is raised.
       """
        ## see self.MyPlot for parameter details.
        self.myPlot(plt.loglog, plotnum, ptitle, xlabel,ylabel,\
                    x, y,plotCol, label,legendAlpha, pltaxis, \
                    maxNX, maxNY)

    ############################################################
    ##
    def semilogX(self, plotnum, ptitle, xlabel, ylabel, x, y, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], maxNX=10, maxNY=10):
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
                
            Returns:
                | Nothing
        
            Raises:
                | No exception is raised.
       """
        ## see self.MyPlot for parameter details.
        self.myPlot(plt.semilogx, plotnum,ptitle,xlabel,ylabel,\
                    x, y,plotCol, label,legendAlpha, pltaxis, \
                    maxNX, maxNY)

    ############################################################
    ##
    def semilogY(self, plotnum, ptitle, xlabel, ylabel, x, y, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], maxNX=10, maxNY=10):
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
                | ptitle (string): plot title
                | xlabel (string): x axis label
                | ylabel (string): y axis label
                | x (np.array[N,] or [N,1]): abscissa
                | y (np.array[N,] or [N,M]): ordinates - could be M columns
                | plotCol ([strings]): plot line style, list with M entries, use default if []
                | label  ([strings]): legend label for ordinate, list withM entries
                | legendAlpha (float): transparancy for legend
                | pltaxis ([xmin, xmax, ymin,ymax]): scale for x,y axes. default if all zeros.
                | maxNX (int): draw maxNX+1 tick labels on x axis
                | maxNY (int): draw maxNY+1 tick labels on y axis
                
            Returns:
                | Nothing
        
            Raises:
                | No exception is raised.
       """
        ## see self.MyPlot for parameter details.
        self.myPlot(plt.semilogy, plotnum,ptitle,xlabel,ylabel,\
                    x, y,plotCol, label,legendAlpha, pltaxis, \
                    maxNX, maxNY)

    ############################################################
    ##
    def myPlot(self, plotcommand,plotnum,ptitle,xlabel,ylabel, 
                    x, y, plotCol=[],label=[],legendAlpha=0.0,\
                    pltaxis=[0, 0, 0, 0], maxNX=0, maxNY=0):
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
        
        #use add_subplot to keep the previous subplot
        if(ptitle is not None):
            sbp=self.fig.add_subplot(self.nrow, self.ncol, plotnum,title=ptitle) 
        else:
            sbp=self.fig.add_subplot(self.nrow, self.ncol, plotnum) 
        plt.grid(True)
        if xlabel is not None:
            plt.xlabel(xlabel)  
        if ylabel is not None:
            plt.ylabel(ylabel)  
        
        if maxNX >0:
            sbp.xaxis.set_major_locator(mpl.ticker.MaxNLocator(maxNX))
        if maxNY >0:
            sbp.yaxis.set_major_locator(mpl.ticker.MaxNLocator(maxNY))
        if not label:
            for i in range(yy.shape[1]):
                plotcommand(xx, yy[:, i], plotCol[i],label=None)
        else:
            for i in range(yy.shape[1]):
                plotcommand(xx,yy[:,i],plotCol[i],label=label[i])
            leg = plt.legend(loc='best', fancybox=True)
            leg.get_frame().set_alpha(legendAlpha)
            self.bbox_extra_artists.append(leg)

        # use scientific format on axes
        #yfm = sbp.yaxis.get_major_formatter()
        #yfm.set_powerlimits([ -3, 3])

        formy = None

        formy = plt.ScalarFormatter()
        formy.set_powerlimits((-3, 4))
        formy.set_scientific(True)
        sbp.yaxis.set_major_formatter(formy)
            
        #scale the axes
        if sum(pltaxis)!=0:
            plt.axis(pltaxis)

    ############################################################
    ## 
    def polar(self, plotnum, ptitle, theta, r, \
                    plotCol=[], label=[],labelLocation=[-0.1, 0.1], \
                    legendAlpha=0.0, \
                    rscale=[0, 0], rgrid=[0, 0], thetagrid=[30], \
                    direction='counterclockwise', zerooffset=0):
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
                | ptitle (string): plot title
                | theta (np.array[N,] or [N,1]): angular abscissa
                | r (np.array[N,] or [N,M]): radial ordinates - could be M columns
                | plotCol ([strings]): plot line style, list with M entries, use default if []
                | label  ([strings]): legend label, list with M entries
                | labelLocation ([x,y]): where the legend should located
                | legendAlpha (float): transparancy for legend
                | rscale ([rmin, rmax]): plotting limits. default if all 0
                | rgrid ([rinc, rmax]): radial grid default if all 0. if rinc=0 then rmax is number of ntervals.
                | thetagrids (float): theta grid interval [degrees]
                | direction (string)= 'counterclockwise' or 'clockwise'
                | zerooffset (float) = rotation offset where zero should be [rad]
                
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
        
        sbplt=plt.subplot(self.nrow,self.ncol,plotnum,polar=True) 
        plt.grid(True)
        
        rmax=0
        if not label:
            for i in range(rr.shape[1]):
                # negative val :forcing positive and phase shifting
                ttt = tt + numpy.pi*(rr[:, i] < 0).reshape(-1, 1)
                rrr = numpy.abs(rr[:, i])
                plt.polar(ttt, rrr, plotCol[i],)
                rmax=numpy.maximum(rrr.max(), rmax)
        else:
            for i in range(rr.shape[1]):
                # negative val :forcing positive and phase shifting
                ttt = tt + numpy.pi*(rr[:, i] < 0).reshape(-1, 1)
                rrr = numpy.abs(rr[:, i])
                plt.polar(ttt,rrr,plotCol[i],label=label[i])
                rmax=numpy.maximum(rrr.max(), rmax)

            fontP = mpl.font_manager.FontProperties()
            fontP.set_size('small')
            leg = plt.legend(loc='upper left',
                    bbox_to_anchor=(labelLocation[0], labelLocation[1]),
                    prop = fontP, fancybox=True)
            leg.get_frame().set_alpha(legendAlpha)
            self.bbox_extra_artists.append(leg)
            
        sbplt.set_theta_direction(direction)
        sbplt.set_theta_offset(zerooffset)
        
        #set up the grids
        plt.thetagrids(range(0, 360, thetagrid[0]))
        if sum(rgrid)!=0:
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
        if sum(rscale)!=0:
            sbplt.set_rmin(rscale[1])
            sbplt.set_rmax(rscale[0])
            
        sbplt.set_title(ptitle, verticalalignment ='bottom',\
                  horizontalalignment='center'      )


    ############################################################
    ##
    def showImage(self, plotnum, img,  ptitle="", cmap=plt.cm.gray, fsize=12):
        """Creates a subplot and show the image using the colormap provided.

            Args:
                | plotnum (int): subplot number
                | img (np.ndarray): numpy 2d array                
                | ptitle (string): plot title (optional)
                | cmap: matplotlib colormap, default gray (optional)
                | fsize: title font size, default 12pt (optional)
                
            Returns:
                | Nothing
        
            Raises:
                | No exception is raised.
        """

        sbp=plt.subplot(self.nrow, self.ncol, plotnum)   
        
        plt.imshow(img, cmap)
        plt.axis('off')
        plt.title(ptitle, fontsize=fsize)
        

    def plot3d(self, plotnum, ptitle, xlabel, ylabel, zlabel, x, y, z, \
               plotCol=[], label=None, legendAlpha=0.0):
        """3D plot on linear scales for x y z input sets.
        
        Given an existing figure, this function plots in a specified subplot position. 
        The function arguments are described below in some detail. 
        
        Note that multiple 3D data sets can be plotted simultaneously by adding additional columns to
        the input coordinates of vertices, each column representing a different function in the plot. 
        This is convenient if large arrays of data must be plotted. If more than one column is present,
        the label argument can contain the legend labels for each of the columns/lines. 
        
            Args:
                | plotnum (int): subplot number
                | ptitle (string): plot title
                | xlabel (string): x axis label
                | ylabel (string): y axis label
                | x (np.array[N,] or [N,M]): x coordinates of vertices
                | y (np.array[N,] or [N,M]): y coordinates of vertices
                | z (np.array[N,] or [N,M]): z coordinates of vertices
                | plotCol ([strings]): plot line style, list with M entries, use default if []
                | label  ([strings]): legend label for ordinate, list with M entries
                | legendAlpha (float): transparancy for legend
        """
        
        # if required convert 1D arrays into 2D arrays                
        if x.ndim < 2:
            x = x.reshape(-1,1)
        if y.ndim < 2:
            y = y.reshape(-1,1)
        if z.ndim < 2:
            z = z.reshape(-1,1)
        
        plotCol = self.buildPlotCol(plotCol, x.shape[-1])        
        
        ax = self.fig.add_subplot(self.nrow,self.ncol, plotnum, projection='3d')
        
        for i in range(x.shape[-1]):
            ax.plot(x[:,i], y[:,i], z[:,i], plotCol[i])
        

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        
        if label:        
            leg = plt.legend(label, loc='best', fancybox=True)
            leg.get_frame().set_alpha(legendAlpha)
            self.bbox_extra_artists.append(leg)

        plt.title(ptitle)

     
################################################################
################################################################
##
## plot graphs and confirm the correctness of the functions

if __name__ == '__main__':

    ##create some data
    xLinS=numpy.linspace(0, 10, 50).reshape(-1, 1)
    yLinS=1.0e7 * numpy.random.random(xLinS.shape[0]).reshape(-1, 1)

    yLinA=yLinS
    yLinA = numpy.hstack((yLinA, \
            1.0e7 * numpy.random.random(xLinS.shape[0]).reshape(-1, 1)))
    yLinA = numpy.hstack((yLinA, \
            1.0e7 * numpy.random.random(xLinS.shape[0]).reshape(-1, 1)))

    A = Plotter(1, 2, 2,'Array Plots',figsize=(12,8))
    A.plot(1, "Array Linear","X", "Y", xLinS, yLinA,\
           label=['A1', 'A2', 'A3'],legendAlpha=0.5, maxNX=10, maxNY=2)
    A.logLog(2, "Array LogLog","X", "Y", xLinS, yLinA,\
             label=['A1', 'A2', 'A3'],legendAlpha=0.5)
    A.semilogX(3, "Array SemilogX","X", "Y", xLinS, yLinA,\
               label=['A1', 'A2', 'A3'],legendAlpha=0.5)
    A.semilogY(4, "Array SemilogY","X", "Y", xLinS, yLinA,\
               label=['A1', 'A2', 'A3'],legendAlpha=0.5)
    A.saveFig('A.png')
    #A.saveFig('A.eps')
    
    S = Plotter(2, 2, 2,'Single Plots',figsize=(12,8))
    S.plot(1, "Single Linear","X", "Y", xLinS, yLinS,\
           label=['Single'],legendAlpha=0.5)
    S.logLog(2, "Single LogLog","X", "Y", xLinS, yLinS,\
             label=['Single'],legendAlpha=0.5)
    S.semilogX(3, "Single SemilogX","X", "Y", xLinS, yLinS,\
               label=['Single'],legendAlpha=0.5)
    S.semilogY(4, "Single SemilogY","X", "Y", xLinS, yLinS,\
               label=['Single'],legendAlpha=0.5)
    S.saveFig('S.png', dpi=300)
    #S.saveFig('S.eps')

    r = numpy.arange(0, 3.01, 0.01).reshape(-1, 1)
    theta = 2*numpy.pi*r
    r2 = numpy.hstack((r,r**2))
    P = Plotter(3, 2, 2,'Polar Plots', figsize=(12,8))
    P.polar(1, "Single Polar",theta, r,\
           label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0.5,3])
    P.polar(2, "Array Polar",theta, r2,\
           label=['A', 'B'],legendAlpha=0.5,rscale=[2,6],rgrid=[2,6],\
           thetagrid=[45], direction=u'clockwise', zerooffset=0)
    P.polar(3, "Single Polar",theta, r,\
           label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0,3], \
           direction=u'clockwise', zerooffset=numpy.pi/2)
    P.polar(4, "Array Polar",theta, r2,\
           label=['A', 'B'],legendAlpha=0.5,rscale=[0,9],rgrid=[0,6],\
           thetagrid=[45], direction=u'counterclockwise', zerooffset=-numpy.pi/2)
    P.saveFig('P.png')
    #P.saveFig('P.eps')
    

    
    #test/demo to show that multiple plots can be done in the same subplot, on top of older plots
    xLinS=numpy.linspace(0, 10, 50).reshape(-1, 1)
    M= Plotter(1, 1, 1,'Multi-plots',figsize=(12,8))
    #it seems that all attempts to plot in same subplot space must use same ptitle.
    yLinS=numpy.random.random(xLinS.shape[0]).reshape(-1, 1)
    M.plot(1, None,"X", "Y", xLinS, yLinS,plotCol=['b'], label=['A1'])
    yLinS=numpy.random.random(xLinS.shape[0]).reshape(-1, 1)
    M.plot(1, None,"X", "Y", xLinS, yLinS,plotCol=['g'], label=['A2'])
    yLinS=numpy.random.random(xLinS.shape[0]).reshape(-1, 1)
    M.plot(1, None,"X", "Y", xLinS, yLinS,plotCol=['r'], label=['A3'])
    yLinS=numpy.random.random(xLinS.shape[0]).reshape(-1, 1)
    M.plot(1, None,"X", "Y", xLinS, yLinS,plotCol=['c'], \
           label=['A4'],legendAlpha=0.5, maxNX=10, maxNY=2)
    M.saveFig('M.png')
    #M.saveFig('M.eps')


    xv,yv = numpy.mgrid[-5:5:21j, -5:5:21j]
    z = numpy.sin(numpy.sqrt(xv**2 + yv**2))
    P = Plotter(4, 2, 2,'Images & Array Linear', figsize=(12, 8))
    P.showImage(1, z, ptitle='winter colormap, font 10pt', cmap=plt.cm. winter, fsize=10)
    P.showImage(2, z, ptitle='prism colormap, default font ', cmap=plt.cm. prism)
    P.showImage(3, z, ptitle='default gray colormap, font 8pt', fsize=8)
    P.plot(4, "Array Linear","x", "z", xv[:, 1],  z)
    P.saveFig('I.png')
#    P.saveFig('I.eps')

    
    #3D plot example
    def parametricCurve(z, param1 = 2, param2 = 1):
        r = z**param1 + param2
        theta = numpy.linspace(-4 * numpy.pi, 4 * numpy.pi, 100)
        return (r * numpy.sin(theta), r * numpy.cos(theta))
        
    P3D = Plotter(5, 1, 1,'Plot 3D Single', figsize=(12,8))
    z = numpy.linspace(-2, 2, 100)
    x, y = parametricCurve(z)    
    
    P3D.plot3d(1, 'Parametric Curve', 'X', 'Y', 'Z', x.T, y.T, z.T)
    P3D.saveFig('3D.png')
    
    P3D = Plotter(6, 1, 1,'Plot 3D Single', figsize=(12,8))
    P3D.plot3d(1, 'Parametric Curve', 'X', 'Y', 'Z', x.T, y.T, z.T, label=['parametric curve'], legendAlpha=0.5)
    P3D.saveFig('3DwithLabel.png')

    P3D = Plotter(7, 2, 2,'Plot 3D Aspects', figsize=(12,8))
    P3D.plot(1, 'Top View', 'X', 'Y', x.T, y.T)
    P3D.plot(2, 'Side View Along Y Axis', 'X', 'Z', x.T, z.T)
    P3D.plot(3, 'Side View Along X Axis', 'Y', 'Z', y.T, z.T)
    P3D.plot3d(4, '3D View', 'X', 'Y', 'Z', x.T, y.T, z.T)
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

    P3D.plot3d(1, 'Parametric Curve', 'X', 'Y', 'Z', x.T, y.T, z.T, label=label, legendAlpha=0.5)
    P3D.saveFig('M3D.png')

    print('module ryplot done!')
