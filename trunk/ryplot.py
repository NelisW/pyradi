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

# Contributor(s): ______________________________________.
################################################################
"""
This module provides functions for plotting cartesian and polar plots. This class provides a 
basic plotting capability, with a minimum number of lines. These are all wrapper functions, 
based on existing functions in other Python classes.

See the __main__ function for examples of use.
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__='CJ Willers'
__all__=['plotter']

import numpy
import math
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


class plotter:
    """ Encapsulates a plotting environment, optimized for 
    radiometry plots.
    
    This class provides a wrapper around Matplotlib to provide a plotting 
    environment specialised towards radiometry results.  These functions
    were developed to provide well labelled plots by entering only one or two lines.
    
    Provision is made for plots containing subplots (i.e. multiple plots on the same figure),
    linear scale and log scale plots, and cartesian and polar plots.
    """
    
    ############################################################
    ##
    def __init__(self,fignumber=0,subpltnrow=1,subpltncol=1,\
                 figuretitle='', 
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
            
        self.figtitle=plt.gcf().text(.5,.95,figuretitle,\
                    horizontalalignment='center',\
                    fontproperties=FontProperties(size=16))
                    
        self.bbox_extra_artists=[]
        self.bbox_extra_artists.append(self.figtitle)
        


    ############################################################
    ##
    def BuildPlotCol(self, plotCol, n):
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
    def SaveFig(self, filename='mpl.png',dpi=100,bbox_inches='tight',\
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
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, 
                        pad_inches=pad_inches,\
                        bbox_extra_artists= self.bbox_extra_artists)

    ############################################################
    ##
    def GetPlot(self):
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
    def Plot(self, plotnum, ptitle, xlabel, ylabel, x, y, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], MaxNX=0, MaxNY=0):
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
                | MaxNX (int): draw MaxNX+1 tick labels on x axis
                | MaxNY (int): draw MaxNy+1 tick labels on y axis
                
            Returns:
                | Nothing
        
            Raises:
                | No exception is raised.
       """
        ## see self.MyPlot for parameter details.
        self.MyPlot(plt.plot, plotnum, ptitle, xlabel, ylabel, \
                    x, y,plotCol, label,legendAlpha, pltaxis, \
                    MaxNX, MaxNY)

    ############################################################
    ##
    def LogLog(self, plotnum, ptitle, xlabel, ylabel, x, y, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], MaxNX=0, MaxNY=0):
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
                | MaxNX (int): draw MaxNX+1 tick labels on x axis
                | MaxNY (int): draw MaxNy+1 tick labels on y axis
                
            Returns:
                | Nothing
        
            Raises:
                | No exception is raised.
       """
        ## see self.MyPlot for parameter details.
        self.MyPlot(plt.loglog, plotnum, ptitle, xlabel,ylabel,\
                    x, y,plotCol, label,legendAlpha, pltaxis, \
                    MaxNX, MaxNY)

    ############################################################
    ##
    def SemilogX(self, plotnum, ptitle, xlabel, ylabel, x, y, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], MaxNX=0, MaxNY=0):
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
                | MaxNX (int): draw MaxNX+1 tick labels on x axis
                | MaxNY (int): draw MaxNy+1 tick labels on y axis
                
            Returns:
                | Nothing
        
            Raises:
                | No exception is raised.
       """
        ## see self.MyPlot for parameter details.
        self.MyPlot(plt.semilogx, plotnum,ptitle,xlabel,ylabel,\
                    x, y,plotCol, label,legendAlpha, pltaxis, \
                    MaxNX, MaxNY)

    ############################################################
    ##
    def SemilogY(self, plotnum, ptitle, xlabel, ylabel, x, y, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], MaxNX=0, MaxNY=0):
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
                | MaxNX (int): draw MaxNX+1 tick labels on x axis
                | MaxNY (int): draw MaxNy+1 tick labels on y axis
                
            Returns:
                | Nothing
        
            Raises:
                | No exception is raised.
       """
        ## see self.MyPlot for parameter details.
        self.MyPlot(plt.semilogy, plotnum,ptitle,xlabel,ylabel,\
                    x, y,plotCol, label,legendAlpha, pltaxis, \
                    MaxNX, MaxNY)

    ############################################################
    ##
    def MyPlot(self, plotcommand,plotnum,ptitle,xlabel,ylabel, 
                    x, y, plotCol=[],label=[],legendAlpha=0.0,\
                    pltaxis=[0, 0, 0, 0], MaxNX=0, MaxNY=0):
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
                | MaxNX (int): draw MaxNX+1 tick labels on x axis
                | MaxNY (int): draw MaxNy+1 tick labels on y axis
                
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

        plotCol = self.BuildPlotCol(plotCol, yy.shape[1])

        sbp=plt.subplot(self.nrow, self.ncol, plotnum,title=ptitle) 
        plt.grid(True)
        plt.xlabel(xlabel)  
        plt.ylabel(ylabel)  
        if MaxNX >0:
            sbp.xaxis.set_major_locator(mpl.ticker.MaxNLocator(MaxNX))
        if MaxNY >0:
            sbp.yaxis.set_major_locator(mpl.ticker.MaxNLocator(MaxNY))
        #print(label)
        if not label:
            for i in range(yy.shape[1]):
                plotcommand(xx, yy[:, i], plotCol[i],)
        else:
            for i in range(yy.shape[1]):
                plotcommand(xx,yy[:,i],plotCol[i],label=label[i])
            leg = plt.legend(loc='best', fancybox=True)
            leg.get_frame().set_alpha(legendAlpha)
            self.bbox_extra_artists.append(leg)

        #scale the axes
        if sum(pltaxis)!=0:
            plt.axis(pltaxis)

    ############################################################
    ## 
    def Polar(self, plotnum, ptitle, theta, r, \
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
            
        plotCol = self.BuildPlotCol(plotCol, rr.shape[1])
        
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



#from matplotlib.font_manager import FontProperties
#
#   fontP = FontProperties()
#   fontP.set_size('small')
#   legend([plot1], "title", prop = fontP)

      
     
################################################################
################################################################
##
## plot graphs and confirm the correctness of the functions

if __name__ == '__main__':

    ##create some data
    xLinS=numpy.linspace(0, 10, 50).reshape(-1, 1)
    yLinS=numpy.random.random(xLinS.shape[0]).reshape(-1, 1)

    yLinA=yLinS
    yLinA = numpy.hstack((yLinA, \
            numpy.random.random(xLinS.shape[0]).reshape(-1, 1)))
    yLinA = numpy.hstack((yLinA, \
            numpy.random.random(xLinS.shape[0]).reshape(-1, 1)))

    A = plotter(1, 2, 2,'Array Plots')
    A.Plot(1, "Array Linear","X", "Y", xLinS, yLinA,\
           label=['A1', 'A2', 'A3'],legendAlpha=0.5, MaxNX=10, MaxNY=2)
    A.LogLog(2, "Array LogLog","X", "Y", xLinS, yLinA,\
             label=['A1', 'A2', 'A3'],legendAlpha=0.5)
    A.SemilogX(3, "Array SemilogX","X", "Y", xLinS, yLinA,\
               label=['A1', 'A2', 'A3'],legendAlpha=0.5)
    A.SemilogY(4, "Array SemilogY","X", "Y", xLinS, yLinA,\
               label=['A1', 'A2', 'A3'],legendAlpha=0.5)
    A.SaveFig('A.png')
    
    S = plotter(2, 2, 2,'Single Plots',figsize=(12,9))
    S.Plot(1, "Single Linear","X", "Y", xLinS, yLinS,\
           label=['Single'],legendAlpha=0.5)
    S.LogLog(2, "Single LogLog","X", "Y", xLinS, yLinS,\
             label=['Single'],legendAlpha=0.5)
    S.SemilogX(3, "Single SemilogX","X", "Y", xLinS, yLinS,\
               label=['Single'],legendAlpha=0.5)
    S.SemilogY(4, "Single SemilogY","X", "Y", xLinS, yLinS,\
               label=['Single'],legendAlpha=0.5)
    S.SaveFig('S.png', dpi=300)

    r = numpy.arange(0, 3.01, 0.01).reshape(-1, 1)
    theta = 2*numpy.pi*r
    r2 = numpy.hstack((r,r**2))
    P = plotter(3, 2, 2,'Polar Plots', figsize=(12,8))
    P.Polar(1, "Single Polar",theta, r,\
           label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0.5,3])
    P.Polar(2, "Array Polar",theta, r2,\
           label=['A', 'B'],legendAlpha=0.5,rscale=[2,6],rgrid=[2,6],\
           thetagrid=[45])
    P.Polar(3, "Single Polar",theta, r,\
           label=['Single'],legendAlpha=0.5,rscale=[0,3],rgrid=[0,3])
    P.Polar(4, "Array Polar",theta, r2,\
           label=['A', 'B'],legendAlpha=0.5,rscale=[0,9],rgrid=[0,6],\
           thetagrid=[45])
    P.SaveFig('P.png')
    P.SaveFig('P.eps')
