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


## This class provides a basic plotting capability, with a 
## minimum number of lines.

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
import math
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


class plotter:
    """ Encapsulates a plotting environment, optimized for 
    radiometry plots.
    """
    
    ############################################################
    ##
    def __init__(self,fignumber=0,subpltnrow=1,subpltncol=1,\
                 figuretitle='', 
                 figsize=(9,9)):
        """Class constructor parameters:
          fignumber: the plt figure number, must be supplied
          subpltnrow=1: subplot number of rows
          subpltncol=1: subplot number of columns
          figuretitle='': the overall heading for the figure
          figsize is the size in inches
          dpi is the resolution of the bitmap
          is required to indicate the filee type required.
        """
    
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
        """returns a sequence of default colour styles of 
           appropriate length
        """
        if not plotCol:
            return [self.plotCol[i % len(self.plotCol)] \
                                         for i in range(n)]
        else:
            return plotCol

    ############################################################
    ##
    def SaveFig(self, filename='mpl.png',dpi=100,bbox_inches='tight',\
                pad_inches=0.1):
        """Save the plot to a disk file, using name supplied in 
           constructor
           filename='': output filename to write plot, file ext 
           dpi: the resolution of the graph in dots per inche
           bbox_inches
       """
        if len(filename)>0:
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, 
                        pad_inches=pad_inches,\
                        bbox_extra_artists= self.bbox_extra_artists)

    ############################################################
    ##
    def GetPlot(self):
        """Returns the current plot
        """
        return self.fig

    ############################################################
    ##
    def Plot(self, plotnum, ptitle, xlabel, ylabel, x, y, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], MaxNX=0, MaxNY=0):
        ## see self.MyPlot for parameter details.
        self.MyPlot(plt.plot, plotnum, ptitle, xlabel, ylabel, \
                    x, y,plotCol, label,legendAlpha, pltaxis, \
                    MaxNX, MaxNY)

    ############################################################
    ##
    def LogLog(self, plotnum, ptitle, xlabel, ylabel, x, y, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], MaxNX=0, MaxNY=0):
        ## see self.MyPlot for parameter details.
        self.MyPlot(plt.loglog, plotnum, ptitle, xlabel,ylabel,\
                    x, y,plotCol, label,legendAlpha, pltaxis, \
                    MaxNX, MaxNY)

    ############################################################
    ##
    def SemilogX(self, plotnum, ptitle, xlabel, ylabel, x, y, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], MaxNX=0, MaxNY=0):
        ## see self.MyPlot for parameter details.
        self.MyPlot(plt.semilogx, plotnum,ptitle,xlabel,ylabel,\
                    x, y,plotCol, label,legendAlpha, pltaxis, \
                    MaxNX, MaxNY)

    ############################################################
    ##
    def SemilogY(self, plotnum, ptitle, xlabel, ylabel, x, y, \
                    plotCol=[], label=[],legendAlpha=0.0, \
                    pltaxis=[0, 0, 0, 0], MaxNX=0, MaxNY=0):
        ## see self.MyPlot for parameter details.
        self.MyPlot(plt.semilogy, plotnum,ptitle,xlabel,ylabel,\
                    x, y,plotCol, label,legendAlpha, pltaxis, \
                    MaxNX, MaxNY)

    ############################################################
    ##
    def MyPlot(self, plotcommand,plotnum,ptitle,xlabel,ylabel, 
                    x, y, plotCol=[],label=[],legendAlpha=0.0,\
                    pltaxis=[0, 0, 0, 0], MaxNX=0, MaxNY=0):
        """Create a subplot and plot the data as required.
            Function parameters:
                plotnum: subplot number
                ptitle: plot title
                xlabel: x axis label
                ylabel: y axis label
                x: abscissa
                y: ordinates - could be N columns
                plotCol []:plot line style, list with N entries, 
                        use default if []
                label []: legend label for ordinate, list with 
                        N entries
                legendAlpha=0.0: transparancy for legend
                pltaxis:scale for x,y axes.default if all zeros.
                MaxNX: draw MaxNX+1 tick labels on x axis
                MaxNY: draw MaxNy+1 tick labels on y axis
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
        """Create a subplot and plot the data as required.
            Function parameters:
                plotnum: subplot number
                ptitle: plot title
                theta: angular abscissa
                r: radial ordinates - could be N columns
                plotCol []:plot line style, list with N entries, 
                        use default if []
                label []: legend label, list with N entries
                labelLocation[]: where the legend should located
                legendAlpha=0.0: transparancy for legend
                rscale[]:plotting limits: [rmin, rmax] default if all 0
                rgrid[]: radial grid [rinc, rmax] default if all 0.
                         if rinc==0 then rmax is number of ntervals.
                thetagrids[]: theta grid interval
                direction = 'counterclockwise' or 'clockwise'
                zerooffset = rotation offset where zero should be [rad]
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
