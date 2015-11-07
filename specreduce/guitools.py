"""
Module containing generic graphical user interface widgets.
"""

# Ensure python 2.5 compatibility
from __future__ import with_statement
import matplotlib.cm

# General imports
import numpy as np

# Gui library imports
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import CirclePolygon, Rectangle


def zscale(image, contrast=1.0):
    """Implementation of the IRAF zscale algorithm to find vmin and vmax parameters for the dynamic range of a display. It finds the image values near the median image value without the time consuming process of computing a full image histogram."""

    from scipy import optimize
    #import matplotlib.pyplot as plt

    # Get ordered list of points
    I=np.sort(image.flatten())

    # Get number of points
    npoints=len(I)

    # Find the midpoint (median)
    midpoint=(npoints-1)/2

    # Fit a linear function
    # I(i) = intercept + slope * (i - midpoint)
    fitfunc = lambda p, x: p[0]*x+p[1]
    errfunc = lambda p, x, y: fitfunc(p, x) - y

    # Initial guess for the parameters
    p0 = [(I[-1]-I[0])/npoints,I[midpoint]]

    # Fit
    i=np.arange(len(I))
    p1, success = optimize.leastsq(errfunc, p0[:], args=(i, I))

#    plt.plot(i,I,'r+')
#    plt.plot(i,fitfunc(p1,i))
#    plt.show()

    if success in [1,2,3,4]:
        slope=p1[0]
        z1=I[midpoint]+(slope/contrast)*(1-midpoint)
        z2=I[midpoint]+(slope/contrast)*(npoints-midpoint)
    else:
        z1=np.min(image)
        z2=np.max(image)

    return z1, z2


class MplCanvas(FigureCanvas):
    """Base class for embedding a matplotlib canvas in a PyQt4 GUI.
    """

    def __init__(self):
        """Default constructor."""

        # Initialize base class
        FigureCanvas.__init__(self,Figure())

        # Set resize policy
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)

        # Set geometry
        FigureCanvas.updateGeometry(self)

    def set_focus(self, event):
        self.setFocus()

    def connectMatplotlibKeyEvents(self):
        """Bind events to event handlers."""

        self.key_press_id = self.figure.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.key_release_id = self.figure.canvas.mpl_connect('key_release_event', self.onKeyRelease)

    def connectMatplotlibMouseEvents(self):
        """Bind events to event handlers."""

        self.button_press_id = self.figure.canvas.mpl_connect('button_press_event', self.onButtonPress)
        self.button_release_id = self.figure.canvas.mpl_connect('button_release_event', self.onButtonRelease)

    def connectMatplotlibMouseMotion(self):
        self.mouse_motion_id = self.figure.canvas.mpl_connect('motion_notify_event', self.set_focus)

    def disconnectMatplotlibKeyEvents(self):
        """Unbind events."""

        self.figure.canvas.mpl_disconnect(key_press_id)
        self.figure.canvas.mpl_disconnect(key_release_id)

    def disconnectMatplotlibMouseEvents(self):
        """Unbind events."""

        self.figure.canvas.mpl_disconnect(button_press_id)
        self.figure.canvas.mpl_disconnect(button_release_id)

    def onButtonPress(self, event):
        """Overload this function to implement mousebutton press events."""
        pass

    def onButtonRelease(self, event):
        """Overload this function to implement mousebutton release events."""
        pass

    def onKeyPress(self, event):
        """Overload this function to implement mousebutton press events."""
        print "I'm here:", event, dir(event)
        pass

    def onKeyRelease(self, event):
        """Overload this function to implement mousebutton release events."""
        pass

class ImageDisplay(MplCanvas):
    """Class for displaying FITS images using matplotlib imshow() embedded in a Qt 4 GUI.
    With extra methods for overplotting patches."""

    def __init__(self):
        """Default constructor."""

        # Initialize base class
        MplCanvas.__init__(self)

        # Add central axes instance
        self.axes = self.figure.add_subplot(111)

        # Connect mouse events
        self.connectMatplotlibMouseEvents()

        # Keep track of all patches
        self.patches={}
        self.zorder=10

        # Set display parameters
        self.vmin=None
        self.vmax=None
        self.interpolation=None
        self.cmap=None
        self.scale='minmax'
        self.contrast=1.0
        self.origin='lower'
        self.aspect='auto'

    def onButtonPress(self, event):
        """Emit signal on selecting valid image position."""

        if event.xdata and event.ydata:
            self.emit(QtCore.SIGNAL("positionSelected(float, float)"),
	            float(event.xdata), float(event.ydata))

    def setColormap(self, cmap_name):
        """Set colormap based on name."""
        try:
            self.cmap=matplotlib.cm.get_cmap(cmap_name)
        except:
            raise SpecError('Cannot get colormap instance for specified cmap')

    def setScale(self):
        # Set scale parameters
        if self.scale=='minmax':
            self.vmin=np.min(self.image)
            self.vmax=np.max(self.image)
        elif self.scale=='zscale':
            self.vmin,self.vmax=zscale(self.image,self.contrast)
        else:
            self.vmin=None
            self.vmax=None

    def loadImage(self, image):
        """Load image array."""

        # Set image
        self.image=image

        # Set vmin and vmax parameters
        self.setScale()

    def drawImage(self):
        """Draw image to canvas."""

        # Display image
        self.axes.imshow(self.image, cmap=self.cmap, aspect=self.aspect, vmin=self.vmin, vmax=self.vmax,interpolation=self.interpolation, origin=self.origin)

    def addPatch(self, label, patch):
        """Add a matplotlib *patch* instance with a given *label*."""

        # There shall be one and only one patch for each label
        if label in self.patches:
            del self.patches[label]

        # Add patch to list
        self.patches[label]=patch

        self.zorder+=1

    def removePatch(self, label):
        """Remove patch instance referenced by *label* from figure."""

        # Remove patch if it exists
        if label in self.patches:
            del self.patches[label]

    def addCircle(self, label, x, y, r, color='y', lw=1):
        """Add circle patch at postion (*x*,*y*) with radius *r*
        using a line with color *color* and thickness *lw*."""

        circ=CirclePolygon((x,y),radius=r,ec=color,zorder=self.zorder,lw=lw,fill=False)

        # Add patch to figure
        self.addPatch(label,circ)

    def addSquare(self, label, x, y, r, color='y', lw=1):
        """Add square patch at postion (*x*,*y*) with radius *r*
        using a line with color *color* and thickness *lw*."""

        # Calculate coordinates
        xl=x-r
        yl=y-r
        w=2*r
        h=2*r

        # Create Rectangle patch instance
        rect=Rectangle((xl,yl),width=w,height=h,ec=color,zorder=self.zorder,lw=lw,fill=False)

        # Add patch to figure
        self.addPatch(label,rect)

    def addRectangle(self, label, x1, y1, x2, y2, color='y', lw=1):
        """Add rectangle patch from (*x1*,*y1*) to (*x2*,*y2*)
        using a line with color *color* and thickness *lw*."""

        # Calculate coordinates
        w=x2-x1
        h=y2-y1

        # Create Rectangle patch instance
        rect=Rectangle((x1,y1),width=w,height=h,ec=color,zorder=self.zorder,lw=lw,fill=False)

        # Add patch to figure
        self.addPatch(label,rect)

    def reset(self):
        # Delete all patches
        self.patches={}

        # Redraw canvas
        self.redraw_canvas()

    def redraw_canvas(self,keepzoom=False):
        if keepzoom:
            # Store current zoom level
            xmin, xmax = self.axes.get_xlim()
            ymin, ymax = self.axes.get_ylim()

        # Clear plot
        self.axes.clear()

        # Draw image
        self.drawImage()

        # Draw patches
        for key in self.patches.keys():
            self.axes.add_patch(self.patches[key])

        if keepzoom:
            # Restore zoom level
            self.axes.set_xlim((xmin,xmax))
            self.axes.set_ylim((ymin,ymax))

        # Force redraw
        self.draw()

