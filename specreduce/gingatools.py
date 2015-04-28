"""
Module containing generic graphical user interface widgets.
"""

# General imports
from astropy.io import fits as pyfits
import numpy as np

# Gui library imports
from ginga import AstroImage
from ginga.qtw.ImageViewCanvasQt import ImageViewCanvas as GingaCanvas
from ginga.misc import log


class GingaImageDisplay(GingaCanvas):
    """Class for displaying FITS images using Ginga embedded in a Qt 4 GUI.
    With extra methods for overplotting patches."""

    def __init__(self):
        """Default constructor."""

        #logger = log.get_logger(log_stderr=True, level=20)
        logger = log.get_logger(null=True)
        
        # Initialize base class
        GingaCanvas.__init__(self, logger=logger)

        # Keep track of all patches
        self.patches={}
        #self.zorder=10

        self.get_bindings().enable_all(True)
        self.enable_draw(True)
        self.enable_edit(True)

        self.set_drawtype('line', color='green', cap='ball',
                          linewidth=2)
        #self.add_callback('draw-event', self.draw_cb)
        #self.add_callback('edit-event', self.edit_cb)
        self.add_callback('drag-drop', self.drop_file)
        #self.add_callback('cursor-down', self.btndown_cb)
        
    ## def onButtonPress(self, event):
    ##     """Emit signal on selecting valid image position."""

    ##     if event.xdata and event.ydata:
    ##         self.emit(QtCore.SIGNAL("positionSelected(float, float)"), 
    ##                   float(event.xdata), float(event.ydata))

    def setColorMap(self, cmap_name):
        """Set colormap based on name."""
        try:
            self.set_color_map(cmap_name)
        except Exception as e:
            raise ValueError('Cannot get colormap instance for specified cmap: %s' % (
                str(e)))

    def setScale(self, algname, **params):
        self.set_autocut_params(algname, **params)
    
    def loadImage(self, data_np):
        """Load image array."""

        # Set image
        self.image = data_np

        aimg = AstroImage.AstroImage(self.logger)
        aimg.set_data(self.image)

        self.set_image(aimg)

    def drawImage(self):
        """Draw image to canvas."""
        pass
    
    def addPatch(self, label, patch):
        """Add a matplotlib *patch* instance with a given *label*."""

        # There shall be one and only one patch for each label
        if label in self.patches:
            del self.patches[label]

        # Add patch to list
        self.patches[label]=patch

        #self.zorder+=1
        if isinstance(patch, GingaCanvasObject):
            self.add(patch)
        else:
            self.logger.warn("Skipping unsupported patch %s" % (str(patch)))
      
    def removePatch(self, label):
        """Remove patch instance referenced by *label* from figure."""

        # Remove patch if it exists
        if label in self.patches:
            del self.patches[label]

    def addCircle(self, label, x, y, r, color='y', lw=1):
        """Add circle patch at postion (*x*,*y*) with radius *r*
        using a line with color *color* and thickness *lw*."""

        Circle = self.getDrawClass('circle')
        circ = Circle(x,y,radius=r,color=color,linewidth=lw,fill=False)

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
        Rectangle = self.getDrawClass('rectangle')
        rect=Rectangle(xl,yl,xl+w,xl+h,color=color,linewidth=lw,fill=False)

        # Add patch to figure
        self.addPatch(label,rect)

    def addRectangle(self, label, x1, y1, x2, y2, color='y', lw=1):
        """Add rectangle patch from (*x1*,*y1*) to (*x2*,*y2*)
        using a line with color *color* and thickness *lw*."""

        # Calculate coordinates
        w=x2-x1
        h=y2-y1

        # Create Rectangle patch instance
        Rectangle = self.getDrawClass('rectangle')
        rect=Rectangle(xl,yl,xl+w,xl+h,color=color,linewidth=lw,fill=False)

        # Add patch to figure
        self.addPatch(label,rect)

    def reset(self):
        # Delete all patches
        self.patches={}

        # Redraw canvas
        #self.redraw_canvas()
        self.deleteAllObjects()

    def drop_file(self, fitsimage, paths):
        fileName = paths[0]
        self.load_file(fileName)

    def load_file(self, filepath):
        image = AstroImage.AstroImage(logger=self.logger)
        image.load_file(filepath)

        self.set_image(image)

    def draw_cb(self, canvas, tag):
        obj = canvas.getObjectByTag(tag)
        # <-- obj drawn by user 
        return True

    def edit_cb(self, canvas, obj):
        # <-- lines edited on image
        return True

    def btndown_cb(self, canvas, event, data_x, data_y):
        self.logger.info("mouse clicked at data (x=%f, y=%f)" % (
            data_x, data_y))
        return True

    def redraw_canvas(self,keepzoom=False):
        # in ginga this is a nop because it keeps track of redrawing screen
        pass
    
    # add some dummpy matplotlib compatibility methods here?
    
