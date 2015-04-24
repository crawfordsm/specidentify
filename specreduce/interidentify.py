
"""INTERIDENTIFY provides an interactive method for identifying
lines in an arc image.  The tasks displays the full image, a
line extracted from the image, and residuals to the fit of that line.
The task will display the total image so the user can extract the lines
to be fit.  Or the user can automate the process so only certain lines are
fit by the user.  On the next tab, the task displays the arc line
and the fit to the line including what lines have been detected and
are being used for the fit.  Finally the task displays the residual in
the fit and the user can select different options to be displayed.

Author                 Version      Date
-----------------------------------------------
S. M. Crawford (SAAO)    1.0       10 Oct 2009

TODO
----


LIMITATIONS
-----------


"""

# Ensure Python 2.5 compatibility
from __future__ import with_statement

# General imports
import os
import sys
import copy
import numpy as np
from astropy.io import fits as pyfits

# Gui library imports
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT

# specreduce imports
try:
    from gingatools import GingaImageDisplay
    have_ginga = True
except ImportError:
    have_ginga = False

from guitools import ImageDisplay, MplCanvas
from guitools import MplCanvas

from PySpectrograph.Spectra import Spectrum, apext

import WavelengthSolution
import spectools as st
#import AutoIdentify as ai


class InterIdentifyWindow(QtGui.QMainWindow):

    """Main application window."""

    def __init__(self, xarr, specarr, slines, sfluxes, ws, hmin=150, wmin=400, mdiff=20,
                 filename=None, res=2.0, dres=0.1, dc=20, ndstep=20, sigma=5, smooth=0, niter=5, istart=None,
                 nrows=1, rstep=100, method='Zeropoint', ivar=None, cmap='gray', scale='zscale', contrast=1.0,
                 subback=0, textcolor='green', log=None, verbose=True, prefer_ginga=True):
        """Default constructor."""

        # set up the variables
        if istart is None:
            self.y1 = int(0.5 * len(specarr))
        else:
            self.y1 = istart
        self.y2 = self.y1 + nrows
        self.specarr = specarr
        self.xarr = xarr
        self.ivar = ivar
        self.slines = slines
        self.sfluxes = sfluxes
        self.hmin = hmin
        self.wmin = wmin
        self.ws = ws
        self.res = res
        self.dres = dres
        self.mdiff = mdiff
        self.sigma = sigma
        self.niter = int(niter)
        self.nrows = nrows
        self.rstep = rstep
        self.dc = dc
        self.ndstep = ndstep
        self.method = method
        self.cmap = cmap
        self.scale = scale
        self.contrast = contrast
        self.smooth = smooth
        self.subback = subback
        self.filename = filename
        self.ImageSolution = {}
        self.textcolor = textcolor
        self.log = log
        self.verbose = verbose
        self.prefer_ginga = prefer_ginga

        # Setup widget
        QtGui.QMainWindow.__init__(self)

        # Set main widget
        self.main = QtGui.QWidget(self)

        # Set window title
        self.setWindowTitle("InterIdentify")

        # create the Image page
        if have_ginga and self.prefer_ginga:
            displayClass = gingaImageWidget
        else:
            displayClass = imageWidget
        self.imagePage = displayClass(self.specarr, y1=self.y1, y2=self.y2, hmin=self.hmin, wmin=self.wmin, cmap=self.cmap,
                                     rstep=self.rstep, name=self.filename, scale=self.scale, contrast=self.contrast, log=self.log)

        # set up the arc page
        self.farr = apext.makeflat(self.specarr, self.y1, self.y2)
        self.farr = st.flatspectrum(self.xarr, self.farr, order=self.subback)

        # set up variables
        self.arcdisplay = ArcDisplay(xarr, self.farr, slines, sfluxes, self.ws, specarr=self.specarr,
                                     res=self.res, dres=self.dres, dc=self.dc, ndstep=self.ndstep, xp=[], wp=[],
                                     method=self.method, smooth=self.smooth, niter=self.niter, mdiff=self.mdiff,
                                     sigma=self.sigma, textcolor=self.textcolor, log=self.log, verbose=self.verbose)
        self.arcPage = arcWidget(
            self.arcdisplay,
            hmin=hmin,
            wmin=wmin,
            y1=self.y1,
            y2=self.y2,
            name=self.filename)
        # set up the residual page
        self.errPage = errWidget(self.arcdisplay, hmin=hmin, wmin=wmin)

        # create the tabs
        self.tabWidget = QtGui.QTabWidget()
        self.tabWidget.addTab(self.imagePage, 'Image')
        self.tabWidget.addTab(self.arcPage, 'Arc')
        self.tabWidget.addTab(self.errPage, 'Residual')

        # layout the widgets
        mainLayout = QtGui.QVBoxLayout(self.main)
        mainLayout.addWidget(self.tabWidget)
        # self.setLayout(mainLayout)

        # Set focus to main widget
        # self.main.setFocus()

        # Set the main widget as the central widget
        self.setCentralWidget(self.main)

        # Destroy widget on close
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Close when config dialog is closed
        # self.connect(self.conf, QtCore.SIGNAL('destroyed()'),
        #              self, QtCore.SLOT('close()'))
        self.connect(self.tabWidget, QtCore.SIGNAL('currentChanged(int)'),
                     self.currentChanged)
        self.connect(self.imagePage, QtCore.SIGNAL('regionChange(int,int)'),
                     self.regionChange)
        self.connect(self.imagePage, QtCore.SIGNAL('runauto(int, int, int)'),
                     self.runauto)
        self.connect(self.arcPage, QtCore.SIGNAL('savews()'), self.saveWS)
        self.connect(self.arcdisplay, QtCore.SIGNAL('quit()'), self.close)

    def keyPressEvent(self, event):
        # print "Key Pressed:", event.key
        if event.key == 'q':
            self.close()

    def currentChanged(self, event):
        # print event
        pass

    def regionChange(self, y1, y2):
        self.y1 = y1
        self.y2 = y2
        self.farr = apext.makeflat(self.specarr, self.y1, self.y2)
        self.farr = st.flatspectrum(self.xarr, self.farr, order=self.subback)
        # set up variables
        self.ws = self.newWS(0.5 * (self.y1 + self.y2))
        self.arcdisplay = ArcDisplay(
            self.xarr,
            self.farr,
            self.slines,
            self.sfluxes,
            self.ws,
            specarr=self.specarr,
            res=self.res,
            dres=self.dres,
            smooth=self.smooth,
            niter=self.niter,
            sigma=self.sigma,
            xp=[],
            wp=[],
            textcolor=self.textcolor,
            log=self.log,
            verbose=self.verbose)
        self.arcPage = arcWidget(
            self.arcdisplay,
            hmin=self.hmin,
            wmin=self.wmin,
            y1=self.y1,
            y2=self.y2)
        self.connect(self.arcPage, QtCore.SIGNAL('savews()'), self.saveWS)
        # set up the residual page
        self.errPage = errWidget(
            self.arcdisplay,
            hmin=self.hmin,
            wmin=self.wmin)
        # reset the pages
        self.tabWidget.removeTab(2)
        self.tabWidget.removeTab(1)
        self.tabWidget.insertTab(1, self.arcPage, 'Arc')
        self.tabWidget.insertTab(2, self.errPage, 'Residual')

    def saveWS(self):
        self.ws = self.arcdisplay.ws
        value = 0.0

        k = 0.5 * (self.y1 + self.y2)
        xp = np.array(self.arcdisplay.xp)
        wp = np.array(self.arcdisplay.wp)
        if len(xp > 0):
            w = self.arcdisplay.ws.value(xp)
            value = (wp - w).std()

        if self.log is not None:
            msg = 'Saving WS value for row %i with rms=%f for %i lines' % (
                k, value, len(self.arcdisplay.wp))
            self.log.message(msg)

        # create a new wavelength solution
        nws = copy.deepcopy(self.ws)
        if len(xp > 0):
            nws = WavelengthSolution.WavelengthSolution(
                self.ws.x_arr,
                self.ws.w_arr,
                order=self.ws.order,
                function=self.ws.function)
            try:
                nws.fit()
            except Exception as e:
                if self.log is not None:
                    self.log.warning(
                        "Unable to save wavelength solution because %s" %
                        e)
                return
        self.ImageSolution[k] = nws
        # for k in self.ImageSolution: print k,self.ImageSolution[k].coef

    def newWS(self, y):
        """Determine the WS closest to the values given by y1 and y2"""
        keys = np.array(self.ImageSolution.keys())
        try:
            i = abs(keys - y).argmin()
            ws = self.ImageSolution[keys[i]]
            nws = WavelengthSolution.WavelengthSolution(
                ws.x_arr,
                ws.w_arr,
                order=ws.order,
                function=ws.function)
            nws.fit()
            return nws
        except:
            return self.ws

    def runauto(self, istart, nrows, rstep):
        """ Autoidentify the rest of the lines and produce the image solution"""
        self.ImageSolution = self.arcdisplay.autoidentify(
            istart=istart,
            nrows=nrows,
            rstep=rstep,
            oneline=False)


class imageWidget(QtGui.QWidget):

    def __init__(self, imarr, y1=None, y2=None, nrows=1, rstep=100, hmin=150, wmin=400,
                 name=None, cmap='Gray', scale='zscale', contrast=0.1, log=None, parent=None):
        super(imageWidget, self).__init__(parent)

        self.y1 = y1
        self.y2 = y2
        self.x1 = 0
        self.x2 = len(imarr[0])
        self.nrows = nrows
        self.rstep = rstep
        self.log = log

        # Add FITS display widget with mouse interaction and overplotting
        self.imdisplay = ImageDisplay()
        self.imdisplay.setMinimumHeight(hmin)
        self.imdisplay.setMinimumWidth(wmin)

        # Set colormap
        self.imdisplay.setColormap(cmap)

        # Set scale mode for dynamic range
        self.imdisplay.scale = scale
        self.imdisplay.contrast = contrast
        self.imdisplay.aspect = 'auto'
        self.imdisplay.loadImage(imarr)
        self.imdisplay.drawImage()

        self.y1line, = self.imdisplay.axes.plot(
            [self.x1, self.x2], [self.y1, self.y1], ls='-', color='#00FF00')
        self.y2line, = self.imdisplay.axes.plot(
            [self.x1, self.x2], [self.y2, self.y2], ls='-', color='#00FF00')

        # Add navigation toolbars for each widget to enable zooming
        self.toolbar = NavigationToolbar2QT(self.imdisplay, self)

        # set up the information panel
        self.infopanel = QtGui.QWidget()

        # add the name of the file
        self.NameLabel = QtGui.QLabel("Filename:")
        self.NameLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.NameValueLabel = QtGui.QLabel("%s" % name)
        self.NameValueLabel.setFrameStyle(
            QtGui.QFrame.Panel | QtGui.QFrame.Sunken)

        # add the rows that are extracted
        self.y1Label = QtGui.QLabel("Y1:")
        self.y1Label.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.y1ValueEdit = QtGui.QLineEdit("%6i" % self.y1)
        self.y2Label = QtGui.QLabel("Y2:")
        self.y2Label.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.y2ValueEdit = QtGui.QLineEdit("%6i" % self.y2)
        self.updateButton = QtGui.QPushButton("Update")
        self.updateButton.clicked.connect(self.updatesection)

        # add the update for automatically updating it
        self.nrLabel = QtGui.QLabel("nrows:")
        self.nrLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.nrValueEdit = QtGui.QLineEdit("%5i" % self.nrows)
        self.nsLabel = QtGui.QLabel("rstep:")
        self.nsLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.nsValueEdit = QtGui.QLineEdit("%6i" % self.rstep)
        self.nextButton = QtGui.QPushButton("Next")
        self.nextButton.clicked.connect(self.nextsection)

        self.autoButton = QtGui.QPushButton("Auto-Identify")
        self.autoButton.clicked.connect(self.runauto)

        # set up the info panel layout
        infoLayout = QtGui.QGridLayout(self.infopanel)
        infoLayout.addWidget(self.NameLabel, 0, 0, 1, 1)
        infoLayout.addWidget(self.NameValueLabel, 0, 1, 1, 5)
        infoLayout.addWidget(self.y1Label, 1, 0, 1, 1)
        infoLayout.addWidget(self.y1ValueEdit, 1, 1, 1, 1)
        infoLayout.addWidget(self.y2Label, 1, 2, 1, 1)
        infoLayout.addWidget(self.y2ValueEdit, 1, 3, 1, 1)
        infoLayout.addWidget(self.updateButton, 1, 4, 1, 1)
        infoLayout.addWidget(self.nrLabel, 2, 0, 1, 1)
        infoLayout.addWidget(self.nrValueEdit, 2, 1, 1, 1)
        infoLayout.addWidget(self.nsLabel, 2, 2, 1, 1)
        infoLayout.addWidget(self.nsValueEdit, 2, 3, 1, 1)
        infoLayout.addWidget(self.nextButton, 2, 4, 1, 1)
        infoLayout.addWidget(self.autoButton, 3, 0, 1, 1)

        # Set up the layout
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.imdisplay)
        mainLayout.addWidget(self.toolbar)
        mainLayout.addWidget(self.infopanel)
        self.setLayout(mainLayout)

    def updatesection(self):
        self.y1 = int(self.y1ValueEdit.text())
        self.y2 = int(self.y2ValueEdit.text())
        self.nrows = int(self.nrValueEdit.text())
        self.rstep = int(self.nsValueEdit.text())
        if abs(self.y1 - self.y2) != self.nrows:
            if self.log:
                self.log.warning(
                    "Warning: Update y2 to increase the row sampling")
        self.y1line.set_ydata([self.y1, self.y1])
        self.y2line.set_ydata([self.y2, self.y2])
        self.imdisplay.draw()
        self.emit(QtCore.SIGNAL("regionChange(int,int)"), self.y1, self.y2)

    def nextsection(self):
        self.nrows = int(self.nrValueEdit.text())
        self.rstep = int(self.nsValueEdit.text())
        self.y1 = self.y1 + self.rstep
        self.y2 = self.y1 + self.nrows
        self.y1ValueEdit.setText('%6i' % self.y1)
        self.y2ValueEdit.setText('%6i' % self.y2)
        self.updatesection()

    def runauto(self):
        if self.log is not None:
            self.log.message("Running Auto")
        self.emit(
            QtCore.SIGNAL("runauto(int, int, int)"),
            self.y1,
            self.nrows,
            self.rstep)


class gingaImageWidget(QtGui.QWidget):
    # ?? Can this be unified with imageWidget?

    def __init__(self, imarr, y1=None, y2=None, nrows=1, rstep=100, hmin=150, wmin=400,
                 name=None, cmap='Gray', scale='zscale', contrast=0.1, log=None, parent=None):
        super(gingaImageWidget, self).__init__(parent)

        self.y1 = y1
        self.y2 = y2
        self.x1 = 0
        self.x2 = len(imarr[0])
        self.nrows = nrows
        self.rstep = rstep
        self.log = log

        # Add FITS display widget with mouse interaction and overplotting
        self.imdisplay = GingaImageDisplay()
        #self.imdisplay.setMinimumHeight(hmin)
        #self.imdisplay.setMinimumWidth(wmin)
        self.imdisplay.configure(wmin, hmin)

        # Set colormap
        ## self.imdisplay.setColormap(cmap)

        # Set scale mode for dynamic range
        ## self.imdisplay.scale = scale
        ## self.imdisplay.contrast = contrast
        ## self.imdisplay.aspect = 'auto'
        self.imdisplay.loadImage(imarr)
        self.imdisplay.drawImage()
        Line = self.imdisplay.getDrawClass('line')
        self.y1line = Line(self.x1, self.y1, self.x2, self.y1,
                           #lines='-',
                           linewidth=2, color='green')
        self.imdisplay.add(self.y1line)
        self.y2line = Line(self.x1, self.y2, self.x2, self.y2,
                           #ls='-',
                           linewidth=2, color='green')
        self.imdisplay.add(self.y2line)

        ## # Add navigation toolbars for each widget to enable zooming
        ## self.toolbar = NavigationToolbar2QT(self.imdisplay, self)

        # set up the information panel
        self.infopanel = QtGui.QWidget()

        # add the name of the file
        self.NameLabel = QtGui.QLabel("Filename:")
        self.NameLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.NameValueLabel = QtGui.QLabel("%s" % name)
        self.NameValueLabel.setFrameStyle(
            QtGui.QFrame.Panel | QtGui.QFrame.Sunken)

        # add the rows that are extracted
        self.y1Label = QtGui.QLabel("Y1:")
        self.y1Label.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.y1ValueEdit = QtGui.QLineEdit("%6i" % self.y1)
        self.y2Label = QtGui.QLabel("Y2:")
        self.y2Label.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.y2ValueEdit = QtGui.QLineEdit("%6i" % self.y2)
        self.updateButton = QtGui.QPushButton("Update")
        self.updateButton.clicked.connect(self.updatesection)

        # add the update for automatically updating it
        self.nrLabel = QtGui.QLabel("nrows:")
        self.nrLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.nrValueEdit = QtGui.QLineEdit("%5i" % self.nrows)
        self.nsLabel = QtGui.QLabel("rstep:")
        self.nsLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.nsValueEdit = QtGui.QLineEdit("%6i" % self.rstep)
        self.nextButton = QtGui.QPushButton("Next")
        self.nextButton.clicked.connect(self.nextsection)

        self.autoButton = QtGui.QPushButton("Auto-Identify")
        self.autoButton.clicked.connect(self.runauto)

        # set up the info panel layout
        infoLayout = QtGui.QGridLayout(self.infopanel)
        infoLayout.addWidget(self.NameLabel, 0, 0, 1, 1)
        infoLayout.addWidget(self.NameValueLabel, 0, 1, 1, 5)
        infoLayout.addWidget(self.y1Label, 1, 0, 1, 1)
        infoLayout.addWidget(self.y1ValueEdit, 1, 1, 1, 1)
        infoLayout.addWidget(self.y2Label, 1, 2, 1, 1)
        infoLayout.addWidget(self.y2ValueEdit, 1, 3, 1, 1)
        infoLayout.addWidget(self.updateButton, 1, 4, 1, 1)
        infoLayout.addWidget(self.nrLabel, 2, 0, 1, 1)
        infoLayout.addWidget(self.nrValueEdit, 2, 1, 1, 1)
        infoLayout.addWidget(self.nsLabel, 2, 2, 1, 1)
        infoLayout.addWidget(self.nsValueEdit, 2, 3, 1, 1)
        infoLayout.addWidget(self.nextButton, 2, 4, 1, 1)
        infoLayout.addWidget(self.autoButton, 3, 0, 1, 1)

        # Set up the layout
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.imdisplay.get_widget())
        ## mainLayout.addWidget(self.toolbar)
        mainLayout.addWidget(self.infopanel)
        self.setLayout(mainLayout)

    def updatesection(self):
        self.y1 = int(self.y1ValueEdit.text())
        self.y2 = int(self.y2ValueEdit.text())
        self.nrows = int(self.nrValueEdit.text())
        self.rstep = int(self.nsValueEdit.text())
        if abs(self.y1 - self.y2) != self.nrows:
            if self.log:
                self.log.warning(
                    "Warning: Update y2 to increase the row sampling")
        ## self.y1line.set_ydata([self.y1, self.y1])
        ## self.y2line.set_ydata([self.y2, self.y2])
        self.y1line.y1 = self.y1
        self.y1line.y2 = self.y1
        self.y2line.y1 = self.y2
        self.y2line.y2 = self.y2
        self.imdisplay.redraw(whence=3)
        self.emit(QtCore.SIGNAL("regionChange(int,int)"), self.y1, self.y2)

    def nextsection(self):
        self.nrows = int(self.nrValueEdit.text())
        self.rstep = int(self.nsValueEdit.text())
        self.y1 = self.y1 + self.rstep
        self.y2 = self.y1 + self.nrows
        self.y1ValueEdit.setText('%6i' % self.y1)
        self.y2ValueEdit.setText('%6i' % self.y2)
        self.updatesection()

    def runauto(self):
        if self.log is not None:
            self.log.message("Running Auto")
        self.emit(
            QtCore.SIGNAL("runauto(int, int, int)"),
            self.y1,
            self.nrows,
            self.rstep)


class arcWidget(QtGui.QWidget):

    def __init__(self, arcdisplay, hmin=150, wmin=450, name=None,
                 x1=0, w1=0, y1=None, y2=None, parent=None):
        super(arcWidget, self).__init__(parent)

        # Add FITS display widget with mouse interaction and overplotting
        self.arcdisplay = arcdisplay
        self.arcdisplay.arcfigure.setMinimumHeight(hmin)
        self.arcdisplay.arcfigure.setMinimumWidth(wmin)
        self.arcdisplay.plotArc()

        # Add navigation toolbars for each widget to enable zooming
        self.toolbar = NavigationToolbar2QT(self.arcdisplay.arcfigure, self)

        # set up the information panel
        self.infopanel = QtGui.QWidget()

        # add the name of the file
        self.NameLabel = QtGui.QLabel("Filename:")
        self.NameLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.NameValueLabel = QtGui.QLabel("%s" % name)
        self.NameValueLabel.setFrameStyle(
            QtGui.QFrame.Panel | QtGui.QFrame.Sunken)

        # add the rows that are extracted
        self.y1Label = QtGui.QLabel("Y1:")
        self.y1Label.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.y1ValueLabel = QtGui.QLabel("%6i" % y1)
        self.y1ValueLabel.setFrameStyle(
            QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        self.y2Label = QtGui.QLabel("Y2:")
        self.y2Label.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.y2ValueLabel = QtGui.QLabel("%6i" % y2)
        self.y2ValueLabel.setFrameStyle(
            QtGui.QFrame.Panel | QtGui.QFrame.Sunken)

        # add in what the value is for a x and w position
        self.x1Label = QtGui.QLabel("X1:")
        self.x1Label.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.w1Label = QtGui.QLabel("w1:")
        self.w1Label.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.x1ValueLabel = QtGui.QLabel("%6.2f" % x1)
        self.x1ValueLabel.setFrameStyle(
            QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        w1 = self.arcdisplay.ws.value(x1)
        self.w1ValueEdit = QtGui.QLineEdit("%6i" % w1)
        self.addButton = QtGui.QPushButton("Add")
        self.addButton.clicked.connect(self.addpoints)

        # add in radio buttons for pixel or wavelength
        self.pixelradio = QtGui.QRadioButton("Pixel")
        self.wavelengthradio = QtGui.QRadioButton("Wavelength")
        self.pixelradio.setChecked(True)

        # add in information about the order and type of solution
        self.funcLabel = QtGui.QLabel("Function:")
        self.funcLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.funcComboBox = QtGui.QComboBox()
        self.funcComboBox.addItems(self.arcdisplay.ws.func_options)
        self.funcComboBox.setCurrentIndex(
            self.arcdisplay.ws.func_options.index(
                self.arcdisplay.ws.function))
        # self.funcComboBox."%s" % self.arcdisplay.ws.function)
        self.orderLabel = QtGui.QLabel("Order:")
        self.orderLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.orderValueEdit = QtGui.QLineEdit("%2i" % self.arcdisplay.ws.order)
        self.updateButton = QtGui.QPushButton("Update")
        self.updateButton.clicked.connect(self.updatefunction)

        # provide a method for automatically fitting the line
        self.methodComboBox = QtGui.QComboBox()
        self.methodComboBox.addItems(['None']) #ai.autoidentify_options)
        self.methodComboBox.setCurrentIndex(0)
            #ai.autoidentify_options.index(
            #    self.arcdisplay.method))
        self.runButton = QtGui.QPushButton("Run")
        self.runButton.clicked.connect(self.runauto)

        self.saveButton = QtGui.QPushButton("Save")
        self.saveButton.clicked.connect(self.savews)

        # provide the full layout of the information panel
        infoLayout = QtGui.QGridLayout(self.infopanel)
        infoLayout.addWidget(self.NameLabel, 0, 0, 1, 1)
        infoLayout.addWidget(self.NameValueLabel, 0, 1, 1, 5)
        infoLayout.addWidget(self.y1Label, 1, 0, 1, 1)
        infoLayout.addWidget(self.y1ValueLabel, 1, 1, 1, 1)
        infoLayout.addWidget(self.y2Label, 1, 2, 1, 1)
        infoLayout.addWidget(self.y2ValueLabel, 1, 3, 1, 1)
        infoLayout.addWidget(self.x1Label, 2, 0, 1, 1)
        infoLayout.addWidget(self.x1ValueLabel, 2, 1, 1, 1)
        infoLayout.addWidget(self.w1Label, 2, 2, 1, 1)
        infoLayout.addWidget(self.w1ValueEdit, 2, 3)
        infoLayout.addWidget(self.addButton, 2, 4, 1, 1)
        infoLayout.addWidget(self.funcLabel, 3, 0, 1, 1)
        infoLayout.addWidget(self.funcComboBox, 3, 1, 1, 1)
        infoLayout.addWidget(self.orderLabel, 3, 2, 1, 1)
        infoLayout.addWidget(self.orderValueEdit, 3, 3, 1, 1)
        infoLayout.addWidget(self.updateButton, 3, 4, 1, 1)
        infoLayout.addWidget(self.methodComboBox, 4, 0, 1, 1)
        infoLayout.addWidget(self.runButton, 4, 1, 1, 1)
        infoLayout.addWidget(self.saveButton, 4, 4, 1, 1)
        # infoLayout.addWidget(self.pixelradio, 3, 0, 1, 2)
        # infoLayout.addWidget(self.wavelengthradio, 3, 2, 1, 2)

        # Set up the layout
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.arcdisplay.arcfigure)
        mainLayout.addWidget(self.toolbar)
        mainLayout.addWidget(self.infopanel)
        self.setLayout(mainLayout)

        self.connect(
            self.arcdisplay,
            QtCore.SIGNAL('keyPressEvent'),
            self.keyPressEvent)
        self.connect(
            self.arcdisplay,
            QtCore.SIGNAL('updatex(float)'),
            self.updatexlabel)
        self.connect(
            self.funcComboBox,
            QtCore.SIGNAL('activated(QString)'),
            self.updatefunction)
        self.connect(
            self.methodComboBox,
            QtCore.SIGNAL('activated(QString)'),
            self.updatemethod)

    def keyPressEvent(self, event):
        pass
        # print "Arc Widget, keyPress:", event

    def updatexlabel(self, value):
        try:
            self.x1ValueLabel.setText("%6.2f" % value)
            self.w1ValueEdit.setText("%6.2f" % self.arcdisplay.ws.value(value))
        except TypeError:
            pass

    def addpoints(self):
        """Add the x and w points to the list of matched points"""
        x = float(self.x1ValueLabel.text())
        w = float(self.w1ValueEdit.text())
        # x=[1904.5, 1687.22, 3124.349, 632.5705]
        # w=[4671.225, 4624.275, 4916.512, 4383.901]
        self.arcdisplay.addpoints(x, w)

    def updatefunction(self):
        """Update the values for the function"""
        self.arcdisplay.ws.order = int(self.orderValueEdit.text())
        self.arcdisplay.ws.function = self.funcComboBox.currentText()
        self.arcdisplay.ws.set_func()
        self.arcdisplay.findfit()

    def updatemethod(self):
        """Update the values for the method for autoidenitfy"""
        self.arcdisplay.method = self.methodComboBox.currentText()

    def runauto(self):
        """Run autoidenity on one line"""

        self.arcdisplay.dc = 0.5 * self.arcdisplay.rms * self.arcdisplay.ndstep
        self.arcdisplay.autoidentify()

    def savews(self):
        """Save the wcs to the """
        self.emit(QtCore.SIGNAL("savews()"))


class errWidget(QtGui.QWidget):

    def __init__(self, arcdisplay, hmin=150, wmin=450, name=None, parent=None):
        super(errWidget, self).__init__(parent)

        # Add FITS display widget with mouse interaction and overplotting
        self.arcdisplay = arcdisplay
        self.arcdisplay.errfigure.setMinimumHeight(hmin)
        self.arcdisplay.errfigure.setMinimumWidth(wmin)
        self.arcdisplay.plotErr()

        # Add navigation toolbars for each widget to enable zooming
        self.toolbar = NavigationToolbar2QT(self.arcdisplay.errfigure, self)

        # set up the information panel
        self.infopanel = QtGui.QWidget()

        # add the name of the file
        self.NameLabel = QtGui.QLabel("Filename:")
        self.NameLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.NameValueLabel = QtGui.QLabel("%s" % name)
        self.NameValueLabel.setFrameStyle(
            QtGui.QFrame.Panel | QtGui.QFrame.Sunken)

        # add in the rejection parameters
        self.sigmaLabel = QtGui.QLabel("Sigma:")
        self.sigmaLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.sigmaValueEdit = QtGui.QLineEdit(
            "%2.1f" %
            self.arcdisplay.ws.thresh)
        self.niterLabel = QtGui.QLabel("Niter:")
        self.niterLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.niterValueEdit = QtGui.QLineEdit("%i" % self.arcdisplay.ws.niter)
        self.rejectButton = QtGui.QPushButton("Reject")
        self.rejectButton.clicked.connect(self.rejectpoints)

        # add the labels for the results
        self.aveLabel = QtGui.QLabel("Average:")
        self.aveLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.aveValueLabel = QtGui.QLabel("")
        self.aveValueLabel.setFrameStyle(
            QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        self.stdLabel = QtGui.QLabel("Std(A):")
        self.stdLabel.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.stdValueLabel = QtGui.QLabel("")
        self.stdValueLabel.setFrameStyle(
            QtGui.QFrame.Panel | QtGui.QFrame.Sunken)

        # provide the full layout of the information panel
        infoLayout = QtGui.QGridLayout(self.infopanel)
        infoLayout.addWidget(self.NameLabel, 0, 0, 1, 1)
        infoLayout.addWidget(self.NameValueLabel, 0, 1, 1, 5)
        infoLayout.addWidget(self.aveLabel, 1, 0)
        infoLayout.addWidget(self.aveValueLabel, 1, 1)
        infoLayout.addWidget(self.stdLabel, 1, 2)
        infoLayout.addWidget(self.stdValueLabel, 1, 3)
        infoLayout.addWidget(self.sigmaLabel, 2, 0)
        infoLayout.addWidget(self.sigmaValueEdit, 2, 1)
        infoLayout.addWidget(self.niterLabel, 2, 2)
        infoLayout.addWidget(self.niterValueEdit, 2, 3)
        infoLayout.addWidget(self.rejectButton, 2, 4)

        # Set up the layout
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.arcdisplay.errfigure)
        mainLayout.addWidget(self.toolbar)
        mainLayout.addWidget(self.infopanel)
        self.setLayout(mainLayout)

        self.connect(
            self.arcdisplay,
            QtCore.SIGNAL('fitUpdate()'),
            self.fitUpdate)

    def fitUpdate(self):
        if len(self.arcdisplay.xp) <= 1:
            return
        try:
            xp = np.array(self.arcdisplay.xp)
            wp = np.array(self.arcdisplay.wp)
            w = self.arcdisplay.ws.value(xp)
            value = (wp - w).mean()
            self.aveValueLabel.setText("%4.2g" % value)
            value = (wp - w).std()
            self.stdValueLabel.setText("%4.2g" % value)
        except Exception as e:
            if self.arcdisplay.log is not None:
                self.arcdisplay.log.message(e)
            pass

    def rejectpoints(self):
        self.arcdisplay.ws.set_thresh(float(self.sigmaValueEdit.text()))
        self.arcdisplay.ws.set_niter(int(self.niterValueEdit.text()))
        self.arcdisplay.findfit()


class ArcDisplay(QtGui.QWidget):

    """Class for displaying Arc Spectra using matplotlib and embedded in a Qt 4 GUI.
    """

    def __init__(self, xarr, farr, slines, sfluxes, ws, xp=[], wp=[], mdiff=20, specarr=None,
                 res=2.0, dres=0.1, dc=20, ndstep=20, sigma=5, smooth=0, niter=5, method='MatchZero',
                 textcolor='green', log=None, verbose=True):
        """Default constructor."""
        QtGui.QWidget.__init__(self)

        # Initialize base class
        self.arcfigure = MplCanvas()
        self.errfigure = MplCanvas()

        # Add central axes instance
        self.axes = self.arcfigure.figure.add_subplot(111)
        self.erraxes = self.errfigure.figure.add_subplot(111)

        # Connect mouse events
        self.arcfigure.connectMatplotlibMouseMotion()
        self.arcfigure.mpl_connect('button_press_event', self.onButtonPress)
        self.arcfigure.mpl_connect('key_press_event', self.onKeyPress)

        self.errfigure.connectMatplotlibMouseMotion()
        self.errfigure.mpl_connect('button_press_event', self.onButtonPress)
        self.errfigure.mpl_connect('key_press_event', self.onKeyPress)

        # load the data
        self.xarr = xarr
        self.farr = farr
        self.slines = slines
        self.sfluxes = sfluxes
        self.ws = ws
        self.orig_ws = copy.deepcopy(ws)
        self.specarr = specarr
        self.mdiff = mdiff
        self.sigma = sigma
        self.niter = int(niter)
        self.smooth = int(smooth)
        self.res = res
        self.dres = dres
        self.dc = dc
        self.sections = 6
        self.ndstep = ndstep
        self.method = method
        self.textcolor = textcolor
        self.log = log
        self.verbose = True

        # if asked, smooth the data
        if self.smooth > 0:
            self.farr = st.smooth_spectra(
                self.xarr,
                self.farr,
                sigma=self.smooth)

        self.xp = xp
        self.wp = wp

        self.rms = res

        # set up the artificial spectra
        self.spectrum = Spectrum.Spectrum(
            self.slines,
            self.sfluxes,
            dw=self.dres,
            stype='line',
            sigma=self.res)
        self.swarr = self.spectrum.wavelength
        self.sfarr = self.spectrum.flux * \
            self.farr.max() / self.spectrum.flux.max()

        # set up the wavelength solution
        if self.ws.function == 'line':
            self.ws.set_xarr(self.xarr)
            self.ws.farr = self.farr
            self.ws.spectrum = self.spectrum

        # set up the list of deleted points
        self.dxp = []
        self.dwp = []

        # set up other variables
        self.isArt = False
        self.isFeature = False

        # Set display parameters
        self.xmin = self.xarr.min()
        self.xmax = self.xarr.max()
        self.ymin = self.farr.min()
        self.ymax = self.farr.max()

    def help(self):
        helpoutput = """
 ? - Print this file     q - Quit the program
 c - centroid on line    x - print the current position
 a - Display spectrum    l - display features
 b - identify features   f - fit solution
 p - print features      P - print solution
 z - zeropoint fit       Z - find zeropoint and dispersion
 r - redraw spectrum     R - reset values
 e - add closest line    L - show detected peaks
 d - delete feature      u - undelete feature
 X - fit full X-cor
 """
        print helpoutput

    def onKeyPress(self, event):
        """Emit signal on key press"""
        if event.key == '?':
            # return the help file
            self.help()
        elif event.key == 'q':
            # exit the task
            self.emit(QtCore.SIGNAL("quit()"))
        elif event.key == 'c':
            # return the centroid
            if event.xdata:
                self.log.message(str(event.xdata), with_header=False)
                cx = st.mcentroid(
                    self.xarr,
                    self.farr,
                    xc=event.xdata,
                    xdiff=self.mdiff)
                self.emit(QtCore.SIGNAL("updatex(float)"), cx)
        elif event.key == 'x':
            # return the x position
            if event.xdata:
                self.log.message(str(event.xdata), with_header=False)
                self.emit(QtCore.SIGNAL("updatex(float)"), event.xdata)
        elif event.key == 'R':
            # reset the fit
            self.reset()
        elif event.key == 'f':
            # find the fit
            self.findfit()
            self.emit(QtCore.SIGNAL("fitUpdate()"))
        elif event.key == 'b':
            # auto-idenitfy features
            self.isFeature = True
            self.findfeatures()
        elif event.key == 'z':
            # Assume the solution is correct and find the zeropoint
            # that best matches it from cross correllation
            self.findzp()
        elif event.key == 'Z':
            # Assume the solution is correct and find the zeropoint
            # that best matches it from cross correllation
            self.findzpd()
        elif event.key == 'X':
            # Assume the solution is almost correct
            # Fit the full solution using the cross correlation coefficient
            self.findxcorfit()

        elif event.key == 'e':
            # find closest feature from existing fit and line list
            # and match it
            self.addclosestline(event.xdata)
        elif event.key == 'i':
            # reset identified features
            pass
        elif event.key == 't':
            # reset identified features
            self.isFeature = True
            self.testfeatures()
        elif event.key == 'l':
            # plot the features from existing list
            if self.isFeature:
                self.isFeature = False
                self.redraw_canvas()
            else:
                self.isFeature = True
                self.plotFeatures()
                self.redraw_canvas()
        elif event.key == 'L':
            # plot the sources that are detected
            self.plotDetections()
        elif event.key == 'p':
            # print information about features
            for i in range(len(self.xp)):
                print self.xp[i], self.wp[i]
        elif event.key == 'P':
            # print information about features
            print self.ws.coef
        elif event.key == 'r':
            # redraw graph
            self.redraw_canvas()
        elif event.key == 'a':
            # draw artificial spectrum
            self.isArt = not self.isArt
            self.redraw_canvas()
        elif event.key == 'd':
            # Delete feature
            save = False
            y = None
            if event.canvas == self.errfigure:
                y = event.ydata
                save = True
            self.deletepoints(event.xdata, y=y, save=save)
            self.redraw_canvas(keepzoom=True)
        elif event.key == 'u':
            # undelete
            self.undeletepoints(event.xdata, y=event.ydata)
            self.redraw_canvas(keepzoom=True)
        elif event.key:
            self.emit(QtCore.SIGNAL("keyPressEvent(string)"), event.key)

    def onButtonPress(self, event):
        """Emit signal on selecting valid image position."""

        if event.xdata and event.ydata:
            self.emit(QtCore.SIGNAL("positionSelected(float, float)"),
                      float(event.xdata), float(event.ydata))

    def plotArc(self):
        """Draw image to canvas."""

        # plot the spectra
        self.spcurve, = self.axes.plot(
            self.xarr, self.farr, linewidth=0.5, linestyle='-', marker='None', color='b')

    def plotArt(self):
        """Plot the artificial spectrum"""
        self.isArt = True
        warr = self.ws.value(self.xarr)
        asfarr = st.interpolate(
            warr,
            self.swarr,
            self.sfarr,
            left=0.0,
            right=0.0)
        asfarr = asfarr * self.farr.max() / asfarr.max()
        self.fpcurve, = self.axes.plot(self.xarr, asfarr, linewidth=0.5, linestyle='-',
                                       marker='None', color='r')

    def plotDetections(self):
        """Plot the lines that are detected"""
        xp, xf = st.findpoints(
            self.xarr, self.farr, self.sigma, self.niter, sections=self.sections)
        print xp
        self.axes.plot(xp, xf, ls='', marker='|', ms=20, color='#000000')

    def plotFeatures(self):
        """Plot features identified in the line list"""
        fl = np.array(self.xp) * 0.0 + 0.25 * self.farr.max()
        self.splines = self.axes.plot(
            self.xp,
            fl,
            ls='',
            marker='|',
            ms=20,
            color=self.textcolor)
        # set up the text position
        tsize = 0.83
        self.ymin, self.ymax = self.axes.get_ylim()
        ppp = (self.ymax - self.ymin) / (self.arcfigure.figure.get_figheight()
                                         * self.arcfigure.figure.get_dpi())
        f = self.ymax - 10 * tsize * ppp
        for x, w in zip(self.xp, self.wp):
            w = '%6.2f' % float(w)
            self.axes.text(
                x,
                f,
                w,
                size='small',
                rotation='vertical',
                color=self.textcolor)

    def plotErr(self):
        """Draw image to canvas."""
        if self.xp and self.wp:
            # plot the spectra
            w = self.ws.value(np.array(self.xp))
            self.errcurve, = self.erraxes.plot(
                self.xp, self.wp - w, linewidth=0.5, linestyle='', marker='o', color='b')
        if self.dxp and self.dwp:
            # plot the spectra
            dw = self.ws.value(np.array(self.dxp))
            self.delerrcurve, = self.erraxes.plot(
                self.dxp, self.dwp - dw, linewidth=0.5, linestyle='', marker='x', color='b')

    def set_wdiff(self):
        """Derive a value for wdiff"""
        try:
            self.wdiff = self.mdiff * self.ws.coef[1]
        except:
            self.wdiff = self.mdiff

    def testfeatures(self):
        """Run the test matching algorithm"""
        self.set_wdiff()
        res = max(self.res * 0.25, 2)
        xp, wp = st.crosslinematch(self.xarr, self.farr, self.slines, self.sfluxes, self.ws,
                                   res=res, mdiff=self.mdiff, wdiff=20, sigma=self.sigma,
                                   niter=self.niter, sections=self.sections)
        for x, w in zip(xp, wp):
            if w not in self.wp and w > -1:
                self.xp.append(x)
                self.wp.append(w)
        self.plotFeatures()
        self.redraw_canvas()

    def findfeatures(self):
        """Given a set of features, find other features that might
           correspond to those features
        """
        self.set_wdiff()

        # xp, wp=st.findfeatures(self.xarr, self.farr, self.slines, self.sfluxes,
        # self.ws, mdiff=self.mdiff, wdiff=self.wdiff, sigma=self.sigma,
        # niter=self.niter, sections=3)
        xp, wp = st.crosslinematch(self.xarr, self.farr, self.slines, self.sfluxes, self.ws,
                                   res=self.res, mdiff=self.mdiff, wdiff=20,
                                   sections=self.sections, sigma=self.sigma, niter=self.niter)
        for x, w in zip(xp, wp):
            if w not in self.wp and w > -1:
                self.xp.append(x)
                self.wp.append(w)
        # for i in range(len(self.xp)): print self.xp[i], self.wp[i]
        # print
        self.plotFeatures()
        self.redraw_canvas()

    def addclosestline(self, x):
        """Find the closes line to the centroided position and
           add it
        """
        cx = st.mcentroid(self.xarr, self.farr, xc=x, xdiff=self.mdiff)
        w = self.ws.value(cx)
        d = abs(self.slines - w)
        w = self.slines[d.argmin()]

        self.xp.append(x)
        self.wp.append(w)
        self.plotFeatures()
        self.redraw_canvas()

    def findzp(self):
        """Find the zeropoint for the source and plot of the new value
        """
        dc = 0.5 * self.rms * self.ndstep
        self.ws = st.findzeropoint(self.xarr, self.farr, self.swarr, self.sfarr,
                                   self.ws, dc=dc, ndstep=self.ndstep, inttype='interp')
        self.plotArt()
        self.redraw_canvas()

    def findzpd(self):
        """Find the zeropoint and dispersion for the source and plot of the new value
        """
        dc = 0.5 * self.rms * self.ndstep
        # fixed at 0.1 of the dispersion
        dd = 0.1 * self.ws.coef[1]

        # set upt he docef values
        dcoef = self.ws.coef * 0.0
        dcoef[0] = dc
        dcoef[1] = dd
        self.ws = st.findxcor(self.xarr, self.farr, self.swarr, self.sfarr, self.ws,
                              dcoef=dcoef, ndstep=self.ndstep, best=False, inttype='interp')
        self.plotArt()
        self.redraw_canvas()

    def findxcorfit(self):
        """Maximize the normalized correlation coefficient using the full wavelength solution.
        """
        self.ws = st.fitxcor(
            self.xarr,
            self.farr,
            self.swarr,
            self.sfarr,
            self.ws,
            interptype='interp')
        self.plotArt()
        self.redraw_canvas()

    def findfit(self):
        if len(self.xp) < self.ws.order:
            raise SpecError(
                "Insufficient sources number of points for fit")
            return
        try:
            self.ws = st.findfit(
                np.array(
                    self.xp), np.array(
                    self.wp), ws=self.ws, thresh=self.ws.thresh)
        except SpecError as e:
            self.log.warning(e)
            return

        del_list = []
        for i in range(len(self.ws.func.mask)):
            if self.ws.func.mask[i] == 0:
                self.deletepoints(self.ws.func.x[i], w=self.ws.func.y[i],
                                  save=True)
        self.rms = self.ws.sigma(self.ws.x_arr, self.ws.w_arr)
        self.redraw_canvas()

    def autoidentify(self, rstep=1, istart=None, nrows=1, oneline=True):
        """Run the autoidentify method for the current line"""
        # update the line list such that it is only the line list of selected
        # lines
        if self.wp:
            slines = np.array(self.wp)
            sfluxes = self.farr[self.xp]
            # sfluxes=np.zeros(len(slines))
            # for i in range(len(slines)):
            #    try:
            #       sfluxes[i]=self.sfluxes[self.slines==slines[i]][0]
            #    except:
            #       if sfluxes.mean()==0:
            #            sfluxes[i]=1
            #       else:
            #            sfluxes[i]=sfluxes.mean()

        else:
            slines = self.slines
            sfluxes = self.sfluxes

        #iws = ai.AutoIdentify(self.xarr, self.specarr, slines, sfluxes, self.ws, farr=self.farr,
        #                      method=self.method, rstep=rstep, istart=istart, nrows=nrows,
        #                      res=self.res, dres=self.dres, mdiff=self.mdiff, sigma=self.sigma,
        #                     smooth=self.smooth, niter=self.niter, dc=self.dc, ndstep=self.ndstep,
        #                      oneline=oneline, log=self.log, verbose=self.verbose)
        iws = None
        if oneline:
            self.ws = iws
        else:
            return iws

    def addpoints(self, x, w):
        """Add points to the line list
        """
        if isinstance(x, list) and isinstance(w, list):
            self.xp.extend(x)
            self.wp.extend(w)
        else:
            self.xp.append(x)
            self.wp.append(w)

    def deletepoints(self, x, y=None, w=None, save=False):
        """ Delete points from the line list
        """
        dist = (np.array(self.xp) - x) ** 2

        # assumes you are using the error plot
        if y is not None:
            w = self.ws.value(np.array(self.xp))
            norm = self.xarr.max() / abs(self.wp - w).max()
            dist += norm * (self.wp - w - y) ** 2
            # print y, norm, dist.min()
            # print y, dist.min()
        elif w is not None:
            norm = self.xarr.max() / abs(self.wp - w).max()
            dist += norm * (self.wp - w)**2
        in_minw = dist.argmin()

        if save:
            self.dxp.append(self.xp[in_minw])
            self.dwp.append(self.wp[in_minw])
        self.xp.__delitem__(in_minw)
        self.wp.__delitem__(in_minw)

    def undeletepoints(self, x, y=None):
        """ Delete points from the line list
        """
        if len(self.dxp) < 1:
            return
        if len(self.dxp) == 1:
            self.xp.append(self.dxp[0])
            self.wp.append(self.dwp[0])
            self.dxp.__delitem__(0)
            self.dwp.__delitem__(0)
            return

        dist = (self.dxp - x) ** 2
        if y is not None:
            w = self.ws.value(np.array(self.dxp))
            # dist += (self.dwp-w-y)**2
        in_minw = dist.argmin()

        self.xp.append(self.dxp[in_minw])
        self.wp.append(self.dwp[in_minw])
        self.dxp.__delitem__(in_minw)
        self.dwp.__delitem__(in_minw)

        return

    def reset(self):
        self.ws = copy.deepcopy(self.orig_ws)
        self.redraw_canvas()

    def redraw_canvas(self, keepzoom=False):
        if keepzoom:
            # Store current zoom level
            xmin, xmax = self.axes.get_xlim()
            ymin, ymax = self.axes.get_ylim()

        # Clear plot
        self.axes.clear()

        # Draw image
        self.plotArc()

        # if necessary, redraw the features
        if self.isFeature:
            self.plotFeatures()

        # if necessary, draw the artificial spectrum
        if self.isArt:
            self.plotArt()

        # Restore zoom level
        if keepzoom:
            self.axes.set_xlim((self.xmin, self.xmax))
            self.axes.set_ylim((self.ymin, self.ymax))

        # Force redraw
        self.arcfigure.draw()

        self.err_redraw_canvas()

    def err_redraw_canvas(self, keepzoom=False):
        if keepzoom:
            # Store current zoom level
            xmin, xmax = self.erraxes.get_xlim()
            ymin, ymax = self.erraxes.get_ylim()
        else:
            self.xmin, self.xmax = self.axes.get_xlim()

        # Clear plot
        self.erraxes.clear()

        # Draw image
        self.plotErr()

        # Restore zoom level
        if keepzoom:
            self.erraxes.set_xlim((xmin, xmax))
            self.erraxes.set_ylim((ymin, ymax))
        else:
            self.erraxes.set_xlim((self.xmin, self.xmax))

        self.errfigure.draw()

        self.emit(QtCore.SIGNAL("fitUpdate()"))


def InterIdentify(xarr, specarr, slines, sfluxes, ws, mdiff=20, rstep=1, filename=None,
                  function='poly', order=3, sigma=3, smooth=0, niter=5, res=2, dres=0.1, dc=20, ndstep=20,
                  istart=None, method='Zeropoint', scale='zscale', cmap='gray', contrast=1.0,
                  subback=0, textcolor='green', log=None, verbose=True):

    # Create GUI
    App = QtGui.QApplication(sys.argv)
    aw = InterIdentifyWindow(xarr, specarr, slines, sfluxes, ws, rstep=rstep, mdiff=mdiff, sigma=sigma, niter=niter,
                             res=res, dres=dres, dc=dc, ndstep=ndstep, istart=istart, method=method, smooth=smooth,subback=subback,
                             cmap=cmap, scale=scale, contrast=contrast, filename=filename, textcolor=textcolor, log=log)
    aw.show()
    # Start application event loop
    exit = App.exec_()
    imsol = aw.ImageSolution.copy()

    # Check if GUI was executed succesfully
    if exit != 0:
        raise SpecError(
            'InterIdentify GUI has unexpected exit status ' +
            str(exit))
    del aw
    return imsol
