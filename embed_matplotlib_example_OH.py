#!/imaging/local/software/anaconda/latest/x86_64/bin/python
"""
This demo demonstrates how to embed a matplotlib (mpl) plot 
into a PyQt4 GUI application, including:
* Using the navigation toolbar
* Adding data to the plot
* Dynamically modifying the plot's properties
* Processing mpl events
* Saving the plot to a file from a menu
The main goal is to serve as a basis for developing rich PyQt GUI
applications featuring mpl plots (using the mpl OO API).
Eli Bendersky (eliben@gmail.com)
License: this code is in the public domain
Last modified: 19.01.2009
"""

# https://github.com/eliben/code-for-blog/blob/master/2009/qt_mpl_bars.py

import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy as np

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class AppForm(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('Demo: PyQt with matplotlib')

        # self.create_menu()
        self.create_main_frame()
        # self.create_status_bar()

        # self.textbox.setText('1 2 3 4')
        self.textbox.setText('0 0 0.75 0.75')

        # coordinates from mouse clicks
        self.data_mouse = []
        
        self.on_draw()

        # # new
        # self.setMouseTracking(True)
        
        # this will print cursor position continuously:

        # while 1:
        #     print QCursor().pos()
        #     self.curs_pos()

    # def save_plot(self):
    #     file_choices = "PNG (*.png)|*.png"
        
    #     path = unicode(QFileDialog.getSaveFileName(self, 
    #                     'Save file', '', 
    #                     file_choices))
    #     if path:
    #         self.canvas.print_figure(path, dpi=self.dpi)
    #         self.statusBar().showMessage('Saved to %s' % path, 2000)
    
    # def on_about(self):
    #     msg = """ A demo of using PyQt with matplotlib:
        
    #      * Use the matplotlib navigation bar
    #      * Add values to the text box and press Enter (or click "Draw")
    #      * Show or hide the grid
    #      * Drag the slider to modify the width of the bars
    #      * Save the plot to a file using the File menu
    #      * Click on a bar to receive an informative message
    #     """
    #     QMessageBox.about(self, "About the demo", msg.strip())
    
    # def on_pick(self, event):
    #     # The event received here is of the type
    #     # matplotlib.backend_bases.PickEvent
    #     #
    #     # It carries lots of information, of which we're using
    #     # only a small amount here.
    #     # 
    #     box_points = event.artist.get_bbox().get_points()
    #     msg = "You've clicked on a bar with coords:\n %s" % box_points
        
    #     QMessageBox.information(self, "Click!", msg)

    # new from http://stackoverflow.com/questions/19825650/python-pyqt4-how-to-detect-the-mouse-click-position-anywhere-in-the-window
    def mousePressEvent(self, QMouseEvent):
        print QMouseEvent.pos()

    # def mouseReleaseEvent(self, QMouseEvent):
    #     cursor =QCursor()
    #     print cursor.pos()

    ## new method
    def curs_pos(self, event):
        """ Print cursor position in canvas
        """
        # cursor position
        x = np.array([event.xdata, event.ydata])
        self.data_mouse = [0, 0, x[0], x[1]]
        # coordinates from text box, shifted to origin
        xt = np.array([self.data_text[2] - self.data_text[0], self.data_text[3] - self.data_text[1]])
        print "Cursor pos", x[0], x[1]
        print "Text pos", xt[0], xt[1]
        if (x[0] !=None and x[1] != None):
            corr = np.dot(x,xt) / np.sqrt(np.dot(x,x)*np.dot(xt,xt))
            angle = np.arccos( corr )
            print "Correlation", corr
            print "Angle: ", angle, " rad; ", 180*angle/np.pi, " deg"
            
            self.on_draw()
            # print QCursor().cursor.pos()

    
    def on_draw(self):
        """ Redraws the figure
        """
        # str = unicode(self.textbox.text())
        # # self.data = map(int, str.split())
        # self.data = map(float, str.split())
        
        # x = range(len(self.data))

        # clear the axes and redraw the plot anew
        #
        self.axes.clear()
        self.axes.grid(True)
        # self.axes.grid(self.grid_cb.isChecked())

        # not sure what this is
        # print self.mpl_toolbar.x()
        ## this seems to be cursor position in canvas?
        # print self.cursor().pos()
        
        # self.axes.bar(
        #     left=x, 
        #     height=self.data, 
        #     # width=self.slider.value() / 100.0, 
        #     align='center', 
        #     alpha=0.44,
        #     picker=5)

        str = unicode(self.textbox.text())
        # self.data = map(int, str.split())
        self.data_text = map(float, str.split())

        # self.axes.plot(x)
        X = [ self.data_text, self.data_mouse ]
        colours = ['k', 'r'] # arrow colours
        for [x,c] in zip(X,colours):
            if (x!=[]):
                self.axes.arrow(x[0], x[1], x[2], x[3], head_width=0.05, head_length=0.1, fc=c, ec=c)
        # self.axes.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
        
        self.canvas.draw()

    
    def create_main_frame(self):
        self.main_frame = QWidget()
        
        # Create the mpl Figure and FigCanvas objects. 
        # 5x4 inches, 100 dots-per-inch
        #
        self.dpi = 200
        self.fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setParent(self.main_frame)

        ## new from https://github.com/matplotlib/matplotlib/issues/707/
        self.canvas.setFocusPolicy( Qt.ClickFocus )
        self.canvas.setFocus()

                
        # Since we have only one plot, we can use add_axes 
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.axes = self.fig.add_subplot(111)
        self.axes.set_xlim([-1, 1])        
        self.axes.set_ylim([-1, 1])
        self.axes.grid(True, 'major')
        
        # # Bind the 'pick' event for clicking on one of the bars
        # #
        # self.canvas.mpl_connect('pick_event', self.on_pick)
        
        # Create the navigation toolbar, tied to the canvas
        #
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        ## new
        self.canvas.mpl_connect('button_press_event', self.curs_pos)
        
        # Other GUI controls
        # 
        self.textbox = QLineEdit()
        self.textbox.setMinimumWidth(200)
        self.connect(self.textbox, SIGNAL('editingFinished ()'), self.on_draw)
        
        self.draw_button = QPushButton("&Draw")
        self.connect(self.draw_button, SIGNAL('clicked()'), self.on_draw)        
        
        # self.grid_cb = QCheckBox("Show &Grid")
        # self.grid_cb.setChecked(False)
        # self.connect(self.grid_cb, SIGNAL('stateChanged(int)'), self.on_draw)
        
        # slider_label = QLabel('Bar width (%):')
        # self.slider = QSlider(Qt.Horizontal)
        # self.slider.setRange(1, 100)
        # self.slider.setValue(20)
        # self.slider.setTracking(True)
        # self.slider.setTickPosition(QSlider.TicksBothSides)
        # self.connect(self.slider, SIGNAL('valueChanged(int)'), self.on_draw)
        
        #
        # Layout with box sizers
        # 
        hbox = QHBoxLayout()
        
        # for w in [  self.textbox, self.draw_button, self.grid_cb,
        #             slider_label, self.slider]:
        for w in [  self.textbox, self.draw_button]:
            hbox.addWidget(w)
            hbox.setAlignment(w, Qt.AlignVCenter)
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox)
        
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

        ## NEW (use cursor position)        
        # self.connect(self.canvas, SIGNAL('clicked()'), self.curs_pos)
    
    # def create_status_bar(self):
    #     self.status_text = QLabel("This is a demo")
    #     self.statusBar().addWidget(self.status_text, 1)
        
    # def create_menu(self):        
    #     self.file_menu = self.menuBar().addMenu("&File")
        
    #     load_file_action = self.create_action("&Save plot",
    #         shortcut="Ctrl+S", slot=self.save_plot, 
    #         tip="Save the plot")
    #     quit_action = self.create_action("&Quit", slot=self.close, 
    #         shortcut="Ctrl+Q", tip="Close the application")
        
    #     self.add_actions(self.file_menu, 
    #         (load_file_action, None, quit_action))
        
    #     self.help_menu = self.menuBar().addMenu("&Help")
    #     about_action = self.create_action("&About", 
    #         shortcut='F1', slot=self.on_about, 
    #         tip='About the demo')
        
    #     self.add_actions(self.help_menu, (about_action,))

    # def add_actions(self, target, actions):
    #     for action in actions:
    #         if action is None:
    #             target.addSeparator()
    #         else:
    #             target.addAction(action)

    # def create_action(  self, text, slot=None, shortcut=None, 
    #                     icon=None, tip=None, checkable=False, 
    #                     signal="triggered()"):
    #     action = QAction(text, self)
    #     if icon is not None:
    #         action.setIcon(QIcon(":/%s.png" % icon))
    #     if shortcut is not None:
    #         action.setShortcut(shortcut)
    #     if tip is not None:
    #         action.setToolTip(tip)
    #         action.setStatusTip(tip)
    #     if slot is not None:
    #         self.connect(action, SIGNAL(signal), slot)
    #     if checkable:
    #         action.setCheckable(True)
    #     return action


def main():
    app = QApplication(sys.argv)
    form = AppForm()
    form.show()
    app.exec_()

    return form


if __name__ == "__main__":
    form = main()