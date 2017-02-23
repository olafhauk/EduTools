#!/imaging/local/software/anaconda/latest/x86_64/bin/python
"""
Plot two vectors in 2D and print correlation and angle
One vector defined by values in text box
Second vector will change with mouse clicks in plot are
"""

# based on:
# https://github.com/eliben/code-for-blog/blob/master/2009/qt_mpl_bars.py
# and lot's of googling

import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy as np
from numpy.random import * # for use in interactive displays

import scipy.linalg

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

plot_colors = ['b', 'g', 'r', 'c', 'm', 'k', 'w']

class AppForm(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('Regression demo')

        # What do display: 0 -> vectors, 1 -> regression
        self.display_option = 0

        # self.create_menu()

        self.create_main_frame()
        # self.create_status_bar()        

        # coordinates from mouse clicks
        self.data_mouse = []
        
        self.on_draw()        

        # # new
        # self.setMouseTracking(True)
        
        # this will print cursor position continuously:

        # while 1:
        #     print QCursor().pos()
        #     self.curs_pos()

    def display_method(self, text):
        # switch between display options (combobox)
        print text
        if (text=='Vectors'):
            self.display_option = 0
        elif (text=='Regression'):
            self.display_option = 1
        elif (text=='Correlation'):
            self.display_option = 2
        
        self.create_main_frame()
        self.on_draw()
    
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
        """ get cursor position in canvas
        """
        # cursor position
        x = np.array([event.xdata, event.ydata])
        self.data_mouse = [0, 0, x[0], x[1]]
        # coordinates from text box, shifted to origin        
        print "Cursor pos", x[0], x[1]
        self.on_draw()

    
    def on_draw(self):
        """ Redraws the figure
        """
        # str = unicode(self.textbox.text())
        # # self.data = map(int, str.split())
        # self.data = map(float, str.split())

        print "Draw"
        
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

        str_txt = unicode(self.textbox_lim.text())
        # self.data = map(int, str.split())
        text_lim = map(float, str_txt.split())
        self.x_lim = text_lim

        x = np.arange(self.x_lim[0],self.x_lim[1],self.x_lim[2]) # x-axis for plots
        x = x * self.slider_scale.value()/10. # scale x-axis with slider value
        n = len(x) # for use in interactive displays

        ## display VECTORS
        if self.display_option==0:

            ## Vector Plot
            if self.data_mouse==[]:
                self.data_mouse = [0,0,0.9,0]
            X = [ np.array([0,0,0,0.9]), np.array(self.data_mouse) ]            
            scale_fac = self.slider_scale.value() / 10.
            X[0] = X[0] * scale_fac # rescale; mouse coordinates are already scaled
            colours = ['k', 'r'] # arrow colours            
            for [x,c] in zip(X,colours):
                if (x!=[]):                    
                    # self.axes.arrow(x[0], x[1], x[2], x[3], head_width=0.05, head_length=0.1, fc=c, ec=c)
                    self.axes.arrow(x[0], x[1], x[2], x[3], fc=c, ec=c)
                    self.axes.set_xlim(self.x_lim[0]*scale_fac, self.x_lim[1]*scale_fac)
                    self.axes.set_ylim(self.x_lim[0]*scale_fac, self.x_lim[1]*scale_fac)
            self.X = X[0] # keep reference vector
                    
            # angle between vectors
            corr = np.dot(X[0],X[1]) / np.sqrt(np.dot(X[0],X[0])*np.dot(X[1],X[1]))
            angle = np.arccos( corr )
            disp_text = "  %.2f rad / %.2f deg, r = %.2f" % (angle, 180*angle/np.pi, corr)
            self.axes.text(x=self.x_lim[0]*scale_fac,y=-0.9*self.x_lim[1]*scale_fac,s=disp_text)
            # myarc = matplotlib.patches.Arc(xy=(0,0), width=1, height=1, angle=180, theta1=0, theta2=360)
            # self.axes.add_patch(myarc)
            print "Correlation", corr
            print "Angle: ", angle, " rad; ", 180*angle/np.pi, " deg"
                
        ## display REGRESSION
        elif self.display_option==1: # regression option chosen
            # textbox 2 (function execution)
            # textbox 1
            
            # insert defaults where necessary
            if len(text_lim)==0:
                self.x_lim.append(-1.)
            if len(text_lim)==1:
                self.x_lim.append(1.)
            if len(text_lim)==2:
                self.x_lim.append(0.01)  

            if hasattr(self, 'functext'):                
                legend = []
                ylist = [] # list of results
                for ff in self.functext:
                    txt_str = unicode(ff.text())
                    print txt_str
                    if (txt_str != ''):
                        y = np.array( eval(txt_str) )
                        self.axes.plot(x,y)
                        self.axes.set_xlim(x[0],x[-1])
                        ylist.append(y)
                        legend.append(txt_str)
                
                if len(ylist)>1:
                    ymat = np.array(ylist)
                    corrmat = np.corrcoef(ymat)
                    print corrmat
                    
                    nr, nc = corrmat.shape
                    extent = [-0.5, nc-0.5, nr-0.5, -0.5]
                    self.h_imshow = self.axes2.imshow(corrmat, extent=extent, origin='upper',
                                                         interpolation='nearest', vmin=-1, vmax=1)
                    self.axes2.set_xlim(extent[0], extent[1])
                    self.axes2.set_ylim(extent[2], extent[3])
                    self.axes2.xaxis.set_ticks(np.arange(0,nc,1))
                    self.axes2.yaxis.set_ticks(np.arange(0,nr,1))

                    # add colorbar
                    divider2 = make_axes_locatable(self.axes2)
                    cax2 = divider2.append_axes("right", size="20%", pad=0.05)
                    cb = self.fig.colorbar(self.h_imshow, cax=cax2)
                    cb.set_label(r"Correlation", size=8)
                    cb.ax.tick_params(labelsize=6)

                    # Fit Regression, explain first function by other functions
                    X = ymat[1:,:].T
                    pinvX = scipy.linalg.pinv(X)
                    b = pinvX.dot(ymat[0,:].T)
                    ye = X.dot(b)
                    print ye.shape, b.shape, X.shape
                    self.axes.plot(x,ye,linestyle='--')

                    legend.append('Pred')
                    self.axes.legend(legend, loc=2, prop={'size': 6})

        ## display CORRELATION
        elif self.display_option==2:
            if len(text_lim)==0:
                self.x_lim.append(-1.)
            if len(text_lim)==1:
                self.x_lim.append(1.)
            if len(text_lim)==2:
                self.x_lim.append(0.1)  

            x = np.arange(self.x_lim[0],self.x_lim[1],self.x_lim[2]) # variable 1
            y = np.arange(self.x_lim[0],self.x_lim[1],self.x_lim[2]) # variable 2

            x = x * self.slider_scale.value()/10. # scale x-axis with slider value
            y = y * self.slider_scale.value()/10. # scale x-axis with slider value

            n = len(x) # for use in interactive display

            text_list = [] # text strings to evaluate as functions for scatter plots
            if (hasattr(self, 'corrtext')):
                for ff in self.corrtext:
                    if len(ff.text())>0:
                        text_list.append( '[' + unicode(ff.text()) + ']' )
                    else:
                        text_list.append('[x,y]')

                legend = []
                zlist = [] # list of results
                for [ti,tt] in enumerate(text_list):
                    if not(tt==[]):
                        z = np.array( eval(tt) )
                        zlist.append(y)                        
                        self.axes.scatter(z[0,], z[1,], marker='x', s=10, c=plot_colors[ti])
                        corr = np.corrcoef(z[0,], z[1,])
                        legend.append( "%s %.3f" % (tt, corr[0,1]) )

                min_x, max_x = np.min(z[0,]), np.max(z[0,])
                min_y, max_y = np.min(z[1,]), np.max(z[1,])
                    
                self.axes.set_xlim(min_x, max_x)
                self.axes.set_ylim(min_y, max_y)                

                self.axes.legend(legend, loc=2, prop={'size': 6})
    
        self.canvas.draw()



    def on_add(self):
        """ Add text box (for function plotting) to figure
        """
        print "Adding"

        if self.display_option==1: # Regression
        
            if not(hasattr(self, 'functext')):
                self.functext = [] # initialise list

            n = len(self.functext)

            self.functext.append(QLineEdit()) # textbox for functions to be executed
            self.functext[n].setMinimumWidth(200)
            self.create_main_frame()
            self.on_draw()
            self.connect(self.functext[n], SIGNAL('editingFinished ()'), self.on_draw)
        
        elif self.display_option == 2: # Correlation
            
            if not(hasattr(self, 'corrtext')):
                    self.corrtext = [] # initialise list

            n = len(self.corrtext)

            self.corrtext.append(QLineEdit()) # textbox for functions to be executed
            self.corrtext[n].setMinimumWidth(200)
            self.create_main_frame()
            self.on_draw()
            self.connect(self.corrtext[n], SIGNAL('editingFinished ()'), self.on_draw)



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
        if self.display_option==0 or self.display_option==2:
            self.axes = self.fig.add_subplot(111)
            self.axes.set_xlim([-1, 1])
            self.axes.set_ylim([-1, 1])
            self.axes.grid(True, 'major')
            self.axes.tick_params(axis='both', which='major', labelsize=6)            

        elif self.display_option==1: # only for regression
            self.axes = self.fig.add_subplot(211)
            self.axes.set_xlim([-1, 1])
            self.axes.set_ylim([-1, 1])
            self.axes.grid(True, 'major')
            self.axes.tick_params(axis='both', which='major', labelsize=6)

            self.axes2 = self.fig.add_subplot(212)
            self.axes2.set_xlim([-1, 1])
            self.axes2.set_ylim([-1, 1])
            self.axes2.grid(True, 'major')
            self.axes2.tick_params(axis='both', which='major', labelsize=6)

            # self.fig2 = Figure((5.0, 4.0), dpi=self.dpi)
            # self.canvas2 = FigureCanvas(self.fig2)
            # self.canvas2.setParent(self.main_frame)
            # self.axes2 = self.fig2.add_subplot(111)
            # self.axes2.tick_params(axis='both', which='major', labelsize=6)
        
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

        ## Text for x-axis limits (min, max, step)
        if not(hasattr(self, 'textbox_lim')):            
            self.textbox_lim = QLineEdit()
            self.textbox_lim.setText('-1 1 0.01')
            self.textbox_lim.setMinimumWidth(200)
            self.connect(self.textbox_lim, SIGNAL('editingFinished ()'), self.on_draw)        
        
        self.draw_button = QPushButton("&Draw")
        self.connect(self.draw_button, SIGNAL('clicked()'), self.on_draw)

        self.add_button = QPushButton("&Add") # add textbox for function plotting
        self.connect(self.add_button, SIGNAL('clicked()'), self.on_add)        

        # Menu box for display options (vectors. regression, correlation)
        if not(hasattr(self, 'comboBox')):
            self.comboBox = QComboBox(self)
            self.comboBox.addItem("Vectors")
            self.comboBox.addItem("Regression")
            self.comboBox.addItem("Correlation")

            self.comboBox.activated[str].connect(self.display_method)

        ## Slider
        if not(hasattr(self, 'slider_scale')):
            self.slider_label = QLabel('Scaling:')
            self.slider_scale = QSlider(Qt.Horizontal)
            self.slider_scale.setRange(1, 1000)
            self.slider_scale.setValue(10)
            self.slider_scale.setTracking(True)
            # self.slider_scale.setTickPosition(QSlider.TicksBothSides)
            self.slider_scale.setTickPosition(QSlider.TicksBelow)
            self.slider_scale.setTickInterval(50)
            self.connect(self.slider_scale, SIGNAL('valueChanged(int)'), self.on_draw)
            
        # Create layout within canvas

        hbox = QHBoxLayout()   # buttons, sliders, etc.
        hbox2 = QHBoxLayout()  # text boxes for regression
        hbox3 = QHBoxLayout()  # text boxes for correlation
        
        # for w in [  self.textbox, self.draw_button, self.grid_cb,
        #             slider_label, self.slider]:
        attr_list = [self.textbox_lim, self.draw_button, self.add_button,
                        self.slider_label, self.slider_scale, self.comboBox]        

        for w in attr_list:
            hbox.addWidget(w)
            hbox.setAlignment(w, Qt.AlignVCenter)

        # initialise or add text boxes       
        keep_dispopt = self.display_option
        self.display_option==1 # Regression
        if not(hasattr(self, 'functext')):
            self.functext = []
            self.functext.append(QLineEdit()) # textbox for functions to be executed
            self.functext[0].setMinimumWidth(200)            
            self.connect(self.functext[0], SIGNAL('editingFinished ()'), self.on_draw)
        else:
            for w in self.functext:
                hbox2.addWidget(w)
                hbox2.setAlignment(w, Qt.AlignVCenter)

        self.display_option==2 # Correlation
        if not(hasattr(self, 'corrtext')):
            self.corrtext = []
            self.corrtext.append(QLineEdit()) # textbox for functions to be executed
            self.corrtext[0].setMinimumWidth(200)            
            self.connect(self.corrtext[0], SIGNAL('editingFinished ()'), self.on_draw)
        else:
            for w in self.corrtext:
                hbox3.addWidget(w)
                hbox3.setAlignment(w, Qt.AlignVCenter)

        self.display_option = keep_dispopt
                
        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        # if self.display_option==1: # only for regression
        #     vbox.addWidget(self.canvas2)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)
        
        # self.grid_cb = QCheckBox("Show &Grid")
        # self.grid_cb.setChecked(False)
        # self.connect(self.grid_cb, SIGNAL('stateChanged(int)'), self.on_draw)
        
        
        
        #
        # Layout with box sizers
        #         

        ## NEW (use cursor position)        
        # self.connect(self.canvas, SIGNAL('clicked()'), self.curs_pos)
    
    # def create_status_bar(self):
    #     self.status_text = QLabel("This is a demo")
    #     self.statusBar().addWidget(self.status_text, 1)
        
    # def create_menu(self):
    #     self.options_menu = self.menuBar().addMenu("&Options")
        
    #     # load_file_action = self.create_action("&Save plot",
    #     #     shortcut="Ctrl+S", slot=self.save_plot, 
    #     #     tip="Save the plot")
    #     # quit_action = self.create_action("&Quit", slot=self.close, 
    #     #     shortcut="Ctrl+Q", tip="Close the application")

    #     regress_action = QAction("&Regression", self)
        
    #     self.connect(regress_action, SIGNAL("triggered()"), self.do_regression)

    #     self.options_menu.addAction(regress_action)
        
    #     # self.add_actions(self.file_menu, 
    #     #     [load_file_action])
        
    #     # self.help_menu = self.menuBar().addMenu("&Help")
    #     # about_action = self.create_action("&About", 
    #     #     shortcut='F1', slot=self.on_about, 
    #     #     tip='About the demo')
        
    #     # self.add_actions(self.help_menu, (about_action,))

    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def create_action(  self, text, slot=None, shortcut=None, 
                        icon=None, tip=None, checkable=False, 
                        signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            self.connect(action, SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action


def main():
    app = QApplication(sys.argv)
    form = AppForm()
    form.show()
    app.exec_()

    return form


if __name__ == "__main__":
    form = main()