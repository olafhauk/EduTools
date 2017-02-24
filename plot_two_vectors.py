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


    def covcorr(self, text):
        # switch between correlation and covariance display
        print text
        if (text=='Disp Corr'):
            self.display_covcorr = 0
        elif (text=='Disp Cov'):
            self.display_covcorr = 1        
        
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

        str_txt = unicode(self.textbox_lim.text())
        # self.data = map(int, str.split())
        text_lim = map(float, str_txt.split())
        self.x_lim = text_lim

        # symmetrical x-range
        x = np.arange(self.x_lim[0],self.x_lim[1]+self.x_lim[2],self.x_lim[2]) # x-axis for plots
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

                # get slider values for use in interactive display
                S = []
                for ss in self.funcsliders:
                    S.append(ss.value()/1000.)
                S = np.array(S)

                for [fi,ff] in enumerate(self.functext):
                    txt_str = unicode(ff.text())
                    print txt_str
                    if (txt_str != ''):
                        y = np.array( eval(txt_str) )
                        self.axes.plot(x,y,c=plot_colors[fi])
                        self.axes.set_xlim(x[0],x[-1])
                        ylist.append(y)
                        legend.append(txt_str)                
                
                if len(ylist)>1:
                    ymat = np.array(ylist)
                    corrmat = np.corrcoef(ymat) # correlation matrix
                    print corrmat
                    print ymat.shape
                    covmat = np.cov(ymat) # covariance matrix

                    # correlation or covariance to display
                    if self.display_covcorr==0:
                        disp_mat = corrmat
                        axes2_title = 'Correlation'
                    else:
                        disp_mat = covmat
                        axes2_title = 'Covariance'
                    
                    nr, nc = corrmat.shape
                    extent = [-0.5, nc-0.5, nr-0.5, -0.5]
                    self.h_imshow = self.axes2.imshow(disp_mat, extent=extent, origin='upper',
                                                         interpolation='nearest', vmin=-1, vmax=1)                    

                    self.axes2.set_xlim(extent[0], extent[1])
                    self.axes2.set_ylim(extent[2], extent[3])
                    self.axes2.xaxis.set_ticks(np.arange(0,nc,1))
                    self.axes2.yaxis.set_ticks(np.arange(0,nr,1))
                    self.axes2.set_title(axes2_title, {'fontsize': 6})

                    # add colorbar
                    divider2 = make_axes_locatable(self.axes2)
                    cax2 = divider2.append_axes("right", size="20%", pad=0.05)
                    cb = self.fig.colorbar(self.h_imshow, cax=cax2)                    
                    cb.ax.tick_params(labelsize=6)

                    # Fit Regression, explain first function by other functions
                    X = ymat[1:,:].T
                    pinvX = scipy.linalg.pinv(X)
                    b = pinvX.dot(ymat[0,:].T)
                    ye = X.dot(b)
                    print ye.shape, b.shape, X.shape
                    self.axes.plot(x,ye,c='k',linestyle='--')

                    # plot parameter estimates
                    print b
                    nb = len(b)
                    b_x = np.arange(0,nb) + .6 # start at 1, because 0 is predicted variable
                    self.h_bar = self.axes3.bar(b_x,b,.8)
                    self.axes3.set_xlim(0.3, b_x[-1]+1.2)
                    self.axes3.xaxis.set_ticks(np.arange(1,nb+1,1))

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

            # get slider values for interactive display
            S = []
            for ss in self.funcsliders:
                S.append(ss.value()/1000.)
            S = np.array(S)

            # symmetrical x-/y-ranges
            x = np.arange(self.x_lim[0],self.x_lim[1]+self.x_lim[2],self.x_lim[2]) # variable 1
            y = np.arange(self.x_lim[0],self.x_lim[1]+self.x_lim[2],self.x_lim[2]) # variable 2

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
                        cov = np.dot(z[0,],z[1,])
                        corr = np.corrcoef(z[0,], z[1,])
                        legend.append( "%s %.2f %.2f" % (tt, corr[0,1], cov) )

                min_x, max_x = np.min(z[0,]), np.max(z[0,])
                min_y, max_y = np.min(z[1,]), np.max(z[1,])
                    
                self.axes.set_xlim(min_x, max_x)
                self.axes.set_ylim(min_y, max_y)                

                self.axes.legend(legend, loc=2, prop={'size': 6})
    
        self.fig.tight_layout()
        self.canvas.draw()



    def on_add_func(self):
        """ Add text box (for function plotting) to figure
        """
        print "Adding function"

        if self.display_option==1: # Regression
        
            if not(hasattr(self, 'functext')):
                self.functext = [] # initialise list
                self.funclabels = []

            n = len(self.functext)

            disp_text = str(n) + ":" # text for label

            self.functext.append(QLineEdit()) # textbox for functions to be executed
            self.functext[n].setMinimumWidth(200)
            self.connect(self.functext[n], SIGNAL('editingFinished ()'), self.on_draw)
            
            self.funclabels.append(QLabel(disp_text))

        elif self.display_option == 2: # Correlation
            
            if not(hasattr(self, 'corrtext')):
                    self.corrtext = [] # initialise list
                    self.corrlabels = []

            n = len(self.corrtext)

            disp_text = str(n) + ":" # text for label

            self.corrtext.append(QLineEdit()) # textbox for functions to be executed
            self.corrtext[n].setMinimumWidth(200)            
            self.connect(self.corrtext[n], SIGNAL('editingFinished ()'), self.on_draw)

            self.corrlabels.append(QLabel(disp_text))

        self.create_main_frame()


    def on_add_slider(self):
        """ Add sliders (for function plotting) to figure
        """

        print "Adding slider"

        # noise sliders
        if not(hasattr(self, 'funcsliders')):
            self.funcsliders = []
            self.fslidelabels = []
        
        n = len(self.funcsliders)

        disp_text = str(n) + ":" # text for label

        self.funcsliders.append(QSlider(Qt.Horizontal))
        self.funcsliders[n].setRange(0, 1000)
        self.funcsliders[n].setValue(0)
        self.funcsliders[n].setTracking(True)
        self.funcsliders[n].setTickPosition(QSlider.TicksBelow)
        self.funcsliders[n].setTickInterval(50)
        self.connect(self.funcsliders[n], SIGNAL('valueChanged(int)'), self.on_draw)

        self.fslidelabels.append(QLabel(disp_text))

        self.create_main_frame()
        self.on_draw()


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
            self.axes.set_xlim([-1., 1.])
            self.axes.set_ylim([-1., 1.])
            self.axes.grid(True, 'major')
            self.axes.tick_params(axis='both', which='major', labelsize=6)            

        elif self.display_option==1: # only for regression
            self.axes = self.fig.add_subplot(211)
            self.axes.set_xlim([-1, 1])
            self.axes.set_ylim([-1, 1])
            self.axes.grid(True, 'major')
            self.axes.tick_params(axis='both', which='major', labelsize=6)

            self.axes2 = self.fig.add_subplot(223)                        
            # self.axes2.grid(True, 'major')
            self.axes2.tick_params(axis='both', which='major', labelsize=6)            

            self.axes3 = self.fig.add_subplot(224)
            self.axes3.grid(True, 'major')
            self.axes3.tick_params(axis='both', which='major', labelsize=6)
            self.axes3.set_title("Estimates", {'fontsize': 6})
        
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

        self.add_func_button = QPushButton("&+Func") # add textbox for function plotting
        self.connect(self.add_func_button, SIGNAL('clicked()'), self.on_add_func)

        self.add_slider_button = QPushButton("&+Slider") # add slider for function plotting
        self.connect(self.add_slider_button, SIGNAL('clicked()'), self.on_add_slider)

        # Menu box for display options (vectors. regression, correlation)
        if not(hasattr(self, 'comboBox')):
            self.comboBox = QComboBox(self)
            self.comboBox.addItem("Vectors")
            self.comboBox.addItem("Regression")
            self.comboBox.addItem("Correlation")

            self.comboBox.activated[str].connect(self.display_method)

        # menu for correlation/covariance
        if not(hasattr(self, 'corrBox')):
            self.corrBox = QComboBox(self)
            self.corrBox.addItem("Disp Corr")
            self.corrBox.addItem("Disp Cov")
            self.display_covcorr = 0 # Default: Correlation

            self.corrBox.activated[str].connect(self.covcorr)

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

        hbox1 = QHBoxLayout()   # buttons, sliders, etc.
        hbox2 = QHBoxLayout()  # text boxes for regression
        hbox3 = QHBoxLayout()  # sliders for noise in regression
        hbox4 = QHBoxLayout()  # text boxes for correlation        
               
        attr_list = [self.textbox_lim, self.draw_button, self.add_func_button,
                    self.add_slider_button, self.slider_label, self.slider_scale,
                    self.corrBox, self.comboBox]

        # HBOX1 for Draw/Add buttons etc.
        for w in attr_list:
            hbox1.addWidget(w)
            hbox1.setAlignment(w, Qt.AlignVCenter)

        ## initialise or add TEXT BOXES
        
        # REGRESSION text and slider boxes
        if not(hasattr(self, 'functext')):
            self.funclabel = QLabel('Regr:') # for box with regression functions

            self.funclabels = []
            self.funclabels.append(QLabel('0:'))

            self.functext = []
            self.functext.append(QLineEdit()) # textbox for functions to be executed
            self.functext[0].setMinimumWidth(200)
            self.connect(self.functext[0], SIGNAL('editingFinished ()'), self.on_draw)

            
        if not(hasattr(self, 'funcsliders')):
            # regression sliders
            self.fslidelabels = []
            self.fslidelabels.append(QLabel('0:'))

            self.funcsliders = []
            self.funcsliders.append(QSlider(Qt.Horizontal))
            self.funcsliders[0].setRange(0, 1000)
            self.funcsliders[0].setMinimumWidth(200)
            self.funcsliders[0].setValue(0)
            self.funcsliders[0].setTracking(True)
            self.funcsliders[0].setTickPosition(QSlider.TicksBelow)
            self.funcsliders[0].setTickInterval(50)
            self.connect(self.funcsliders[0], SIGNAL('valueChanged(int)'), self.on_draw)

        # add function text boxes to box
        hbox2.addWidget(self.funclabel)
        for aa in range(len(self.functext)):
            w = self.funclabels[aa]
            hbox2.addWidget(w)

            w = self.functext[aa]
            hbox2.addWidget(w)
            hbox2.setAlignment(w, Qt.AlignVCenter)
        
        # add function sliders to box
        for aa in range(len(self.fslidelabels)):
            w = self.fslidelabels[aa]
            hbox3.addWidget(w)

            w = self.funcsliders[aa]
            hbox3.addWidget(w)
            hbox3.setAlignment(w, Qt.AlignVCenter)


        # CORRELATION text box        
        if not(hasattr(self, 'corrtext')):
            self.corrlabel = QLabel('Corr:') # for box with regression functions

            self.corrlabels = []
            self.corrlabels.append(QLabel('0:'))

            self.corrtext = []
            self.corrtext.append(QLineEdit()) # textbox for functions to be executed
            self.corrtext[0].setMinimumWidth(200)            
            self.connect(self.corrtext[0], SIGNAL('editingFinished ()'), self.on_draw)        
        
        # add correlation text boxes to box
        hbox4.addWidget(self.corrlabel)
        for aa in range(len(self.corrtext)):
            w = self.corrlabels[aa]
            hbox4.addWidget(w)

            w = self.corrtext[aa]
            hbox4.addWidget(w)
            hbox4.setAlignment(w, Qt.AlignVCenter)

                
        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        # if self.display_option==1: # only for regression
        #     vbox.addWidget(self.canvas2)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        
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