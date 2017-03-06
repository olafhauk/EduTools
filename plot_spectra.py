#!/imaging/local/software/anaconda/latest/x86_64/bin/python
"""
Plot filtered data of a spike, filter spectrum etc.
OH Mar 2017
"""

# based on plot_two_vectors.py

import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy as np
# for use in text functions
from numpy import sin,sinh,cos,cosh,tan,tanh,exp,exp2,expm1,arcsin,arcsinh,arccos,arccosh,arctan,arctanh,arctan2
from numpy.random import * # for use in interactive displays

import scipy.linalg

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from time import sleep

plot_colors = ['b', 'g', 'r', 'c', 'm', 'k', 'w']

class AppForm(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('Filter demo')

        # What do display: 0 -> vectors, 1 -> regression
        self.display_option = 0

        # by default don't show moving filter kernel
        self.show_movie = 0

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
        if (text=='Filters'):
            self.display_option = 0
        # elif (text=='Regression'):
        #     self.display_option = 1
        # elif (text=='Correlation'):
        #     self.display_option = 2
        
        self.create_main_frame()
        self.on_draw()


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

        self.axes2.clear()
        self.axes3.clear()

        str_txt = unicode(self.textbox_lim.text())
        # self.data = map(int, str.split())
        text_lim = map(float, str_txt.split())
        self.x_lim = text_lim

        # scaling of x-axis from slider
        scale_fac = self.slider_scale.value() / 10.

        # symmetrical x-range
        x = np.arange(self.x_lim[0],self.x_lim[1]+self.x_lim[2],self.x_lim[2]) # x-axis for plots
        x = x * scale_fac # scale x-axis with slider value
        n = len(x) # for use in interactive displays

        ## display FILTERS
        if self.display_option==0:

            SF = 1000/self.x_lim[2] # sampling frequency
            F = self.funcsliders[0].value() # frequency of signal

            FWHM = self.funcsliders[1].value() # width of Gaussian filter kernel (ms)
            print("FWHM: %f" % (FWHM))
            n_steps = 500 # number of steps for convolution with filter kernel

            ## Signal time course
            y = np.zeros([n,1])
            y[x==0] = 1

            txt_str = unicode( self.functext[0].text() )
            if txt_str == '':
                y = np.sin(x*2.*np.pi*F/SF)
            else:
                y = np.array( eval(txt_str) )

            self.axes.plot(x,y) # plot signal time course

            self.axes.set_xlim(self.x_lim[0]*scale_fac, self.x_lim[1]*scale_fac)
            y1 = min(min(y),0)
            y2 = max(max(y),0)
            self.axes.set_ylim(y1, y2)

            ## convolve with Gaussian
            # x-values where convolution kernel to be applied
            self.steps = np.arange(min(x), max(x), (max(x)-min(x))/n_steps)

            self.x = x
            self.y = y
            self.FWHM = FWHM
            self.ss = min(x)
            self.step = (max(x)-min(x))/n_steps
            self.y_conv = [] # convolution of kernel and signal, created in plot_gauss()

            # FFT of Gauss kernel
            # compute and plot filter kernel
            txt_str = unicode( self.functext[1].text() )

            if txt_str == '':
                filt_kern = np.exp(-x**2/(2*FWHM**2)) # filter kernel
                filt_kern = SF*filt_kern / np.sum(filt_kern) # normalise kernel to unit area
            else:
                filt_kern = np.array( eval(txt_str) )
                
            y_max = np.max(y)
            self.axes.plot(x,y_max*filt_kern/np.max(filt_kern))

            # Plot Gaussian kernel frequency spectrum
            fft_g = np.fft.fft(filt_kern)
            fft_g2 = fft_g[0:np.floor(n/2)]
            psd_g = np.abs(fft_g2)**2 / (SF*n)
            psd_g[1:-2] = 2*psd_g[1:-2]

            freqs = np.arange(0,len(psd_g)*(SF/n),SF/n)

            self.axes3.plot(freqs, psd_g/max(psd_g))
            
            if self.show_movie==1:
                # start timer to plot moving Gaussian kernels
                self.timer = QTimer(self)
                self.connect(self.timer, SIGNAL("timeout()"), self.plot_gauss)
                self.timer.start(1)
                    
            # Plot signal frequency spectrum
            fft_y = np.fft.fft(y)
            fft_y2 = fft_y[0:np.floor(n/2)]
            psd_y = np.abs(fft_y2)**2 / (SF*n)
            psd_y[1:-2] = 2*psd_y[1:-2]

            freqs = np.arange(0,len(psd_y)*(SF/n),SF/n)

            self.axes2.plot(freqs, psd_y/max(psd_y))

            # compute and plot "filtered" spectrum
            psd_gy = np.multiply(psd_y, psd_g)
            self.axes2.plot(freqs, psd_gy/max(psd_gy))

        # self.fig.tight_layout()
        self.canvas.draw()


    def plot_gauss(self):
        self.axes.clear()

        # variables available to external text function
        ss = self.ss # current x-value for convolution
        x = self.x - ss # adjust for computation of current kernel
        y = self.y
        FWHM = self.FWHM        
        steps = self.steps

        # plot signal time course
        self.axes.plot(self.x,y)

        # compute and plot filter kernel
        txt_str = unicode( self.functext[1].text() )

        if txt_str == '':
            filt_kern = np.exp(-x**2/(2*FWHM**2)) # filter kernel            
        else:
            filt_kern = np.array( eval(txt_str) )

        filt_kern = filt_kern / np.sum(filt_kern) # normalise kernel to unit area
            
        self.axes.plot(self.x,filt_kern/np.max(filt_kern))

        # compute and plot convolution
        self.y_conv.append(y.dot(filt_kern))
        n_conv = len(self.y_conv)
        self.axes.plot(self.steps[:n_conv], self.y_conv)

        # increase step, cancel loop when end reached
        self.ss = self.ss + self.step
        self.canvas.draw()
        if self.ss >= max(self.x):
            self.timer.stop()


    def on_movie(self):
        """ Show moving filter kernel
        """

        if self.show_movie==0:
            self.show_movie = 1
            self.on_draw()
        else:
            self.timer.stop()
            self.show_movie = 0        


    def on_add_func(self):
        """ Add text box (for function plotting) to figure
        """
        print "Adding function"

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
        self.funcsliders[n].setRange(0, 100)
        self.funcsliders[n].setValue(20)
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
        if self.display_option==0:            
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
            self.axes3.set_title("Kernel Spectrum", {'fontsize': 6})
        
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
            self.textbox_lim.setText('-1000 1000 1') # x-axis scale of signal time course in ms
            self.textbox_lim.setMinimumWidth(200)
            self.connect(self.textbox_lim, SIGNAL('editingFinished ()'), self.on_draw)        
        
        self.movie_button = QPushButton("&Movie")
        self.connect(self.movie_button, SIGNAL('clicked()'), self.on_movie)

        self.add_func_button = QPushButton("&+Func") # add textbox for function plotting
        self.connect(self.add_func_button, SIGNAL('clicked()'), self.on_add_func)

        self.add_slider_button = QPushButton("&+Slider") # add slider for function plotting
        self.connect(self.add_slider_button, SIGNAL('clicked()'), self.on_add_slider)

        # Menu box for display options (Filters, etc.)
        if not(hasattr(self, 'comboBox')):
            self.comboBox = QComboBox(self)
            self.comboBox.addItem("Filters")            

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

        hbox1 = QHBoxLayout()   # buttons, sliders, etc.
        hbox2 = QHBoxLayout()  # text boxes for regression
        hbox3 = QHBoxLayout()  # sliders for noise in regression
        hbox4 = QHBoxLayout()  # text boxes for correlation        
               
        attr_list = [self.textbox_lim, self.movie_button, self.add_func_button,
                    self.add_slider_button, self.slider_label, self.slider_scale,
                    self.comboBox]

        # HBOX1 for Draw/Add buttons etc.
        for w in attr_list:
            hbox1.addWidget(w)
            hbox1.setAlignment(w, Qt.AlignVCenter)

        ## initialise or add TEXT BOXES
        
        # REGRESSION text and slider boxes
        if not(hasattr(self, 'functext')):            

            self.funclabels = []
            self.funclabels.append(QLabel('Signal:'))
            self.funclabels.append(QLabel('Kernel:'))

            self.functext = []
            
            # function text for signal time course
            self.functext.append(QLineEdit("sin(x*2.*np.pi*F/SF)")) # textbox for functions to be executed
            self.functext[0].setMinimumWidth(200)
            self.connect(self.functext[0], SIGNAL('editingFinished ()'), self.on_draw)

            # function text for filter kernel time course
            self.functext.append(QLineEdit("exp(-x**2/(2*FWHM**2))")) # textbox for functions to be executed
            self.functext[1].setMinimumWidth(200)
            self.connect(self.functext[1], SIGNAL('editingFinished ()'), self.on_draw)

            
        if not(hasattr(self, 'funcsliders')):
            # regression sliders
            self.fslidelabels = []
            self.fslidelabels.append(QLabel('Freq:'))
            self.fslidelabels.append(QLabel('FWHM:'))

            self.funcsliders = []
            # Slider for signal frequency
            self.funcsliders.append(QSlider(Qt.Horizontal))
            self.funcsliders[0].setRange(0, 100)
            self.funcsliders[0].setMinimumWidth(200)
            self.funcsliders[0].setValue(20)
            self.funcsliders[0].setTracking(True)
            self.funcsliders[0].setTickPosition(QSlider.TicksBelow)
            self.funcsliders[0].setTickInterval(50)
            self.connect(self.funcsliders[0], SIGNAL('valueChanged(int)'), self.on_draw)

            # Slider for Gaussian filter kernel FWHM
            self.funcsliders.append(QSlider(Qt.Horizontal))
            self.funcsliders[1].setRange(1, 50)
            self.funcsliders[1].setMinimumWidth(200)
            self.funcsliders[1].setValue(20)
            self.funcsliders[1].setTracking(True)
            self.funcsliders[1].setTickPosition(QSlider.TicksBelow)
            self.funcsliders[1].setTickInterval(50)
            self.connect(self.funcsliders[1], SIGNAL('valueChanged(int)'), self.on_draw)

        # add function text boxes to box
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
    app.setQuitOnLastWindowClosed(False)
    form = AppForm()
    form.show()
    app.exec_()

    return form


if __name__ == "__main__":
    form = main()