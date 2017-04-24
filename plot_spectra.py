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
from numpy import sin,sinh,cos,cosh,tan,tanh,exp,exp2,expm1,arcsin,arcsinh,arccos,arccosh
from numpy import arctan,arctanh,arctan2,pi
from numpy import arange
# note: pi is now np.pi, don't use as index of p
from numpy.random import * # for use in interactive displays

import scipy.linalg
from scipy.fftpack import fft

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from wavelets_mne import * # e.g. morelet, cwt

from time import sleep

print "!!!"
print "This is still under development."
print "!!!"

plot_colors = ['b', 'g', 'r', 'c', 'm', 'k', 'w']

# global variables
global x_global
x_global = [] # keep track of x-axis for plotting etc.
global sfreq # sampling frequency

def box(p=0, w=0):
    # draw a box car function
    # p: float, position on x-axis (ms)
    # w: float, width of peak (ms)    
    # returns: numpy array with box
    # uses self.prec: in case x is a scalar, minimum distance for peak
    global x_global    

    x = np.array(x_global)

    # x = np.arange(-100,100)

    n = x.shape    

    if n!=(): # if array

        y = np.zeros(n)

        if w==0: # delta peak wanted, find nearest index
            i = np.argmin(np.abs(x-p))
            y[i] = 1
        else: # make boxcar
            y[np.where((x<=(p+w/2)) & (x>=(p-w/2)))] = 1.

    else: # if just one scalar number
        if w==0: # delta peak wanted, find nearest index
            if np.abs(x-p) < self.prec:
                y = np.array(1.)
        elif (x<=(p+w/2)) & (x>=(p-w/2)): # make boxcar            
            y = np.array(1.)
        else:
            y = 0

    return y


def morlet(freq, x=x_global, n_cycles=7, sigma=None, zero_mean=False):
    # run morlet wavelet with global variables
    # calls my_morlet from wavelets_mne.py
    # freq: float, frequency (Hz)
    # x: array, x values
    # note: the shift on x-axis determined by comparing x with x_global 
    # clunky, but necessary for consistency in interactive text functions
    # returns: array, wavelet
    global sfreq # sample frequency
    global x_global # original x values

    x_ori = np.array(x_global)

    # desired shift of wavelet along x-axis
    x_shift = x_ori[0] - x[0]

    W = my_morlet(sfreq, freq, x_shift, x_global, n_cycles=n_cycles, sigma=sigma, zero_mean=zero_mean)

    return W
            


class AppForm(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('Filter demo')

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
        if (text=='Filters'):
            self.display_option = 0
        elif (text=='Connectivity'):
            self.display_option = 1
        elif (text=='TimeFrequency'):
            self.display_option = 2
        
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
        global x_global # x-axis for plotting
        global sfreq # sampling frequency
      
        # x = range(len(self.data))

        # clear the axes and redraw the plot anew
        #
        if self.display_option==0:
            self.axes.clear()
            self.axes.grid(True)

            self.axes2.clear()
            # self.axes3.clear()
        elif self.display_option==1:
            self.axes2.clear()
            for ax in self.trial_axes:
                ax.clear()
                ax.set_xlim([self.x_lim[0], self.x_lim[1]])
                ax.grid(True, 'major')
                ax.tick_params(axis='both', which='major', labelsize=6)
        elif self.display_option==2:
            self.axes.clear()
            self.axes2.clear()        

        str_txt = unicode(self.textbox_lim.text())
        # self.data = map(int, str.split())
        text_lim = map(float, str_txt.split())
        self.x_lim = text_lim

        # scaling of from slider
        scale_fac = self.slider_scale.value() / 200.
        self.scale_fac = scale_fac

        # symmetrical x-range
        x = np.arange(self.x_lim[0],self.x_lim[1]+self.x_lim[2],self.x_lim[2]) # x-axis for plots
        # x = x * scale_fac # scale x-axis with slider value
        n = len(x) # for use in interactive displays

        # current sampling frequency (Hz)
        sfreq = 1000./self.x_lim[2] # global
        self.sfreq = sfreq
        SF = sfreq # sampling rate, for interactive text
        
        x_ori = x # keep because x may change locally
        x_global = x # global, for use in functions (QTimer)

        ## display FILTERS
        opt = self.display_option

        if opt==0:

            # get slider values for interactive display
            S = []
            for ss in self.funcsliders:
                S.append(ss.value()/1000.)
            S = np.array(S)

            F = self.funcsliders[1].value() # frequency of signal
            print "Frequency: %.2f: " % F

            W = self.funcsliders[0].value()*10. # FWHM variable for interactive text
            self.FWHM = W # for use in functions (historically FWHM)
            print "Width: %.2f: " % W

            n_steps = 50 # number of steps for convolution with filter kernel
            step_min = x[30] # where to start/stop convoluting
            step_max = x[-30]
            # x-values where convolution kernel to be applied
            self.steps = np.arange(step_min, step_max, (step_max-step_min)/n_steps)

            ## Signal time course
            y = np.zeros([n,1])
            y[x==0] = 1

            txt_str = unicode( self.functext[opt][0].text() )
            
            y = np.array( eval(txt_str) )

            self.axes.plot(x,y) # plot signal time course            

            self.x = x
            self.y = y
            self.ss = self.steps[0]
            self.stepsize = self.steps[1]-self.steps[0]
            self.step = 0 # count steps in convolution movie
            self.y_conv = [] # convolution of kernel and signal, created in plot_gauss()

            # FFT of Gauss kernel
            # compute and plot filter kernel
            txt_str = unicode( self.functext[opt][1].text() )

            filt_kern = np.array( eval(txt_str) )
                
            y_max = np.max(y)
            filt_kern = y_max*filt_kern/np.max(filt_kern)

            self.axes.plot(x,filt_kern)

            # Plot filter kernel frequency spectrum
            fft_k = fft(filt_kern)
            fft_k2 = fft_k[0:int(np.floor(n/2))]
            psd_k = np.abs(fft_k2)**2 / (sfreq*n)
            psd_k[1:-2] = 2*psd_k[1:-2]

            freqs = np.arange(0,len(psd_k)*(sfreq/n),sfreq/n)

            pl_idx = np.arange(0,freqs.shape[0]/4)  # how much to plot

            # self.axes3.plot(freqs[pl_idx], psd_g[pl_idx]/max(psd_g))
            
            if self.show_movie==1:

                # pre-compute convolution values for all x-values
                # before going into movie loop
                
                self.y_conv = [] # convolution time series
                self.filt_kern = [] # moving filter kernels at different time points
                
                x_ori = self.x

                # compute one "original" kernel that will be shifted around
                filt_kern = np.array( eval(txt_str) )
                filt_kern = filt_kern / np.sum(np.abs(filt_kern)) # normalise kernel to unit area

                # for interactive use, x will vary in the following
                print "Convoluting..."
                for xx in x_ori:
                    if np.mod(xx,100)==0:
                        print xx

                    kern_shift = shift_kernel(filt_kern, x_ori, xx, sfreq)

                    self.y_conv.append(y.dot(kern_shift))

                    # keep filter kernels at step points                
                    samp_dist = 1000./self.sfreq # sample distance (ms)
                    if np.min(np.abs(self.steps-xx)) < samp_dist/2.: # a hack to find step points
                        self.filt_kern.append(kern_shift)

                x = x_ori # back to normal


                # start timer to plot moving Gaussian kernels
                self.timer = QTimer(self)
                self.connect(self.timer, SIGNAL("timeout()"), self.plot_kernel)
                self.timer.start(10)

                # plot again so it stays in display
                if hasattr(self, 'y_conv_plot'):
                    self.axes.plot(self.x_conv,self.y_conv_plot)
                
                self.show_movie = 0
                    
            # Plot signal frequency spectrum
            fft_y = fft(y)
            fft_y2 = fft_y[0:int(np.floor(n/2))]
            psd_y = np.abs(fft_y2)**2 / (sfreq*n)
            psd_y[1:-2] = 2*psd_y[1:-2]

            freqs = np.arange(0,len(psd_y)*(sfreq/n),sfreq/n)
            pl_idx = np.arange(0,freqs.shape[0]/4)  # how much to plot

            self.axes2.plot(freqs[pl_idx], psd_y[pl_idx]/max(psd_y), c=plot_colors[0])

            # plot kernel FFT spectrum
            self.axes2.plot(freqs[pl_idx], psd_k[pl_idx]/max(psd_k), c=plot_colors[1])

            self.axes.set_xlim(self.x_lim[0], self.x_lim[1])            

            y1 = scale_fac*min(np.r_[y,filt_kern,0])*1.05
            y2 = scale_fac*max(np.r_[y,filt_kern,0])*1.05
            self.axes.set_ylim(y1, y2)

        if opt==1: # Connectivity

            # get slider values for interactive display
            S = []
            for ss in self.funcsliders:
                S.append(ss.value()/1000.)
            S = np.array(S)

            F = self.funcsliders[0].value() / 10. # frequency of signal

            print "Frequency: %.2f" % F

            # Noise
            N = 3*self.funcsliders[1].value() # noise for signal delay

            ## Signal time courses (samples, conditions, trials)
            data = np.zeros([n,2,self.trials_n])
            
            # get text for signal formula
            sig_str = unicode( self.functext[opt][0].text() )

            # get text for phase differences
            phase_str = unicode( self.functext[opt][1].text() )
            
            # create time courses per trial
            # 2 signals per trial with time shift
            delay_x = [] # delays (ms) across trials
            for t in range(0,self.trials_n):
                x = x_ori       
                data[:,0,t] = np.array( eval(sig_str) )

                 # add delay with noise to data (N controlled by slider)
                delay_x.append(np.float( phase_str ) + N*np.random.randn())
                x = x_ori + delay_x[-1]
                data[:,1,t] = np.array( eval(sig_str) )

                pp = 2*t # plot window
                self.trial_axes[pp].plot(x_ori,data[:,0,t],linewidth=0.5)
                self.trial_axes[pp].plot(x_ori,data[:,1,t],linewidth=0.5)

            print "Mean/Std delay (ms): %.f / %.f" % (np.mean(delay_x), np.std(delay_x))

            
            data_fft = fft(data,axis=0) # FFT along first axis
            data_fft = data_fft[0:int(np.floor(n/2.)),:,:]
            data_psd = np.abs(data_fft)**2 / (sfreq*n)
            data_psd[1:-2,:,:] = 2*data_psd[1:-2,:,:]

            mean_psd = np.mean(data_psd,axis=2) # over trials
            mean_psd = np.mean(mean_psd,axis=1) # over conditions
            m_idx = np.argmax(mean_psd) # peak frequency for coherence

            # (cross) spectral density between conditions
            cross_1 = np.multiply(data_fft[:,0,:], data_fft[:,0,:].conj())
            cross_2 = np.multiply(data_fft[:,1,:], data_fft[:,1,:].conj())
            cross_12 = np.multiply(data_fft[:,0,:], data_fft[:,1,:].conj())

            # Coherency
            coh = np.divide(cross_12,np.sqrt(np.multiply(cross_1,cross_2)))

            # magnitude squared coherence
            coh2 = np.abs(coh)

            freqs = np.linspace(0,int(np.floor(sfreq/2.)),int(np.round(n/2.)))
            print "Maximum frequency: ", freqs[m_idx]
            freqs = freqs / (freqs[-1]/2) # normalise because it'll be plotted with vectors
            freqs = freqs - freqs[0] # centre around 0

            # real and imaginary parts of spectrum (normalised)
            r_p = np.divide( np.real(data_fft), np.abs(data_fft) )
            i_p = np.divide( np.imag(data_fft), np.abs(data_fft) )

            # mean coherency at peak frequency across trials
            mean_coh = coh[m_idx,:].mean()

            R = [] # for lagend plotting
            legend_txt = [] # legend for trials in summary plot
            for nn in range(0,self.trials_n):
                pp = 2*nn + 1 # plot window

                # plot into separate sub-plots across trials
                self.trial_axes[pp].plot(freqs,data_psd[:,0,nn],linewidth=0.5,c="lightgrey")
                self.trial_axes[pp].plot(freqs,data_psd[:,1,nn],linewidth=0.5,c="lightgrey")

                self.trial_axes[pp].arrow(0,0,r_p[m_idx,0,nn],i_p[m_idx,0,nn],fc="blue", ec="blue", linewidth=0.5)
                self.trial_axes[pp].arrow(0,0,r_p[m_idx,1,nn],i_p[m_idx,1,nn],fc="green",ec="green", linewidth=0.5)

                # plot coherency in trial
                self.trial_axes[pp].arrow(0,0,np.real(coh[m_idx,nn]),np.imag(coh[m_idx,nn]),fc="black",ec="black")
                
                self.trial_axes[pp].set_xlim([-1.1,1.1])
                self.trial_axes[pp].set_ylim([-1.1,1.1])

                # plot coherencies across trials and mean into one window
                c = plot_colors[nn]
                self.axes2.arrow(0,0,np.real(coh[m_idx,nn]),np.imag(coh[m_idx,nn]),linewidth=0.5,fc=c,ec=c)
                                
                R.append(Rectangle((0,0), 1, 1, fc=c)) # dummy for legend plotting (legend doesn't work for arrow())
                legend_txt.append("%d" % (nn+1)) # start at 1
            
            # plot mean coherency across trials
            self.axes2.arrow(0,0,np.real(mean_coh),np.imag(mean_coh),fc="black",ec="black",linewidth=2)
            self.axes2.set_xlim([-1.1,1.1])
            self.axes2.set_ylim([-1.1,1.1])
            self.axes2.legend(R,legend_txt,prop={'size': 3})


        if opt==2: # Time-Frequency

            # get slider values for interactive display
            S = []
            for ss in self.funcsliders:
                S.append(ss.value())
            S = np.array(S)
            S[1] = S[1]*10. # used for Fmax in TF plot

            F = self.funcsliders[0].value() # frequency for signal

            # get text for wavelet frequencies
            freqs_str = unicode( self.functext[opt][1].text() )

            freqs = np.array( eval(freqs_str) )
            n_f = len(freqs)

            # get signal for wavelet analysis
            sig_str = unicode( self.functext[opt][0].text() )

            data = np.zeros([1,n])
            data[0,:] = np.array( eval(sig_str) )

            # Plot signal
            self.axes.plot(x, data[0,:], linewidth=1, c='black')

            n_cycles = np.zeros(n_f)
            n_cycles[freqs<=5] = 2
            n_cycles[(freqs>5) & (freqs<=20)] = 3
            n_cycles[freqs>20] = 7

            print sfreq
            Ws = morlet_mne(sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, zero_mean=True)
            
            # plot some wavelets
            n_plot = 3 # how many example wavelets to plot
            f_p = int(np.floor(n_f/3.)) # up to which frequency to plot wavelets
            s_d = x[1]-x[0] # sample distance
            legend = ['signal'] # figure legend for frequencies, first line is signal
            # plot up to third of max frequency, otherwise too small
            for ii in range(0,f_p,f_p/n_plot):
                W = Ws[np.round(ii)]
                n_w = len(W) # to be symmetrical around zero
                width = (n_w-1)*s_d
                x_w = np.linspace(-width/2,+width/2,n_w)                
                legend.append("%.2fHz" % freqs[ii])
                self.axes.plot(x_w, np.real(W), linewidth=0.4, c=plot_colors[len(legend)-1])
            self.axes.legend(legend, loc=2, prop={'size': 4})

            out = cwt(data, Ws)
            
            self.axes2.imshow(np.abs(out[0,:,:]), aspect='auto', extent=[x[0],x[-1], freqs[0],freqs[-1]],
                                origin='lower')

        # self.fig.tight_layout()
        self.canvas.draw()


    def plot_kernel(self):
        # plot kernel for convolution movie
        global x_global
        global sfreq

        # leave loop if end of steps reached
        if self.step > len(self.steps)-1:
            self.show_movie = 0
            self.timer.stop()
            return
        
        self.axes.clear()

        # variables available to external text function
        ss = self.steps[self.step] # current x-value for convolution
        x_ori = self.x
        y = self.y
        FWHM = self.FWHM        
        steps = self.steps

        # plot signal time course
        self.axes.plot(self.x,y)

        # compute and plot filter kernel
        opt = self.display_option
        txt_str = unicode( self.functext[opt][1].text() )

        # # compute convolution values for all x-values (but only plot up to current step)
        # if self.step == 0:

        #     self.y_conv = [] # convolution time series
        #     self.filt_kern = [] # moving filter kernels at different time points
            
        #     for xx in x_ori:
        #         x = x_ori - xx # move kernel along x-axis
        #         if txt_str == '':
        #             filt_kern = np.exp(-x**2/(2*FWHM**2)) # filter kernel            
        #         else:
        #             filt_kern = np.array( eval(txt_str) )

        #         filt_kern = filt_kern / np.sum(filt_kern) # normalise kernel to unit area

        #         self.y_conv.append(y.dot(filt_kern))

        #         # keep filter kernels at step points                
        #         samp_dist = 1000./self.SF # sample distance (ms)
        #         if np.min(np.abs(steps-xx)) < samp_dist/2.: # a hack to find step points
        #             self.filt_kern.append(filt_kern)
                
        ss_idx = np.argmin(np.abs(x_ori-ss)) # index up to which to plot
        from_idx = np.argmin(np.abs(x_ori-self.steps[0])) # index from which to plot convolution

        max_y = np.max(self.y)
        filt_kern = (max_y/np.max(self.filt_kern[self.step]))*self.filt_kern[self.step]

        # plot current filter kernel
        self.axes.plot(x_ori,filt_kern)

        self.axes.set_xlim(self.x_lim[0], self.x_lim[1])            

        y1 = self.scale_fac*min(np.r_[y,filt_kern,0])*1.05
        y2 = self.scale_fac*max(np.r_[y,filt_kern,0])*1.05
        self.axes.set_ylim(y1, y2)

        # plot convolution up to current step
        self.x_conv = x_ori[from_idx:ss_idx] # keep for later plotting

        self.y_conv_plot = self.y_conv[from_idx:ss_idx]
        self.axes.plot(self.x_conv, self.y_conv_plot)        
            
        # increase step, cancel loop when end reached        
        self.ss = self.ss + self.stepsize
        self.canvas.draw()

        self.step = self.step + 1

        
    def on_edit(self):
        """ refreshes display when editing of textboxes finished
        """

        if hasattr(self, 'y_conv_plot'):
            delattr(self, 'y_conv_plot') # remove convolution from plot
            delattr(self, 'x_conv')

        self.on_draw()


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
        self.show_movie = 0 # by default don't show moving filter kernel
       
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
        if self.display_option==0: # filter and convolution
            self.axes = self.fig.add_subplot(211)
            self.axes.grid(True, 'major')
            self.axes.tick_params(axis='both', which='major', labelsize=6)

            self.axes2 = self.fig.add_subplot(212)                        
            # self.axes2.grid(True, 'major')
            self.axes2.tick_params(axis='both', which='major', labelsize=6)            

            # self.axes3 = self.fig.add_subplot(224)
            # self.axes3.grid(True, 'major')
            # self.axes3.tick_params(axis='both', which='major', labelsize=6)
            # self.axes3.set_title("Kernel Spectrum", {'fontsize': 6})
        elif self.display_option==1: # Coherence
            self.trials_n = 5 # number of simulated signal trials
            self.trial_axes = []
            for nn in range(0,self.trials_n):
                # axes for signals and polar representations
                self.trial_axes.append( self.fig.add_subplot(self.trials_n, 3, 3*nn+1) )
                self.trial_axes.append( self.fig.add_subplot(self.trials_n, 3, 3*nn+2) )

            self.axes2 = self.fig.add_subplot(233)
            self.axes2.grid(True, 'major')
            self.axes2.tick_params(axis='both', which='major', labelsize=6)            
            
        elif self.display_option==2: # Time-Frequency

            self.axes = self.fig.add_subplot(211)
            self.axes.tick_params(axis='both', which='major', labelsize=6)

            self.axes2 = self.fig.add_subplot(212)
            self.axes2.tick_params(axis='both', which='major', labelsize=6)

        
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
            self.textbox_lim.setText('-1000 1000 5') # x-axis scale of signal time course in ms
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
            self.comboBox.addItem("Connectivity")
            self.comboBox.addItem("TimeFrequency")

            self.comboBox.activated[str].connect(self.display_method)       

        ## Slider
        if not(hasattr(self, 'slider_scale')):
            self.slider_label = QLabel('Scaling:')
            self.slider_scale = QSlider(Qt.Horizontal)
            self.slider_scale.setRange(1, 1000)
            self.slider_scale.setValue(200)
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


        if not(hasattr(self, 'funcsliders')):
            # sliders
            self.fslidelabels = []
            self.fslidelabels.append(QLabel('S0')) # will be specified below

            # add one additional slider
            self.fslidelabels.append(QLabel('S1:'))

            self.funcsliders = []
            # Slider
            self.funcsliders.append(QSlider(Qt.Horizontal))
            self.funcsliders[0].setRange(0, 100)
            self.funcsliders[0].setMinimumWidth(200)
            self.funcsliders[0].setValue(10)
            self.funcsliders[0].setTracking(True)
            self.funcsliders[0].setTickPosition(QSlider.TicksBelow)
            self.funcsliders[0].setTickInterval(50)
            self.connect(self.funcsliders[0], SIGNAL('valueChanged(int)'), self.on_draw)

            # Slider
            self.funcsliders.append(QSlider(Qt.Horizontal))
            self.funcsliders[1].setRange(0, 100)
            self.funcsliders[1].setMinimumWidth(200)
            self.funcsliders[1].setValue(20)
            self.funcsliders[1].setTracking(True)
            self.funcsliders[1].setTickPosition(QSlider.TicksBelow)
            self.funcsliders[1].setTickInterval(50)
            self.connect(self.funcsliders[1], SIGNAL('valueChanged(int)'), self.on_draw)


        ## initialise or add TEXT BOXES       
        
        options = [0,1,2] # number of display option        

        if not(hasattr(self, 'functext')): # at the very beginning
            self.functext = []
            self.funclabels = []
            [self.functext.append([]) for ii in options] # text boxes for functions of different options
            [self.funclabels.append([]) for ii in options] # correponding display labels
 
        # If text specified, keep it
        # apparently widgets get deleted by their parents
        if self.functext[0]==[]:
            txt0 = "box(0,W)"
            txt1 = "morlet(F, x)"
        else:
            txt0 = self.functext[0][0].text()
            txt1 = self.functext[0][1].text()

        self.functext[0] = []
        self.functext[0].append(QLineEdit(txt0))
        self.functext[0].append(QLineEdit(txt1))

        self.functext[0][0].setMinimumWidth(200)
        self.connect(self.functext[0][0], SIGNAL('editingFinished ()'), self.on_edit)
        self.functext[0][1].setMinimumWidth(200)
        self.connect(self.functext[0][1], SIGNAL('editingFinished ()'), self.on_edit)

        self.funclabels[0] = []
        self.funclabels[0].append(QLabel('Signal:'))
        self.funclabels[0].append(QLabel('Kernel:'))

        
        if self.functext[1]==[]:
            txt0 = "sin(2*pi*(F/1000.)*x)"
            txt1 = "50"
        else:
            txt0 = self.functext[1][0].text()
            txt1 = self.functext[1][1].text()

        self.functext[1] = []
        self.functext[1].append(QLineEdit(txt0))
        self.functext[1].append(QLineEdit(txt1))

        self.functext[1][0].setMinimumWidth(200)
        self.connect(self.functext[1][1], SIGNAL('editingFinished ()'), self.on_edit)
        self.functext[1][1].setMinimumWidth(200)
        self.connect(self.functext[1][1], SIGNAL('editingFinished ()'), self.on_edit)

        self.funclabels[1] = []
        self.funclabels[1].append(QLabel('Signal:'))
        self.funclabels[1].append(QLabel('Delay (ms):'))            


        if self.functext[2]==[]:
            txt0 = "sin(2*pi*(F/1000.)*x)"
            txt1 = "arange(5,100,2)"
        else:
            txt0 = self.functext[2][0].text()
            txt1 = self.functext[2][1].text()

        self.functext[2] = []
        self.functext[2].append(QLineEdit(txt0))
        self.functext[2].append(QLineEdit(txt1))

        self.functext[2][0].setMinimumWidth(200)
        self.connect(self.functext[2][0], SIGNAL('editingFinished ()'), self.on_edit)
        self.functext[2][1].setMinimumWidth(200)
        self.connect(self.functext[2][1], SIGNAL('editingFinished ()'), self.on_edit)

        self.funclabels[2] = []
        self.funclabels[2].append(QLabel('Signal:'))
        self.funclabels[2].append(QLabel('Freqs:'))

            
        if self.display_option==0: # Filter
            self.fslidelabels[0].setText('Width (W):')
            self.fslidelabels[1].setText('Freq (F):')
        elif self.display_option==1: # Connectivity
            self.fslidelabels[0].setText('Freq (F):')
            self.fslidelabels[1].setText('Noise (N):')
            self.funcsliders[0].setValue(20)
            self.funcsliders[1].setValue(0)

        elif self.display_option==2:
            self.fslidelabels[0].setText('Freq (F):')
            self.fslidelabels[1].setText('S[1]:')


        # add function text boxes to box
        opt = self.display_option # current display option       

        functext = self.functext[opt] # text box and label for current option
        funclabels = self.funclabels[opt]        
        
        for [fi,ff] in enumerate(functext): # labels and text boxes            
            w = self.funclabels[opt][fi]
            hbox2.addWidget(w) # add label text
            
            hbox2.addWidget(ff) # add corresponding text box
            hbox2.setAlignment(ff, Qt.AlignVCenter)
        
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