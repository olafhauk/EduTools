# PyQt tutorials
# https://wiki.python.org/moin/PyQt/Tutorials

# PyQt4 examples
# https://pythonspot.com/en/pyqt4/


###
# Buttons
###

# import sys
# from PyQt4.QtCore import pyqtSlot
# from PyQt4.QtGui import *
 
# # create our window
# app = QApplication(sys.argv)
# w = QWidget()
# w.setWindowTitle('Button click example @pythonspot.com')
 
# # Create a button in the window
# btn = QPushButton('Click me', w)
 
# # Create the actions
# @pyqtSlot()
# def on_click():
#     print('clicked')
 
# @pyqtSlot()
# def on_press():
#     print('pressed')
 
# @pyqtSlot()
# def on_release():
#     print('released')
 
# # connect the signals to the slots
# btn.clicked.connect(on_click)
# btn.pressed.connect(on_press)
# btn.released.connect(on_release)
 
# # Show the window and run the app
# w.show()


###
# Progress Bar
###

# import sys
# from PyQt4.QtGui import *
# from PyQt4.QtCore import *
# from PyQt4.QtCore import pyqtSlot,SIGNAL,SLOT
 
# class QProgBar(QProgressBar):
 
#     value = 0
 
#     @pyqtSlot()
#     def increaseValue(progressBar):
#         progressBar.setValue(progressBar.value)
#         progressBar.value = progressBar.value+1
 
# # Create an PyQT4 application object.
# a = QApplication(sys.argv)       
 
# # The QWidget widget is the base class of all user interface objects in PyQt4.
# w = QWidget()
 
# # Set window size. 
# w.resize(320, 240)
 
# # Set window title  
# w.setWindowTitle("PyQT4 Progressbar @ pythonspot.com ") 
 
# # Create progressBar. 
# bar = QProgBar(w)
# bar.resize(320,50)    
# bar.setValue(0)
# bar.move(0,20)
 
# # create timer for progressBar
# timer = QTimer()
# bar.connect(timer,SIGNAL("timeout()"),bar,SLOT("increaseValue()"))
# timer.start(400) 
 
# # Show window
# w.show()


###
# Combo box
###

import sys
from PyQt4.QtGui import *
 
# Create an PyQT4 application object.
a = QApplication(sys.argv)       
 
# The QWidget widget is the base class of all user interface objects in PyQt4.
w = QMainWindow()
 
# Set window size. 
w.resize(320, 100)
 
# Set window title  
w.setWindowTitle("PyQT Python Widget!") 
 
# Create combobox
combo = QComboBox(w)
combo.addItem("Python")
combo.addItem("Perl")
combo.addItem("Java")
combo.addItem("C++")
combo.move(20,20)
 
# Show window
w.show()

sys.exit(a.exec_())