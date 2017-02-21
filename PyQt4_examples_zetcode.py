#!/imaging/local/software/anaconda/latest/x86_64/bin/python
"""
ZetCode PyQt4 tutorial 

In this example, we create a simple
window in PyQt4.

author: Jan Bodnar
website: zetcode.com 
last edited: October 2011
"""

# http://zetcode.com/gui/pyqt4/

#!/usr/bin/python
# -*- coding: utf-8 -*-


###
# simple window
###

# import sys
# from PyQt4 import QtGui


# def main():
    
#     app = QtGui.QApplication(sys.argv)

#     w = QtGui.QWidget()
#     w.resize(250, 150)
#     w.move(300, 300)
#     w.setWindowTitle('Simple')
#     w.show()
    
#     sys.exit(app.exec_())


# if __name__ == '__main__':
#     main()



###
# picture
###

#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ZetCode PyQt4 tutorial 

In this example, we dispay an image
on the window. 

author: Jan Bodnar
website: zetcode.com 
last edited: September 2011
"""

import sys
from PyQt4 import QtGui, QtCore

class Example(QtGui.QWidget):
    
    def __init__(self):
        super(Example, self).__init__()
        
        self.initUI()
        
    def initUI(self):      

        hbox = QtGui.QHBoxLayout(self)
        pixmap = QtGui.QPixmap("redrock.png")

        lbl = QtGui.QLabel(self)
        lbl.setPixmap(pixmap)

        hbox.addWidget(lbl)
        self.setLayout(hbox)
        
        self.move(300, 200)
        self.setWindowTitle('Red Rock')
        self.show()        
        
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()    