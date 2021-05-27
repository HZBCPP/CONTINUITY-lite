#!/usr/bin/env python3
import sys 
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog

from interface_functions_visualization import *

##########################################################################################################################################
'''  
     CONTINUITY QC interface : 
     Check registration and tractography results and display circle and brain connectome
'''  
##########################################################################################################################################

if __name__ == '__main__':

    print("CONTINUITY QC interface")

    app = QtWidgets.QApplication(sys.argv)
    window = Ui_visu()
    window.show()
    app.exec_()