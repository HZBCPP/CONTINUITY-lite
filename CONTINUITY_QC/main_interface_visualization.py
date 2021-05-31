#!/usr/bin/env python3
import sys 
import argparse
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

	# *****************************************
	# Parameters
	# *****************************************

	parser = argparse.ArgumentParser(description='CONTINUITY script for visualization interface')
	parser.add_argument("default_config_filename", help = "File with all efault parameters", type = str) 
	parser.add_argument("user_json_filename", help = "File with all parameters given by the user", type = str) 

	args = parser.parse_args()


	user_filename = args.user_json_filename
	default_config_filename = args.default_config_filename

	qt_args = []
	qt_args.append(sys.argv[-1])
	qt_args.append(default_config_filename)
	qt_args.append(user_filename)
	sys.argv = qt_args

	#print(qt_args)
	app = QtWidgets.QApplication(sys.argv)
	window = Ui_visu()
	window.show()
	app.exec_()