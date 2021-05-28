#!/usr/bin/env python3
import json
import os 
import sys 
import subprocess
from termcolor import colored
import time
import datetime

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QCheckBox, QGridLayout, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib.widgets import Cursor
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne
from mne.viz import circular_layout, plot_connectivity_circle
import nrrd

from functools import partial

sys.path.insert(1, os.path.split(os.getcwd())[0])  # if you want to open the second interface alone
sys.path.insert(1, os.getcwd())                    # if you want to open the second interface with the first interface

# sys.path.insert(1, '/proj/NIRAL/tools/CONTINUITY/')

from CONTINUITY_functions import *


##########################################################################################################################################
'''  
     CONTINUITY QC interface: functions file
'''  
##########################################################################################################################################

class Ui_visu(QtWidgets.QTabWidget):

    # *****************************************
    # Initialisation
    # *****************************************

    def __init__(self):
        super(Ui_visu, self).__init__()
    
        if os.path.exists('./CONTINUITY_QC/interface_visualization.ui'):  # if you open the second interface with the first interface
            uic.loadUi('./CONTINUITY_QC/interface_visualization.ui', self)
        else:                                                             # if you open the second interface alone
            uic.loadUi('./interface_visualization.ui', self)

        # Write default values on interface:    
        self.setup_default_values()

        self.show()



    # *****************************************
    # Setup default value
    # *****************************************

    def setup_default_values(self):
        # Open json files: 
        global user_json_filename
        user_json_filename = "../CONTINUITY_ARGS/args_main_CONTINUITY.json"
        if os.path.exists("./CONTINUITY_ARGS/args_main_CONTINUITY.json"): # if you open the second interface with the first interface
            user_json_filename = "./CONTINUITY_ARGS/args_main_CONTINUITY.json"
       
        global default_json_filename
        default_json_filename = "../CONTINUITY_ARGS/args_main_CONTINUITY_completed_test.json" #./CONTINUITY_ARGS/args_setup.json"
        if os.path.exists("./CONTINUITY_ARGS/args_main_CONTINUITY_completed_test.json"):  # if you open the second interface with the first interface
            default_json_filename = "./CONTINUITY_ARGS/args_main_CONTINUITY_completed_test.json" #./CONTINUITY_ARGS/args_setup.json"
 
        
        # Json file which contains values given by the user: 
        with open(user_json_filename, "r") as user_Qt_file:
            global json_user_object
            json_user_object = json.load(user_Qt_file)

        # Json file which contains defaults values
        with open(default_json_filename, "r") as default_Qt_file:
            global json_setup_object
            json_setup_object = json.load(default_Qt_file)

        # Setup default path to access of created files for Slicer and Slicer:
        self.OUTPUT_path_textEdit.setText(json_user_object['Parameters']["OUT_PATH"]["value"])
        self.slicer_textEdit.setText(json_user_object['Executables']["slicer"]["value"])


        # Setup default path to visualize connectivity matrix and brain/circle connectome:
        path = os.path.join(json_user_object['Parameters']["OUT_PATH"]["value"], json_user_object['Arguments']["ID"]["value"], "Tractography", 'new_parcellation_table')
        self.parcellation_table_textEdit.setText(path)

        global overlapName
        overlapName = ""
        if json_user_object['Parameters']["overlapping"]["value"]: 
            overlapName = "_overlapping" 

        # Setup default path to save the connectivity matrix with a specific name:
        global loopcheckName
        loopcheckName = ""
        if json_user_object['Parameters']["loopcheck"]["value"]: 
            loopcheckName = "_loopcheck"
            
        path = os.path.join(json_user_object['Parameters']["OUT_PATH"]["value"], json_user_object['Arguments']["ID"]["value"], "Tractography" )
                                                                                                                                #, "Network" + overlapName + loopcheckName)
        self.connectivity_matrix_textEdit.setText(path)

        # Colormap circle connectome: strength of each node
        self.layout_colormap = QGridLayout()
        self.colormap_circle_groupBox.setLayout(self.layout_colormap)

        # Circle connectome
        self.Layoutcircle = QGridLayout()
        self.circle_connectome_groupBox.setLayout(self.Layoutcircle)
        self.fig_file_textEdit.setText(json_user_object['Parameters']["OUT_PATH"]["value"])   

        # Normalize matrix
        self.Layout_normalize_matrix = QGridLayout()
        self.normalize_matrix_groupBox.setLayout(self.Layout_normalize_matrix)
        self.path_normalize_matrix_textEdit.setText(json_user_object['Parameters']["OUT_PATH"]["value"])

        # 2D brain connectome
        self.Layout_brain_connectome = QGridLayout()
        self.brain_connectome_groupBox.setLayout(self.Layout_brain_connectome) 
        self.Layout_brain_connectome.setContentsMargins(0, 0, 0, 0) 

        self.num_slice_axial_label.setText("Slice " + str(self.num_slice_axial_horizontalSlider.value()))
        self.num_slice_sagittal_label.setText("Slice " + str(self.num_slice_sagittal_horizontalSlider.value()))
        self.num_slice_coronal_label.setText("Slice " + str(self.num_slice_coronal_horizontalSlider.value()))

        # 3D brain connectome
        self.Layout_brain_connectome_3D = QGridLayout()
        self.brain_connectome_3D_groupBox.setLayout(self.Layout_brain_connectome_3D)   

        # VTK fiber
        self.Layout_vtk_fiber = QGridLayout()
        self.Layout_vtk_fiber_groupBox.setLayout(self.Layout_vtk_fiber)



    # *****************************************
    # Write in user json file
    # *****************************************  
    
    def update_user_json_file():
        with open(user_json_filename, "w+") as user_Qt_file:
            user_Qt_file.write(json.dumps(json_user_object, indent=4)) 



    # *****************************************
    # Function to run a specific command
    # *****************************************

    def run_command(text_printed, command):
        # Display command 
        print(colored("\n"+" ".join(command)+"\n", 'blue'))
        # Run command and display output and error
        run = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = run.communicate()
        print(text_printed, "out: ", colored("\n" + str(out) + "\n", 'green')) 
        print(text_printed, "err: ", colored("\n" + str(err) + "\n", 'red'))



    # *****************************************
    # Executables: write the path for Slicer (given by the user) in the user information json file
    # *****************************************  

    def slicer_path_button_clicked(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()" , "", "ALL Files (*)", options=QFileDialog.Options())
        if fileName:
            self.slicer_textEdit.setText(fileName)
            json_user_object['Executables']["slicer"]["value"] = fileName
            Ui_visu.update_user_json_file()



    # *****************************************
    # Write the path of output folder (i.e output of tractography) in user json file
    # ****************************************

    def OUTPUT_path_button_clicked(self):
        DirName= QtWidgets.QFileDialog.getExistingDirectory(self)
        if DirName:
            self.OUTPUT_path_textEdit.setText(DirName)
            json_user_object['Parameters']["OUT_PATH"]["value"] = DirName
            Ui_visu.update_user_json_file()



    # *****************************************
    # Open a script to open Slicer with specific parameters and without: 
    # *****************************************

    def open_slicer_clicked(self): 
        if os.path.exists("./CONTINUITY_QC/slicer_QC.py"):
            Ui_visu.run_command("Open slicer with specific parameters", [sys.executable, "./CONTINUITY_QC/slicer_QC.py", user_json_filename])
        else: 
            Ui_visu.run_command("Open slicer with specific parameters", [sys.executable, "./slicer_QC.py", user_json_filename])


    def open_slicer_only(self):
        Ui_visu.run_command("Open Slicer without configuration", [json_user_object['Executables']["slicer"]["value"]])



    # *****************************************
    # Convert name of data in comboBox (QT) to name of param in script
    # *****************************************

    def convert_name_data(Qt_param):
        dict_param = {"B0":"B0", "T1 registered":"T1_registered", "T2 registered":"T2_registered", "FA":"FA", "AD":"AD", "labeled image":"labeled_image"}
        return dict_param[Qt_param]



    # *****************************************
    # Save type of files which will displayed in Slicer (in the view controllers)
    # *****************************************  

    def update_param(self, comboBox_name):
        json_user_object['View_Controllers'][comboBox_name]["value"] = Ui_visu.convert_name_data( eval("self." + comboBox_name + "_comboBox.currentText()")) 
        print(eval("self." + comboBox_name + "_comboBox.currentText()"))
        Ui_visu.update_user_json_file()


        
    # *****************************************
    # Activated if an view controllers parameters is modify
    # ***************************************** 

    def view_controllers_params_clicked(self):
        comboBox_name = self.sender().objectName()[:-9] # 9 caracter in '_comboBox'
        Ui_visu.update_param(self, comboBox_name) 



    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # *************************************  Connectivity matrix visualization  ************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************

        
    # *****************************************
    # Select the parcellation table file WITH subcortical regions  
    # ***************************************** 

    def parcellation_table_pushButton_clicked(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()" , "", "ALL Files (*)", options=QFileDialog.Options())
        if fileName:
            self.parcellation_table_textEdit.setText(fileName)
            json_user_object['Arguments']["PARCELLATION_TABLE"]["value"] = fileName
            Ui_visu.update_user_json_file()



    # *****************************************
    # Display the connectivity matrix with a specific normalization
    # ***************************************** 

    def display_normalize_matrix_pushButton_clicked(self):
        matrix = os.path.join(self.connectivity_matrix_textEdit.toPlainText(), "fdt_network_matrix")

        # Create the last part of the title: 
        overlapName = ""
        if json_user_object['Parameters']["overlapping"]["value"]: 
            overlapName = "_overlapping"

        end_name = ' without Loopcheck and without Overlapping'
        if   len(overlapName)>3 and len(loopcheckName)>3: end_name = ' with Loopcheck and with Overlapping'
        elif len(overlapName)<3 and len(loopcheckName)>3: end_name = ' without Loopcheck and with Overlapping'
        elif len(overlapName)>3 and len(loopcheckName)<3: end_name = ' with Loopcheck and without Overlapping'

        # Remove previous plot:
        for i in reversed(range(self.Layout_normalize_matrix.count())): 
            self.Layout_normalize_matrix.itemAt(i).widget().setParent(None)

        # Create global configuration for figure:
        self.fig_normalize_matrix = plt.figure(num=None)
        self.canvas = FigureCanvas(self.fig_normalize_matrix)
        self.Layout_normalize_matrix.addWidget(self.canvas)

        # Add figure and axes:
        global ax_matrix
        ax_matrix = self.fig_normalize_matrix.add_subplot(1,1,1)
        ax_matrix.set_xlabel('Seeds')
        ax_matrix.set_ylabel('Targets')

        # Title:
        #start = 'Connectivity matrix for ' + json_user_object['Arguments']["ID"]["value"] + "\n  "
        #self.fig_normalize_matrix.suptitle(start +end_name, fontsize=10)
        self.fig_normalize_matrix.set_zorder(1)

        # Specific normalization: 
        global a_matrix 
        if self.type_of_normalization_comboBox.currentText()   == "No normalization":         a_matrix = no_normalization(matrix)
        elif self.type_of_normalization_comboBox.currentText() == "Whole normalization":      a_matrix = whole_normalization(matrix)
        elif self.type_of_normalization_comboBox.currentText() == "Row region normalization": a_matrix = row_region_normalization(matrix)
        elif self.type_of_normalization_comboBox.currentText() == "Row column normalization": a_matrix = row_column_normalization(matrix)

        # Specific symmetrization: 
        if self.type_of_symmetrization_comboBox.currentText()   == "Average": a_matrix = average_symmetrization(a_matrix)
        elif self.type_of_symmetrization_comboBox.currentText() == "Maximum": a_matrix = maximum_symmetrization(a_matrix)
        elif self.type_of_symmetrization_comboBox.currentText() == "Minimum": a_matrix = minimum_symmetrization(a_matrix)

        min_a, max_a  = (np.min(a_matrix), np.max(a_matrix))

        self.min_a_norm_label.setText(str(min_a))
        self.max_a_norm_label.setText(str("{:e}".format(max_a)))

        # Plotting the correlation matrix:
        vmin, vmax = (0,0)
        check_before_display = True
        
        if self.vmin_vmax_percentage_checkBox.isChecked(): 
            if not(self.vmax_normalize_matrix_spinBox.value() <=  100 and self.vmin_normalize_matrix_spinBox.value() >= 0):
                self.error_label.setText('<font color="red">Please select 2 values between 0 to 100</font> ')
                check_before_display = False
        
        elif self.vmin_vmax_real_values_checkBox.isChecked(): 
            max_a = float(max_a)
            if not(self.vmax_normalize_matrix_spinBox.value() <=  float(max_a) and self.vmin_normalize_matrix_spinBox.value() >= min_a):
                self.error_label.setText( '<font color="red">Please select 2 values between ' + str(min_a) + ' to ' + str("%.7f" % (max_a))+ '</font>'  )
                check_before_display = False

        elif self.vmin_vmax_regions_checkBox.isChecked():
            nb = np.shape(a_matrix)[0] #number of regions
            max_region_display = float(1/(nb*(nb-1)))  
             
            if not(self.vmax_normalize_matrix_spinBox.value() <=  max_region_display and self.vmin_normalize_matrix_spinBox.value() >= min_a):
                self.error_label.setText('<font color="red">Please select 2 values between ' + str(min_a) + ' to ' + str("%.7f" % (max_region_display)) + '</font>')
                check_before_display = False

        else:
            self.error_label.setText( '<font color="red">Please select a type of range </font>'  )


        if check_before_display:
            self.error_label.setText(' ')

            im = ax_matrix.imshow(a_matrix, interpolation='nearest', vmin = self.vmin_normalize_matrix_spinBox.value(), 
                                                                      vmax = self.vmax_normalize_matrix_spinBox.value())

            divider = make_axes_locatable(ax_matrix)
            cax = divider.new_vertical(size="3%", pad=0.7, pack_start=True)
            self.fig_normalize_matrix.add_axes(cax)
            self.fig_normalize_matrix.colorbar(im, cax=cax, orientation="horizontal")


            
            # Defining the cursor
            cursor = Cursor(ax_matrix, horizOn=True, vertOn=True, useblit=True, color = 'red', linewidth = 1)

            # Creating an annotating box
            global annot
            annot = ax_matrix.annotate("", xy=(0,0), xytext=(-20,-50), xycoords='data',textcoords="offset points",
                    bbox=dict(boxstyle='round4', fc='linen',ec='r',lw=2, alpha=1),
                    arrowprops=dict(arrowstyle='fancy'))

            annot.set_visible(False)
           
            global my_fig_matrix, lhor, lver
            my_fig_matrix = self.fig_normalize_matrix
            lhor = ax_matrix.axhline(0)
            lver = ax_matrix.axvline(0)

            lhor.set_ydata(-1)
            lver.set_xdata(-1)


            # Get the parcellation table with Cortical and Subcortical regions: 
            with open(os.path.join(self.parcellation_table_textEdit.toPlainText()), "r") as table_json_file:
                table_json_object = json.load(table_json_file)

            # Get data points for connected and unconnected points: 
            global list_name_matrix
            list_name_unordered, list_MatrixRow, list_name_matrix = ([], [], [])
           
            for key in table_json_object:    
                list_name_unordered.append(key["name"])
                list_MatrixRow.append(key["MatrixRow"])

            # Sort regions by VisuHierarchy number: 
            sorted_indices = np.argsort(list_MatrixRow)
            

            for i in range(len(list_MatrixRow)):
                index = sorted_indices[i]
                list_name_matrix.append(list_name_unordered[index])


            self.fig_normalize_matrix.tight_layout(pad=0)            

            self.fig_normalize_matrix.canvas.mpl_connect('button_press_event', self.cursor_mouse_move)



    def cursor_mouse_move(self,event):
       
        if not event.inaxes:
            return

        # *****************************************
        # Left click
        # *****************************************

        if event.button == 1:  
            x, y = event.xdata, event.ydata
            numrows, numcols = len(a_matrix[0]), len(a_matrix[1])

            col = int(x+0.5)
            row = int(y+0.5)

            if col>=0 and col<numcols and row>=0 and row<numrows:
                z = a_matrix[row][col]
                
            annot.xy = (x,y)
            text = "Column " + str(col) + ": " + str(list_name_matrix[col]) + " \nRow " + str(row) + ": " + str(list_name_matrix[row]) + " \nValue: " + str(z) 
            text_small = "Col: " + str(list_name_matrix[col]) + " \nRow: " + str(list_name_matrix[row])

            annot.set_text(text_small)
            annot.set_visible(True)

            # update the line positions
            lhor.set_ydata(y)
            lver.set_xdata(x)

            self.label_matrix.setText(text)

            self.fig_normalize_matrix.canvas.draw()


        # *****************************************
        # Left click
        # *****************************************
  

        elif event.button == 3: 
            print("test")

            annot.set_visible(False)

            lhor.set_ydata(-1)
            lver.set_xdata(-1)

            self.fig_normalize_matrix.canvas.draw()

            
    # *****************************************
    # Select the path to GET the connectivity matrix
    # ***************************************** 

    def get_connectivity_matrix_pushButton_clicked(self):
        DirName= QtWidgets.QFileDialog.getExistingDirectory(self)
        if DirName:
            self.connectivity_matrix_textEdit.setText(DirName)


    # *****************************************
    # Select the path to SAVE the connectivity matrix
    # ***************************************** 

    def connectivity_matrix_pushButton_clicked(self):
        DirName= QtWidgets.QFileDialog.getExistingDirectory(self)
        if DirName:
            self.path_normalize_matrix_textEdit.setText(DirName)



    # *****************************************
    # Save the connectivity matrix
    # ***************************************** 

    def save_normalize_matrix_pushButton_clicked(self):
        path_name = os.path.join(self.path_normalize_matrix_textEdit.toPlainText(), 'Connectivity_matrix_normalized_' + self.type_of_normalization_comboBox.currentText() 
                                                                                     + "_symmetrization_by_" + self.type_of_symmetrization_comboBox.currentText()+ '.pdf')
        # Save and display:
        self.fig_normalize_matrix.savefig(path_name, format='pdf')
        self.save_normalize_matrix_textEdit.setText("Figure saved here: " + path_name)


    # *****************************************
    # Checkbox to selecte the type of colormap range
    # ***************************************** 

    def checkBox_vmin_vmax_cliked(self): 
        checkbox_name = self.sender().objectName()

        if checkbox_name == "vmin_vmax_percentage_checkBox":
            if self.vmin_vmax_percentage_checkBox.isChecked(): 
                self.vmin_vmax_real_values_checkBox.setChecked(False)
                self.vmin_vmax_regions_checkBox.setChecked(False)
                Ui_visu.display_normalize_matrix_pushButton_clicked(self)


        elif checkbox_name == "vmin_vmax_real_values_checkBox":
            if self.vmin_vmax_real_values_checkBox.isChecked(): 
                self.vmin_vmax_percentage_checkBox.setChecked(False)
                self.vmin_vmax_regions_checkBox.setChecked(False)
                Ui_visu.display_normalize_matrix_pushButton_clicked(self)

        elif checkbox_name == "vmin_vmax_regions_checkBox":
            if self.vmin_vmax_regions_checkBox.isChecked(): 
                self.vmin_vmax_real_values_checkBox.setChecked(False)
                self.vmin_vmax_percentage_checkBox.setChecked(False)
                Ui_visu.display_normalize_matrix_pushButton_clicked(self)

        

    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # ********************************************************  Circle connectome  ********************************************************************* 
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************

    # *****************************************
    # Plot circle connectome
    # ***************************************** 

    def plot_circle_connectome(self):

        # *****************************************
        # Extract name of each regions and create a circular layout
        # *****************************************
       
        # Get the parcellation table with cortical and subcortical regions:
        with open(os.path.join(self.parcellation_table_textEdit.toPlainText()), "r") as table_json_file:
            table_json_object = json.load(table_json_file)

        # Get the number of separation required = number of different 'VisuHierarchy' label:  "VisuHierarchy": "seed.left."
        global label_names
        list_VisuHierarchy, label_names, VisuOrder_associated, VisuHierarchy_associated = ([], [], [], [])
        number_of_subcortical_regions = 0 

        for key in table_json_object:
            # List with all regions names:
            label_names.append(key["name"])
            VisuHierarchy_associated.append(key["VisuHierarchy"])


        # Build 'VisuOrder_associated' list and 'list_VisuHierarchy': 
        for key in table_json_object:
            # Subcortical regions: currently all with -1 and if I didn't do that I have an issue to display them after sort their key["VisuOrder"]
            if key["VisuOrder"] == -1: 
                number_of_subcortical_regions+= 1
                if key['name'].endswith('_R'):
                    VisuOrder_associated.append(int(len(label_names)+ number_of_subcortical_regions))
                else:  #'-L'
                    VisuOrder_associated.append(int(len(label_names)+30 + number_of_subcortical_regions))
            
            # Cortical regions: 
            else:
                VisuOrder_associated.append(int(key["VisuOrder"]))

            if key["VisuHierarchy"] not in list_VisuHierarchy:
                list_VisuHierarchy.append(key["VisuHierarchy"])

           
        # Build 'list_of_list_VisuHierarchy'
        list_of_list_VisuHierarchy = [[] for i in range(len(list_VisuHierarchy))]
        for key in table_json_object:
            index = list_VisuHierarchy.index(key["VisuHierarchy"])
            list_of_list_VisuHierarchy[index].append(key["name"])

        # Sort regions by VisuHierarchy number: 
        sorted_indices = np.argsort( VisuOrder_associated)

        # Build 'node_order' and 'VisuHierarchy_order'
        global node_order
        node_order, VisuHierarchy_order = ([],[])

        for i in range(len(VisuOrder_associated)):
            index = sorted_indices[i]
            node_order.append(label_names[index])
            VisuHierarchy_order.append(VisuHierarchy_associated[index])
        

        # Build 'list_boundaries' and 'list_name_boundaries': 
        global name_boundaries
        list_boundaries, name_boundaries, list_name_boundaries, i  = ([0], [], [], 0)

        while i < len(VisuHierarchy_order)-1:   
            current_elem = VisuHierarchy_order[i]
            next_elem = VisuHierarchy_order[i+1]

            if current_elem != next_elem: #boundary
                list_boundaries.append(i+1)
                list_name_boundaries.append(current_elem)
            i += 1


        # Get the middle of each number of element for each region 
        middle = [0 for x in range(len(list_of_list_VisuHierarchy))]
        for i in range(len(list_of_list_VisuHierarchy)):   
            middle[i] = int(len(list_of_list_VisuHierarchy[i])/2)


        # Build 'name_boundaries': only the name of each group of region
        list_of_cpt = [0 for x in range(len(list_VisuHierarchy))]
        for key in table_json_object:

            index = list_VisuHierarchy.index(key["VisuHierarchy"])
            list_of_cpt[index] += 1

            if list_of_cpt[index] == middle[index]: 
                name_boundaries.append(key["VisuHierarchy"])
            else: 
                name_boundaries.append("")

        # Create a circular layout:
        global node_angles
        node_angles = circular_layout(label_names, node_order, start_pos=90, group_boundaries=list_boundaries)
    

        # *****************************************
        # Get the normalize connectivity matrix
        # *****************************************
        
        matrix = os.path.join(self.connectivity_matrix_textEdit.toPlainText(), "fdt_network_matrix")

        # Specific normalization: 
        if self.type_of_normalization_circle_comboBox.currentText()   == "No normalization":         connectivity_score = no_normalization(matrix)
        elif self.type_of_normalization_circle_comboBox.currentText() == "Whole normalization":      connectivity_score = whole_normalization(matrix)
        elif self.type_of_normalization_circle_comboBox.currentText() == "Row region normalization": connectivity_score = row_region_normalization(matrix)
        elif self.type_of_normalization_circle_comboBox.currentText() == "Row column normalization": connectivity_score = row_column_normalization(matrix)

        # Specific symmetrization: 
        if self.type_of_symmetrization_circle_comboBox.currentText()   == "Average": connectivity_score = average_symmetrization(connectivity_score)
        elif self.type_of_symmetrization_circle_comboBox.currentText() == "Maximum": connectivity_score = maximum_symmetrization(connectivity_score)
        elif self.type_of_symmetrization_circle_comboBox.currentText() == "Minimum": connectivity_score = minimum_symmetrization(connectivity_score)
        
        # Transform a list of list into a numpy array:
        global connectivity_matrix
        connectivity_matrix = np.array(connectivity_score)

        global number_total_line
        number_total_line = np.count_nonzero(np.abs(connectivity_matrix)) #Doc plot_connectivity_circle: n_lines strongest connections (strength=abs(con))
        max_value = np.amax(connectivity_matrix)


        # *****************************************
        # Strenght of each node with a specific colormap
        # *****************************************

        # Remove previous plot for the colormap associated to node features: 
        for i in reversed(range(self.layout_colormap.count())): 
            self.layout_colormap.itemAt(i).widget().setParent(None)
        
        # New plot for the colormap associated to node features: 
        self.fig2 = plt.figure()
        self.canvas = FigureCanvas(self.fig2)
        self.layout_colormap.addWidget(self.canvas)
        
        # Set information for the colormap associated to node features: 
        ax = self.fig2.add_axes([0.1, 0.4, 0.8, 0.4]) # add_axes([xmin,ymin,dx,dy]) 
        vmax = self.vmax_colorbar_spinBox.value() / 100
        vmin = self.vmin_colorbar_spinBox.value() / 100
    
        # Display colorbar: 
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdBu),  cax=ax, orientation='horizontal', ticks=[0, vmax/4, vmax/2, 3*vmax/4, 1])

        # Set type of colormap to color each node: 
        RdBu = plt.cm.get_cmap('RdBu', 12)

        # Compute the strenght of each node: node strength is the sum of weights of links connected to the node
        if self.node_features_comboBox.currentText() == "Strength":
            # Sum of each column, each line and all elem: 
            waytotal_column = sum_column(connectivity_matrix)
            waytotal_line   = sum_line(connectivity_matrix)
            waytotal        = sum_all(connectivity_matrix)
         
            i = 0
            list_val = []
            for line in connectivity_matrix:
                j = 0 
                for val in line:
                    instrength = waytotal_column[j]   #is = sum(CIJ,1);    % instrength = column sum of CIJ: the instrength is the sum of inward link weights 
                    outstrength = waytotal_line[i]    #os = sum(CIJ,2);    % outstrength = row sum of CIJ:   the outstrength is the sum of outward link weights
                    list_val.append(instrength + outstrength) #str = is+os;   % strength = instrength+outstrength
                    j=j+1
                i=i+1

        # Compute the degree of each node: node degree is the number of links connected to the node
        elif self.node_features_comboBox.currentText() == "Degree": 
            # matrix binarised:  ensure CIJ is binary (CIJ = double(CIJ~=0))
            connectivity_matrix_binarize = np.where(connectivity_matrix > 0, 1, 0)  # > threshold, upper,lower

            waytotal_column = sum_column(connectivity_matrix_binarize)
            waytotal_line   = sum_line(connectivity_matrix_binarize)

            i = 0
            list_val = []
            for line in connectivity_matrix_binarize:
                j = 0
                for val in line:
                    indegree = waytotal_column[j]   # id = sum(CIJ,1);    % indegree = column sum of CIJ: the indegree is the number of inward links
                    outdegree = waytotal_line[i]    # od = sum(CIJ,2);    % outdegree = row sum of CIJ:   the outdegree is the number of outward links.  
                    list_val.append(indegree + outdegree)  # deg = id+od; % degree = indegree+outdegree
                    j=j+1
                i=i+1

        # To normalized between 0 to 1: 
        max_strenght, min_strenght  = (np.amax(list_val), np.amin(list_val))

        # Normalized printed color:
        norm_map = plt.Normalize(vmin, vmax)

        global label_color
        label_color = [] #list
        for i in range(len(list_val)):
            strenght_norm =(list_val[i] - min_strenght) / (max_strenght - min_strenght)  # norm between 0 to 1 

            # Adjust the color to the range of the colormap: 
            if strenght_norm < vmax and strenght_norm > vmin: 
                label_color.append(RdBu(norm_map(strenght_norm)))
            else:
                label_color.append((0,0,0)) #black node


        # *****************************************
        # Display circle connectome
        # *****************************************

        # Remove previous circle plot:
        for i in reversed(range(self.Layoutcircle.count())): 
            self.Layoutcircle.itemAt(i).widget().setParent(None)
        
        # New circle connectome plot: 
        self.fig = plt.figure(facecolor='SlateGray')
        self.canvas = FigureCanvas(self.fig)
        self.Layoutcircle.addWidget(self.canvas)

        global n_lines,node_angles_copy, axes 
        n_lines = int( (self.n_lines_spinBox.value()/100) * number_total_line)
        node_angles_copy = node_angles

        fig, axes = plot_connectivity_circle(connectivity_matrix, label_names, 
                                             n_lines = n_lines,linewidth = self.linewidth_spinBox.value(),
                                             vmin = (self.vmin_connectome_spinBox.value() / 100), vmax = (self.vmax_connectome_spinBox.value() / 100),
                                             node_angles = node_angles, node_colors = tuple(label_color), 
                                             fig = self.fig, show = False,
                                             colorbar_pos = (- 0.1, 0.1), facecolor='SlateGray', 
                                             fontsize_names = self.textwidth_spinBox.value(),
                                             colormap = self.colormap_connectome_comboBox.currentText(),
                                             padding = self.padding_spinBox.value(), 
                                             node_linewidth = self.nodelinewidth_spinBox.value(),node_edgecolor='Black' )

        # Udpate text in the interface
        if self.wait_label.text() != "done! ": 
            self.wait_label.setText("done! ")
        else:   
            self.wait_label.setText("done again! ")
        
        self.nb_line_label.setText(str(int( (self.n_lines_spinBox.value()/100) * number_total_line)) + " lines displayed")

        
        # Remove previous node label: 
        loop = len(axes.texts)
        for i in range(loop):
            axes.texts.remove(axes.texts[0])

        # Convert to radian and 
        node_angles = node_angles_copy * np.pi / 180
        node_angles_copy_event = node_angles


        # Draw new node labels: 
        angles_deg = 180 * node_angles_copy_event / np.pi

        for name, angle_rad, angle_deg in zip(name_boundaries, node_angles, angles_deg):
            if angle_deg >= 270:
                ha = 'left'
            else:
                # Flip the label, so text is always upright
                angle_deg += 180
                ha = 'right'

            axes.text(angle_rad, 10.4, str(name), size=self.textwidth_spinBox.value(),
                      rotation=angle_deg, rotation_mode='anchor', horizontalalignment=ha, verticalalignment='center', color='white')
               
        self.fig.canvas.draw()

        global indices, con_thresh, con_abs_util
        indices = np.tril_indices(len(label_names), -1)
        connectivity_matrix_modif = connectivity_matrix
        connectivity_matrix_modif = connectivity_matrix_modif[indices]

        con_thresh = np.sort(np.abs(connectivity_matrix_modif).ravel())[-n_lines ]

        # Get the connections which we are drawing and sort by connection strength this will allow to draw the strongest connections first
        con_abs = np.abs(connectivity_matrix_modif)
        con_abs_util = np.abs(connectivity_matrix)
        con_draw_idx = np.where(con_abs >= con_thresh)[0]
        
        # Connectivity_matrix_modif = connectivity_matrix_modif[con_draw_idx]
        con_abs = con_abs[con_draw_idx]
        indices = [ind[con_draw_idx] for ind in indices]

        # Now sort them
        sort_idx = np.argsort(con_abs)
        indices = [ind[sort_idx] for ind in indices]

        # Callback function: 
        callback =  partial(self.get_nodes, fig=self.fig, axes=axes ,indices=indices, n_nodes=len(label_names),  node_angles=node_angles_copy)
        self.fig.canvas.mpl_connect('button_press_event', callback)

        global my_fig
        my_fig = self.fig
        self.all_nodes_listWidget.clear()

        while i < len(VisuHierarchy_order):   
            current_elem = VisuHierarchy_order[i]
       
            #print('region',current_elem, list_of_list_VisuHierarchy[list_VisuHierarchy.index(current_elem)])
            self.all_nodes_listWidget.addItems(list_of_list_VisuHierarchy[list_VisuHierarchy.index(current_elem)])

            for i in range(self.all_nodes_listWidget.count()):
                item = self.all_nodes_listWidget.item(i)        

            i += 1

        self.all_nodes_listWidget.currentItemChanged.connect(self.get_item)



    # *****************************************
    # Get the name of the node selected by the user in the list
    # *****************************************
       
    def get_item(self, item):
        self.all_nodes_listWidget.blockSignals(True)
        
        Ui_visu.display_nodes(self, item.text(),update_manually_fig=True, 
            fig=my_fig, axes=axes, indices=indices, n_nodes=len(label_names), node_angles=node_angles_copy)

        self.all_nodes_listWidget.blockSignals(False)



    # *****************************************
    # Get the name of the node clicked by the user if left click and display all connections if right click
    # *****************************************

    def get_nodes(self, event, fig=None, axes=None, indices=None, n_nodes=0, node_angles=None, ylim=[9, 10]):
        # Convert to radian  
        node_angles = node_angles * np.pi / 180
        node_angles_copy_event = node_angles

        # *****************************************
        # Left click
        # *****************************************

        if event.button == 1:  
            # Click must be near node radius
            if event.ydata != "None": 
                if not ylim[0] <= event.ydata <= ylim[1]:
                    return

            # Set angles between [0, 2*pi]
            node_angles = node_angles % (np.pi * 2)
            node = np.argmin(np.abs(event.xdata - node_angles))
    
            Ui_visu.display_nodes(self, label_names[node], update_manually_fig=False, 
                    fig=my_fig, axes=axes, indices=indices, n_nodes=len(label_names), node_angles=node_angles_copy)


        # *****************************************
        # Right click
        # *****************************************

        elif event.button == 3: 
            # Remove previous node label: 
            loop = len(axes.texts)
            for i in range(loop):
                axes.texts.remove(axes.texts[0])

            # Draw new node labels: 
            angles_deg2 = 180 * node_angles_copy_event / np.pi
            for name, angle_rad, angle_deg in zip(name_boundaries, node_angles, angles_deg2):
                if angle_deg >= 270:
                    ha = 'left'
                else:
                    # Flip the label, so text is always upright
                    angle_deg += 180
                    ha = 'right'

                axes.text(angle_rad, 10.4, str(name), size=self.textwidth_spinBox.value(),
                      rotation=angle_deg, rotation_mode='anchor', horizontalalignment=ha, verticalalignment='center', color='white')
            
            fig.canvas.draw()

            self.clicked_nodes_label.setText("")
            self.nodes_associated_plainTextEdit.setPlainText("")



    # *****************************************
    # Display the label node and the labels of all its connected nodes + lines 
    # *****************************************

    def display_nodes(self, selected_node, update_manually_fig=False, fig=None, axes=None, indices=None, n_nodes=0, node_angles=None):
        # Convert to radian  
        node_angles = node_angles * np.pi / 180
        node_angles_copy_event = node_angles

        # Set angles between [0, 2*pi]
        node_angles = node_angles % (np.pi * 2)

        # Create a list with only the label of connected regions: 
        label_names_update, text = ([], "")

        for target in range(len(con_abs_util[0])):
            
            if label_names[target] == selected_node: 
                label_names_update.append(str(label_names[target]))

            elif con_abs_util[label_names.index(selected_node), target] >= con_thresh:
                text += '  ' + label_names[target] + '\n' 
                label_names_update.append(str(label_names[target]))
            else: 
                label_names_update.append(' ')

        # Display the name of the name and the list of connected regions:
        self.clicked_nodes_label.setText('Node selected: <font color="Turquoise">' + str(selected_node) + " </font>")
        self.nodes_associated_plainTextEdit.setPlainText(text)

        # Remove previous node label: 
        loop = len(axes.texts)
        for i in range(loop):
            axes.texts.remove(axes.texts[0])

        # Draw new node labels: 
        angles_deg = 180 * node_angles_copy_event / np.pi

        for name, angle_rad, angle_deg in zip(label_names_update, node_angles, angles_deg):
            if angle_deg >= 270:
                ha = 'left'
            else: # Flip the label, so text is always upright
                angle_deg += 180
                ha = 'right'

            if name != selected_node:
                axes.text(angle_rad, 10.4, str(name), size=self.textwidth_spinBox.value(),
                      rotation=angle_deg, rotation_mode='anchor', horizontalalignment=ha, verticalalignment='center', color='white')
            else: 
                axes.text(angle_rad, 10.4, str(name), size=self.textwidth_spinBox.value(),
                      rotation=angle_deg, rotation_mode='anchor', horizontalalignment=ha, verticalalignment='center', color='Turquoise')
               
        # Display lines in case of the user select a region in a list (because of no click, this function isn't handle by mne)
        if update_manually_fig: 
            patches = axes.patches
            node = label_names.index(selected_node)
            for ii, (x, y) in enumerate(zip(indices[0], indices[1])):
                patches[ii].set_visible(node in [x, y])

        fig.canvas.draw()



    # *****************************************
    # Setup the path to save the circle connectome
    # *****************************************

    def fig_file_pushButton_clicked(self):
        DirName= QtWidgets.QFileDialog.getExistingDirectory(self)
        if DirName:
            self.fig_file_textEdit.setText(DirName)



    # *****************************************
    # Save the circle connectome
    # *****************************************

    def save_circle_connectome_pushButton_clicked(self): 
        path_name = os.path.join(self.fig_file_textEdit.toPlainText(), 'Circle_connectome_' + self.type_of_normalization_comboBox.currentText() 
                                                                                     + "_symmetrization_by_" + self.type_of_symmetrization_comboBox.currentText()+ '.pdf')
        # Display and save: 
        self.save_circle_connectome_textEdit.setText("Figure saved here: " + path_name)
        self.fig.savefig(path_name, format='pdf', facecolor='black')



    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # ***********************************************  Brain connectome 2D  ****************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************

    # *****************************************
    # Display the axial/sagittal/coronal background of the brain connectome
    # *****************************************

    def background_axial_brain_connectome(self): 
        self.im1.set_data(self.imarray_axial[ self.num_slice_axial_horizontalSlider.value()])
        self.fig_brain_connectome.canvas.draw_idle()
        plt.close()
        self.num_slice_axial_label.setText("Slice " + str(self.num_slice_axial_horizontalSlider.value()))

    def background_sagittal_brain_connectome(self): 
        self.im2.set_data(self.imarray_sagittal[ self.num_slice_sagittal_horizontalSlider.value()])
        self.fig_brain_connectome.canvas.draw_idle()
        plt.close()
        self.num_slice_sagittal_label.setText("Slice " + str(self.num_slice_sagittal_horizontalSlider.value()))

    def background_coronal_brain_connectome(self): 
        self.im3.set_data(self.imarray_coronal[ self.num_slice_coronal_horizontalSlider.value()])
        self.fig_brain_connectome.canvas.draw_idle()
        plt.close()
        self.num_slice_coronal_label.setText("Slice " + str(self.num_slice_coronal_horizontalSlider.value()))
       


    # *****************************************
    # Display the brain connectome
    # ***************************************** 

    def display_brain_connectome_pushButton_clicked(self):
        now = datetime.datetime.now()
        print(now.strftime("Display brain connectome function: %H:%M %m-%d-%Y"))
        start = time.time()

        # *****************************************
        # Setup the view
        # *****************************************

        # Remove previous plot:
        for i in reversed(range(self.Layout_brain_connectome.count())): 
            self.Layout_brain_connectome.itemAt(i).widget().setParent(None)
        
        # Create figure:
        self.fig_brain_connectome = plt.figure(num=None)
        self.canvas = FigureCanvas(self.fig_brain_connectome)
        self.Layout_brain_connectome.addWidget(self.canvas)
        
        # Set title:
        outputfilename = 'Brain connectome of subject ' + json_user_object['Arguments']["ID"]["value"] + ' (connectivity matrix normalized (row-region))'
        self.fig_brain_connectome.suptitle(outputfilename, fontsize=10)

        self.ax1 = self.fig_brain_connectome.add_subplot(1,3,1) #axial
        self.ax2 = self.fig_brain_connectome.add_subplot(1,3,2) #sagittal left or right
        self.ax3 = self.fig_brain_connectome.add_subplot(1,3,3) #coronal

        # Set subtitles: 
        self.ax1.title.set_text("Axial")
        self.ax2.title.set_text("Sagittal right")
        if self.sagittal_left_checkBox.isChecked():
            self.ax2.title.set_text("Sagittal left")
        self.ax3.title.set_text("Coronal")


        # *****************************************
        # Setup the background: slice brain image
        # *****************************************

        # Find localization of images: 
        self.imarray, header = nrrd.read("./CONTINUITY_QC/template_brain_connectome_2D.nrrd")  #T0054-1-1-6yr-T1_SkullStripped_scaled.nrrd
        
        # Modify the matrix to select a specific orrientation: 
        self.imarray_axial          = np.rot90(self.imarray, k=1, axes=(0, 2))  
        self.imarray_sagittal_left  = np.flip(np.rot90(self.imarray, k=3, axes=(1, 2)) , axis=1)  
        self.imarray_sagittal_right = np.rot90(self.imarray_sagittal_left, k=2, axes=(0, 2)) 
        self.imarray_coronal        = np.rot90(self.imarray_sagittal_right, k=1, axes=(0, 2))  

        # Sagittal view: right or left: 
        self.imarray_sagittal = self.imarray_sagittal_right
        if self.sagittal_left_checkBox.isChecked():
            self.imarray_sagittal = self.imarray_sagittal_left

        # Plot background with a specific slice:
        self.im1 = self.ax1.imshow(self.imarray_axial[self.num_slice_axial_horizontalSlider.value()]) 
        self.im2 = self.ax2.imshow(self.imarray_sagittal[self.num_slice_sagittal_horizontalSlider.value()])
        self.im3 = self.ax3.imshow(self.imarray_coronal[self.num_slice_coronal_horizontalSlider.value()]) 


        # *****************************************
        # Extract data points in connectivity matrix and display points
        # *****************************************

        # Get the parcellation table with Cortical and Subcortical regions: 
        with open(os.path.join(self.parcellation_table_textEdit.toPlainText()), "r") as table_json_file:
            table_json_object = json.load(table_json_file)

        # Get data points for connected and unconnected points: 
        global list_name_2D_connectome
        list_name_unordered, list_coord_unordered, list_MatrixRow, list_name_2D_connectome, list_coord_2D_connectome = ([], [], [], [], [])
       
        for key in table_json_object:    
            list_name_unordered.append(key["name"])
            list_coord_unordered.append(key["coord"])
            list_MatrixRow.append(key["MatrixRow"])


        # Sort regions by VisuHierarchy number: 
        sorted_indices = np.argsort(list_MatrixRow)
        

        for i in range(len(list_MatrixRow)):
            index = sorted_indices[i]
            list_name_2D_connectome.append(list_name_unordered[index])
            list_coord_2D_connectome.append(list_coord_unordered[index])



        list_x               , list_y               , list_z                = ([], [], [])
        list_x_sagittal_left , list_y_sagittal_left , list_z_sagittal_left  = ([], [], [])
        list_x_sagittal_right, list_y_sagittal_right, list_z_sagittal_right = ([], [], [])
        list_x_coronal       , list_y_coronal       , list_z_coronal        = ([], [], [])
        
        # Extract data point for each view (axial, sagittal, coronal)
        for element in list_coord_2D_connectome:   
            # Header of nrrd-file: array([146, 190, 165])
            x = -(element[0]) + 146/2
            y = -(element[1]) + 165/2
            z = -(element[2]) + 190/2

            # Axial and coronal:
            list_x.append(x)
            list_y.append(y)
            list_z.append(z)

            # Sagittal left:
            if x>= 146/2 : 
                list_x_sagittal_left.append(x)   
                list_y_sagittal_left.append(-y + 190)
                list_z_sagittal_left.append(z)

                list_x_sagittal_right.append(float('nan'))
                list_y_sagittal_right.append(float('nan'))
                list_z_sagittal_right.append(float('nan'))

            # Sagittal right:
            else: 
                y_sagittal_left = float('nan')
                z_sagittal_left = float('nan')
                list_x_sagittal_left.append(float('nan'))
                list_y_sagittal_left.append(float('nan'))    
                list_z_sagittal_left.append(float('nan'))

                list_x_sagittal_right.append(x)    
                list_y_sagittal_right.append(y) 
                list_z_sagittal_right.append(z)



        # *****************************************
        # Get the normalize connectivity matrix: 
        # *****************************************

        matrix = os.path.join(self.connectivity_matrix_textEdit.toPlainText(), "fdt_network_matrix")

        # Specific normalization: 
        if self.type_of_normalization_brain_comboBox.currentText()   == "No normalization":         a = no_normalization(matrix)
        elif self.type_of_normalization_brain_comboBox.currentText() == "Whole normalization":      a = whole_normalization(matrix)
        elif self.type_of_normalization_brain_comboBox.currentText() == "Row region normalization": a = row_region_normalization(matrix)
        elif self.type_of_normalization_brain_comboBox.currentText() == "Row column normalization": a = row_column_normalization(matrix)

        # Specific symmetrization: 
        if self.type_of_symmetrization_brain_comboBox.currentText()   == "Average": a = average_symmetrization(a)
        elif self.type_of_symmetrization_brain_comboBox.currentText() == "Maximum": a = maximum_symmetrization(a)
        elif self.type_of_symmetrization_brain_comboBox.currentText() == "Minimum": a = minimum_symmetrization(a)
            
        # Transform in a numpy array:
        a = np.array(a)


        # *****************************************
        # Display lines with colors depending of informations in connectivity matrix
        # *****************************************

        # Colorbar vmin and vmaw used to setup the normalization of each color: 
        vmin_axial    = self.min_colorbar_axial_brain_connectome_doubleSpinBox.value()    / 100
        vmax_axial    = self.max_colorbar_axial_brain_connectome_doubleSpinBox.value()    / 100
        vmin_sagittal = self.min_colorbar_sagittal_brain_connectome_doubleSpinBox.value() / 100
        vmax_sagittal = self.max_colorbar_sagittal_brain_connectome_doubleSpinBox.value() / 100
        vmin_coronal  = self.min_colorbar_coronal_brain_connectome_doubleSpinBox.value()  / 100
        vmax_coronal  = self.max_colorbar_coronal_brain_connectome_doubleSpinBox.value()  / 100

        # To normalize colors:
        norm_axial    = mpl.colors.Normalize(vmin=vmin_axial,    vmax=vmax_axial)
        norm_sagittal = mpl.colors.Normalize(vmin=vmin_sagittal, vmax=vmax_sagittal)
        norm_coronal  = mpl.colors.Normalize(vmin=vmin_coronal,  vmax=vmax_coronal)

        # Min and max of the connectivity matrix: 
        mmin, mmax = (np.min(a),np.max(a))

        for i in range(np.shape(a)[0]):
            for j in range(np.shape(a)[1]):

                # Normalize:
                my_norm = (a[i,j] - mmin) / (mmax - mmin) #value between 0 to 1 
       
                # Specific threshold for axial lines (give by the range of the colorbar):
                if my_norm <= vmax_axial and my_norm >= vmin_axial:
                    point1 = [list_x[i], list_y[i],list_z[i]]
                    point2 = [list_x[j], list_y[j],list_z[j]]

                    x_values = [point1[0], point2[0]]
                    y_values = [point1[1], point2[1]]

                    # Display lines for axial view: 
                    index = list_coord_2D_connectome.index([])
                    cax1 = self.ax1.plot(x_values, y_values, lw=1.5, color= plt.cm.RdBu(norm_axial(my_norm)), gid=" lines axial")

                    if not self.plot_unconnected_points_CheckBox.isChecked(): 
                        # Plot connected points for axial view: 
                        cax1 = self.ax1.plot(list_x[i], list_y[i] , 'brown', marker=".", markersize=8, gid=" points axial ") 
                        cax1 = self.ax1.plot(list_x[j], list_y[j] , 'brown', marker=".", markersize=8, gid=" points axial ") 
                    
                # Specific threshold for coronal lines (give by the range of the colorbar):
                if my_norm <= vmax_coronal and my_norm >= vmin_coronal:
                    point1 = [list_x[i], list_y[i],list_z[i]]
                    point2 = [list_x[j], list_y[j],list_z[j]]

                    x_values = [point1[0], point2[0]]
                    z_values = [point1[2], point2[2]]

                    # Display lines for coronal view: 
                    cax3 = self.ax3.plot(x_values, z_values, lw=1.5, color= plt.cm.RdBu(norm_coronal(my_norm)), gid=" lines coronal")

                    if not self.plot_unconnected_points_CheckBox.isChecked(): 
                        # Plot points for coronal view:
                        cax3 = self.ax3.plot(list_x[i], list_z[i] , 'brown', marker=".", markersize=8, gid=" point coronal" )
                        cax3 = self.ax3.plot(list_x[j], list_z[j] , 'brown', marker=".", markersize=8, gid=" point coronal") 

                # Specific threshold for sagittal slice (give by the range of the colorbar):
                if my_norm <= vmax_sagittal and my_norm >= vmin_sagittal:

                    if self.sagittal_left_checkBox.isChecked():
                        point1_sagittal_left = [list_x_sagittal_left[i], list_y_sagittal_left[i],list_z_sagittal_left[i]]
                        point2_sagittal_left = [list_x_sagittal_left[j], list_y_sagittal_left[j],list_z_sagittal_left[j]]

                        y_values_sagittal_left = [point1_sagittal_left[1], point2_sagittal_left[1]]
                        z_values_sagittal_left = [point1_sagittal_left[2], point2_sagittal_left[2]]

                        # Display lines for sagittal left view: 
                        cax2 = self.ax2.plot(y_values_sagittal_left,  z_values_sagittal_left , lw=1.5, color= plt.cm.RdBu(norm_sagittal(my_norm)), gid=" lines sagittal") 

                        if not self.plot_unconnected_points_CheckBox.isChecked(): 
                            # Plot points for sagittal left  view:
                            cax2 = self.ax2.plot(list_y_sagittal_left[i], list_z_sagittal_left[i], 'brown', marker=".", markersize=8, gid=" point left sagittal" ) #sagittal left
                            cax2 = self.ax2.plot(list_y_sagittal_left[j], list_z_sagittal_left[j], 'brown', marker=".", markersize=8, gid=" point left sagittal") #sagittal left

                    else: 
                        point1_sagittal_right = [list_x_sagittal_right[i], list_y_sagittal_right[i],list_z_sagittal_right[i]]
                        point2_sagittal_right = [list_x_sagittal_right[j], list_y_sagittal_right[j],list_z_sagittal_right[j]]

                        y_values_sagittal_right = [point1_sagittal_right[1], point2_sagittal_right[1]]
                        z_values_sagittal_right = [point1_sagittal_right[2], point2_sagittal_right[2]]

                        # Display lines for sagittal right view: 
                        cax2 = self.ax2.plot(y_values_sagittal_right, z_values_sagittal_right, lw=1.5, color= plt.cm.RdBu(norm_sagittal(my_norm)), gid=" lines right sagittal") 
                        
                        if not self.plot_unconnected_points_CheckBox.isChecked(): 
                            # Plot points for sagittal right  view:
                            cax2 = self.ax2.plot(list_y_sagittal_right[i], list_z_sagittal_right[i], 'brown', marker=".", markersize=8, gid=" point right sagittal") #sagittal right
                            cax2 = self.ax2.plot(list_y_sagittal_right[j], list_z_sagittal_right[j], 'brown', marker=".", markersize=8, gid=" point right sagittal") #sagittal right
                                

        # *****************************************
        # Setup and display colorbar
        # *****************************************

        # Set colorbar position
        p0 = self.ax1.get_position().get_points().flatten()
        p1 = self.ax2.get_position().get_points().flatten()
        p2 = self.ax3.get_position().get_points().flatten()
      
        ax1_cbar = self.fig_brain_connectome.add_axes([p0[0], 0.07, p0[2]-p0[0]-0.03, 0.03])  #add_axes([xmin,ymin,dx,dy]) 
        ax2_cbar = self.fig_brain_connectome.add_axes([p1[0], 0.07, p1[2]-p1[0]-0.03, 0.03])  
        ax3_cbar = self.fig_brain_connectome.add_axes([p2[0], 0.07, p2[2]-p2[0]-0.03, 0.03]) 

        # Display colorbar
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm_axial,    cmap=plt.cm.RdBu), cax=ax1_cbar, orientation='horizontal')
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm_sagittal, cmap=plt.cm.RdBu), cax=ax2_cbar, orientation='horizontal')
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm_coronal,  cmap=plt.cm.RdBu), cax=ax3_cbar, orientation='horizontal')

        print("End display brain connectome: ",time.strftime("%H h: %M min: %S s",time.gmtime( time.time() - start )))

     
        self.fig_brain_connectome.canvas.mpl_connect('button_press_event', self.click_2D_connectome)           
        #plt.show()





    def click_2D_connectome(self, event):
        # Iterating over each data member plotted
        for curve in self.ax1.get_lines():
            # Searching which data member corresponds to current mouse position
            if curve.contains(event)[0]:
                print("axial lines  %s" % curve.get_gid())


        for curve in self.ax2.get_lines():
            # Searching which data member corresponds to current mouse position
            if curve.contains(event)[0]:
                print("sagittal lines %s" % curve.get_gid())


        for curve in self.ax3.get_lines():
            # Searching which data member corresponds to current mouse position
            if curve.contains(event)[0]:
                print("coronal lines   %s" % curve.get_gid())



    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # *****************************************  Brain connectome 3D  **********************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************

    # *****************************************
    # Display the 3D brain connectome THE FIRST TIME
    # ***************************************** 

    def create_brain_connectome_3D_pushButton_clicked(self):
        now = datetime.datetime.now()
        print(now.strftime("Display 3D brain connectome function: %H:%M %m-%d-%Y"))
        start = time.time()

        # *****************************************
        # Get the connectivity matrix
        # *****************************************

        matrix = os.path.join(self.connectivity_matrix_textEdit.toPlainText(), "fdt_network_matrix") 
      
        # Specific normalization: 
        global a 
        if self.type_of_normalization_brain_comboBox.currentText()   == "No normalization":         a = no_normalization(matrix)
        elif self.type_of_normalization_brain_comboBox.currentText() == "Whole normalization":      a = whole_normalization(matrix)
        elif self.type_of_normalization_brain_comboBox.currentText() == "Row region normalization": a = row_region_normalization(matrix)
        elif self.type_of_normalization_brain_comboBox.currentText() == "Row column normalization": a = row_column_normalization(matrix)

        # Specific symmetrization: 
        if self.type_of_symmetrization_brain_comboBox.currentText()   == "Average": a = average_symmetrization(a)
        elif self.type_of_symmetrization_brain_comboBox.currentText() == "Maximum": a = maximum_symmetrization(a)
        elif self.type_of_symmetrization_brain_comboBox.currentText() == "Minimum": a = minimum_symmetrization(a)
      
        # Transform in a numpy array: 
        a = np.array(a)

        # Range colorbar and normalization:
        vmin_3D, vmax_3D = ( self.min_colorbar_brain_3D_connectome_doubleSpinBox.value() / 100, self.max_colorbar_brain_3D_connectome_doubleSpinBox.value() / 100)
        norm_3D = mpl.colors.Normalize(vmin=vmin_3D, vmax=vmax_3D)

        # To normalize colors of lines: 
        global mmin, mmax
        mmin, mmax = (np.min(a),np.max(a))
        self.min_a_3D_label.setText(str(mmin))
        self.max_a_3D_label.setText(str(mmax))
        

        # *****************************************
        # Setup figure parameters: remove the previous plot: 
        # *****************************************

        for i in reversed(range(self.Layout_brain_connectome_3D.count())): 
            self.Layout_brain_connectome_3D.itemAt(i).widget().setParent(None)


        # *****************************************
        # Get path to brain surfaces and read the file
        # *****************************************
        
        SURFACE_template = './CONTINUITY_QC/template_brain_connectome_3D.vtk.vtk'
        if not os.path.exists(SURFACE_template): 
            SURFACE_template = './template_brain_connectome_3D.vtk.vtk'

        # Create template surfaces: visualization independente of subject:  (only cortical surfaces but is good for juste visualized)
        #polydatamerge('./CONTINUITY_QC/surface_template/icbm_avg_mid_sym_mc_left.vtk', './CONTINUITY_QC/surface_template/icbm_avg_mid_sym_mc_right.vtk', SURFACE_template)
        # Compute point for Destrieux: function in CONTINUITY_functions.py and use in CONTINUITY_completed_script.py

        # Read the brain surfaces file:
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(SURFACE_template)
        reader.Update() 


        # *****************************************
        # Add a specific VTK window in the PyQt5 interface
        # *****************************************

        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.Layout_brain_connectome_3D.addWidget(self.vtkWidget)

        # Create the widget
        balloonRep = vtk.vtkBalloonRepresentation()
        balloonRep.SetBalloonLayoutToImageRight()

        global balloonWidget
        balloonWidget = vtk.vtkBalloonWidget()
        balloonWidget.SetRepresentation(balloonRep)
    
        # Setup output view:
        output = reader.GetOutput()
        output_port = reader.GetOutputPort()
        scalar_range = output.GetScalarRange()

        # Create the mapper for brain surfaces:
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(output_port)
        mapper.SetScalarRange(scalar_range)

        # Create the actor for brains surfaces:
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(self.opacity_3D_spinBox.value())

        # Create the renderer: 
        self.ren = vtk.vtkRenderer()

        #balloonWidget.AddBalloon(actor, 'This is the brain surface', None)
        self.ren.AddActor(actor) #brain surfaces


        # *****************************************
        # Get point data thanks to parcellation table
        # *****************************************

        # Get the parcellation table with Cortical and Subcortical regions: 
        with open(os.path.join(self.parcellation_table_textEdit.toPlainText()), "r") as table_json_file:
            table_json_object = json.load(table_json_file)

        # Get data points for connected and unconnected points: 
        list_x, list_y, list_z, list_points, list_name_unordered, list_MatrixRow, list_name = ([], [], [], [], [], [], [])
       
        for key in table_json_object:    
            list_name_unordered.append(key["name"])
            list_MatrixRow.append(key["MatrixRow"])

        # Sort regions by VisuHierarchy number: 
        sorted_indices = np.argsort(list_MatrixRow)
        

        for i in range(len(list_MatrixRow)):
            index = sorted_indices[i]

            list_name.append(list_name_unordered[index])

            for key in table_json_object:
                if key["name"] == list_name_unordered[index]:
                    list_x.append(key["coord"][0])
                    list_y.append(key["coord"][1])
                    list_z.append(key["coord"][2])

                    # for map()
                    point = []
                    point.append(key["coord"][0])
                    point.append(key["coord"][1])
                    point.append(key["coord"][2])

                    list_points.append(point)

        # Set 1 if the point is connected and 0 otherwise
        list_visibility_point = []

        if not self.plot_unconnected_points_3D_CheckBox.isChecked(): 
            for i in range(np.shape(a)[0]):
                is_connected = False

                for j in range(np.shape(a)[1]):
                    # Normalize the value in connectivity matrix: 
                    my_norm = (a[i,j] - mmin) / (mmax - mmin) # value between 0 to 1 

                    # To be coherent with the range of colorbar: 
                    if my_norm > vmin_3D and my_norm < vmax_3D: 
                        is_connected = True
                
                if not is_connected: 
                    list_visibility_point.append(0)
                else: 
                    list_visibility_point.append(1)
        else: 
            list_visibility_point = [1] * np.shape(a)[0]


        for i in range(len(list_x)): 
            # *****************************************
            # Creates points thanks to parcellation table 
            # *****************************************

            # Create the polydata where we will store all the geometric data (points and lines):
            pointPolyData = vtk.vtkPolyData()

            # Create points and the topology of the point (a vertex):
            points = vtk.vtkPoints()
            vertices = vtk.vtkCellArray()

            # Add all point: 
            id = points.InsertNextPoint(list_x[i],list_y[i],list_z[i])
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(id)

            # Add the points to the polydata container:
            pointPolyData.SetPoints(points)
            pointPolyData.SetVerts(vertices)

            # *****************************************
            # Points colors
            # *****************************************

            # Setup colors parameters for point:
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)

            # Add color: 
            colors.InsertNextTypedTuple((0,0,255))
            
            # Add color points to the polydata container: 
            pointPolyData.GetPointData().SetScalars(colors)

            # Create the mapper for point:
            mapper_points = vtk.vtkPolyDataMapper()  
            mapper_points.SetInputData(pointPolyData)

            # Create the actor for points:
            actor_point = vtk.vtkActor()
            actor_point.SetMapper(mapper_points)
            actor_point.GetProperty().SetPointSize(self.point_size_3D_spinBox.value())
            actor_point.GetProperty().SetRenderPointsAsSpheres(1)

            # Set visibility: 
            actor_point.SetVisibility(list_visibility_point[i])                

            # Add point to renderer
            balloonWidget.AddBalloon(actor_point, 'This is ' + str(list_name[i]) , None)
            self.ren.AddActor(actor_point) 


        # *****************************************
        # Creates lines thanks to connectivity matrix and add colors
        # *****************************************

        # Create the color map
        colorLookupTable = vtk.vtkLookupTable()
        colorLookupTable.SetTableRange(vmin_3D, vmax_3D)
        n = 255  #n: number of colors
        colorLookupTable.SetNumberOfTableValues(n)
        colorLookupTable.Build()

        # Add value inside the color map:
        my_colormap = plt.cm.get_cmap('RdBu') #RdBu
        for i in range(n):
            my_color = list(my_colormap(i/n)) # tuple: R, G, B, 1 
            my_color = my_color[:-1] # R G B 
            colorLookupTable.SetTableValue(i, my_color[0], my_color[1], my_color[2], 1)

        for i in range(np.shape(a)[0]):
            for j in range(np.shape(a)[1]):                

                # Normalize the value in connectivity matrix: 
                my_norm = (a[i,j] - mmin) / (mmax - mmin) # value between 0 to 1 
            
                # Create a container and store the lines in it: 
                lines = vtk.vtkCellArray()

                # Create the polydata where we will store all the geometric data (points and lines):
                linesPolyData = vtk.vtkPolyData()

                # To access to 2 points: 
                two_points = vtk.vtkPoints()
                two_points.InsertNextPoint(list_x[i],list_y[i],list_z[i])
                two_points.InsertNextPoint(list_x[j],list_y[j],list_z[j])

                linesPolyData.SetPoints(two_points)

                # Create each lines: 
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0,0)
                line.GetPointIds().SetId(1,1)
                lines.InsertNextCell(line)

                # Add the lines to the polydata container:
                linesPolyData.SetLines(lines)

                # Setup colors parameters for lines: 
                colors_line = vtk.vtkUnsignedCharArray()
                colors_line.SetNumberOfComponents(3)

                # Add color:
                my_color = [0.0, 0.0, 0.0]
                colorLookupTable.GetColor(my_norm, my_color)
       
                # Create the mapper per line:
                mapper_lines = vtk.vtkPolyDataMapper()  
                mapper_lines.SetInputData(linesPolyData)#tubes.GetOutput())   
             
                mapper_lines.SetScalarModeToUseCellData()
                mapper_lines.SetColorModeToMapScalars()
                mapper_lines.Update()   

                mapper_lines.SetLookupTable(colorLookupTable)
                mapper_lines.SetScalarRange(vmin_3D, vmax_3D)
                mapper_lines.Update()  

                # Create one actor per line:
                actor_lines = vtk.vtkActor()
                actor_lines.SetMapper(mapper_lines)
                actor_lines.GetProperty().SetColor(my_color[0], my_color[1], my_color[2])

                actor_lines.GetProperty().SetLineWidth(self.linewidth_3D_spinBox.value())

                # To be coherent with the range of colorbar: 
                actor_lines.SetVisibility(0)
                if my_norm > vmin_3D and my_norm < vmax_3D:
                    actor_lines.SetVisibility(1)

                # Add to the renderer:
                balloonWidget.AddBalloon(actor_lines, 'This is the line between ' + str(list_name[i]) + " and " + str(list_name[j]), None)
                self.ren.AddActor(actor_lines)    

        # Add the color map:
        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetNumberOfLabels(8)
        scalarBar.SetLookupTable(colorLookupTable)
        self.ren.AddActor2D(scalarBar)

        # Set color of the background: 
        namedColors = vtk.vtkNamedColors()
        self.ren.SetBackground(namedColors.GetColor3d("SlateGray")) 


        # *****************************************
        # Add a window with an interactor and start visualization
        # *****************************************

        # Create a window and an interactor
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

         # add the custom style
        style = MouseInteractorHighLightActor()
        style.SetDefaultRenderer(self.ren)
        self.iren.SetInteractorStyle(style)

        balloonWidget.SetInteractor(self.iren)
        balloonWidget.EnabledOn()

        # Start visualization
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().Zoom(1.3)
        self.iren.Initialize()
        print("End display 3D brain connectome: ",time.strftime("%H h: %M min: %S s",time.gmtime( time.time() - start )))


    # *****************************************
    # Update the 3D brain connectome if the user change the range: min/max
    # ***************************************** 

    def update_3D_connectome(self): 
        # New range colorbar: 
        vmin_3D,vmax_3D  = (self.min_colorbar_brain_3D_connectome_doubleSpinBox.value() / 100, self.max_colorbar_brain_3D_connectome_doubleSpinBox.value() / 100)

         # Create the new color map:
        colorLookupTable1 = vtk.vtkLookupTable()
        colorLookupTable1.SetTableRange(vmin_3D, vmax_3D)
        n = 255  #n: number of colors
        colorLookupTable1.SetNumberOfTableValues(n)
        colorLookupTable1.Build()

        # Add value inside the new color map:
        my_colormap = plt.cm.get_cmap('RdBu') #RdBu
        for i in range(n):
            my_color = list(my_colormap(i/n)) # tuple: R, G, B, 1 
            my_color = my_color[:-1] # R G B 
            colorLookupTable1.SetTableValue(i, my_color[0], my_color[1], my_color[2], 1)
        
        # Compute again the list with all atribute to know with point to hidde:
        list_visibility_point = []

        if not self.plot_unconnected_points_3D_CheckBox.isChecked(): 
            for i in range(np.shape(a)[0]):
                is_connected = False

                for j in range(np.shape(a)[1]):
                    # Normalize the value in connectivity matrix: 
                    my_norm = (a[i,j] - mmin) / (mmax - mmin) # value between 0 to 1 

                    # To be coherent with the range of colorbar: 
                    if my_norm > vmin_3D and my_norm < vmax_3D: 
                        is_connected = True
                
                if not is_connected:  
                    list_visibility_point.append(0)
                else: 
                    list_visibility_point.append(1)
        else: 
            list_visibility_point = [1] * np.shape(a)[0]


        # Compute the list with all atribute to know with lines to hidde: 
        list_visibility_lines,list_color  = ([],[])
        for i in range(np.shape(a)[0]):
            for j in range(np.shape(a)[1]):                

                # Normalize the value in connectivity matrix: 
                my_norm = (a[i,j] - mmin) / (mmax - mmin) # value between 0 to 1 

                # To be coherent with the range of colorbar: 
                my_color = [0.0, 0.0, 0.0]

                if my_norm > vmin_3D and my_norm < vmax_3D:
                    list_visibility_lines.append(1)
                    # Add color:
                    colorLookupTable1.GetColor(my_norm, my_color)
                    list_color.append(my_color)

                else: 
                    list_visibility_lines.append(0)
                    list_color.append(my_color)

        # Update actor:
        actors = self.ren.GetActors()
        actors.InitTraversal()

        iNumberOfActors = actors.GetNumberOfItems()
        for i in range(iNumberOfActors): 
            if i == 0: 
                actors.GetNextProp().VisibilityOn() # skip brain surfaces actor which is the first actor

            elif i != 0 and i < len(list_visibility_point)+1: # actor point
                if list_visibility_point[i-1] == 0: 
                    actors.GetNextProp().VisibilityOff()
                else:
                    actors.GetNextProp().VisibilityOn() 

            else: #actor line
                if list_visibility_lines[i-1 - len(list_visibility_point)] == 0: 
                    actors.GetNextProp().VisibilityOff()
                else: 
                    my_actor = actors.GetNextProp()
                    my_actor.VisibilityOn()
                    # Add new color: 
                    my_actor.GetProperty().SetColor(list_color[i-1 - len(list_visibility_point)][0], 
                                                    list_color[i-1 - len(list_visibility_point)][1], 
                                                    list_color[i-1 - len(list_visibility_point)][2])

        actors_2D = self.ren.GetActors2D()
        actors_2D.InitTraversal()
        iNumberOfActors_2D = actors_2D.GetNumberOfItems()
        for i in range(iNumberOfActors_2D):
            actors_2D.GetNextProp().SetLookupTable(colorLookupTable1)

        # Update visualization
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().Zoom(1.3)
        self.iren.Initialize()


    
    # *****************************************
    # Update points size 
    # ***************************************** 

    def update_points_size(self):
        # Update actor:
        actors = self.ren.GetActors()
        actors.InitTraversal()

        for i in range(np.shape(a)[0]+1):  
            if i == 0:  #actor for brain surfaces: skip
                actors.GetNextProp().VisibilityOn()
            else: # actor points
                actors.GetNextProp().GetProperty().SetPointSize(self.point_size_3D_spinBox.value())

        # Update visualization
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().Zoom(1.3)
        self.iren.Initialize()


    
    # *****************************************
    # Update line width
    # ***************************************** 

    def update_lines_size(self):

        # Update actor:
        actors = self.ren.GetActors()
        actors.InitTraversal()

        iNumberOfActors = actors.GetNumberOfItems()

        for i in range(iNumberOfActors):  
            if i == 0 or i < np.shape(a)[0]+1:  #actor for brain surfaces + point: skip
                actors.GetNextProp().VisibilityOn()
            else: # actors lines
                actors.GetNextProp().GetProperty().SetLineWidth(self.linewidth_3D_spinBox.value())

        # Update visualization
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().Zoom(1.3)
        self.iren.Initialize()
   


    # *****************************************
    # Update the opacity of the brain surface: first actor! 
    # *****************************************

    def updata_opacity(self):
        # Update actor:
        actors = vtk.vtkPropCollection() 
        actors = self.ren.GetViewProps()
        actors.InitTraversal()

        # Brain surface: first actor
        actors.GetNextProp().GetProperty().SetOpacity(self.opacity_3D_spinBox.value())

        # Update visualization
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().Zoom(1.3)
        self.iren.Initialize()



    # *****************************************
    # Display brain surfaces with a specific view 
    # *****************************************

    def view_3D(self):

        # Axial view:
        if self.view_3D_comboBox.currentText() == "Axial": 
            self.ren.GetActiveCamera().SetViewUp(0, 1, 0)
            self.ren.GetActiveCamera().SetFocalPoint(0.0, 0.0, 0.0)  
            self.ren.GetActiveCamera().SetPosition(0,0,1)  
     
        # Sagittal view:
        elif self.view_3D_comboBox.currentText() == "Sagittal": 
            self.ren.GetActiveCamera().SetViewUp(1, 0, 0)
            self.ren.GetActiveCamera().SetFocalPoint(0.0, 0.0, 0.0)
            self.ren.GetActiveCamera().SetPosition(1,0,-1)  

        # Coronal view: 
        else: 
            self.ren.GetActiveCamera().SetViewUp(0, 1, 0 )
            self.ren.GetActiveCamera().SetFocalPoint(0.0, 0.0, 0.0)
            self.ren.GetActiveCamera().SetPosition(0,-1,0) 
           
        # Start visualization
        self.ren.GetActiveCamera().Elevation(30)        
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().Zoom(1.3)
        self.iren.Initialize()
        


    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # ****************************************  Display vtk file  **************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************

    # *****************************************
    # Display a vtk file given by the user
    # *****************************************
    
    def selected_file_clicked(self):
        # Remove the previous plot: 
        for i in reversed(range(self.Layout_vtk_fiber.count())): 
            self.Layout_vtk_fiber.itemAt(i).widget().setParent(None)

        # Add a specific vtk window
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.Layout_vtk_fiber.addWidget(self.vtkWidget)

        # Read the source file
        reader = vtk.vtkPolyDataReader() #vtkPolyDataReader
        reader.SetFileName(self.select_vtk_file_textEdit.toPlainText())
        reader.Update()  

        # Setup output view
        output = reader.GetOutput()
        output_port = reader.GetOutputPort()
        scalar_range = output.GetScalarRange()

        # Create the mapper 
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(output_port)
        mapper.SetScalarRange(scalar_range)

        # Create the Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Create the Renderer and a window
        self.ren = vtk.vtkRenderer()
        self.ren.AddActor(actor)

        # Set color of the background: 
        namedColors = vtk.vtkNamedColors()
        self.ren.SetBackground(namedColors.GetColor3d("SlateGray")) 
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        # Start visualization
        self.ren.ResetCamera()
        self.iren.Initialize()



    # *****************************************
    # Selected a vtk file to display it
    # *****************************************

    def select_vtk_file_pushButton_clicked(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()" , "", "ALL Files (*)", options=QFileDialog.Options())
        if fileName:
            self.select_vtk_file_textEdit.setText(fileName) 


  
# *****************************************
# AddObserver for the 3D connectome
# *****************************************

class MouseInteractorHighLightActor(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, parent=None):
        self.AddObserver("LeftButtonPressEvent", self.leftButtonPressEvent)
        self.LastPickedActor = None
        self.LastPickedProperty = vtk.vtkProperty()

    def leftButtonPressEvent(self, obj, event):
        clickPos = self.GetInteractor().GetEventPosition()

        picker = vtk.vtkPropPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())
      
        # get the new
        self.NewPickedActor = picker.GetActor()

        # If something was selected
        if self.NewPickedActor:

            # If we picked something before, reset its property
            if self.LastPickedActor:
                self.LastPickedActor.GetProperty().DeepCopy(self.LastPickedProperty)

            # Save the property of the picked actor so that we can
            # restore it next time
            self.LastPickedProperty.DeepCopy(self.NewPickedActor.GetProperty())
        
            # save the last picked actor
            self.LastPickedActor = self.NewPickedActor

        self.OnLeftButtonDown()
        return




'''
class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)
'''





'''
    # Display a vtk file WITHOUT Qt interface
    file_name = "./input_CONTINUITY/stx_T0054-1-1-6yr-T1_SkullStripped_scaled_BiasCorr_corrected_multi_atlas_white_surface_rsl_left_327680_native_ITKspace.vtk"

    # Read the source file.
    import vtk
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()  # Needed because of GetScalarRange
    output = reader.GetOutput()
    output_port = reader.GetOutputPort()
    scalar_range = output.GetScalarRange()

    # Create the mapper that corresponds the objects of the vtk file
    # into graphics elements
    mapper = vtkDataSetMapper()
    mapper.SetInputConnection(output_port)
    mapper.SetScalarRange(scalar_range)

    # Create the Actor
    actor = vtkActor()
    actor.SetMapper(mapper)

    # Create the Renderer
    renderer = vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1) # Set background 

    # Create the RendererWindow
    renderer_window = vtkRenderWindow()
    renderer_window.AddRenderer(renderer)

    # Create the RendererWindowInteractor and display the vtk_file
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderer_window)

    interactor.Initialize()
    interactor.Start()
'''