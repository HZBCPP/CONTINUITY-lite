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


'''
# For each subcortical region: 
# 1- copy in a separate folder (one folder per region)
# 2- merge all vtk file of each region 

# Get data: 
my_path = "/Human3/AutoSeg_Rerun_CONTE1and2/Data"

# Create output folders:
OUT_sc = os.path.join(OUT_PATH,'sc surf') 
if not os.path.exists(OUT_sc):
	os.mkdir(OUT_sc)

OUT_sc_organize = os.path.join(OUT_PATH,'sc surf organize') 
if not os.path.exists(OUT_sc_organize):
	os.mkdir(OUT_sc_organize)

# Get a list with all folder in my_path
filenames= os.listdir (my_path)

# 1- copy in a separate folder (one folder per region): 
for filename in filenames: # loop through all the files and folders for each subject      filename = ID = neo-0590-1-1-2year
	
	if os.path.isdir( os.path.join(my_path, filename)): # check if the object is a folder or not
		# Get SALT dir: 
		new_path = os.path.join(my_path, filename, 'AutoSegTissue_1year_v2-MultiAtlas', 'SALT')

		# Extract vtk file: 
		vtk_files = glob.glob(new_path + "/**/*_surfSPHARM.vtk", recursive = True)

		# Copy each vtk file in its good folder: folder corresponding to its associate region
		for file in vtk_files:
			sub_file = file.split("/")
			name = sub_file[-1]

			# Get the name of its region
			if name.startswith( str( filename + "-T1_SkullStripped_scaled_label_")):
				start = name.find("-T1_SkullStripped_scaled_label_") + len("-T1_SkullStripped_scaled_label_")
				end = name.find("_",len(filename) + len("-T1_SkullStripped_scaled_label_") )

				if str(end) == "-1": #no underscore found
					end = name.find(".", len(filename) + len("-T1_SkullStripped_scaled_label_") ) 

				region = name[start:end]

				# Copy:
				OUT_sc_organize_region = os.path.join(OUT_sc_organize,region) 
				if not os.path.exists(OUT_sc_organize_region):
					os.mkdir(OUT_sc_organize_region)

				shutil.copy(file, OUT_sc_organize_region) 


# 2- merge all vtk file of each region 
# Get folder for all region:
folder_region= os.listdir(OUT_sc_organize)

# For each region: 
for OUT_sc_organize_my_region in folder_region: #ThalR

	# Get vtk file
	vtk_files = glob.glob(os.path.join(OUT_sc_organize,OUT_sc_organize_my_region) + "/**/*_surfSPHARM.vtk", recursive = True)

	# Create the file which is an average/template for this region: 
	surface_merge = os.path.join(OUT_sc_organize, OUT_sc_organize_my_region, 'surface_merge' + OUT_sc_organize_my_region + '.vtk')

	# Merge each file :
	for f in range(len(vtk_files)):
		if f == 0: 
			first_vtk_file = vtk_files[f]

		elif f == 1: # create first merge
			command = ['/NIRAL/tools/bin_linux64/MeshMath',vtk_files[f],  surface_merge, '-avgMesh', vtk_files[f], first_vtk_file]
			# Display command 
			print(colored("\n"+" ".join(command)+"\n", 'blue'))
			# Run command and display output and error
			run = subprocess.Popen(command, stdout=0, stderr=subprocess.PIPE)
			out, err = run.communicate()
			print('merge surfaces', "out: ", colored("\n" + str(out) + "\n", 'green')) 
			print('merge surfaces', "err: ", colored("\n" + str(err) + "\n", 'red'))


		else :
			command = ['/NIRAL/tools/bin_linux64/MeshMath', vtk_files[f], surface_merge, '-avgMesh', vtk_files[f], surface_merge]
			# Display command 
			print(colored("\n"+" ".join(command)+"\n", 'blue'))
			# Run command and display output and error
			run = subprocess.Popen(command, stdout=0, stderr=subprocess.PIPE)
			out, err = run.communicate()
			print('merge surfaces', "out: ", colored("\n" + str(out) + "\n", 'green')) 
			print('merge surfaces', "err: ", colored("\n" + str(err) + "\n", 'red'))

'''