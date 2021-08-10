#!/usr/bin/env python3
import argparse
import json
import os 
import sys 
import traceback
import shutil
import subprocess
import time
import datetime
from termcolor import colored
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import nibabel as nib

import dipy 

from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response, auto_response_msmt, mask_for_response_msmt, response_from_mask_msmt
from dipy.reconst.shm import CsaOdfModel
import dipy.reconst.shm as shm

from dipy.data import default_sphere, small_sphere
from dipy.direction import peaks_from_model, ProbabilisticDirectionGetter

from dipy.tracking import utils
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion, BinaryStoppingCriterion
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk, save_vtk_streamlines, load_vtk
from dipy.io.vtk import load_polydata


from CONTINUITY_functions import *

##########################################################################################################################################

    # CONTINUITY completed script for tractography

##########################################################################################################################################

# *****************************************
# Parameters
# *****************************************

parser = argparse.ArgumentParser(description='CONTINUITY script for tractography')
parser.add_argument("user_json_filename", help = "File with all parameters given by the user", type = str) 
args = parser.parse_args()

# Read json file
with open(args.user_json_filename, "r") as user_Qt_file:
    json_user_object = json.load(user_Qt_file)
'''
for categories, infos in json_user_object.items():
    for key in infos: 
        print(key, ": ", json_user_object[categories][key]["value"])
'''

# Arguments: argparse
noGUI                      = json_user_object["Arguments"]["noGUI"]['value']
cluster                    = json_user_object["Arguments"]["cluster"]['value']
ID                         = json_user_object["Arguments"]["ID"]['value']
DWI_DATA                   = json_user_object["Arguments"]["DWI_DATA"]['value']
DWI_DATA_bvecs             = json_user_object["Arguments"]["DWI_DATA_bvecs"]['value']
DWI_DATA_bvals             = json_user_object["Arguments"]["DWI_DATA_bvecs"]['value']
T1_DATA                    = json_user_object["Arguments"]["T1_DATA"]['value']
T2_DATA                    = json_user_object["Arguments"]["T2_DATA"]['value']
BRAINMASK                  = json_user_object["Arguments"]["BRAINMASK"]['value']
PARCELLATION_TABLE         = json_user_object["Arguments"]["PARCELLATION_TABLE"]['value']
PARCELLATION_TABLE_NAME    = json_user_object["Arguments"]["PARCELLATION_TABLE_NAME"]['value']
labelSetName               = json_user_object["Arguments"]["labelSetName"]['value']
WM_L_Surf                  = json_user_object["Arguments"]["WM_L_Surf"]['value']
WM_R_Surf                  = json_user_object["Arguments"]["WM_R_Surf"]['value']
SURFACE_USER               = json_user_object["Arguments"]["SURFACE_USER"]['value']
WM_L_Surf_NON_REGISTRATION = json_user_object["Arguments"]["WM_L_Surf_NON_REGISTRATION"]['value']
WM_R_Surf_NON_REGISTRATION = json_user_object["Arguments"]["WM_R_Surf_NON_REGISTRATION"]['value']
SALTDir                    = json_user_object["Arguments"]["SALTDir"]['value']
labeled_image              = json_user_object["Arguments"]["labeled_image"]['value']
KWMDir                     = json_user_object["Arguments"]["KWMDir"]['value']
wm_mask					   = json_user_object["Arguments"]["wm_mask"]['value']
gm_mask					   = json_user_object["Arguments"]["gm_mask"]['value']

# Parameters: 
cluster_command_line                    = json_user_object["Parameters"]["cluster_command_line"]['value']
tractography_model                      = json_user_object["Parameters"]["tractography_model"]['value']
only_registration                       = json_user_object["Parameters"]["only_registration"]['value']
only_bedpostx                           = json_user_object["Parameters"]["only_bedpostx"]['value']
run_bedpostx_gpu                        = json_user_object["Parameters"]["run_bedpostx_gpu"]['value']
run_probtrackx2_gpu                     = json_user_object["Parameters"]["run_probtrackx2_gpu"]['value'] 
filtering_with_tcksift					= json_user_object["Parameters"]["filtering_with_tcksift"]['value']
optimisation_with_tcksift2				= json_user_object["Parameters"]["optimisation_with_tcksift2"]['value']
#multi_shell_DWI                         = json_user_object["Parameters"]["multi_shell_DWI"]['value']
act_option				                = json_user_object["Parameters"]["act_option"]['value']
UPSAMPLING_DWI                          = json_user_object["Parameters"]["UPSAMPLING_DWI"]['value']
DO_REGISTRATION                         = json_user_object["Parameters"]["DO_REGISTRATION"]['value']
INTEGRATE_SC_DATA                       = json_user_object["Parameters"]["INTEGRATE_SC_DATA"]['value']
INTEGRATE_SC_DATA_by_generated_sc_surf  = json_user_object["Parameters"]["INTEGRATE_SC_DATA_by_generated_sc_surf"]['value']
EXTRA_SURFACE_COLOR                     = json_user_object["Parameters"]["EXTRA_SURFACE_COLOR"]['value']
ignoreLabel                             = json_user_object["Parameters"]["ignoreLabel"]['value']
left_right_surface_need_to_be_combining = json_user_object["Parameters"]["left_right_surface_need_to_be_combining"]['value']
subcorticals_region_names               = json_user_object["Parameters"]["subcorticals_region_names"]['value']
subcorticals_region_labels              = json_user_object["Parameters"]["subcorticals_region_labels"]['value']
surface_already_labeled                 = json_user_object["Parameters"]["surface_already_labeled"]['value']
cortical_label_left                     = json_user_object["Parameters"]["cortical_label_left"]['value']
cortical_label_right                    = json_user_object["Parameters"]["cortical_label_right"]['value']
first_fixed_img                         = json_user_object["Parameters"]["first_fixed_img"]['value']
first_moving_img                        = json_user_object["Parameters"]["first_moving_img"]['value']
second_fixed_img                        = json_user_object["Parameters"]["second_fixed_img"]['value']
second_moving_img                       = json_user_object["Parameters"]["second_moving_img"]['value']
first_metric_weight                     = json_user_object["Parameters"]["first_metric_weight"]['value']
first_radius                            = json_user_object["Parameters"]["first_radius"]['value']
second_metric_weight                    = json_user_object["Parameters"]["second_metric_weight"]['value']
second_radius                           = json_user_object["Parameters"]["second_radius"]['value']
deformation_field_sigma                 = json_user_object["Parameters"]["deformation_field_sigma"]['value']
gradient_field_sigma                    = json_user_object["Parameters"]["gradient_field_sigma"]['value']
SyN_param                               = json_user_object["Parameters"]["SyN_param"]['value']
iteration1                              = json_user_object["Parameters"]["iteration1"]['value']
iteration2                              = json_user_object["Parameters"]["iteration2"]['value']
iteration3                              = json_user_object["Parameters"]["iteration3"]['value']
nb_threads                              = json_user_object["Parameters"]["nb_threads"]['value']
nb_jobs_bedpostx_gpu                    = json_user_object["Parameters"]["nb_jobs_bedpostx_gpu"]['value']
overlapping                             = json_user_object["Parameters"]["overlapping"]['value']
nb_fibers                               = json_user_object["Parameters"]["nb_fibers"]['value']
nb_fiber_per_seed                       = json_user_object["Parameters"]["nb_fiber_per_seed"]['value']
steplength                              = json_user_object["Parameters"]["steplength"]['value']
sampvox                                 = json_user_object["Parameters"]["sampvox"]['value']
loopcheck                               = json_user_object["Parameters"]["loopcheck"]['value']
sx  									= json_user_object["Parameters"]["sx"]['value']
sy  									= json_user_object["Parameters"]["sy"]['value']
sz  									= json_user_object["Parameters"]["sz"]['value']
nb_iteration_GenParaMeshCLP  			= json_user_object["Parameters"]["nb_iteration_GenParaMeshCLP"]['value']
spharmDegree  			                = json_user_object["Parameters"]["spharmDegree"]['value']
subdivLevel  			                = json_user_object["Parameters"]["subdivLevel"]['value']
list_bval_that_will_be_deleted          = json_user_object["Parameters"]["list_bval_that_will_be_deleted"]['value']
list_bval_for_the_tractography          = json_user_object["Parameters"]["list_bval_for_the_tractography"]['value']
size_of_bvals_groups_DWI                = json_user_object["Parameters"]["size_of_bvals_groups_DWI"]['value']
wm_fa_thr                               = json_user_object["Parameters"]["wm_fa_thr"]['value']
gm_fa_thr                               = json_user_object["Parameters"]["gm_fa_thr"]['value']
csf_fa_thr                              = json_user_object["Parameters"]["csf_fa_thr"]['value']
gm_md_thr                               = json_user_object["Parameters"]["gm_md_thr"]['value']
csf_md_thr                              = json_user_object["Parameters"]["csf_md_thr"]['value']
OUT_PATH                                = json_user_object["Parameters"]["OUT_PATH"]['value']

# Executables
pathUnu                   = json_user_object["Executables"]["unu"]['value']
pathN4BiasFieldCorrection = json_user_object["Executables"]["N4BiasFieldCorrection"]['value']
pathBRAINSFit_CMD         = json_user_object["Executables"]["BRAINSFit"]['value']
pathdtiprocess            = json_user_object["Executables"]["dtiprocess"]['value']
pathDtiestim              = json_user_object["Executables"]["dtiestim"]['value']
pathANTS_CMD              = json_user_object["Executables"]["ANTS"]['value']
pathITK_TRANSTOOL_EXE     = json_user_object["Executables"]["ITKTransformTools_v1"]['value']
pathPOLY_TRANSTOOL_EXE    = json_user_object["Executables"]["polydatatransform_v1"]['value']
pathWARP_TRANSFORM        = json_user_object["Executables"]["WarpImageMultiTransform"]['value']
DWIConvertPath            = json_user_object["Executables"]["DWIConvert"]['value']
FSLPath                   = json_user_object["Executables"]["fsl"]['value'] 
bedpostx_gpuPath          = json_user_object["Executables"]["bedpostx_gpu"]['value'] 
probtrackx2_gpuPath       = json_user_object["Executables"]["probtrackx2_gpu"]['value'] 
ExtractLabelSurfaces      = json_user_object["Executables"]["ExtractLabelSurfaces"]['value']
MRtrixPath                = json_user_object["Executables"]["MRtrix"]['value'] 
SegPostProcessCLPPath     = json_user_object["Executables"]["SegPostProcessCLP"]['value']
GenParaMeshCLPPath        = json_user_object["Executables"]["GenParaMeshCLP"]['value']
ParaToSPHARMMeshCLPPath   = json_user_object["Executables"]["ParaToSPHARMMeshCLP"]['value'] 

writeSeedListScript       = os.path.realpath(os.path.dirname(__file__)) + "/writeSeedList.py" 

# Environment variables: 
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nb_threads)
os.environ["OMP_NUM_THREADS"] = str(nb_threads)


# *****************************************
# Create folder and complete log file
# *****************************************

OUT_FOLDER = os.path.join(OUT_PATH,ID) #ID
if not os.path.exists( OUT_FOLDER ):
    os.mkdir(OUT_FOLDER)


# Log file:
log_file = os.path.join(OUT_FOLDER,"log.txt")

# Context manager that copies stdout and any exceptions to a log file
class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

with Tee(log_file):

	OUT_INPUT_CONTINUITY_DWISPACE = os.path.join(OUT_FOLDER,"Input_CONTINUITY_DWISpace") #ID --> Input_CONTINUITY_DWISpace
	if not os.path.exists(OUT_INPUT_CONTINUITY_DWISPACE): os.mkdir(OUT_INPUT_CONTINUITY_DWISPACE)
	
	OUT_SALT = os.path.join(OUT_FOLDER, "Salt") #ID --> SALT
	if not os.path.exists(OUT_SALT): os.mkdir(OUT_SALT)

	OUT_T1TODWISPACE = os.path.join(OUT_FOLDER,"T1ToDWISpace") #ID --> T1ToDWISpace
	if not os.path.exists(OUT_T1TODWISPACE): os.mkdir(OUT_T1TODWISPACE)

	OUT_00_QC_VISUALIZATION = os.path.join(OUT_T1TODWISPACE,"00_QC_Visualization") #ID --> T1ToDWISpace --> 00_QC_Visualization
	if not os.path.exists(OUT_00_QC_VISUALIZATION): os.mkdir(OUT_00_QC_VISUALIZATION)

	OUT_INTERMEDIATEFILES = os.path.join(OUT_T1TODWISPACE,"IntermediateFiles") #ID --> T1ToDWISpace --> IntermediateFiles
	if not os.path.exists(OUT_INTERMEDIATEFILES): os.mkdir(OUT_INTERMEDIATEFILES)

	OUT_DTI = os.path.join(OUT_INTERMEDIATEFILES,"DTI") #ID --> T1ToDWISpace --> IntermediateFiles --> DTI
	if not os.path.exists(OUT_DTI): os.mkdir(OUT_DTI)
	
	OUT_SURFACE = os.path.join(OUT_DTI, "Surface") #ID --> T1ToDWISpace --> IntermediateFiles --> DTI --> Surface
	if not os.path.exists(OUT_SURFACE): os.mkdir(OUT_SURFACE)

	OUT_WARPS = os.path.join(OUT_T1TODWISPACE, "Warps") #ID --> T1ToDWISpace --> WARP
	if not os.path.exists(OUT_WARPS): os.mkdir(OUT_WARPS)

	OUT_SLICER = os.path.join(OUT_FOLDER, "InputDataForSlicer") #ID --> INPUTDATA: for visualization
	if not os.path.exists(OUT_SLICER): os.mkdir(OUT_SLICER)

	OUT_TRACTOGRAPHY = os.path.join(OUT_FOLDER, "Tractography") #ID --> Tractography
	if not os.path.exists(OUT_TRACTOGRAPHY): os.mkdir(OUT_TRACTOGRAPHY)

	OUT_DIFFUSION = os.path.join(OUT_TRACTOGRAPHY, "Diffusion") #ID --> Tractography --> Diffusion
	if not os.path.exists(OUT_DIFFUSION): os.mkdir(OUT_DIFFUSION)

	OUT_DWI = os.path.join(OUT_FOLDER,"DWI_files") #ID --> DWI files
	if not os.path.exists(OUT_DWI): os.mkdir(OUT_DWI)


	# *****************************************
	# Function to run a specific command
	# *****************************************

	def run_command(text_printed, command):
		# Display command:
	    print(colored("\n"+" ".join(command)+"\n", 'blue'))
	    # Run command and display output and error:
	    run = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	    out, err = run.communicate()
	    print(text_printed, "out: ", colored("\n" + str(out) + "\n", 'green')) 
	    print(text_printed, "err: ", colored("\n" + str(err) + "\n", 'red'))



	# *****************************************
	# Function to convert DWI in nifti format to nrrd format 
	# *****************************************

	# Convert DWI nifti input to nrrd:  
	[path, afile] = os.path.split(DWI_DATA) # split path and file name(nrrd)

	if len(list_bval_that_will_be_deleted) != 0: 

		if not afile.endswith('nii.gz'): 
			# DWI data in nrrd format: need to be converted 

			print("*****************************************")
			print("Preprocessing: Convert DWI image to nifti format")
			print("*****************************************")           

			DWI_nifti = os.path.join(OUT_DWI, ID + "_DWI_before_remove_bvals.nii.gz")
			if os.path.exists(DWI_nifti):
				print("DWI_nifti file: Found Skipping Convert DWI image to nifti format ")
			else:
				print("Convert DWI image to nifti format ")

				run_command("DWIConvert: convert DWI to nifti format", [DWIConvertPath, 
                                                                        "--inputVolume", DWI_DATA, #input data 
                                                                        "--conversionMode", "NrrdToFSL", 
                                                                        "--outputVolume", DWI_nifti, 
                                                                        "--outputBValues", os.path.join(OUT_DWI, "bvals"), 
                                                                        "--outputBVectors", os.path.join(OUT_DWI, "bvecs")])
			# Update the path of DWI: 
			DWI_DATA_bvals = os.path.join(OUT_DWI, "bvals")
			DWI_DATA_bvecs = os.path.join(OUT_DWI, "bvecs")
			DWI_DATA       = DWI_nifti

			
		# DWI data in nifti format 
		print("*****************************************")
		print("Remove bval from DWI")
		print("*****************************************")

		# Find all b-values: 
		all_bval_with_duplicates = []
		bval_file = open(DWI_DATA_bvals, 'r')     
		for line in bval_file:
			line = int(line.strip('\n') )
			all_bval_with_duplicates.append(line)
	
		# Write txt file with all bval that will be deleted: 
		txt_file_with_bval_that_will_be_deleted = os.path.join(OUT_DWI, "txt_file_with_bval_that_will_be_deleted.txt")

		with open(txt_file_with_bval_that_will_be_deleted, 'w') as filebval:
			for listitem in list_bval_that_will_be_deleted:
				filebval.write('%s\n' % listitem)

				# Write other nerest b-values: 
				for i in range(size_of_bvals_groups_DWI*2):
					if int((listitem-size_of_bvals_groups_DWI)) + i in all_bval_with_duplicates:
						filebval.write('%s\n' % int((listitem-size_of_bvals_groups_DWI)) + i)


		# Filtering DWI: 
		remove_bval_from_DWI(txt_file_with_bval_that_will_be_deleted, DWI_DATA, DWI_DATA_bvecs, DWI_DATA_bvals, OUT_DWI, ID, FSLPath)

		# Update the path of DWI: 
		DWI_DATA_bvals = os.path.join(OUT_DWI, ID + '_DWI_filtered.bval')
		DWI_DATA_bvecs = os.path.join(OUT_DWI, ID + '_DWI_filtered.bvec')
		DWI_DATA       = os.path.join(OUT_DWI, ID + '_DWI_filtered.nii.gz')



	[path, afile] = os.path.split(DWI_DATA) # split path and file name(nrrd)
	if afile.endswith('nii.gz'):

		print("*****************************************")
		print("Preprocessing: Convert DWI FSL2Nrrd")
		print("*****************************************") 

		output_nrrd = os.path.join(OUT_DWI, afile[:-7] + '.nrrd') #filtered or not
		if os.path.exists(output_nrrd):
			print("DWI_nifti file: Found Skipping Convert DWI image to nifti format ")
		else:

			run_command("DWIConvert: convert input image in nifti format to nrrd format", [DWIConvertPath, "--inputVolume", DWI_DATA, 
														                             				       "--conversionMode", "FSLToNrrd", 
														                             				       "--outputVolume", output_nrrd, 
														                             				       "--inputBValues",DWI_DATA_bvals, "--inputBVectors",DWI_DATA_bvecs])
		# New path :
		DWI_DATA = output_nrrd


	# nothing to remove but the script need to have a list of bvals so need to convert a nrrd file 
	else: 
		# DWI data in nrrd format: need to be converted 
		print("*****************************************")
		print("Preprocessing: Convert DWI image to nifti format (nothing to remove)")
		print("*****************************************")           

		DWI_nifti = os.path.join(OUT_DWI, ID + "_DWI.nii.gz")
		if os.path.exists(DWI_nifti):
			print("DWI_nifti file: Found Skipping Convert DWI image to nifti format ")
		else:
			print("Convert DWI image to nifti format ")

			run_command("DWIConvert: convert DWI to nifti format", [DWIConvertPath, 
                                                                    "--inputVolume", DWI_DATA, #input data 
                                                                    "--conversionMode", "NrrdToFSL", 
                                                                    "--outputVolume", DWI_nifti, 
                                                                    "--outputBValues", os.path.join(OUT_DWI, "bvals"), 
                                                                    "--outputBVectors", os.path.join(OUT_DWI, "bvecs")])
		# Update the path of DWI: 
		DWI_DATA_bvals = os.path.join(OUT_DWI, "bvals")
		DWI_DATA_bvecs = os.path.join(OUT_DWI, "bvecs")



	########################################################################
	'''   
	    CONTINUITY script 1: Prepare files for T1 to DWISpace Registration

	 1- Pre-registration: (Up)sampled DWI, DWI BrainMask (upsample data is an option) using UNU
	 2- Preparing files for registration: 
	    - B0/DTI Image Generation: calculate the (up)sampled/masked DTI and the B0 directly from the (up)sampled DWI. Get the masked B0 and DTI image using dtiest. 
	    - B0 Bias Corrected Image generation: bias Correct the B0 Image to match the T2 Bias Corrected Image using N4BiasFieldCorrection
	    - FA generation using DTI process

	(from the script of Maria Bagonis (Nov 2019) 
	'''
	########################################################################

	print("**********************************************************************************")
	print("Script 1: prepare files for T1 to DWI space registration")
	print("**********************************************************************************")

	# Protect script: 
	if not DO_REGISTRATION:
		INTEGRATE_SC_DATA = False
		INTEGRATE_SC_DATA_by_generated_sc_surf = False

		if not left_right_surface_need_to_be_combining: 
			surface_already_labeled = True



	if DO_REGISTRATION:
		print("Starting Pre-registration: (Up)sampled DWI, DWI BrainMask, T1 DWISpace, DWISpace T1 surfaces")

		# Create different names for DWI_NRRD and DWI_MASK according to the value of UPSAMPLING_DWI and find the position of the gradient 
		if UPSAMPLING_DWI:
		    DWI_NRRD = os.path.join(OUT_INPUT_CONTINUITY_DWISPACE, ID + "_DWI_resample.nrrd")
		    DWI_MASK = os.path.join(OUT_INPUT_CONTINUITY_DWISPACE, ID + "_Original_DWI_BrainMask_resample.nrrd")

		    # Check the header file (line “kinds”) to verify if gradient is the first dimension or the last dimension of data → change unu resample option
		    command = [pathUnu, "head", DWI_DATA]   # NOT use run_command function because I use 'out' after
		    print(colored("\n"+" ".join(command)+"\n", 'blue'))
		    run = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		    out, err = run.communicate()
		    print("Unu check header, out: ", colored("\n" + str(out) + "\n", 'green')) 
		    print("Unu check header, err: ", colored("\n" + str(err) + "\n", 'red'))
	
		    for i in out.splitlines():
		        section = i.split()
		        if "b'kinds:" in str(section):
		            grad_first = False                # Case: [b'kinds:', b'domain', b'domain', b'domain', b'vector'] --> gradient in last position
		            if "b'vector" in str(section[1]): # Case: [b'kinds:', b'vector', b'domain', b'domain', b'domain'] --> gradient in first position
		                grad_first = True               
		else:
		    DWI_NRRD = os.path.join(OUT_INPUT_CONTINUITY_DWISPACE, ID + "_DWI_original.nrrd")
		    DWI_MASK = os.path.join(OUT_INPUT_CONTINUITY_DWISPACE, ID + "_Original_DWI_BrainMask_original.nrrd")


		# Interpolation / upsampling DWI
		if os.path.exists( DWI_NRRD ):
		    print("Files Found: Skipping Upsampling DWI")
		elif UPSAMPLING_DWI:
			print("*****************************************")
			print("*****************************************")

			command = [pathUnu, "resample", "-i", DWI_DATA, "-s", "x2", "x2", "x2", "=", "-k", "cubic:0,0.5"]
			if grad_first: 
				command = [pathUnu, "resample", "-i", DWI_DATA, "-s", "=", "x2", "x2", "x2", "-k", "cubic:0,0.5"]       
			p1 = subprocess.Popen(command, stdout=subprocess.PIPE)

			command = [pathUnu,"3op", "clamp", "0",'-', "10000000"]
			p2 = subprocess.Popen(command, stdin=p1.stdout, stdout=subprocess.PIPE)

			command = [pathUnu,"save", "-e", "gzip", "-f", "nrrd", "-o", DWI_NRRD]
			p3 = subprocess.Popen(command,stdin=p2.stdout, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
			print( colored("\n"+" ".join(command)+"\n", 'blue'))
			out, err = p3.communicate()
			print("Resample DWI out: ", colored("\n" + str(out) + "\n", 'green'))
			print("Resample DWI err: ", colored("\n" + str(err) + "\n", 'red')) 

		else: # no Upsampling DWI

			command = [pathUnu,"3op", "clamp", "0", "10000000", DWI_DATA]
			p2 = subprocess.Popen(command, stdout=subprocess.PIPE)
			print( colored("\n"+" ".join(command)+"\n", 'blue'))


			command = [pathUnu,"save", "-e", "gzip", "-f", "nrrd", "-o", DWI_NRRD]
			p3 = subprocess.Popen(command,stdin=p2.stdout, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
			print( colored("\n"+" ".join(command)+"\n", 'blue'))
			out, err = p3.communicate()
			print("No resample DWI out: ", colored("\n" + str(out) + "\n", 'green'))
			print("No resample DWI err: ", colored("\n" + str(err) + "\n", 'red'))     
		   

		# Interpolation / upsampling DWI MASK
		if os.path.exists( DWI_MASK ):
		    print("Files Found: Skipping Upsampling DWIMask")
		elif UPSAMPLING_DWI:
			print("*****************************************")
			print("Upsampling DWI MASK")
			print("*****************************************")

			command = [pathUnu, "resample", "-i", BRAINMASK, "-s", "x2", "x2", "x2", "-k", "cheap"]
			p1 = subprocess.Popen(command, stdout=subprocess.PIPE)

			command = [pathUnu,"save", "-e", "gzip", "-f", "nrrd", "-o", DWI_MASK]
			p2 = subprocess.Popen(command,stdin=p1.stdout, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
			print( colored("\n"+" ".join(command)+"\n", 'blue'))
			out, err = p2.communicate()
			print("Pipe resample DWI Mask out: ", colored("\n" + str(out) + "\n", 'green'))
			print("Pipe resample DWI Mask err: ", colored("\n" + str(err) + "\n", 'red')) 

		else: # no Upsampling DWI
		    command = [pathUnu,"save", "-e", "gzip", "-f", "nrrd", "-o", DWI_MASK, "-i", BRAINMASK]
		    p2 = subprocess.Popen(command, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		    print( colored("\n"+" ".join(command)+"\n", 'blue'))
		    out, err = p2.communicate()
		    print("Pipe no resample DWI_Mask out: ", colored("\n" + str(out) + "\n", 'green'))
		    print("Pipe no resample DWI_Mask err: ", colored("\n" + str(err) + "\n", 'red')) 


		print("*****************************************")
		print("Preparing files for registration")
		print("*****************************************")

		# Create different name according to upsampling parameter:
		if UPSAMPLING_DWI:
		    B0_NRRD             = os.path.join(OUT_00_QC_VISUALIZATION, ID + "_DTI_B0_resample.nrrd")
		    AD_NRRD             = os.path.join(OUT_00_QC_VISUALIZATION, ID + "_DTI_AD_resample.nrrd")
		    DTI_NRRD            = os.path.join(OUT_DTI, ID + "_DTI_DTI_resample.nrrd")
		    IDWI_NRRD           = os.path.join(OUT_DTI, ID + "_DTI_IDTI_resample.nrrd")
		    B0_BiasCorrect_NRRD = os.path.join(OUT_DTI, ID + "_DTI_B0_BiasCorrect_resample.nrrd")
		    FA_NRRD             = os.path.join(OUT_DTI, ID + "_DTI_FA_resample.nrrd")
		else:
		    B0_NRRD             = os.path.join(OUT_00_QC_VISUALIZATION, ID + "_DTI_B0_original.nrrd")
		    AD_NRRD             = os.path.join(OUT_00_QC_VISUALIZATION, ID + "_DTI_AD_original.nrrd")
		    DTI_NRRD            = os.path.join(OUT_DTI, ID + "_DTI_DTI_original.nrrd")
		    IDWI_NRRD           = os.path.join(OUT_DTI, ID + "_DTI_IDTI_original.nrrd")
		    B0_BiasCorrect_NRRD = os.path.join(OUT_DTI, ID + "_DTI_B0_BiasCorrect_original.nrrd")
		    FA_NRRD             = os.path.join(OUT_DTI, ID + "_DTI_FA_original.nrrd")


		# Calculate the (up)sampled/masked DTI and the B0 directly from the (up)sampled DWI. Get the masked B0 and DTI image using dtiest
		if os.path.exists( B0_NRRD ):
		    print("Files Found: Skipping B0/DTI Image Generation")
		else:
		    # Estimate tensor in a set of DWIs     
		    run_command("Dtiestim BO/DTI Image generation", [pathDtiestim, "--dwi_image", DWI_NRRD, 
		                                                                   "-M", DWI_MASK, 
		                                                                   "-t", '0', #threshold: -t 0 turns off the automatic masking performed in dtiestim
		                                                                   "--B0", B0_NRRD, #output: average baseline image (–B0) which is the average of all the B0s
		                                                                   "--tensor_output", DTI_NRRD, #output
												                           "-m", "wls",  #method: weighted least squares    
												                           "--idwi", IDWI_NRRD, #output:  geometric mean of the diffusion images.
												                           "--correction nearest"])
		    # Add BO_NRRD in INPUTDATA folder for visualization 
		    shutil.copy(B0_NRRD, OUT_SLICER) 


		# Bias Correct the B0 Image to match the T2 Bias Corrected Image: bias correction algorithm    
		if os.path.exists( B0_BiasCorrect_NRRD ):
		    print("B0 Bias Corrected Image Found: Skipping Correction")
		else:
			print("*****************************************")
			print("Bias Correct the B0 Image to match the T2 Bias Corrected Image")
			print("*****************************************")

			run_command("N4BiasFieldCorrection: BO Bias corrected image", [pathN4BiasFieldCorrection, "-d", "3", "-i", B0_NRRD, "-o", B0_BiasCorrect_NRRD])

			# Add B0_BiasCorrect_NRRD in INPUTDATA folder for visualization 
			shutil.copy(B0_BiasCorrect_NRRD, OUT_SLICER) 
		    

		# FA generation using DTI process
		if os.path.exists( FA_NRRD ):
		    print("FA Image Found: Skipping FA Generation from DTI")
		else:
			print("*****************************************")
			print("FA generation using DTI process")      
			print("*****************************************") 
			
			command = [pathdtiprocess, '--version']
			run = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			out, err = run.communicate()

			print("dtiprocess", "out: ", colored("\n" + str(out) + "\n", 'green')) 
			print("dtiprocess", "err: ", colored("\n" + str(err) + "\n", 'red'))
			
			split = out.split()
			last = str(split[-1])
			print(last)

			if last.endswith("1.0.3"): #version 1.0.3 on Pegasus
				run_command("Dtiprocess: FA generation from DTI", [pathdtiprocess, "--inputDTIVolume", DTI_NRRD, "-f", FA_NRRD, "--lambda1_output", AD_NRRD])
			else:#version 1.0.2 on Longleaf
				run_command("Dtiprocess: FA generation from DTI", [pathdtiprocess, "--dti_image", DTI_NRRD, "-f", FA_NRRD, "--lambda1_output", AD_NRRD])

			# Add FA_NRRD and AD_NRRD in INPUTDATA folder for visualization 
			shutil.copy(FA_NRRD, OUT_SLICER) 
			shutil.copy(AD_NRRD, OUT_SLICER) 



		########################################################################
		'''  
		    CONTINUITY script 2: Register T1 surfaces to DWI space: This script continues to prepare T1 surfaces registration to DWI space
	
		 1- T1 resample in DWI space: - Get a rigid transformation using BrainsFits for the initialization of the affine transform in ANTS
				                      - Perform the Deformable Registration using ANTS (35min)
		 		                      - Make invert matrix and T1 resample in DWI space using WARP_TRANSFORM and ITK_TRANSTOOL_EXE
		 2- Concatenate InvertWarp and InvertAffine transformations using ITK_TRANSTOOL_EXE
		 3- Transform surface (left and right) with InvWarp using POLY_TRANSTOOL_EXE

		(from the script of Maria Bagonis (Nov 2019) 
		'''
		########################################################################

		print("**********************************************************************************")
		print("Script 2: register T1 surfaces to DWI space")
		print("**********************************************************************************")

		#*****************************************
		# OUTPUT 
		#*****************************************

		outWarpPrefix = os.path.join(OUT_WARPS,ID + "_")
		outRigidReg   = os.path.join(OUT_WARPS,"RegRigid.txt")
		T1_OUT_NRRD   = os.path.join(OUT_00_QC_VISUALIZATION, ID + "_T1_SkullStripped_scaled_DWISpace.nrrd")
		INVAffine = os.path.join(OUT_WARPS, ID + "_InvAffine.txt") # output of ITK

		# Outputs of ANTs command:
		Affine    = os.path.join(OUT_WARPS, ID + "_Affine.txt")
		INVWarp   = os.path.join(OUT_WARPS, ID + "_InverseWarp.nii.gz")
		Warp      = os.path.join(OUT_WARPS, ID + "_Warp.nii.gz")


		if os.path.exists(Warp):
			print("ANTS File Found Skipping")
		else:
			moving_volume_ANTS = "T2_DATA"
			if T2_DATA == "": # just T1 data
			    moving_volume_ANTS = "T1_DATA"	

			print("*****************************************")
			print("Get a rigid transformation")
			print("*****************************************")

			# Get a rigid transformation using BrainsFits for the initialization of the affine transform in ANTS (register a 3D image to a reference volume)
			run_command("BRAINSFit", [pathBRAINSFit_CMD, "--fixedVolume", B0_BiasCorrect_NRRD, 
			                                             "--movingVolume", eval(moving_volume_ANTS), 
			                                             "--useRigid", "--initializeTransformMode", "useCenterOfHeadAlign", 
			                                             "--outputTransform", outRigidReg])


			print("*****************************************")
			print("Start ANTs command (~30 min with 1 core)")
			print("*****************************************")

			now = datetime.datetime.now()
			print(now.strftime("Script running ANTs command since: %H:%M %m-%d-%Y"))
			start = time.time()

			# Perform the Deformable Registration using ANTS
			# The T1 to DWI Space WARP(displacement field).nrrd is the output of the ANTS registration of the T1 Image to the DWI
			# Radius of the region = number of layers around a voxel/pixel
			command = [pathANTS_CMD, "3", "-m", "CC[", eval(first_fixed_img), ",", eval(first_moving_img), ",", str(first_metric_weight), ",", str(first_radius),"]",
			                              "-m", "CC[", eval(second_fixed_img),",", eval(second_moving_img),",", str(second_metric_weight),",", str(second_radius),"]",
			                              "-r", "Gauss[",str(gradient_field_sigma),",",str(deformation_field_sigma),"]", 
			                              "-i", str(iteration1), "x", str(iteration2), "x", str(iteration3), 
			                              "-t", "SyN[",str(SyN_param),"]",
			                              "-o", outWarpPrefix, 
			                              "--initial-affine", outRigidReg, 
			                              "--use-all-metrics-for-convergence",
			                              "num_threads", str(nb_threads), 
			                              "--verbose", 'True'] 
			if T2_DATA == "": # just T1 data
				command = [pathANTS_CMD, "3", "-m", "CC[", eval(first_fixed_img),",",eval(first_moving_img),",",str(first_metric_weight),",",str(first_radius),"]",
				                              "-r", "Gauss[", str(gradient_field_sigma),",", str(deformation_field_sigma),"]", 
				                              "-i", str(iteration1), "x", str(iteration2), "x", str(iteration3), 
				                              "-t", "SyN[",str(SyN_param),"]",
				                              "-o", outWarpPrefix, 
				                              "--initial-affine", outRigidReg, 
				                              "--use-all-metrics-for-convergence",
				                              "num_threads", str(nb_threads), 
				                              "--verbose", 'True']

			run_command("ANTs command", command)

			print("ANTs command: ",time.strftime("%H h: %M min: %S s",time.gmtime( time.time() - start )))



			print("*****************************************")
			print("T1 resample in DWI space")
			print("*****************************************")

			# Warp an image (T1_DATA) from one space (B0_BiasCorrect_NRRD) to one other space (T1_OUT_NRRD)    3,moving img,output img,reference img           
			run_command("WARP_TRANSFORM: T1 resample in DWI space", [pathWARP_TRANSFORM, "3", T1_DATA, T1_OUT_NRRD, "-R", B0_BiasCorrect_NRRD, Warp, Affine])
			
			# Add T1_OUT_NRRD in INPUTDATA folder for visualization 
			shutil.copy(T1_OUT_NRRD, OUT_SLICER)



			print("*****************************************")
			print("Make invert matrix")	   
			print("*****************************************")
			                                                # 		                                input  output
			run_command("ITK_TRANSTOOL_EXE: Make invert matrix", [pathITK_TRANSTOOL_EXE, "invert", Affine,INVAffine])


		print("*****************************************")
		print("Concatenate InvertWarp and InvertAffine")
		print("*****************************************")

		ConcatedWarp = os.path.join(OUT_WARPS, ID + "_ConcatenatedInvWarp.nrrd")
		if os.path.exists(ConcatedWarp):
			print("ConcatedWarp File Found Skipping")
		else:	                                                         #outputTransform ,Ref img       , input (INVWarp link with displacement)
			run_command("Concatenate", [pathITK_TRANSTOOL_EXE, "concatenate", ConcatedWarp, "-r", T1_DATA, INVAffine, INVWarp, "displacement"])


		print("*****************************************")
		print("Transform surface with InvWarp")
		print("*****************************************")

		# Output of the next functions: 
		RSL_WM_L_Surf = os.path.join(OUT_00_QC_VISUALIZATION, "stx_" + ID + 
		                                        "-T1_SkullStripped_scaled_BiasCorr_corrected_multi_atlas_white_surface_rsl_left_327680_native_DWIspace.vtk")
		RSL_WM_R_Surf = os.path.join(OUT_00_QC_VISUALIZATION, "stx_" + ID + 
		                                        "-T1_SkullStripped_scaled_BiasCorr_corrected_multi_atlas_white_surface_rsl_right_327680_native_DWIspace.vtk")
		if os.path.exists(RSL_WM_L_Surf):
			print("RSL_WM_L_Surf File Found Skipping")
		else:                               #, landmark file,input      , displacement file
			command = [pathPOLY_TRANSTOOL_EXE, "--fiber_file", WM_L_Surf, "-D", ConcatedWarp, "-o", RSL_WM_L_Surf, "--inverty", "--invertx"]
			run_command("POLY_TRANSTOOL_EXE: Transform WM left surface with InvWarp", command)
		
		if os.path.exists(RSL_WM_R_Surf):
			print("RSL_WM_R_Surf File Found Skipping")
		else:
			command = [pathPOLY_TRANSTOOL_EXE, "--fiber_file", WM_R_Surf, "-D", ConcatedWarp, "-o", RSL_WM_R_Surf, "--inverty", "--invertx"]
			run_command("POLY_TRANSTOOL_EXE: Transform WM right surface with InvWarp", command)		

		# Add T1_OUT_NRRD in INPUTDATA folder for visualization 
		shutil.copy(RSL_WM_L_Surf, OUT_SLICER)	
		shutil.copy(RSL_WM_R_Surf, OUT_SLICER)	
       


	########################################################################
	'''
		CONTINUITY script 3:
	
	A-If the user want to integrate subcortical data:
	 	0- Validation of subcortical region list
		1- Apply label: for each sc region label the SALT file with the Atlas label value (Create SPHARM surface labeled with the new atlas label)
		2- Combine the labeled subcorticals using polydatamerge
		3- Create Subcortical.vtk: move combining subcortical surface into DWISpace using POLY_TRANSTOOL_EXE- 
    B- labelization of cortical surfaces		 
	C- Create Cortical.vtk: combine left and right surface (in structural space if Not Registration, in DWI space if Registration) using polydatamerge
	D- If the user want to integrate subcortical data: Combine Subcortical.vtk with Cortical.vtk using polydatamerge

	(from the script of Maria Bagonis (Nov 2019)
	'''
	########################################################################

	print("**********************************************************************************")
	print("Script 3: Label_Combine_WARPtoDWISpace_SALTSubcort")
	print("**********************************************************************************")

	# *****************************************
	# OUTPUT
	# *****************************************

	OUT_LABELS = os.path.join(OUT_SALT, "Labels_" + PARCELLATION_TABLE_NAME)
	if not os.path.exists(OUT_LABELS):
		os.mkdir(OUT_LABELS)

	# Copy the original parcellation table to be able to build an other specific with only good subcortical regions ( = with good KWM and SALT files)
	only_matrix_parcellation_table = os.path.join(OUT_TRACTOGRAPHY, 'only_matrix_parcellation_table' )
	
	if not os.path.exists(only_matrix_parcellation_table): 
		Destrieux_point_already_computed = False
		shutil.copy(PARCELLATION_TABLE, only_matrix_parcellation_table)
	else: 
		Destrieux_point_already_computed = True 




	if subcorticals_region_names == []: 
		#Open parcellation table file with subcortical regions:
		with open(PARCELLATION_TABLE) as data_file:
			data = json.load(data_file)

		for seed in data:
			try: 
				if seed['subcortical']:
					subcorticals_region_names.append(seed['name'])

			except: 
				if INTEGRATE_SC_DATA:  
					print("no 'subcortical' param in your parcellation table")


	if INTEGRATE_SC_DATA:  
		print("*****************************************")
		print("Integration of subcortical data ")
		print("*****************************************")

		# Copy to have only regions with good KWM and SALT files
		subcorticals_list_names_checked, subcorticals_list_labels_checked = (subcorticals_region_names, subcorticals_region_labels )
			
		# Check if all elements in subcorticals_region_names are referenced in the parcellation table with subcortical data:
		for region in subcorticals_region_names:  
			
			#Open parcellation table file with subcortical regions:
			with open(PARCELLATION_TABLE) as data_file:    
				data = json.load(data_file)

			data_region_find = False
			for seed in data:
				if seed['name'] == region: 
					data_region_find = True

			# Check if the label !=0
			if INTEGRATE_SC_DATA_by_generated_sc_surf:
				index = subcorticals_list_names_checked.index(region)
				if subcorticals_list_labels_checked[index] == 0: 
					data_region_find = False
			

			# After check all the json file: 
			if data_region_find == False:
				print(" NO information about ", region, "in your parcellation table with subcortical data")
				
				if INTEGRATE_SC_DATA_by_generated_sc_surf:
					print( "OR the label for this region = 0: this region won't be integrate ")
					subcorticals_list_labels_checked = subcorticals_region_labels.remove(index)
				
				subcorticals_list_names_checked = subcorticals_list_names_checked.remove(region)

		
		if INTEGRATE_SC_DATA_by_generated_sc_surf:

			print("*****************************************")
			print("Generation of subcortical surfaces (~30 min )")
			print("*****************************************")
			
			now = datetime.datetime.now()
			print (now.strftime("Generation of subcortical surfaces: %H:%M %m-%d-%Y"))
			start = time.time()

			if len(subcorticals_list_names_checked) == len(subcorticals_list_labels_checked): 
				# Generate subcortical surfaces: 
				generating_subcortical_surfaces(OUT_FOLDER, ID, labeled_image, subcorticals_list_labels_checked, subcorticals_list_names_checked, 
					                                                           SegPostProcessCLPPath, GenParaMeshCLPPath, ParaToSPHARMMeshCLPPath,
					                                                           sx,sy,sz, nb_iteration_GenParaMeshCLP,spharmDegree, subdivLevel)
				print("Generation of subcortical surfaces: ",time.strftime("%H h: %M min: %S s",time.gmtime(time.time() - start)))

				# Update the localization of SALT surfaces: 
				SALTDir = os.path.join(OUT_FOLDER, 'my_SALT') 

				number_of_points = get_number_of_points(SALTDir)
				create_kwm_files(OUT_FOLDER, PARCELLATION_TABLE, subcorticals_list_names_checked, number_of_points)

				# Update the localization of KWM files: 
				KWMDir = os.path.join(OUT_FOLDER, 'my_KWM') 
			
			else: 
				print("ERROR: You have to provide one label per subcortical regions (0 if you don't want to integrate this region)")
				exit()

			# Add labeled_image in INPUTDATA folder for visualization 
			shutil.copy(labeled_image, OUT_SLICER) 


		else: # user provide SALT and KWM dir
			number_of_points = get_number_of_points(SALTDir)

			KWM_file = open( os.path.join(KWMDir, subcorticals_list_names_checked[0] + "_" + str(number_of_points) + "_KWM.txt"), 'r')     

			# Get number of points
			first_line = KWM_file.readline(70) 
			first_line_list = first_line.split("=") 
			number_of_points =int(first_line_list[1].strip()) #"NUMBER_OF_POINTS=1002"
		


		print("*****************************************")
		print("Apply label after validation of subcortical regions")
		print("*****************************************")

		# For each region label the SALT file with the Atlas label value. Create SPHARM surfaextract_bvalsce labeled with the new atlas label. 
		subcorticals_list_names_checked_with_surfaces = []
		for region in subcorticals_list_names_checked:

			#​The KWM files are intermediate .txt files for labeling the vertices on the respective subcortical SPHARM surfaces with a parcellation specific label number.
			KWMFile = os.path.join(KWMDir, region + "_" + str(number_of_points) + "_KWM.txt")
			SPHARMSurf = os.path.join(SALTDir, ID + "-T1_SkullStripped_scaled_label_" + region + "_pp_surfSPHARM.vtk")

			if not os.path.exists(SPHARMSurf) or not os.path.exists(KWMFile): 
				# Delete info of this region in the new-parcellation-table:
				with open(only_matrix_parcellation_table, 'r') as data_file:
				    data = json.load(data_file)

				for i in range(len(data)):
					if data[i]['name'] == region: 
						data.pop(i)
						break

				with open(only_matrix_parcellation_table, 'w') as data_file:
					data = json.dump(data, data_file, indent = 2)


			else: 
				subcorticals_list_names_checked_with_surfaces.append(region)

				# Create SPHARM surface labeled: 
				SPHARMSurfL = os.path.join(OUT_LABELS, ID + "-T1_SkullStripped_scaled_label_" + region +"_pp_SPHARM_labeled.vtk")

				if os.path.exists(SPHARMSurfL):
					print("For", region, "region: SPHARM surface labeled file: Found Skipping Labeling")
				else: 
					print("For", region, "region: creation SPHARM surface labeled file")
				    # Applies the label in the KWM file to the SPHARM surface: 
					KWMtoPolyData(SPHARMSurf, SPHARMSurfL, KWMFile, labelSetName)

		# Brainstem
		with open(only_matrix_parcellation_table, 'r') as data_file:
			   data = json.load(data_file)

		for i in range(len(data)):
			if data[i]['name'] == 'Brainstem': 
				data.pop(i)
				break

		with open(only_matrix_parcellation_table, 'w') as data_file:
			data = json.dump(data, data_file, indent = 2)


	
		


		print("*****************************************")
		print("Combine the labeled subcorticals")
		print("*****************************************")

		outputSurface = os.path.join(OUT_LABELS, ID + "-" + PARCELLATION_TABLE_NAME + "_Labeled_Subcorticals_Combined_T1Space.vtk")

		print("subcorticals_list_names_checked_with_surfaces", subcorticals_list_names_checked_with_surfaces)

		if os.path.exists( outputSurface ):
			print("OutputSurface file found: Skipping combine the labeled subcorticals ")
		else:
			# Add the first 2 subcortical regions to create an initial output file
			s1 = os.path.join(OUT_LABELS, ID + "-T1_SkullStripped_scaled_label_" + subcorticals_list_names_checked_with_surfaces[0] + "_pp_SPHARM_labeled.vtk")
			s2 = os.path.join(OUT_LABELS, ID + "-T1_SkullStripped_scaled_label_" + subcorticals_list_names_checked_with_surfaces[1] + "_pp_SPHARM_labeled.vtk") 

            # Combine the labeled subcorticals 
			print("For ", subcorticals_list_names_checked_with_surfaces[0], "region: ") 
			polydatamerge_ascii(s1, s2, outputSurface)

			# Add other regions 
			for i in range(2,len(subcorticals_list_names_checked_with_surfaces)):

				toAdd = os.path.join(OUT_LABELS, ID + "-T1_SkullStripped_scaled_label_" + subcorticals_list_names_checked_with_surfaces[i] + "_pp_SPHARM_labeled.vtk")
				print(toAdd)

				# Combine the labeled subcorticals 
				print("For ", subcorticals_list_names_checked_with_surfaces[i], "region: ")
				polydatamerge_ascii(outputSurface, toAdd, outputSurface)


		print("Move combining subcortical surfaces in DWISpace")
		subsAllDWISpace = os.path.join(OUT_SURFACE, "stx_" + ID + "_" + PARCELLATION_TABLE_NAME + "_Labeled_Subcorticals_Combined_DWISpace.vtk" ) 

		if os.path.exists(subsAllDWISpace):
			print("Labeled subcorticals combined DWISpace file: Found Skipping Transformation into DWISpace")
		else: 
			# Transform a PolyData with a displacement field: apply T1 to DWI WARP (ie displacement field)
			#                              , landmark file ,input        , displacement file       , output in DWI space
			command = [pathPOLY_TRANSTOOL_EXE, "--fiber_file",outputSurface, "-D", ConcatedWarp, "-o", subsAllDWISpace, "--inverty", "--invertx"]
			run_command("POLY_TRANSTOOL_EXE: combining subcortical data transform into DWISpace", command)



	else: #no integrate sc data
		subcorticals_list_names_checked_with_surfaces = []
		with open(only_matrix_parcellation_table, 'r') as data_file:
			data = json.load(data_file)

		i = 0 
		while i < len(data)-1:

			if data[i]['name'] in subcorticals_region_names:  
				subcorticals_region_names.remove(data[i]['name'])
				data.pop(i)
				i -= 1
			else: 
				i +=1

		with open(only_matrix_parcellation_table, 'w') as data_file:
			data = json.dump(data, data_file, indent = 2)





	if PARCELLATION_TABLE_NAME == 'Destrieux': 

		print("*****************************************")
		print("Compute one point per region")
		print("*****************************************")
		if not Destrieux_point_already_computed: 
			compute_point_destrieux(only_matrix_parcellation_table, subcorticals_list_names_checked_with_surfaces, ID )





	print("*****************************************")
	print("Labelization of the cortical surfaces ")
	print("*****************************************")

	if not surface_already_labeled:

		if not DO_REGISTRATION:
			# Outputs:
			WM_L_Surf_NON_REGISTRATION_labeled = os.path.join(OUT_00_QC_VISUALIZATION, "stx_" + ID + 
			            "-T1_SkullStripped_scaled_BiasCorr_corrected_multi_atlas_white_surface_rsl_left_327680_native_DWIspace_labeled_" + PARCELLATION_TABLE_NAME + ".vtk")
			WM_R_Surf_NON_REGISTRATION_labeled = os.path.join(OUT_00_QC_VISUALIZATION, "stx_" + ID + 
			            "-T1_SkullStripped_scaled_BiasCorr_corrected_multi_atlas_white_surface_rsl_right_327680_native_DWIspace_labeled_" + PARCELLATION_TABLE_NAME + ".vtk")
		

			print("NON_REGISTRATION: Label the left cortical surface")
			if os.path.exists( WM_L_Surf_NON_REGISTRATION_labeled ):
				print("NON_REGISTRATION: WM left labeled file found: Skipping Labelization of the left cortical surfaces")
			else:
				KWMtoPolyData(WM_L_Surf_NON_REGISTRATION, WM_L_Surf_NON_REGISTRATION_labeled, cortical_label_left, labelSetName)   


			print("NON_REGISTRATION: Label the right cortical surface")
			if os.path.exists( WM_R_Surf_NON_REGISTRATION_labeled ):
				print("NON_REGISTRATION: WM right labeled file found: Skipping Labelization of the right cortical surfaces")
			else:
				KWMtoPolyData(WM_R_Surf_NON_REGISTRATION, WM_R_Surf_NON_REGISTRATION_labeled, cortical_label_right, labelSetName)  		


		else: 
			# Outputs:
			RSL_WM_L_Surf_labeled = os.path.join(OUT_00_QC_VISUALIZATION, "stx_" + ID + 
			           "-T1_SkullStripped_scaled_BiasCorr_corrected_multi_atlas_white_surface_rsl_left_327680_native_DWIspace_labeled_" + PARCELLATION_TABLE_NAME + ".vtk")
			RSL_WM_R_Surf_labeled = os.path.join(OUT_00_QC_VISUALIZATION, "stx_" + ID + 
			           "-T1_SkullStripped_scaled_BiasCorr_corrected_multi_atlas_white_surface_rsl_right_327680_native_DWIspace_labeled_" + PARCELLATION_TABLE_NAME + ".vtk")
			

			print("Label the left cortical surface")
			if os.path.exists( RSL_WM_L_Surf_labeled ):
				print("WM left labeled file found: Skipping Labelization of the left cortical surfaces")
			else:
				KWMtoPolyData(RSL_WM_L_Surf, RSL_WM_L_Surf_labeled, cortical_label_left, labelSetName)  

			print("Label the right cortical surface")
			if os.path.exists( RSL_WM_R_Surf_labeled ):
				print("WM right labeled file found: Skipping Labelization of the right cortical surfaces")
			else:
				KWMtoPolyData(RSL_WM_R_Surf, RSL_WM_R_Surf_labeled, cortical_label_right, labelSetName)

			# Add SURFACE in INPUTDATA folder for visualization 
			shutil.copy(RSL_WM_L_Surf_labeled, OUT_SLICER)
			shutil.copy(RSL_WM_R_Surf_labeled, OUT_SLICER)


	else: #surfaces already labeled
		if not DO_REGISTRATION:
			RSL_WM_L_Surf_NON_REGISTRATION_labeled = WM_L_Surf_NON_REGISTRATION
			RSL_WM_R_Surf_NON_REGISTRATION_labeled = WM_R_Surf_NON_REGISTRATION

		else: 
			RSL_WM_L_Surf_labeled = RSL_WM_L_Surf
			RSL_WM_R_Surf_labeled = RSL_WM_R_Surf

			# Add SURFACE in INPUTDATA folder for visualization 
			shutil.copy(RSL_WM_L_Surf_labeled, OUT_SLICER)
			shutil.copy(RSL_WM_R_Surf_labeled, OUT_SLICER)



	print("*****************************************")
	print("Start: Combine left and right surface in structural and DWI space (do or not do registration) ")
	print("*****************************************")

	# Create cortical.vtk: 
	SURFACE = os.path.join(OUT_SURFACE, "stx_" + ID + "_T1_CombinedSurface_white_" + PARCELLATION_TABLE_NAME + ".vtk")

	if not DO_REGISTRATION:
		if os.path.exists(SURFACE):
			print("NOT REGISTRATION: Combine cortical file: Found Skipping combining cortical left and right surface ")
		else: 
			if left_right_surface_need_to_be_combining:
				# NOT REGISTRATION: combine left and right surface 
				polydatamerge_ascii(WM_L_Surf_NON_REGISTRATION_labeled, WM_R_Surf_NON_REGISTRATION_labeled, SURFACE)
				
			else:
				shutil.copy(SURFACE_USER, SURFACE)
				SURFACE = SURFACE_USER


	else: 
		if os.path.exists(SURFACE):
			print("Combine cortical file: Found Skipping combining cortical left and right surface ")
		else: 
			# Combine left+right surface 
			polydatamerge_ascii(RSL_WM_L_Surf_labeled, RSL_WM_R_Surf_labeled, SURFACE)



	if INTEGRATE_SC_DATA: 
		print("*****************************************")
		print("Start the integration of subcortical data: Combine subcortical with cortical vtk file in DWI Space of choice (Destrieux, AAL, etc)")
		print("*****************************************")

		# Output of the next command: 
		outputSurfaceFullMerge = os.path.join(OUT_INPUT_CONTINUITY_DWISPACE, "stx_" + ID + "_T1_CombinedSurface_white_" + PARCELLATION_TABLE_NAME + 
			                                                                                                                       "_WithSubcorticals.vtk") 

		if os.path.exists(outputSurfaceFullMerge):
			print("Combine cortical and subcortical file: Found Skipping combining cortical and subcortical")
		else: 
			# Integration of subcortical data
			polydatamerge_ascii(subsAllDWISpace, SURFACE, outputSurfaceFullMerge)

			# Add outputSurfaceFullMerge in INPUTDATA folder for visualization 
			shutil.copy(outputSurfaceFullMerge, OUT_SLICER)

		SURFACE = outputSurfaceFullMerge


	# Add SURFACE in INPUTDATA folder for visualization 
	shutil.copy(SURFACE, OUT_SLICER)


	if only_registration: 
		exit()

   
    

	########################################################################
	'''   
	    Run tractography

	 1- Create Diffusion data for bedpostx: conversion of BRAINMASK and DWI data: .nrrd file format to FSL format (.nii.gz) with DWIConvert
	 2- Bedpostx: fit probabilistic diffusion model      - 
	 3- ExtractLabelSurfaces: creation label surface from a VTK surface containing label information
	 4- Creation of seed list: text file listing all path of label surfaces created by ExtractLabelSurfaces using writeSeedList.py
	 5- Probtrackx2: compute tractography according to the seed list created
	 6- Convert T1 image to nifti format using DWIConvert
	 7- Normalize the matrix and save plot as PDF file using plotMatrix.py

	(from the script: "tractographyScriptAppv2.0.sh" in /CIVILITY/src/civility-tractography/scripts) 
	'''
	########################################################################

	print("**********************************************************************************")
	print("Script 4: tractography")
	print("**********************************************************************************")

	# *****************************************
	# Set parameters
	# *****************************************

	if EXTRA_SURFACE_COLOR:
		EXTRA_SURFACE_COLOR = SURFACE

	overlapFlag, overlapName, loopcheckFlag, loopcheckName = ("", "", "", "")
	if overlapping: 
		overlapFlag = "--overlapping" ; overlapName = "_overlapping" 
	if loopcheck: 
		loopcheckFlag = "--loopcheck" ; loopcheckName = "_loopcheck"
	

	# *****************************************
	# Ouput folder
	# *****************************************

	OutSurfaceName = os.path.join(OUT_TRACTOGRAPHY, "OutputSurfaces" + overlapName)
	if not os.path.exists(OutSurfaceName):
	    os.mkdir(OutSurfaceName)


	if not DO_REGISTRATION: 
		DWI_MASK = BRAINMASK
		DWI_NRRD = DWI_DATA



	print("*****************************************")
	print("DWIConvert BRAINMASK and DWI: nrrd to nii")
	print("*****************************************")

	# Outputs:
	DiffusionData      = os.path.join(OUT_DIFFUSION, "data.nii.gz") 
	DiffusionBrainMask = os.path.join(OUT_DIFFUSION, "nodif_brain_mask.nii.gz")

	# DWIConvert BRAINMASK: NrrdToFSL: .nrrd file format to FSL format (.nii.gz)     # Err: "No gradient vectors found " --> it is normal 
	if os.path.exists(DiffusionBrainMask):
	    print("Brain mask FSL file: Found Skipping conversion")
	else: 
		print("DWIConvert BRAINMASK to FSL format")
		
		run_command("DWIConvert BRAINMASK to FSL format(err ok)", [DWIConvertPath, "--inputVolume", DWI_MASK, 
								                                                   "--conversionMode", "NrrdToFSL", 
								                                                   "--outputVolume", DiffusionBrainMask, 
								                                                   "--outputBVectors", os.path.join(OUT_DIFFUSION, "bvecs.nodif"), 
								                                                   "--outputBValues", os.path.join(OUT_DIFFUSION, "bvals.temp")])

	# DWIConvert DWI: Nrrd to FSL format
	if os.path.exists(DiffusionData):
	    print("DWI FSL file: Found Skipping conversion")
	else:
		print("DWIConvert DWI to FSL format")
		
		run_command("DWIConvert DWI to FSL format", [DWIConvertPath, "--inputVolume", DWI_NRRD, 
							                                         "--conversionMode", "NrrdToFSL", 
							                                         "--outputVolume", DiffusionData, 
							                                         "--outputBVectors", os.path.join(OUT_DIFFUSION, "bvecs"), 
							                                         "--outputBValues", os.path.join(OUT_DIFFUSION, "bvals")])


	print("*****************************************")
	print("Create labelSurfaces (~2h30 or 4h with subcortical regions)")
	print("*****************************************")				

	# Outputs:
	labelListNamePath   = os.path.join(OutSurfaceName, "labelListName.txt")
	labelListNumberPath = os.path.join(OutSurfaceName, "labelListNumber.txt")

	if os.path.exists( labelListNamePath):
	    print("labelListName.txt file: Found Skipping create labelSurfaces")
	else:
		now = datetime.datetime.now()
		print (now.strftime("Script running ExtractLabelSurfaces command since: %H:%M %m-%d-%Y"))
		start = time.time()
	
		# Extract Point Data from the vtk file containing labels
		command = [ExtractLabelSurfaces, "--extractPointData", "--translateToLabelNumber",   # /tools/bin_linux64/ExtractLabelSurfaces  version 1.0.4 NOT 1.0
								  	    	 "--labelNameInfo", labelListNamePath, 
								  	    	 "--labelNumberInfo", labelListNumberPath, 
								  	    	 "--useTranslationTable", "--labelTranslationTable", only_matrix_parcellation_table, 
								  	    	 "-a",labelSetName, 
								  	    	 "--vtkLabelFile", str(EXTRA_SURFACE_COLOR), 
								  	    	 "--createSurfaceLabelFiles", 
								  	    	 "--vtkFile", SURFACE,
								  	    	 "--outputSurfaceDirectory", os.path.join(OutSurfaceName, "labelSurfaces"), 
								  	    	 overlapFlag]

		if ignoreLabel != "": 
			command.append("--ignoreLabel")
			command.append(str(ignoreLabel))
			run_command("ExtractLabelSurfaces if 'ignoreLabel' != ''", command)
		else:
	  		run_command("ExtractLabelSurfaces", command)

		print("ExtractLabelSurfaces command: ",time.strftime("%H h: %M min: %S s",time.gmtime(time.time() - start)))



	print("*****************************************")
	print("Write seed list") 
	print("*****************************************")

	# No overwritting:
	if os.path.exists(os.path.join(OutSurfaceName,"seeds.txt")):
		os.remove(os.path.join(OutSurfaceName,"seeds.txt"))

	# Create a text file listing all path of label surfaces created by ExtractLabelSurfaces
	run_command("Write seed list", [sys.executable, writeSeedListScript, OutSurfaceName, only_matrix_parcellation_table])


	NETWORK_DIR = os.path.join(OUT_TRACTOGRAPHY, "Network" + overlapName + loopcheckName)
	if not os.path.exists( NETWORK_DIR ):
		os.mkdir(NETWORK_DIR)



	if tractography_model == "FSL": 

		print("*****************************************")
		print("Run tractography with FSL")
		print("*****************************************")

		# Bedpostx: fit probabilistic diffusion model: automatically determines the number of crossing fibres per voxel
		if os.path.exists( os.path.join(OUT_TRACTOGRAPHY, "Diffusion.bedpostX")):
			print("Bedpostx folder: Found Skipping bedpostx function")
		else:
			print("*****************************************")
			print("Start bedpostx (~ 20 hours)")
			print("*****************************************")

			now = datetime.datetime.now()
			print (now.strftime("Script running bedpostx command since: %H:%M %m-%d-%Y"))
			start = time.time()    


			if not run_bedpostx_gpu: 
				command = [FSLPath + '/bedpostx', OUT_DIFFUSION, "-n", str(nb_fibers)]

			else: 
				command = [bedpostx_gpuPath, OUT_DIFFUSION, "-n", str(nb_fibers), "-NJOBS", str(nb_jobs_bedpostx_gpu) ] 
				#-NJOBS (number of jobs to queue, the data is divided in NJOBS parts, usefull for a GPU cluster, default 4)

			run_command("bedpostx", command)
			print("bedpostx command: ",time.strftime("%H h: %M min: %S s",time.gmtime(time.time() - start)))



		if only_bedpostx: 
			exit()


		# Name define by probtrackx2 tool:
		matrix = "fdt_network_matrix"
		matrixFile = os.path.join(NETWORK_DIR, matrix)

		# Probtrackx2:
		if os.path.exists(matrixFile): 
		    print("fdt_network_matrix found: Found Skipping probtrackx function")
		else:
			print("*****************************************")
			print("Start tractography with probtrackx2 (~3h )")
			print("*****************************************")

			now = datetime.datetime.now()
			print (now.strftime("Script running probtrackx2 command since: %H:%M %m-%d-%Y"))
			start = time.time()

			if not run_probtrackx2_gpu: 
				run_command("probtrackx2", [FSLPath + '/probtrackx2', "-s", os.path.join(OUT_TRACTOGRAPHY, "Diffusion.bedpostX", "merged"), #-s,--samples	
								                             "-m", os.path.join(OUT_TRACTOGRAPHY, "Diffusion.bedpostX", "nodif_brain_mask"), #-m,--mask
								                             "-x", os.path.join(OutSurfaceName, "seeds.txt"), #-x,--seed
								                             "--forcedir", "--network", "--omatrix1", "-V", "0",
								                             "--dir="+NETWORK_DIR, 
								                             "--stop="+os.path.join(OutSurfaceName, "seeds.txt"), 
								                             "-P", str(nb_fiber_per_seed), #-P,--nsamples	Number of samples - default=5000
								                             "--steplength="+str(steplength), 
								                             "--sampvox="+str(sampvox), loopcheckFlag ])
			else: 
				run_command("probtrackx2", [probtrackx2_gpuPath, "-s", os.path.join(OUT_TRACTOGRAPHY, "Diffusion.bedpostX", "merged"), #-s,--samples	
								                             "-m", os.path.join(OUT_TRACTOGRAPHY, "Diffusion.bedpostX", "nodif_brain_mask"), #-m,--mask
								                             "-x", os.path.join(OutSurfaceName, "seeds.txt"), #-x,--seed
								                             "--forcedir", "--network", "--omatrix1", "-V", "0",
								                             "--dir="+NETWORK_DIR, 
								                             "--stop="+os.path.join(OutSurfaceName, "seeds.txt"), 
								                             "-P", str(nb_fiber_per_seed), #-P,--nsamples	Number of samples - default=5000
								                             "--steplength="+str(steplength), 
								                             "--sampvox="+str(sampvox), loopcheckFlag ])


			print("probtrackx2 command: ", time.strftime("%H h: %M min: %S s",time.gmtime(time.time() - start)))


		print("*****************************************")
		print("Normalize connectivity matrix and save plot as PDF file")
		print("*****************************************")

		if not cluster: #can't display before saving on Longleaf
			save_connectivity_matrix('no_normalization', no_normalization(matrixFile), NETWORK_DIR, ID, overlapName, loopcheck)
			save_connectivity_matrix('whole', whole_normalization(matrixFile), NETWORK_DIR, ID, overlapName, loopcheck)
			save_connectivity_matrix('row_region', row_region_normalization(matrixFile), NETWORK_DIR, ID, overlapName, loopcheck)
			save_connectivity_matrix('row_column', row_column_normalization(matrixFile), NETWORK_DIR, ID, overlapName, loopcheck)
		








	# Find all b-values: 
	user_bval_with_nearest_values_without_duplicate,all_bval_with_duplicate = ([],[])

	if len(list_bval_for_the_tractography) == 0: #bval not specify by the user: by default: all bvals used except 0
		list_bval_for_the_tractography = extract_bvals(DWI_DATA_bvals, size_of_bvals_groups_DWI)
		#print("list_bval_for_the_tractography after extraction ", list_bval_for_the_tractography)	

		list_bval_for_the_tractography = [x for x in list_bval_for_the_tractography if x > size_of_bvals_groups_DWI] # remove 0 or nerest values

		bval_file = open(os.path.join(OUT_DWI, "bvals"), 'r')     
		for line in bval_file:
				line = int(line.strip('\n') )
				all_bval_with_duplicate.append(line)
				if not line in user_bval_with_nearest_values_without_duplicate and line != 0:
					user_bval_with_nearest_values_without_duplicate.append(line)
					


	else: # bvals 'grouping' specify by the user: need to add the nereast value 
		bval_file = open(os.path.join(OUT_DWI, "bvals"), 'r')    

		for line in bval_file:
			# Extraction of bval: 
			line = int(line.strip('\n') )
			all_bval_with_duplicate.append(line)

			# Check if user want of not this value 
			for listitem in list_bval_for_the_tractography:
	
				# Other nerest b-values: 
				for i in range(size_of_bvals_groups_DWI*2):
					if int((line - size_of_bvals_groups_DWI)) + i == listitem:
						if not line in user_bval_with_nearest_values_without_duplicate:
							user_bval_with_nearest_values_without_duplicate.append(line)

	print("list_bval_for_the_tractography", list_bval_for_the_tractography)	
	#print("all_bval_with_duplicate", all_bval_with_duplicate)			
	#print("user_bval_with_nearest_values_without_duplicate after conversion: ", user_bval_with_nearest_values_without_duplicate)









    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # ****************************************  Run Mrtrix  ********************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************

	if tractography_model == "MRtrix (default: IFOD2) " or tractography_model == "MRtrix (Tensor-Prob)" or tractography_model == "MRtrix (iFOD1)": 

		print("*****************************************")
		print("Run tractography with " + tractography_model.replace(" ", "") )
		print("*****************************************")

	 	# *****************************************
		# Output folder for MRtrix and DIPY 
		# *****************************************

		add = ""
		if filtering_with_tcksift:     add = '_tcksif'
		if optimisation_with_tcksift2: add = '_tcksif2'

		OUT_MRTRIX = os.path.join(OUT_TRACTOGRAPHY, tractography_model.replace(" ", "") + add) 
		if not os.path.exists(OUT_MRTRIX):
			os.mkdir(OUT_MRTRIX)

		
		# *****************************************
		# Create 5tt   
		# *****************************************	    
		if act_option:
			print("*****************************************")
			print("Convert T1 image to nifti format")
			print("*****************************************")

			if not DO_REGISTRATION:  #user provide directly T1 in DWI space
				T1_OUT_NRRD = T1_DATA

			T1_nifti = os.path.join(NETWORK_DIR, ID + "-T1_SkullStripped_scaled.nii.gz")
			if os.path.exists(T1_nifti):
			    print("T1_nifti file: Found Skipping Convert T1 image to nifti format ")
			else:
				print("Convert T1 image to nifti format ")
				
				run_command("DWIConvert: convert T1 image to nifti format", [DWIConvertPath, "--inputVolume", T1_OUT_NRRD,
																                             "--conversionMode", "NrrdToFSL", 
																                             "--outputVolume", T1_nifti, 
																                             "--outputBValues", os.path.join(OUT_DIFFUSION, "bvals.temp"), 
																                             "--outputBVectors", os.path.join(OUT_DIFFUSION, "bvecs.temp")])

			# First choice: use T1_OUT_NRRD (after conversion in nifti): T1 in DWI space (second choice: use T1_nifti: T1 in structural space + add the transformation: affine )
			fivett_img = os.path.join(OUT_MRTRIX,"5tt.nii.gz")
			if os.path.exists(fivett_img):
			    print("5tt image already compute")
			else: 
				print("Create 5tt image (~20min )")      
				now = datetime.datetime.now()
				print (now.strftime("Script to create 5tt image running since: %H:%M %m-%d-%Y"))
				start = time.time()
				run_command("create 5tt", [sys.executable, MRtrixPath + "/5ttgen", 'fsl', 
																					T1_nifti, # input
																					fivett_img, #output

																					'-scratch', os.path.join(OUT_MRTRIX), 
																					'-nthreads', str(nb_threads) ])
				print("Create 5tt image: ", time.strftime("%H h: %M min: %S s",time.gmtime(time.time() - start)))
				


		print("*****************************************")
		print("Response function estimation: dwi2response")
		print("*****************************************")

		if len(list_bval_for_the_tractography) == 1: # single shell_DWI: 

			# *****************************************
			# Response function estimation: Estimate response function(s) for spherical deconvolution
			# *****************************************

			Response_function_estimation_txt = os.path.join(OUT_MRTRIX,'Response_function_estimation.txt')
			if os.path.exists(Response_function_estimation_txt):
			    print("Response function estimation already compute ")
			else: 
				command = [MRtrixPath + "/dwi2response",'tournier', DiffusionData, # input
													   				Response_function_estimation_txt, #output

													   				'-mask', DiffusionBrainMask, # input
													                '-fslgrad', os.path.join(OUT_DIFFUSION, "bvecs"),os.path.join(OUT_DIFFUSION, "bvals"), # input
													                '-scratch', os.path.join(OUT_MRTRIX),
													                '-nthreads', str(nb_threads) ]
				run_command("Response function estimation (err ok)", command)


		else: # multi shell_DWI: need other file for the next step  
			response_wm_txt = os.path.join(OUT_MRTRIX, "response_wm.txt")
			response_gm_txt = os.path.join(OUT_MRTRIX, "response_gm.txt")
			response_csf_txt = os.path.join(OUT_MRTRIX, "response_csf.txt")

			if os.path.exists(response_wm_txt): 
				print("Multi shell dwi2response msmt_5tt: response_wm_txt, response_wm_txt andresponse_csf_txt already compute ")
			else: 
				print("Multi shell dwi2response msmt_5tt")	

				#dwi2response dhollander dwi.mif response_wm.txt response_gm.txt response_csf.txt

				command = [MRtrixPath + "/dwi2response",'dhollander', DiffusionData, # input
													   			
													   				response_wm_txt,  #output
																	response_gm_txt,  #output
																	response_csf_txt, #output

													   				'-mask', DiffusionBrainMask, # input
													                '-fslgrad', os.path.join(OUT_DIFFUSION, "bvecs"),os.path.join(OUT_DIFFUSION, "bvals"), # input
													                '-scratch', os.path.join(OUT_MRTRIX),
													                '-nthreads', str(nb_threads) ]
				command.append('-shells')
				shells = ""
				for idx, element in enumerate(list_bval_for_the_tractography): 				
					shells += str(element)
					if idx < len(list_bval_for_the_tractography) - 1:
						shells += ","
				command.append(shells)

				run_command("dhollander", command)

				#command = [MRtrixPath +'/mrconvert', DiffusionData, "/work/elodie/testing/diffusionData.mif"]
				#run_command("nifti to mif", command)


				'''
				command = [MRtrixPath + "/dwi2response",'msmt_5tt', 
															DiffusionData, # "/work/elodie/testing/diffusionData.mif", # input
															fivett_img, # input

															response_wm_txt,  #output
															response_gm_txt,  #output
															response_csf_txt, #output

															'-mask', DiffusionBrainMask, #input
													        '-fslgrad', os.path.join(OUT_DIFFUSION, "bvecs"),os.path.join(OUT_DIFFUSION, "bvals"), # input
													        '-scratch', os.path.join(OUT_MRTRIX),
													        '-nthreads', str(nb_threads) ]
				command.append('-shells')
				shells = ""
				for idx, element in enumerate(list_bval_for_the_tractography): 				
					shells += str(element)
					if idx < len(list_bval_for_the_tractography) - 1:
						shells += ","
				command.append(shells)

				run_command("msmt_5tt", command)
				'''
				



		# *****************************************
		# Fibre Orientation Distribution estimation: Estimate fibre orientation distributions from diffusion data using spherical deconvolution
		# *****************************************

		FOD_nii = os.path.join(OUT_MRTRIX, "FOD.nii.gz")
		wmfod_mif = os.path.join(OUT_MRTRIX, "wmfod.mif")
		gm_mif = os.path.join(OUT_MRTRIX, "gm.mif")
		csf_mif = os.path.join(OUT_MRTRIX, "csf.mif")

		if os.path.exists(FOD_nii):
		    print("Fibre Orientation Distribution estimation already compute")
		else: 
			if len(list_bval_for_the_tractography) == 1: # single shell_DWI: 
				command = [MRtrixPath + "/dwi2fod", 'csd',
									    DiffusionData, # input
									    Response_function_estimation_txt, # input
									    FOD_nii, # ouput
									   	'-mask', DiffusionBrainMask, # input
									    '-fslgrad', os.path.join(OUT_DIFFUSION, "bvecs"),os.path.join(OUT_DIFFUSION, "bvals"),# input
									    '-nthreads', str(nb_threads)]
				command.append('-shells')
				shells = ""
				for idx, element in enumerate(list_bval_for_the_tractography):	
					shells += str(element)
					if idx < len(list_bval_for_the_tractography) - 1: 
						shells += ","
				command.append(shells)
			

		if os.path.exists(wmfod_mif):
		    print("White Matter Fibre Orientation Distribution estimation already compute")
		else: 
			if len(list_bval_for_the_tractography) != 1: # multi shell 
				command = [MRtrixPath + "/dwi2fod", 'msmt_csd',
									    DiffusionData, # input

										response_wm_txt,  # input
										wmfod_mif, # ouput

										response_gm_txt,  # input
										gm_mif, # ouput

										response_csf_txt,  # input
										csf_mif, # ouput

									   	'-mask', DiffusionBrainMask, # input
									    '-fslgrad', os.path.join(OUT_DIFFUSION, "bvecs"),os.path.join(OUT_DIFFUSION, "bvals"),# input
									    '-nthreads', str(nb_threads)]
				
				command.append('-shells')
				shells = ""
				for idx, element in enumerate(list_bval_for_the_tractography):			
					shells += str(element)
					if idx < len(list_bval_for_the_tractography) - 1: 
						shells += ","
				command.append(shells)
		
			run_command("FOD estimation", command)


		if len(list_bval_for_the_tractography) != 1:
			FOD_nii = wmfod_mif

			'''
			command = [MRtrixPath +'/mrconvert', wmfod_mif , "/work/elodie/testing/wmfod.nii.gz"]
			run_command("mif to nifti", command)
			
			command = [MRtrixPath +'/mrconvert', gm_mif , "/work/elodie/testing/gm.nii.gz"]
			run_command("mif to nifti", command)
			
			command = [MRtrixPath +'/mrconvert', csf_mif , "/work/elodie/testing/csf.nii.gz"]
			run_command("mif to nifti", command)
				
			'''





		# *****************************************
		# Output folder
		# *****************************************

		# Create Whole-brain streamlines tractography folder: 
		output_track_tckgen = os.path.join(OUT_MRTRIX, "output_track_tckgen") 
		if not os.path.exists(output_track_tckgen): os.mkdir(output_track_tckgen)

		OUT_MRTRIX_tck = os.path.join(output_track_tckgen, "output_track_tckgen_tck") 
		if not os.path.exists(OUT_MRTRIX_tck): os.mkdir(OUT_MRTRIX_tck)

		OUT_MRTRIX_vtk = os.path.join(output_track_tckgen, "output_track_tckgen_vtk") 
		if not os.path.exists(OUT_MRTRIX_vtk): os.mkdir(OUT_MRTRIX_vtk)
		
		# Output tcksift:
		if filtering_with_tcksift:
			tcksift_folder = os.path.join(OUT_MRTRIX, "output_tcksift") 
			if not os.path.exists(tcksift_folder): os.mkdir(tcksift_folder)

			tcksift_tck = os.path.join(tcksift_folder, "output_tcksift_tck") 
			if not os.path.exists(tcksift_tck): os.mkdir(tcksift_tck)

			tcksift_vtk = os.path.join(tcksift_folder, "output_tcksift_vtk") 
			if not os.path.exists(tcksift_vtk): os.mkdir(tcksift_vtk)			

		# Output tcksift2:
		if optimisation_with_tcksift2: 
			tcksift2_txt = os.path.join(OUT_MRTRIX, "output_tcksift2_streamlines_weights_txt") 
			if not os.path.exists(tcksift2_txt): os.mkdir(tcksift2_txt)

		# Output tckEdit:
		weights_folder = os.path.join(OUT_MRTRIX, "output_tckEdit_tracks_and_streamlines_weight") 
		if not os.path.exists(weights_folder): os.mkdir(weights_folder)

		if not optimisation_with_tcksift2: 
			weight_txt = os.path.join(weights_folder, "output_tckEdit_streamlines_weights_txt") 
			if not os.path.exists(weight_txt): os.mkdir(weight_txt)

		weight_tck = os.path.join(weights_folder, "output_tckEdit_tracks_tck ") 
		if not os.path.exists(weight_tck): os.mkdir(weight_tck)


		# *****************************************
		# Open seed.txt file to have the number of regions
		# *****************************************

		number_region_all = 0
		seed_data = open(os.path.join(OutSurfaceName,"seeds.txt"), "r")
		for line in seed_data:  
			number_region_all +=1


		# *****************************************
		# Open seed.txt file to compute all radius
		# *****************************************

		all_seed_with_radius,number_point_per_region, region_name_per_region = ([], [], [])
		seed_data = open(os.path.join(OutSurfaceName,"seeds.txt"), "r")
		for line in seed_data: #for each .asc file: for each region 
			# Extract region name information: 
			line = line.strip('\n')
			number_region = line[-9:-4] #remove '.asc' 
			number_region = number_region.replace("/", "")

			# Compute radius of each seed of this specific region:
			list_coord_seeds, number_point = compute_radius_of_each_seed(str(line)) 
			print('Compute radius for region:', number_region, " ,number of points:", number_point)
	
			all_seed_with_radius.append(list_coord_seeds) # a list of list:  a sublist with all seed and radius per region
			number_point_per_region.append(number_point)
			region_name_per_region.append(number_region)


		# *****************************************
		# Initialize connectome and tractography
		# *****************************************

		now = datetime.datetime.now()
		print(now.strftime("Script running Mrtrix: %H:%M %m-%d-%Y"))
		start = time.time()

		# Create connectome and open seed.txt file to compute tractography:
		connectome = np.zeros( (number_region_all, number_region_all) )
		seed_data = open(os.path.join(OutSurfaceName,"seeds.txt"), "r")

		region = 0 
		while region < number_region_all: 
			# Get seeds and radius of the considering region: 
			list_coord_seeds = all_seed_with_radius[region]
			number_point     = number_point_per_region[region]
			number_region    = region_name_per_region[region]

			# Output: tck file generated by tckgen for this region: 
			output_track_tckgen_tck = os.path.join(OUT_MRTRIX_tck, "output_track_tckgen_tck_" + number_region + ".tck")


			# *****************************************
			# Compute tractography with different option and algorithm
			# *****************************************
			
			if os.path.exists(output_track_tckgen_tck):
			    print("Whole-brain streamlines tractography already compute for region " + number_region)
			else:
				print("Compute tractography for region " + number_region )	

				# Output: tck file generated by tckgen for this region: 
				output_track_tckgen_tck = os.path.join(OUT_MRTRIX_tck, "output_track_tckgen_tck_" + number_region + ".tck")

				'''
				# Add seed coordinates for this region:
				for element in  list_coord_seeds[0 : number_point-1]: 

					output_track_tckgen_tck_seed = os.path.join(OUT_MRTRIX_tck, "output_track_tckgen_tck_" + number_region + "_" + 
													str(element[0]) + ',' + str(element[1]) + ',' + str(element[2]) + ',' + str(element[3]) + ".tck")

					
					# Type of algorithm and their specification:
					if tractography_model == "MRtrix (default: IFOD2) ":
						command = [MRtrixPath + "/tckgen", '-algorithm', 'iFOD2', FOD_nii, output_track_tckgen_tck_seed]	

					elif tractography_model == "MRtrix (Tensor-Prob)":
						command = [MRtrixPath + "/tckgen", '-algorithm', 'Tensor_Prob', DiffusionData, output_track_tckgen_tck_seed]

					elif tractography_model == "MRtrix (iFOD1)": 
						command = [MRtrixPath + "/tckgen", '-algorithm', 'iFOD1', FOD_nii, output_track_tckgen_tck_seed]	

				   
					command.append('-seed_sphere')
					command.append(str(element[0]) + ',' + str(element[1]) + ',' + str(element[2]) + ',' + str(element[3]) ) # X Y Z and radius
										   

					# act option: (not a seed option)
					if act_option: 
						command.append('-act')
						command.append(fivett_img)
						

					# Add common parameters: 
					command.append('-select') #default 1000
					command.append(5000)
					command.append('-max_attempts_per_seed')
					command.append(1000)
					#command.append(nb_fibers) #=2000
					
					command.append('-fslgrad')
					command.append(os.path.join(OUT_DIFFUSION, "bvecs"))
					command.append(os.path.join(OUT_DIFFUSION, "bvals"))
					command.append('-mask')
					command.append(DiffusionBrainMask)
					command.append('-nthreads')
					command.append( str(nb_threads))

				    # Run command:
					run = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
					out, err = run.communicate()
					print("MRtrix tractography", "out: ", colored("\n" + str(out) + "\n", 'green')) 
					print("MRtrix tractography", "err: ", colored("\n" + str(err) + "\n", 'red'))	


					#merge files: 
					# for the first one: init
					if list_coord_seeds.index(element) == 1: 
						output_track_tckgen_tck = output_track_tckgen_tck_seed
					else:
						polydatamerge_ascii(output_track_tckgen_tck_seed, output_track_tckgen_tck, output_track_tckgen_tck)
				'''

				# Type of algorithm and their specification:
				if tractography_model == "MRtrix (default: IFOD2) ":
					command = [MRtrixPath + "/tckgen", '-algorithm', 'iFOD2', FOD_nii, output_track_tckgen_tck]	

				elif tractography_model == "MRtrix (Tensor-Prob)":
					command = [MRtrixPath + "/tckgen", '-algorithm', 'Tensor_Prob', DiffusionData, output_track_tckgen_tck]

				elif tractography_model == "MRtrix (iFOD1)": 
					command = [MRtrixPath + "/tckgen", '-algorithm', 'iFOD1', FOD_nii, output_track_tckgen_tck]	

			    # Add seed coordinates for this region:
				for element in  list_coord_seeds[0 : number_point-1]:  
					command.append('-seed_sphere')
					command.append(str(element[0]) + ',' + str(element[1]) + ',' + str(element[2]) + ',' + str(element[3]) ) # X Y Z and radius
									   

				# act option: (not a seed option)
				if act_option: 
					command.append('-act')
					command.append(fivett_img)
					
				# Add common parameters: 
				#command.append('-select')
				#command.append(nb_fibers) # =2 !
				
				command.append('-fslgrad')
				command.append(os.path.join(OUT_DIFFUSION, "bvecs"))
				command.append(os.path.join(OUT_DIFFUSION, "bvals"))
				command.append('-mask')
				command.append(DiffusionBrainMask)
				command.append('-nthreads')
				command.append( str(nb_threads))

			    # Run command:
				run = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
				out, err = run.communicate()
				print("MRtrix tractography", "out: ", colored("\n" + str(out) + "\n", 'green')) 
				print("MRtrix tractography", "err: ", colored("\n" + str(err) + "\n", 'red'))	   
				

				'''
				# *****************************************
				# Convert tractography tck file to vtk format    FOR VISUALIZATION
				# *****************************************
				
				output_track_tckgen_vtk = os.path.join(OUT_MRTRIX_vtk, "output_track_tckgen_tck_" + number_region  + ".vtk")
				if os.path.exists(output_track_tckgen_vtk):
				    print("conversion to vtk already done")
				else:
					print("Convert tck to vtk")									
					run_command("Convert to vtk", [MRtrixPath + "/tckconvert", output_track_tckgen_tck, output_track_tckgen_vtk])
				'''
				

			# Run tcksift: 
			if filtering_with_tcksift: # Filtering with SIFT 

				# *****************************************
				# tcksift: Filter a whole-brain fibre-tracking data
				# *****************************************

				output_tcksift_tck = os.path.join(tcksift_tck, "output_tcksift_" + number_region  + ".tck")
				if os.path.exists(output_tcksift_tck):
				    print("Tractography already filtered with tcksift")
				else:
					print("Filtering Tractography with tcksift")  
					run_command("tcksift ", [MRtrixPath + "/tcksift", output_track_tckgen_tck, FOD_nii, output_tcksift_tck, '-nthreads', str(nb_threads)])	

				  
				'''
				# *****************************************
				# Convert tcksif tck file to vtk format       FOR VISUALIZATION
				# *****************************************
				
				output_tcksift_vtk = os.path.join(tcksift_vtk,"output_tcksift_" + number_region  + ".vtk")
				if os.path.exists(output_tcksift_vtk):
				    print("conversion to vtk already done")
				else:
					print("Convert tck to vtk")									
					run_command("Convert to vtk", [MRtrixPath + "/tckconvert", output_tcksift_tck, output_tcksift_vtk]) 
				'''
				
			
			# *****************************************
			# Complete connectome: 
			# *****************************************

			region_target = 0
			while region_target < number_region_all: 
				# Get seeds and radius of the target region: 
				list_coord_seeds_target = all_seed_with_radius[region_target]
				number_point_target     = number_point_per_region[region_target]
				number_region_target    = region_name_per_region[region_target]
				print('region',region,'(name', number_region,') ,region target', region_target,'(name', number_region_target,')')

				# Outputs:
				output_tckEdit_tracks_tck = os.path.join(weight_tck, "output_tckEdit_tracks_" + number_region + '_with_' + number_region_target + ".tck" )
				
				if not optimisation_with_tcksift2: 
					weight_txt_file = os.path.join(weight_txt, "output_tckEdit_streamlines_weight_" + number_region + '_with_' + number_region_target + ".txt" )
				else: 
					output_tcksift2_txt = os.path.join(tcksift2_txt, "output_tcksift2_streamlines_weights_" + number_region + '_with_' + number_region_target + ".txt")

				# tckEdit:
				if os.path.exists(output_tckEdit_tracks_tck):
					print('tckEdit already done')
				else:

					# *****************************************
					# tckEdit: Perform various editing operations on track files
					# *****************************************

					# Add input track file for tckEdit: file with all streamlines between the considering region and the target region
					if not filtering_with_tcksift: # NO filtering with SIFT before tckEdit
						command = [MRtrixPath + "/tckedit", output_track_tckgen_tck] 
																	   
					else: # Filtering with SIFT before tckEdit
						command = [MRtrixPath + "/tckedit", output_tcksift_tck] #input track file

					# Add output track file:								
					command.append(output_tckEdit_tracks_tck) 
					command.append('-force')

					# Option '-tck_weights_out': specify the path for an output text scalar file containing streamline weights
					if not optimisation_with_tcksift2: 
						command.append('-tck_weights_out')
						command.append(weight_txt_file)

					# Add seed coordinates for target region:
					for element in list_coord_seeds_target[0 : number_point_target-1]:  
						command.append('-include')
						command.append(	str(element[0]) + ',' + str(element[1]) + ',' + str(element[2]) + ',' + str(element[3]) ) # X Y Z and radius

					# Run command (command line too big to be displayed)
					run = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
					out, err = run.communicate()
					print("tckedit ", "out: ", colored("\n" + str(out) + "\n", 'green')) 
					print("tckedit ", "err: ", colored("\n" + str(err) + "\n", 'red'))	
					

				# tcksift2:
				if optimisation_with_tcksift2: 
					# Extract the number of streamlines:
					run = subprocess.Popen([MRtrixPath + "/tckinfo", '-count', output_tckEdit_tracks_tck], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
					out, err = run.communicate()
				
					for i in out.splitlines():
						section = i.split()
						if "count:" in str(section):  
							number_stream_in_output_tckEdit_tracks_tck = int(section[-1])
							break # to avoid 'total_count' section: [b'total_count:', b'4895']
					print('number of stream in the output file ot tckEdit:',number_stream_in_output_tckEdit_tracks_tck )

					# Do tcksift2: 
					if number_stream_in_output_tckEdit_tracks_tck != 0:
						if os.path.exists(output_tcksift2_txt):
						    print("tcksift2 already done")
						else:

							# *****************************************
							# tcksift2: Optimise per-streamline cross-section multipliers to match a whole-brain tractogram to fixel-wise fibre densities
							# *****************************************

							print("Do optimization algorithm tcksift2: ")		
							run_command("tcksift2", [MRtrixPath + "/tcksift2", output_tckEdit_tracks_tck, FOD_nii, output_tcksift2_txt, '-nthreads', str(nb_threads)])	

						# Open output file: 
						weight_file = open(output_tcksift2_txt, 'r')
						first = weight_file.readline() #first line: command line so skip

					# Not execute tcksift2 because of 0 streamline between this 2 regions:
					else: 
						weight_file = 'nan'
						value = 0 #value in the connectivity matrix
				else: 
					# Open output file: 
					weight_file = open(weight_txt_file, 'r')
					

				# Compute value: Value in connectome matrix between this two region: sum of weight of each streamlines 
				if weight_file != 'nan':
					# Sum each weight of each streamline: (just one line file)
					value = 0
					line = weight_file.readline().strip('\n')
					line_list = line.split(" ")
					line_list = line_list[:-1]

					for i in range(len(line_list)):
						value += float(line_list[i])
		
				# Write connectome: 
				connectome[region,region_target] = value
				print('value for the connectivity matrix:', connectome[region,region_target])
				
				region_target += 1
				print("************************")

			print("*********************************************************************")
			region += 1


		# *****************************************
		# Save the MRtrix connectome
		# *****************************************

		connectome_mrtrix = os.path.join(OUT_MRTRIX,'fdt_network_matrix')
		
		# No overwritting:
		if os.path.exists(connectome_mrtrix):
			os.remove(connectome_mrtrix)

		np.savetxt(connectome_mrtrix, connectome.astype(float),  fmt='%f', delimiter='  ')

		print("End of MRtrix: ",time.strftime("%H h: %M min: %S s",time.gmtime( time.time() - start )))




    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # ****************************************  Run DIPY   *********************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************


	elif tractography_model == "DIPY":
		# Doc: https://dipy.org/documentation/1.4.1./reference/ 

		print("*****************************************")
		print("Run tractography with DIPY")
		print("*****************************************")

		# *****************************************
		# Output folder for MRtrix and DIPY 
		# *****************************************
		now = datetime.datetime.now()
		print(now.strftime("Beginning of DIPY: %H:%M %m-%d-%Y"))
		start = time.time()

		OUT_DIPY = os.path.join(OUT_TRACTOGRAPHY, tractography_model.replace(" ", "")) 
		if not os.path.exists(OUT_DIPY):
			os.mkdir(OUT_DIPY)

	
		print("*****************************************")
		print("Convert DWI image to nifti format")
		print("*****************************************")

		DWI_nifti = os.path.join(OUT_DIPY, ID + "-DWI.nii.gz")
		if os.path.exists(DWI_nifti):
		    print("DWI_nifti file: Found Skipping Convert DWI image to nifti format ")
		else:
			print("Convert DWI image to nifti format ")
			run_command("DWIConvert: convert DWI image to nifti format", [DWIConvertPath, "--inputVolume", DWI_DATA, #DWI_NRRD, #DWI after resampling
															                              "--conversionMode", "NrrdToFSL", 
															                              "--outputVolume", DWI_nifti, 
															                              "--outputBValues", os.path.join(OUT_DIPY, "DWI_nifti_bvals"), 
															                              "--outputBVectors", os.path.join(OUT_DIPY, "DWI_nifti_bvecs")])
		
		
		#*****************************************
		# Data and gradient table
		#*****************************************
		
		data, affine, img = load_nifti(DWI_nifti , return_img=True)

		# Gradient_table: create diffusion MR gradients: loads scanner parameters like the b-values and b-vectors
		gtab = gradient_table(os.path.join(OUT_DIPY, "DWI_nifti_bvals"), os.path.join(OUT_DIPY, "DWI_nifti_bvecs"))

		print("data (: DWI) shape: ", data.shape)







		

		print("*****************************************")
		print("WM mask resample in DWI space")
		print("*****************************************")
		wm_mask_in_DWI_space = os.path.join(OUT_DIPY, "white_matter_mask_in_DWI_space.nrrd")
		if os.path.exists(wm_mask_in_DWI_space):
		    print("wm_mask_in_DWI_space FSL file: Found Skipping warp transform")
		else: 
			run_command("WARP_TRANSFORM: WM mask resample in DWI space", [pathWARP_TRANSFORM, "3", wm_mask, wm_mask_in_DWI_space, "-R", B0_BiasCorrect_NRRD, Warp, Affine])


		#*****************************************
		# Mask to restrict tracking to the white matter (and gray matter):  
		#*****************************************

		wm_mask_in_DWI_space_nifti = os.path.join(OUT_DIPY, "white_matter_mask_in_DWI_space_nifti.nii.gz")
		if os.path.exists(wm_mask_in_DWI_space_nifti):
		    print("WM mask FSL file: Found Skipping conversion")
		else: 
			print("DWIConvert WM to FSL format")
			run_command("DWIConvert ", [DWIConvertPath, "--inputVolume", wm_mask_in_DWI_space, #wm_mask, #BRAINMASK
									                                    "--conversionMode", "NrrdToFSL", 
									                                    "--outputVolume", wm_mask_in_DWI_space_nifti, 
									                                    "--outputBVectors", os.path.join(OUT_DIPY, "wm_mask_in_DWI_space_nifti_bvecs.nodif"), 
									                                    "--outputBValues", os.path.join(OUT_DIPY, "wm_mask_in_DWI_space_nifti_bvals.temp")])

		
		reader = vtk.vtkNrrdReader()
		reader.SetFileName(wm_mask_in_DWI_space)
		reader.Update()

		# Save nifti:
		writer = vtk.vtkNIFTIImageWriter()
		writer.SetInputData(reader.GetOutput())
		writer.SetFileName(os.path.join(OUT_DIPY, "test_white_matter_mask_in_DWI_space_nifti.nii.gz"))
		writer.SetInformation(reader.GetInformation())
		writer.Write()
		

	



		# Load_nifti_data: load only the data array from a nifti file
		data_wm_mask_in_DWI_space_nifti = load_nifti_data(wm_mask_in_DWI_space_nifti)  

		# Reshape to have the same shape for DWI (128, 96, 67, 32) and white matter (128, 96, 67)   (before wm: (128, 96, 67,1)  )
		print("data_wm_mask_in_DWI_space_nifti before dowmsampling", data_wm_mask_in_DWI_space_nifti.shape)
		white_matter = data_wm_mask_in_DWI_space_nifti.reshape(data_wm_mask_in_DWI_space_nifti.shape[0:-1]) 


		# In case of the user don't upsampling the DWI 
		if data_wm_mask_in_DWI_space_nifti.shape != data.shape[0:-1] : # need to downsampling wm_mask

			# Preprocessing
			wm_mask_in_DWI_space_dowmsampling = os.path.join(OUT_DIPY, "wm_mask_in_DWI_space_dowmsampled.nrrd")


			# Interpolation / upsampling DWI
			if os.path.exists( wm_mask_in_DWI_space_dowmsampling ):
				print("Files Found: Skipping downsampling wm mask ")
			else:
				command = [pathUnu, "resample", "-i", wm_mask_in_DWI_space, "-s", "x0.5", "x0.5", "x0.5", "-k", "cubic:0,0.5"]
				    
				p1 = subprocess.Popen(command, stdout=subprocess.PIPE)

				command = [pathUnu,"3op", "clamp", "0",'-', "10000000"]
				p2 = subprocess.Popen(command, stdin=p1.stdout, stdout=subprocess.PIPE)

				command = [pathUnu,"save", "-e", "gzip", "-f", "nrrd", "-o", wm_mask_in_DWI_space_dowmsampling]
				p3 = subprocess.Popen(command,stdin=p2.stdout, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
				print( colored("\n"+" ".join(command)+"\n", 'blue'))
				out, err = p3.communicate()
				print("Resample wm mask out: ", colored("\n" + str(out) + "\n", 'green'))
				print("Resample wm mask err: ", colored("\n" + str(err) + "\n", 'red'))


			# Conversion to nifti again 
			wm_mask_in_DWI_space_dowmsampling_nifti = os.path.join(OUT_DIPY, "wm_mask_in_DWI_space_dowmsampled_nifti.nii.gz")
			if os.path.exists(wm_mask_in_DWI_space_dowmsampling_nifti):
			    print("WM mask FSL file: Found Skipping conversion")
			else: 
				print("DWIConvert WM to FSL format")
				run_command("DWIConvert ", [DWIConvertPath, "--inputVolume", wm_mask_in_DWI_space_dowmsampling,
										                                    "--conversionMode", "NrrdToFSL", 
										                                    "--outputVolume", wm_mask_in_DWI_space_dowmsampling_nifti, 
										                                    "--outputBVectors", os.path.join(OUT_DIPY, "wm_mask_in_DWI_space_dowmsampling_nifti_bvecs.nodif"), 
										                                    "--outputBValues", os.path.join(OUT_DIPY, "wm_mask_in_DWI_space_dowmsampling_nifti_bvals.temp")])

			# Load_nifti_data: load only the data array from a nifti file
			data_wm_mask_in_DWI_space_dowmsampling_nifti = load_nifti_data(wm_mask_in_DWI_space_dowmsampling_nifti)  

			# Reshape to have the same shape for DWI (128, 96, 67, 32) and white matter (128, 96, 67)   (before wm: (128, 96, 67,1)  )
			print("data_wm_mask_in_DWI_space_dowmsampling_nifti after dowmsampling", data_wm_mask_in_DWI_space_dowmsampling_nifti.shape)
			white_matter = data_wm_mask_in_DWI_space_dowmsampling_nifti.reshape(data_wm_mask_in_DWI_space_dowmsampling_nifti.shape[0:-1]) 








		
		if gm_mask != '': # gray matter mask provided by the user
		
			print("*****************************************")
			print("gm mask resample in DWI space")
			print("*****************************************")
			gm_mask_in_DWI_space = os.path.join(OUT_DIPY, "gray_matter_mask_in_DWI_space.nrrd")
			if os.path.exists(gm_mask_in_DWI_space):
			    print("gm_mask_in_DWI_space FSL file: Found Skipping warp transform")
			else: 
				run_command("WARP_TRANSFORM: gm mask resample in DWI space", [pathWARP_TRANSFORM, "3", gm_mask, gm_mask_in_DWI_space, "-R", B0_BiasCorrect_NRRD, Warp, Affine])


			#*****************************************
			# Mask to restrict tracking to the white matter (and gray matter):  
			#*****************************************

			gm_mask_in_DWI_space_nifti = os.path.join(OUT_DIPY, "gray_matter_mask_in_DWI_space_nifti.nii.gz")
			if os.path.exists(gm_mask_in_DWI_space_nifti):
			    print("gm mask FSL file: Found Skipping conversion")
			else: 
				print("DWIConvert gm to FSL format")
				run_command("DWIConvert ", [DWIConvertPath, "--inputVolume", gm_mask_in_DWI_space, #gm_mask, #BRAINMASK
										                                    "--conversionMode", "NrrdToFSL", 
										                                    "--outputVolume", gm_mask_in_DWI_space_nifti, 
										                                    "--outputBVectors", os.path.join(OUT_DIPY, "gm_mask_in_DWI_space_nifti_bvecs.nodif"), 
										                                    "--outputBValues", os.path.join(OUT_DIPY, "gm_mask_in_DWI_space_nifti_bvals.temp")])

			# Load_nifti_data: load only the data array from a nifti file
			data_gm_mask_in_DWI_space_nifti = load_nifti_data(gm_mask_in_DWI_space_nifti)  

			# Reshape to have the same shape for DWI (128, 96, 67, 32) and white matter (128, 96, 67)   (before gm: (128, 96, 67,1)  )
			print("data_gm_mask_in_DWI_space_nifti before dogmsampling", data_gm_mask_in_DWI_space_nifti.shape)
			gray_matter = data_gm_mask_in_DWI_space_nifti.reshape(data_gm_mask_in_DWI_space_nifti.shape[0:-1]) 


			# In case of the user don't upsampling the DWI 
			if data_gm_mask_in_DWI_space_nifti.shape != data.shape[0:-1] : # need to downsampling gm_mask

				# Preprocessing
				gm_mask_in_DWI_space_dowmsampling = os.path.join(OUT_DIPY, "gm_mask_in_DWI_space_dowmsampled.nrrd")


				# Interpolation / upsampling DWI
				if os.path.exists( gm_mask_in_DWI_space_dowmsampling ):
					print("Files Found: Skipping downsampling gm mask ")
				else:
					command = [pathUnu, "resample", "-i", gm_mask_in_DWI_space, "-s", "x0.5", "x0.5", "x0.5", "-k", "cubic:0,0.5"]
					    
					p1 = subprocess.Popen(command, stdout=subprocess.PIPE)

					command = [pathUnu,"3op", "clamp", "0",'-', "10000000"]
					p2 = subprocess.Popen(command, stdin=p1.stdout, stdout=subprocess.PIPE)

					command = [pathUnu,"save", "-e", "gzip", "-f", "nrrd", "-o", gm_mask_in_DWI_space_dowmsampling]
					p3 = subprocess.Popen(command,stdin=p2.stdout, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
					print( colored("\n"+" ".join(command)+"\n", 'blue'))
					out, err = p3.communicate()
					print("Resample gm mask out: ", colored("\n" + str(out) + "\n", 'green'))
					print("Resample gm mask err: ", colored("\n" + str(err) + "\n", 'red'))


				# Conversion to nifti again 
				gm_mask_in_DWI_space_dowmsampling_nifti = os.path.join(OUT_DIPY, "gm_mask_in_DWI_space_dowmsampled_nifti.nii.gz")
				if os.path.exists(gm_mask_in_DWI_space_dowmsampling_nifti):
				    print("gm mask FSL file: Found Skipping conversion")
				else: 
					print("DWIConvert gm to FSL format")
					run_command("DWIConvert ", [DWIConvertPath, "--inputVolume", gm_mask_in_DWI_space_dowmsampling,
											                                    "--conversionMode", "NrrdToFSL", 
											                                    "--outputVolume", gm_mask_in_DWI_space_dowmsampling_nifti, 
											                                    "--outputBVectors", os.path.join(OUT_DIPY, "gm_mask_in_DWI_space_dowmsampling_nifti_bvecs.nodif"), 
											                                    "--outputBValues", os.path.join(OUT_DIPY, "gm_mask_in_DWI_space_dowmsampling_nifti_bvals.temp")])

				# Load_nifti_data: load only the data array from a nifti file
				data_gm_mask_in_DWI_space_dowmsampling_nifti = load_nifti_data(gm_mask_in_DWI_space_dowmsampling_nifti)  

				# Reshape to have the same shape for DWI (128, 96, 67, 32) and white matter (128, 96, 67)   (before gm: (128, 96, 67,1)  )
				print("data_gm_mask_in_DWI_space_dowmsampling_nifti after dowmsampling", data_gm_mask_in_DWI_space_dowmsampling_nifti.shape)
				gray_matter = data_gm_mask_in_DWI_space_dowmsampling_nifti.reshape(data_gm_mask_in_DWI_space_dowmsampling_nifti.shape[0:-1]) 












	


		streamline_vtk = os.path.join(OUT_DIPY,"streamlines.vtk")
		if not os.path.exists(streamline_vtk):

	        #*****************************************
			# Method for getting directions from a diffusion data set
			#*****************************************

			if len(list_bval_for_the_tractography) == 1: # single shell_DWI: 
				print("single shell")
				# auto_response_ssst: Automatic estimation of SINGLE-SHELL single-tissue (ssst) response
				response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=wm_fa_thr) # 0.7: adult 

				# Fit a Constrained Spherical Deconvolution (CSD) model: 
				csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6) 


			else: # multi shell DWI
				print("multi shell")

				# Fiber response function estimation for multi-shell data: 
				ubvals = unique_bvals_tolerance(gtab.bvals)
				print("ubvals", ubvals) #[   0. 1004.] 

				# Automatic estimation of multi-shell multi-tissue (msmt) response: 
				auto_response_wm, auto_response_gm, auto_response_csf = auto_response_msmt(gtab, data, roi_radii=10, wm_fa_thr=wm_fa_thr,  # 0.7
																													gm_fa_thr=gm_fa_thr,  # 0.3
																													csf_fa_thr=csf_fa_thr, # 0.15
																													gm_md_thr=gm_md_thr, # 0.001 
																													csf_md_thr=csf_md_thr) # 0.0032
				response_mcsd = multi_shell_fiber_response(sh_order=8, bvals=ubvals, wm_rf=auto_response_wm, gm_rf=auto_response_gm, csf_rf=auto_response_csf)

				# Fit a Constrained Spherical Deconvolution (CSD) model: 
				mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)
			

			print("*****************************************")
			print("End of getting directions: ",time.strftime("%H h: %M min: %S s",time.gmtime( time.time() - start )))
			print("*****************************************")


	        #*****************************************
			# Stopping criterion: a method for identifying when the tracking must stop: restricting the fiber tracking to areas with good directionality information
			#*****************************************

			# Conversion to nifti  
			brain_nifti = os.path.join(OUT_DIPY, "brain.nii.gz")
			if os.path.exists(brain_nifti):
			    print("brain_nifti mask FSL file: Found Skipping conversion")
			else: 
				print("DWIConvert brain_nifti to FSL format")
				run_command("DWIConvert ", [DWIConvertPath, "--inputVolume", BRAINMASK,
										                                    "--conversionMode", "NrrdToFSL", 
										                                    "--outputVolume", brain_nifti, 
										                                    "--outputBVectors", os.path.join(OUT_DIPY, "brain_nifti_bvecs.nodif"), 
										                                    "--outputBValues", os.path.join(OUT_DIPY, "brain_nifti_bvals.temp")])

			# Load_nifti_data: load only the data array from a nifti file
			data_brain= load_nifti_data(brain_nifti)  

			# Reshape to have the same shape for DWI (128, 96, 67, 32) and gray matter (128, 96, 67)   (before gm: (128, 96, 67,1)  )
			print("data_brain after dowmsampling", data_brain.shape)
			brainmask_array = data_brain.reshape(data_brain.shape[0:-1]) 




			# We use the GFA (similar to FA but ODF based models) of the CSA model to build a stopping criterion.
			# Fit the data to a Constant Solid Angle ODF Model: estimate the Orientation Distribution Function (ODF) at each voxel
			csa_model = CsaOdfModel(gtab, sh_order=6) 
			gfa = csa_model.fit(data, mask= brainmask_array).gfa 


			# Restrict fiber tracking to white matter mask where the ODF shows significant restricted diffusion by thresholding on the Generalized Fractional Anisotropy (GFA)
			# https://dipy.org/documentation/1.4.1./reference/dipy.tracking/#thresholdstoppingcriterion 
			stopping_criterion = ThresholdStoppingCriterion(gfa, .25)  # default value: data < .25

			
			
			#stopping_criterion = BinaryStoppingCriterion(brainmask_array ==1)



			print("*****************************************")
			print("End of stopping criterion method: ",time.strftime("%H h: %M min: %S s",time.gmtime( time.time() - start )))
			print("*****************************************")


	        #*****************************************
			# A set of seeds from which to begin tracking: the seeds chosen will depend on the pathways one is interested in modeling
			#*****************************************

			seed_mask = white_matter #gray_matter #white_matter 
			# Create seeds for fiber tracking from a binary mask: 
			seeds = utils.seeds_from_mask(seed_mask, affine, density=1) 
			print("seeds", seeds ) 

			# The peaks of an ODF are good estimates for the orientation of tract segments at a point in the image
			# peaks_from_model: fit the data and calculated the fiber directions in all voxels of the white matter
			# .peaks_from_model(model, data, sphere, relative_peak_threshold, min_separation_angle, mask=None, return_sh=True, gfa_thr=0, parallel=False ...)
			if len(list_bval_for_the_tractography) == 1: # single shell: 
				peaks = peaks_from_model(csd_model, data, default_sphere, .5, 25, mask=white_matter, return_sh=True, parallel=True) 
			else: 
				peaks = peaks_from_model(mcsd_model, data, default_sphere, .5, 25, mask=white_matter, return_sh=True, parallel=True) 

			# shm_coeff: the spherical harmonic coefficients of the odf: 
			fod_coeff = peaks.shm_coeff


			print("*****************************************")
			print("End of peaks of an ODF method: ",time.strftime("%H h: %M min: %S s",time.gmtime( time.time() - start )))
			print("*****************************************")


			# Discrete Fiber Orientation Distribution (FOD) used by the ProbabilisticDirectionGetter as a PMF for sampling tracking directions.
			# ProbabilisticDirectionGetter: Randomly samples direction of a sphere based on probability mass function (PMF) 
			# from_shcoeff: Probabilistic direction getter from a distribution of directions on the sphere
			prob_dg = ProbabilisticDirectionGetter.from_shcoeff(fod_coeff, max_angle=30., sphere=default_sphere) 

			print("*****************************************")
			print("End of tracking directions: ",time.strftime("%H h: %M min: %S s",time.gmtime( time.time() - start )))
			print("*****************************************")


	        #*****************************************
		    # Generate streamlines
		    #*****************************************

			# Initialization of LocalTracking: Creates streamlines by using local fiber-tracking
			streamlines_generator = LocalTracking(prob_dg, stopping_criterion, seeds, affine, step_size=.5, return_all=False)

			# Generate streamlines object:  streamlines = ArraySequence object
			streamlines = Streamlines(streamlines_generator)
			save_vtk_streamlines(streamlines, streamline_vtk, to_lps=True, binary=False)

			print("*****************************************")
			print("End of generate streamlines: ",time.strftime("%H h: %M min: %S s",time.gmtime( time.time() - start )))
			print("*****************************************")


		else: # tractogram already compute 
			print("Streamlines already computed")
				


        #*****************************************
		# Extract the connectivity matrix
		#*****************************************
	
		matrix = os.path.join(OUT_DIPY, "fdt_network_matrix")

		if not os.path.exists(matrix): 

			print("*****************************************")
			print("Before create connectivity matrix: ",time.strftime("%H h: %M min: %S s",time.gmtime( time.time() - start )))
			print("*****************************************")
			
			# *****************************************
			# Initialize connectome
			# *****************************************

			# Open seed.txt file to have the number of regions:
			number_region_all = 0
			seed_data = open(os.path.join(OutSurfaceName,"seeds.txt"), "r")
			for line in seed_data:  
				number_region_all +=1

			connectome = np.zeros( (number_region_all, number_region_all) )


			# *****************************************
			# Initialize list of region in the same order than Matrix Row:
			# *****************************************

			# Get the parcellation table with Cortical and Subcortical regions: 
			with open(only_matrix_parcellation_table, "r") as table_json_file:
				table_json_object = json.load(table_json_file)

			# Get data points for connected and unconnected points: 
			list_region_unordered, list_matrixrow_unordered, list_region = ([],[],[])

			for key in table_json_object:    
				list_region_unordered.append(int(key["labelValue"]))
				list_matrixrow_unordered.append(key["MatrixRow"])

			# Sort regions by MatrixRow number: 
			sorted_indices = np.argsort(list_matrixrow_unordered)

			for i in range(len(list_matrixrow_unordered)):
				index = sorted_indices[i]
				list_region.append(list_region_unordered[index])

			#print("list_region", list_region)




			# Read the source file
			reader = vtk.vtkPolyDataReader() 
			reader.SetFileName(SURFACE)
			reader.Update()

			points = vtk_to_numpy( reader.GetOutput().GetPoints().GetData() )
			#print("before",points )
			for point in points: 
				point[0] = - point[0] #x
				point[1] = - point[1] #y


			points_vtk = numpy_to_vtk(points, deep=True)
		
			reader.GetOutput().GetPoints().SetData(points_vtk)
			reader.Update()
			
			polydata = reader.GetOutput()
			points = vtk_to_numpy( polydata.GetPoints().GetData() )
			#print("after",points )


			polydata = vtk.vtkPolyData()
			apd = vtk.vtkAppendPolyData()
			apd.AddInputData(reader.GetOutput())

			apd.Update()
			polydata = apd.GetOutput()

			writer = vtk.vtkPolyDataWriter()
			writer.SetFileName(os.path.join(OUT_DIPY,"surface_in_same_space_than_mask.vtk"))

			writer.SetInputData(polydata)
			writer.SetFileTypeToASCII()
			writer.Update()

			try:
				writer.Write()
				print("Merging done!")
			except:
				print("Error while saving file.")
				exit()




			
			
			# Read the source file
			reader = vtk.vtkPolyDataReader() 
			reader.SetFileName(os.path.join(OUT_DIPY,"surface_in_same_space_than_mask.vtk"))
			reader.Update()  

			# Get all points of my surfaces: 
			list_scalar_surface_labeled = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(0)).tolist()
			#print("list_scalar_surface_labeled", list_scalar_surface_labeled) #[11106. 11166. 11112. ... 11166.]
			print("list_scalar_surface_labeled shape ", len(list_scalar_surface_labeled))	#(354708,)

			# Get all scalars of my surface:  
			list_points_surface_labeled = vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).tolist()
			#print("list_points_surface_labeled",list_points_surface_labeled ) #[[-36.6685   85.5245   11.6233 ] ... [-58.3593   46.634   -32.0077 ]]
			print("list_points_surface_labeled shape ", len(list_points_surface_labeled)) #(354708, 3, 1)


			# Streamlines: 
			reader_streamlines = vtk.vtkPolyDataReader() 
			reader_streamlines.SetFileName(streamline_vtk)
			reader_streamlines.Update()
			print("streamlines", reader_streamlines.GetOutput())


			nbCell = reader_streamlines.GetOutput().GetNumberOfCells()
			print("nbCell: ", nbCell) #87407

			array_inter = 0

			print("Before loop fibers: ",time.strftime("%H h: %M min: %S s",time.gmtime( time.time() - start )))

			for i_cell in range(50): #nbCell):

				#vtkfiber = ExtractFiber(reader_streamlines.GetOutput(), i_cell)
				#print("vtkfiber", vtkfiber)
				ids = vtk.vtkIdTypeArray()
				ids.SetNumberOfComponents(1)
				ids.InsertNextValue(i_cell) 

				# extract a subset from a dataset
				selectionNode = vtk.vtkSelectionNode() 
				selectionNode.SetFieldType(0)
				selectionNode.SetContentType(4)
				selectionNode.SetSelectionList(ids) 

				# set containing cell to 1 = extract cell
				selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1) 

				selection = vtk.vtkSelection()
				selection.AddNode(selectionNode)

				# extract the cell from the cluster
				extractSelection = vtk.vtkExtractSelection()
				extractSelection.SetInputData(0, reader_streamlines.GetOutput())
				extractSelection.SetInputData(1, selection)
				extractSelection.Update()

				# convert the extract cell to a polygonal type (a line here)
				geometryFilter = vtk.vtkGeometryFilter()
				geometryFilter.SetInputData(extractSelection.GetOutput())
				geometryFilter.Update()

				# Save fiber in a tube: 
				tf = vtk.vtkTubeFilter()
				tf.SetInputData(geometryFilter.GetOutput())
				tf.SetRadius(0.1)
				tf.SetNumberOfSides(20)
				tf.Update()
				writer = vtk.vtkPolyDataWriter()
				writer.SetFileName(os.path.join(OUT_DIPY,"fiber_tube.vtk"))
				writer.SetInputData(tf.GetOutput())
				writer.Update()
				#print("tube", tf.GetOutput())
				
				try:
					writer.Write()
					#print("Tube created!")
				except:
					print("Error while saving tube.")
					exit()


				reader_tube = vtk.vtkPolyDataReader() 
				reader_tube.SetFileName(os.path.join(OUT_DIPY,"fiber_tube.vtk"))
				reader_tube.Update()



				intersectionPolyDataFilter = vtk.vtkIntersectionPolyDataFilter()
				intersectionPolyDataFilter.SetInputConnection( 0, reader_streamlines.GetOutputPort())
				intersectionPolyDataFilter.SetInputConnection( 1, reader_tube.GetOutputPort())
				intersectionPolyDataFilter.SetTolerance(20)
				intersectionPolyDataFilter.Update()

				print("intersec point:", intersectionPolyDataFilter.GetNumberOfIntersectionPoints(), "intersec line:",intersectionPolyDataFilter.GetNumberOfIntersectionLines())


				# VISUALISATION:
				writer = vtk.vtkPolyDataWriter()
				writer.SetFileName(os.path.join(OUT_DIPY,"intersectionPolyDataFilter.vtk"))

				writer.SetInputData(intersectionPolyDataFilter.GetOutput())
				writer.Update()

				try:
					writer.Write()
					#print("Merging intersectionPolyDataFilter done!")
				except:
					print("Error while saving intersectionPolyDataFilter file.")
					exit()




				reader_intersection = vtk.vtkPolyDataReader() 
				reader_intersection.SetFileName(os.path.join(OUT_DIPY,"intersectionPolyDataFilter.vtk"))
				reader_intersection.Update()

				array_inter += len(vtk_to_numpy(reader_intersection.GetOutput().GetPoints().GetData()).tolist()) #.GetArray(1)).tolist()  # change to find the name  
				#print("array_inter", array_inter)

				# *****************************************
				# Write connectivity matrix
				# *****************************************

				# Add value to the connectivity matrix: 
				if len(vtk_to_numpy(reader_intersection.GetOutput().GetPoints().GetData()).tolist()) !=0: #nb point !=0
					print("points !")

					# remove duplicate ? 
					for first_scalar in vtk_to_numpy(reader_intersection.GetOutput().GetPointData().GetArray(0)).tolist():  #list of scalars in my intersection 
						for second_scalar in vtk_to_numpy(reader_intersection.GetOutput().GetPointData().GetArray(0)).tolist():  #list of scalars in my intersection 
							# Get index in the connectivity matrix 
							first_index = list_region.index(first_scalar)
							second_index = list_region.index(second_scalar)

							connectome[first_index, second_index] += 1
					

			print("After loop fibers: ",time.strftime("%H h: %M min: %S s",time.gmtime( time.time() - start )))
			print("array_inter ", array_inter)

			print("reader_intersection:", reader_intersection.GetOutput())

					

			exit()


			


			'''
			# Save the connectivity matrix 
			np.savetxt(matrix, connectome.astype(float),  fmt='%f', delimiter='  ')
			
			plt.imshow(np.log1p(M), interpolation='nearest')
			plt.savefig(os.path.join(OUT_DIPY, "connectivity.png"))
			'''
		

		print("*****************************************")
		print("End of DIPY: ",time.strftime("%H h: %M min: %S s",time.gmtime( time.time() - start )))
		print("*****************************************")
