#!/usr/bin/env python3
from __future__ import print_function
import argparse
import json
import os 
import subprocess
from termcolor import colored

##########################################################################################################################################
'''  
     CONTINUITY open Slicer scipt: Open Slicer with a python script to load specific files and parameters
'''  
##########################################################################################################################################

# *****************************************
# Parameters
# *****************************************

parser = argparse.ArgumentParser(description='python script for Slicer')
parser.add_argument("user_json_filename", help = "json file with arguments like ID, OUTPATH and parameters for view controllers", type = str)
args = parser.parse_args()

user_json_filename = args.user_json_filename
with open(user_json_filename, "r") as user_Qt_file:
    json_user_object = json.load(user_Qt_file)


# *****************************************
# Main
# *****************************************

if __name__ == '__main__':

    print("Open Slicer with a python script")

    if os.path.exists( "./CONTINUITY_QC/python_script_for_Slicer.py"):
    	command = [json_user_object['Executables']["slicer"]["value"], "--python-script", "./CONTINUITY_QC/python_script_for_Slicer.py", user_json_filename] 
    else:
    	command = [json_user_object['Executables']["slicer"]["value"], "--python-script", "./python_script_for_Slicer.py", user_json_filename] 

    print( colored(" ".join(command), 'blue'))
    slicer_Run = subprocess.Popen(command, stderr=subprocess.PIPE ,universal_newlines=True)
    out, err = slicer_Run.communicate()
    print("slicer_Run out: \n", colored(out, 'green')) 
    print("slicer_Run err: \n", colored(err, 'red'))