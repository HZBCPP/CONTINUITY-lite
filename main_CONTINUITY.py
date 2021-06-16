#!/usr/bin/env python3
import argparse
import json
import os 
import sys 
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from csv import reader, DictReader

from interface_functions import *
from CONTINUITY_functions import *


##########################################################################################################################################

     # CONTINUITY : connectivity tools which include subcortical regions as seed and target for connectivity 

##########################################################################################################################################

def run_command(text_printed, command):
    # Display command:
    print(colored("\n"+" ".join(command)+"\n", 'blue'))
    # Run command and display output and error:
    run = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = run.communicate()
    print(text_printed, "out: ", colored("\n" + str(out) + "\n", 'green')) 
    print(text_printed, "err: ", colored("\n" + str(err) + "\n", 'red'))




if __name__ == '__main__':

    '''
    left = "/work/elodie/CONTINUITY with all data/CONTINUITY_QC/Destrieux_points/icbm_avg_mid_sym_mc_left_hires.vtk"
    right = "/work/elodie/CONTINUITY with all data/CONTINUITY_QC/Destrieux_points/icbm_avg_mid_sym_mc_right_hires.vtk"
    out = "/work/elodie/CONTINUITY/CONTINUITY_QC/icbm_merge.vtk"

    polydatamerge_ascii(left, right, out)

    out_nifti = "/work/elodie/CONTINUITY/CONTINUITY_QC/icbm_merge.nii.gz"

    slicer.util.saveNode(slicer.util.loadVolume(out), out_nifti)




    out_nrrd = "/work/elodie/CONTINUITY/CONTINUITY_QC/icbm_merge.nrrd"
    run_command("template", ["/tools/bin_linux64/DWIConvert", "--inputVolume", out_nifti, 
                                                                                   "--conversionMode", "FSLToNrrd", 
                                                                                   "--outputVolume", out_nrrd] )
    '''
    










    dir_path = os.path.realpath(os.path.dirname(__file__))
   
    # *****************************************
    # Argparse
    # *****************************************

    parser = argparse.ArgumentParser(description='Main CONTINUITY')
    parser.add_argument('-default_config_filename', nargs='?', type=str, help="json with default configuration")
    parser.add_argument('-csv_file'               , nargs='?', type=str, help="csv file with data information for one or several subject") 

    # Intern default configuration json file to add all arguments even if the defaut json given by user is corrupted (= missed arguments)
    default_config_filename = dir_path + "/args_setup.json" #"/work/elodie/testing/args_main_CONTINUITY_completed_test_no_create_SALT.json" #
   
    with open(default_config_filename) as default_file: 
        data_default = json.load(default_file)

    # Add other arguments to allow command line modification:
    for categories, infos in data_default.items():
        if categories == "Arguments": 
            for key in infos:
                if key == "noGUI" or key == "cluster":
                    parser.add_argument('-%s' % key, help=infos[key]['help'], action='store_true') # --> default value = False
                else:
                    parser.add_argument('-%s' % key, type=eval(infos[key]['type']), help=infos[key]['help'], default=infos[key]['default'], metavar='')

    args = vars( parser.parse_args() )

    ''' Display arguments values
    for key, val in args.items():
        print("args:",key ,": '",args[key],"'")  
    print("noGUI:",args['noGUI'], "cluster:",args['cluster'], "csv_file:",args['csv_file'], "default_config_filename:",args['default_config_filename'])
    '''

    # *****************************************
    # Initialization of user file: intern file to store all information 
    # *****************************************

    # 'Real' default configuration file: default configuration given by the user (not intern default configuration file)
    if args['default_config_filename'] != None :
        default_config_filename = args["default_config_filename"]
   
    with open(default_config_filename) as default_file: #args_setup.json"
        data_default = json.load(default_file)    

    # User file
    OUT_HOME = os.getenv("HOME") + "/CONTINUITY_json_file"
    if not os.path.exists( OUT_HOME ):
        os.mkdir(OUT_HOME)

    now = datetime.datetime.now()
    date = str(now.strftime("%H_%M_%S_%m-%d-%Y"))

    user_filename = OUT_HOME + "/user_args_CONTINUITY_" + date + ".json"
    shutil.copy(default_config_filename, user_filename)

    # Initialization of user file with default values in json default file provide by the user 
    with open(user_filename) as user_file:
        global data_user
        data_user = json.load(user_file)

    for categories, infos in data_user.items():
        for key in infos: 
            # change 'default' by 'value'
            d = data_user[categories][key]
            d['value'] = d.pop('default')

        with open(user_filename, "w+") as user_file: 
                user_file.write(json.dumps(data_user, indent=4)) 

    

    # *****************************************
    # Run CONTINUITY thanks to a command line: -noGUI / -cvs_file / -cluster
    # *****************************************

    if args["noGUI"]: 
        
        with open(user_filename) as user_file:
            data_user = json.load(user_file)

        # Write values provide by user (thanks to the command line)in json user file 
        for categories, infos in data_default.items():
            if categories == "Arguments": 
                for key in infos: 
                    if str(args[key]) != " ":
                        data_user[categories][key]['value'] = args[key]

                    with open(user_filename, "w+") as user_file: 
                        user_file.write(json.dumps(data_user, indent=4)) 

        # Find and write localisation of executables            
        executable_path(default_config_filename, user_filename)



        # Create the output folder
        if not os.path.exists( data_user['Parameters']["OUT_PATH"]["value"] ):
            os.mkdir(data_user['Parameters']["OUT_PATH"]["value"])

        OUT_FOLDER = os.path.join(data_user['Parameters']["OUT_PATH"]["value"],data_user['Parameters']["ID"]["value"]) #ID
        if not os.path.exists( OUT_FOLDER ):
            os.mkdir(OUT_FOLDER)
    

        # *****************************************
        # Run CONTINUITY thanks to a command line ONLY: -noGUI
        # *****************************************

        if args['csv_file'] == None: # no csv file provide by the user

            # Test if the user provides all required arguments
            list_of_args_required = []
            with open(user_filename) as user_file:
                data_user = json.load(user_file)

            for categories, infos in data_default.items():
                if categories == "Arguments": 
                    for key in infos: 
                        if data_user[categories][key]['value'] == "required":
                            list_of_args_required.append('-%s' % key) 

            if len(list_of_args_required) != 0:  
                print(str(list_of_args_required)[1:-1] ,"required for CONTINUITY script")
                sys.exit()  

            # Run CONTINUITY script 
            if not args["cluster"]:  # Run localy: -noGUI  
                CONTINUITY(user_filename)
            else: # run in longleaf: -noGUI -cluster 
                cluster(OUT_FOLDER + "/slurm-job", data_user['Parameters']["cluster_command_line"]["value"], 
                        data_user['Parameters']["OUT_PATH"]["value"], data_user['Parameters']["ID"]["value"], user_filename)


        # *****************************************
        # Run CONTINUITY thanks to a command line by providing a csv file:  -noGUI -csv_file 
        # *****************************************

        else:  #args['csv_file'] != ''
            with open(user_filename) as user_file:
                data_user = json.load(user_file)

            with open(args['csv_file'], 'r') as csv_file:
                csv_dict_reader = DictReader(csv_file)

                header = csv_dict_reader.fieldnames
                print("header: ",header )
                
                # Iterate over each row after the header in the csv
                for row in csv_dict_reader:
                    print("info subject:",row)
                    for element in header: 
                        data_user['Arguments'][element]['value'] = row[element]

                        with open(user_filename, "w+") as user_file: 
                            user_file.write(json.dumps(data_user, indent=4)) 

                    # Run CONTINUITY script
                    if not args["cluster"]: # run in longleaf: -noGUI -csv_file -cluster 
                        print("SUBJECT: ", row['ID'] )
                        CONTINUITY(user_filename)
                    else: # Run localy: -noGUI -csv_file
                        cluster(OUT_FOLDER + "/slurm-job", data_user['Parameters']["cluster_command_line"]["value"], 
                                data_user['Parameters']["OUT_PATH"]["value"], data_user['Parameters']["ID"]["value"], user_filename)

        
    # *****************************************
    # Run CONTINUITY thanks to an interface (default)
    # *****************************************

    else: 
        print("PyQt5 CONTINUITY interface")

        qt_args = []
        qt_args.append(sys.argv[-1])
        qt_args.append(default_config_filename)
        qt_args.append(user_filename)
        sys.argv = qt_args

        app = QtWidgets.QApplication(sys.argv)
        window = Ui()
        app.exec_()