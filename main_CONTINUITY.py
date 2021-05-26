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

if __name__ == '__main__':
   
    # *****************************************
    # Argparse
    # *****************************************

    parser = argparse.ArgumentParser(description='Main CONTINUITY')
    parser.add_argument('-default_config_filename', nargs='?', type=str, help="json with default configuration")
    parser.add_argument('-csv_file'               , nargs='?', type=str, help="csv file with data information for one or several subject") 

    # Intern default configuration json file to add all arguments even if the defaut json given by user is corrupted (= missed arguments)
    default_config_filename = "./CONTINUITY_ARGS/args_main_CONTINUITY_completed_test.json" #args_setup.json"
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

    # Display arguments values
    '''
    for key, val in args.items():
        print("args:",key ,": '",args[key],"'")  
    print("noGUI:",args['noGUI'] )
    print("cluster:",args['cluster'] )
    print("csv_file:",args['csv_file'] )
    print("default_config_filename:",args['default_config_filename'] )
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
    user_filename = "./CONTINUITY_ARGS/args_main_CONTINUITY.json" 
    with open(user_filename) as user_file:
        global data_user
        data_user = json.load(user_file)


    # Initialization of user file with default values in json default file provide by the user 
    for categories, infos in data_default.items():
        for key in infos: 
            data_user[categories][key]['value'] = data_default[categories][key]['default']

            with open(user_filename, "w+") as user_file: 
                user_file.write(json.dumps(data_user, indent=4)) 


    # Write a csv file (just for testing)  
    #write_csv_file("./csv_CONTINUITY.csv", default_config_filename)
    

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
                cluster("./slurm-job", data_user['Parameters']["cluster_command_line"]["value"])



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
                        cluster("./slurm-job", data_user['Parameters']["cluster_command_line"]["value"])

        

    # *****************************************
    # Run CONTINUITY thanks to an interface (default)
    # *****************************************

    else: 
        print("PyQt5 CONTINUITY interface")
        app = QtWidgets.QApplication(sys.argv)
        window = Ui()
        app.exec_()
