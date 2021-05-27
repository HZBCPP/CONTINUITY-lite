import os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='CONTINUITY script: writeSeedList script')
parser.add_argument("subject_dir", help = "Subject directorie", type = str) 
parser.add_argument("jsonFile", help = "JsonFile", type = str)
args = parser.parse_args()

def main(args.subject_dir , args.jsonFile ):
	DIR_Surfaces = os.path.join(args.subject_dir, 'labelSurfaces')

	#Open Json file and parse 
	with open(args.jsonFile) as data_file:    
	    data = json.load(data_file)

	#Create file for seedList
	seedPath = args.subject_dir + '/seeds.txt'
	seedList = open(seedPath, 'w')

	#Put all MatrixRow to -1 
	for seed in data:
	  seed['MatrixRow']=-1

	seedID = 0 

	for j in data:
	    filename = os.path.join(DIR_Surfaces, str(j["AAL_ID"]) + ".asc")
	    j['MatrixRow'] = seedID
	    seedID = seedID + 1
	    seedList.write(filename + "\n")
	     
	seedList.close()

	#Update JSON file 
	with open(args.jsonFile, 'w') as txtfile:
	    json.dump(data, txtfile, indent = 2)



if __name__ == '__main__':
	main(args.subject_dir, args.jsonFile )