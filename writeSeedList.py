import os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='CONTINUITY script: writeSeedList script')
parser.add_argument("subject_dir", help = "Subject directorie", type = str) 
parser.add_argument("overlapName", help = "OverlapName", type = str) 
parser.add_argument("jsonFile", help = "JsonFile", type = str)
args = parser.parse_args()

subject_dir = args.subject_dir
overlapName = args.overlapName 
jsonFile = args.jsonFile



def main(subject_dir, overlapName, jsonFile ):
	DIR_Surfaces = os.path.join(subject_dir, 'labelSurfaces')

	#Open Json file and parse 
	with open(jsonFile) as data_file:    
	    data = json.load(data_file)

	#Create file for seedList
	seedPath = subject_dir + '/seeds.txt'
	seedList = open(seedPath, 'w')

	#Put all MatrixRow to -1 
	for seed in data:
	  seed['MatrixRow']=-1

	seedID = 0 

	for j in data:
	    filename = os.path.join(DIR_Surfaces, str(j["AAL_ID"]) + ".asc") # AAL_ID present for Destrieux table too  , file created by ExtractLabelSurfacesl
	    j['MatrixRow'] = seedID
	    seedID = seedID + 1
	    seedList.write(filename + "\n")
	     
	seedList.close()

	#Update JSON file 
	with open(jsonFile, 'w') as txtfile:
	    json.dump(data, txtfile, indent = 2)




if __name__ == '__main__':
	main(subject_dir, overlapName, jsonFile )