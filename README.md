CONTINUITY: CONnectivity Tool with INtegration of sUbcortical regions, registration and visualIzation of TractographY



To install all libraries required for CONTINUITY: 
	- In your terminal, run:  
		cd /tools/CONTINUITY/CONTINUITY_v1.1     ( On Longleaf: cd /proj/NIRAL/tools/CONTINUITY/CONTINUITY_v1.1 ) 

	- Install miniconda by running : 
		bash Miniconda3-latest-Linux-x86_64.sh


	Now, you can restart your terminal at the end

	- On the new terminal, to install libraries, write :  
		cd /tools/CONTINUITY/CONTINUITY_v1.1        ( On Longleaf: cd /proj/NIRAL/tools/CONTINUITY/CONTINUITY_v1.1 ) 
		conda env create -f CONTINUITY_env.yml

	- Then (3min later): 
		conda activate CONTINUITY_env

	- Now, you can run CONTINUITY: 
		python3 main_CONTINUITY.py


--------------------------------------------------------------------------------------------------------------------
For the next time, to run CONTINUITY, you just need to activate your environment:
	conda env create -f CONTINUITY_env.yml