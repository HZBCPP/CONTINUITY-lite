module load freesurfer/6.0.0
export SUBJECTS_DIR=/nas/longleaf/home/ziquanw/fs_out 
sbatch -N 1 -n 8 -t 2-00:00:00 --mem=16g --wrap="recon-all -autorecon1 -autorecon2 -autorecon-pial -s subject_test -openmp 8 -parallel"
