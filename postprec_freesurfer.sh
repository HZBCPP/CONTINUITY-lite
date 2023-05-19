#!/bin/bash

# module load freesurfer

FS_DIR="/newsdd2/dtt/mri_data_proc/training/fs_out"
ANNOT_PREFIX="aparc"
SURFACES=(lh.white rh.white)
ATTRIBUTE_NAME="label"
VOLUMES=(aparc+aseg brain)

echo Starting

for SUBJ in $(ls $FS_DIR); do
    echo "Current subject: $SUBJ"
    echo "Coverting .mgz to .nrrd..."
    for VOL in ${VOLUMES[@]}; do
        VOL_FILE="${FS_DIR}/${SUBJ}/mri/${VOL}.mgz"
        OUT_VOL_FILE="${FS_DIR}/${SUBJ}/mri/${VOL}.nii"
        OUT_NRRD_FILE="${FS_DIR}/${SUBJ}/mri/${VOL}.nrrd"
        mri_convert ${VOL_FILE} ${OUT_VOL_FILE}
        convertITKformats ${OUT_VOL_FILE} ${OUT_VOL_FILE}
        python ./check_nrrd.py ${FS_DIR}/${SUBJ}/mri
    done

    echo "Generating surface label .vtk file..."

    # for SURF in ${SURFACES[@]}; do
    #     python ./convert_surf2vtk.py ${FS_DIR}/${SUBJ}/surf/${SURF} ${ANNOT_PREFIX}
    # done
    
    python ./convert_surf2vtk.py --geometry_fn ${FS_DIR}/${SUBJ}/surf/lh.white --prefix ${ANNOT_PREFIX} --label-offset 10000
    python ./convert_surf2vtk.py --geometry_fn ${FS_DIR}/${SUBJ}/surf/rh.white --prefix ${ANNOT_PREFIX} --label-offset 20000

    echo " done"
done

echo Finished
