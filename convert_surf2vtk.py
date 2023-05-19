import numpy as np
import nibabel as nib
import sys, os, json


def main():
    # geometry_fn = '/work/users/z/i/ziquanw/mri_data_proc/fs_out/subject_test_Yeo400/surf/rh.white'
    geometry_fn = sys.argv[1]
    print(f'Converting {geometry_fn} to VTK format\n')
    label_fn = geometry_fn.replace('white', 'aparc.annot').replace('surf', 'label')
    verts, faces = nib.freesurfer.io.read_geometry(geometry_fn)
    labels = nib.freesurfer.io.read_annot(label_fn)[0]
    ## load atlas table .json
    # prefix = sys.argv[2]    
    # # prefix = 'aparc'
    # continuity_jsons = []
    # geo_fn = geometry_fn.split('/')[-1]
    # LorR = 0 if 'lh' in geometry_fn else 1
    # possible_tag = [['lh-', 'Left-', 'Left_', 'left_', ''],
    #                 ['rh-', 'Right-', 'Right_', 'right_', '']]
    # ctab_fn = geometry_fn.replace('surf', 'label').replace(geo_fn, '%s.annot.ctab' % prefix)
    # if os.path.exists(ctab_fn):
    #     terms, nums = read_ctab(ctab_fn)
    #     target_terms, target_nums = read_ctab('/nas/longleaf/home/ziquanw/training/fs_all.annot.ctab')
    #     for num, term in zip(nums, terms):
    #         if num == 0: continue
    #         pname = term[0]
    #         for tag in possible_tag[LorR]:
    #             term[0] = tag + pname
    #             ind = find_match_ctab(term, target_terms)
    #             if ind is not None: break
            
    #         continuity_jsons.append({
    #             'name': pname.item() if ind is None else term[0].item(),
    #             'labelValue': num.item(),
    #             'AAL_ID': num.item() if ind is None else target_nums[ind].item(),
    #         })
    # with open(geometry_fn + '_atlas_table.json', 'w') as jsonf:
    #     jsonf.write(json.dumps(continuity_jsons, indent=4))
    ##

    invalid = np.where(labels==-1)[0]
    labels[invalid] = 0
    print("There is ", np.unique(labels).shape, " labels in surface")
    new_faces = []
    for face in faces:
        if len([f for f in face if f in invalid]) > 0: continue
        new_faces.append(face)

    lines = '# vtk DataFile Version 4.0\nvtk output\nASCII\nDATASET POLYDATA\n'
    lines += 'POINTS %d float\n' % len(verts)
    for vi, vert in enumerate(verts):
        lines += '%f %f %f ' % (vert[0], vert[1], vert[2])
        if (vi+1) % 3 == 0: lines += '\n'

    lines += '\nPOLYGONS %d %d\n' % (len(new_faces), len(new_faces)*4)
    for fi, face in enumerate(new_faces):
        lines += '3 %d %d %d \n' % (face[0], face[1], face[2])

    lines += 'POINT_DATA %d\nFIELD FieldData 1\nlabel 1 %d float\n' % (len(verts), len(verts))
    for li, label in enumerate(labels):
        lines += '%d ' % label
        if (li+1) % 9 == 0: lines += '\n'
    lines += '\n'
    with open(geometry_fn+ '_labeled.vtk', 'wt') as f:
        f.write(lines)


def read_ctab(ctab_fn):
    ctab = np.loadtxt(ctab_fn, dtype=str)
    out = []
    nums = []
    for line in ctab:
        num = int(line[0])
        term1 = line[1]
        term2 = line[2]+line[3]+line[4]+line[5]
        out.append([term1, term2])
        nums.append(num)
    return np.array(out, dtype=str), np.array(nums)
    
def find_match_ctab(terms, target_terms):
    valid = np.array([True for _ in range(len(target_terms))], dtype=bool)
    for find_dim, term in enumerate(terms):
        valid = valid & np.array([term in target_terms[i, find_dim] for i in range(len(target_terms))], dtype=bool)
    if valid.sum() == 1: 
        return np.where(valid)[0][0]
    else:
        return None

if __name__ == "__main__":
    main()