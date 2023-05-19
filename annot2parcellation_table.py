import nibabel as nib
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lh-annot', '-la', dest='lh_annot', required=True)
    parser.add_argument('--rh-annot', '-ra', dest='rh_annot', required=True)
    parser.add_argument('--output', '-o', dest='output', required=True)
    args = parser.parse_args()

    _, _, lh_names = nib.freesurfer.io.read_annot(args.lh_annot)
    _, _, rh_names = nib.freesurfer.io.read_annot(args.rh_annot)
    lh_names = [name.decode() for name in lh_names]
    rh_names = [name.decode() for name in rh_names]

    lh_table = [
        {
            'VisuOrder': 1,
            'MatrixRow': 1,
            'name': name,
            'VisuHierarchy': 'seed.left.',
            'coord': [0, 0, 0],
            'labelValue': i,
            'AAL_ID': i,
        }
        for i, name in enumerate(lh_names, 10001)
    ]
    rh_table = [
        {
            'VisuOrder': 1,
            'MatrixRow': 1,
            'name': name,
            'VisuHierarchy': 'seed.right.',
            'coord': [0, 0, 0],
            'labelValue': i,
            'AAL_ID': i,
        }
        for i, name in enumerate(rh_names, 20001)
    ]
    table = lh_table + rh_table

    with open(args.output, 'w') as f:
        json.dump(table, f, indent=2)


if __name__ == '__main__':
    main()
