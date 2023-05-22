import nibabel as nib
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lh-annot', '-la', dest='lh_annot', required=True)
    parser.add_argument('--rh-annot', '-ra', dest='rh_annot', required=True)
    parser.add_argument(
        '--lh-label-offset', dest='lh_label_offset', type=int, default=10000
    )
    parser.add_argument(
        '--rh-label-offset', dest='rh_label_offset', type=int, default=20000
    )
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
            'labelValue': str(i),
            'AAL_ID': i,
        }
        for i, name in enumerate(lh_names, 1 + args.lh_label_offset)
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
        for i, name in enumerate(rh_names, 1 + args.rh_label_offset)
    ]
    table = lh_table + rh_table

    with open(args.output, 'w') as f:
        json.dump(table, f, indent=2)


if __name__ == '__main__':
    main()
