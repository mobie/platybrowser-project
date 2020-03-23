import argparse
from elf.io import open_file
from mmpb.segmentation.validation import partition_comparison
from mmpb.release_helper import get_version
from pybdv.metadata import get_data_path

ROOT = '../../data'


def check_segmentations(ref_path, ref_key, seg_path, seg_key):
    with open_file(ref_path, 'r') as f:
        shape = f[ref_key]

    with open_file(seg_path, 'r') as f:
        seg_shape = f[seg_key].shape
        assert shape == seg_shape, "%s, %s" % (str(shape),
                                               str(seg_shape))
    return shape


def compare_seg_to_ref(seg_path, seg_key, version, with_roi, target, max_jobs):
    ref_path = os.path.join(ROOT, version, 'images', 'local',
                            'sbem-6dpf-1-whole-segmented-cells.xml')
    ref_path = get_data_path(ref_path, return_absolute_path=True)

    if ref_path.endswith('.n5'):
        ref_key = 'setup0/timepoint0/so'
    else:
        ref_key = 't00000/s00/0/cells'

    shape = check_segmentations(ref_path, ref_key, seg_path, seg_key)

    halo = [100, 1024, 1024]
    if with_roi:
        roi_begin = [sh // 2 - ha for sh, ha in zip(shape, halo)]
        roi_end = [sh // 2 + ha for sh, ha in zip(shape, halo)]
    else:
        roi_begin = roi_end = None

    tmp_folder = './tmp_partition_comparison_%s' % seg_key
    res = partition_comparison(seg_path, seg_key,
                               ref_path, ref_key,
                               tmp_folder, target, max_jobs,
                               roi_begin=roi_begin, roi_end=roi_end)
    print("Have evaluated segmentation:")
    print(seg_path, ":", seg_key)
    print("against refetence:")
    print(ref_path, ":", ref_key)
    print("Result:")
    print("VI:", vis, "(split)", vim, "(merge)", vis + vim, "(total)")
    print("Adapted Rand error:", ari)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("seg_path", type=str, help="Path to segmentation")
    parser.add_argument("seg_key", type=str, help="Segmentation key")
    parser.add_argument("--version", type=str, default='', help="Reference version.")
    parser.add_argument('--target', type=str, default='slurm')
    parser.add_argument('--max_jobs', type=int, default=200)
    parser.add_argument('--with_roi', default=0, type=int)

    args = parser.parse_args()
    version = args.version
    if version == '':
        version = get_version(ROOT)

    compare_seg_to_ref(args.seg_path, args.seg_key, version,
                       bool(args.with_roi), args.target, args.max_jobs)
