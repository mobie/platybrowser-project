import h5py
import numpy as np
import pandas as pd
# TODO move this to elf
from stomach_registration_targets import save_tif


def exclude_nuclei_from_table(seg_path, table_path, out_path):
    table = pd.read_csv(table_path, sep='\t').values
    ids, vals = table[:, 0], table[:, 1]

    exclusion_ids = [int(k) for k, v in zip(ids, vals) if v != 'None']
    print("Number of nuclei to exclude:", len(exclusion_ids))
    print(exclusion_ids)

    scale = 3
    key = 't00000/s00/%i/cells' % scale
    with h5py.File(seg_path) as f:
        ds = f[key]
        seg = ds[:]
    exclusion_mask = np.isin(seg, exclusion_ids)
    print("Zeroing out", exclusion_mask.sum(), "pixels")
    seg[exclusion_mask] = 0
    seg = (seg > 0).astype('uint8') * 255

    # s0: [.08, .08, .1]
    # s1: [.16, .16, .2]
    # s2: [.32, .32, .4]
    # s3: [.64, .64, .8]
    resolution = [0.64, 0.64, 0.8]
    save_tif(seg, out_path, resolution)


if __name__ == '__main__':
    seg_path = '../../../../data/0.0.0/segmentations/sbem-6dpf-1-whole-segmented-nuclei-labels.h5'
    table_path = '../../EM/nuclei_to_remove.csv'
    out_path = '../../EM/nuclei_new'
    exclude_nuclei_from_table(seg_path, table_path, out_path)
