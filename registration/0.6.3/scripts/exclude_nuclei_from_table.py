import os
import subprocess
import h5py
import numpy as np
import pandas as pd
import imageio


# TODO move this to elf
def save_tif(data, out_path, resolution):
    # write initial tif with imageio
    out_path = out_path + '.tif'
    imageio.volwrite(out_path, data)

    # encode the arguments for the imagej macro:
    # imagej macros can only take a single string as argument, so we need
    # to comma seperate the individual arguments
    assert "," not in out_path, "Can't encode pathname containing a comma"
    arguments = "%s,%i,%f,%f,%f" % (os.path.abspath(out_path), data.shape[0],
                                    resolution[0], resolution[1], resolution[2])

    # call the imagej macro
    fiji_executable = "/g/arendt/EM_6dpf_segmentation/platy-browser-data/software/Fiji.app/ImageJ-linux64"
    script = "/g/arendt/EM_6dpf_segmentation/platy-browser-data/registration/n.n.n/scripts/set_voxel_size.ijm"
    cmd = [fiji_executable, '-batch', '--headless', script, arguments]
    subprocess.run(cmd)


def exclude_nuclei_from_table(seg_path, table_path, out_path):
    table = pd.read_csv(table_path, sep='\t').values
    ids, vals = table[:, 0], table[:, 1]

    exclusion_ids = [int(k) for k, v in zip(ids, vals) if v != 'None']
    print("Number of nuclei to exclude:", len(exclusion_ids))
    print(exclusion_ids)

    scale = 2
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
    resolution = [0.32, 0.32, 0.4]
    save_tif(seg, out_path, resolution)


if __name__ == '__main__':
    seg_path = '../../../data/0.0.0/segmentations/sbem-6dpf-1-whole-segmented-nuclei-labels.h5'
    table_path = '../EM/nuclei_to_remove.csv'
    out_path = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/registration/n.n.n/EM/nuclei'
    exclude_nuclei_from_table(seg_path, table_path, out_path)
