# TODO move this somewhere in the registration folder
import os
import subprocess
import imageio
import numpy as np
import h5py
from pybdv.converter import make_bdv


def save_bdv(data, out_path, resolution):
    make_bdv(data, out_path, resolution=resolution, unit='micrometer')


def save_tif(data, out_path, resolution):
    # write initial tif with imageio
    out_path = out_path + '.tif'
    imageio.volwrite(out_path, data)

    # encode the arguments for the imagej macro:
    # imagej macros can only take a single string as argument, so we need
    # to comma seperate the individual arguments
    assert "," not in out_path, "Can't encode pathname containing a comma"
    arguments = "%s,%i,%f,%f,%f" % (out_path, data.shape[0],
                                    resolution[0], resolution[1], resolution[2])

    # call the imagej macro
    fiji_executable = "/g/almf/software/Fiji.app/ImageJ-linux64"
    script = "/g/arendt/EM_6dpf_segmentation/platy-browser-data/registration/9.9.9/scripts/registration_targets/set_voxel_size.ijm"
    cmd = [fiji_executable, '-batch', '--headless', script, arguments]
    subprocess.run(cmd)


def make_stomach_target_prospr(as_tif=True):
    seg_path = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/prospr/Stomodeum--prosprspaceMEDs.tif'
    med_path = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/prospr/TrpV--prosprspaceMEDs.tif'

    seg = imageio.volread(seg_path) > 0
    med = imageio.volread(med_path) > 0
    assert seg.shape == med.shape
    target = np.logical_or(seg, med)
    target = target.astype('uint8') * 255

    out_dir = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/registration/9.9.9/ProSPr_files_for_registration'
    out_path = os.path.join(out_dir, 'Stomach_forRegistration')
    resolution = [.55, .55, .55]
    if as_tif:
        save_tif(target, out_path, resolution)
    else:
        save_bdv(target, out_path, resolution)


def make_stomach_em_target(as_tif=True):
    seg_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata',
                            'sbem-6dpf-1-whole-segmented-tissue-labels.h5')

    # find the closest scale level
    # print("Target shape:", )
    # with h5py.File('./stomach_prospr_target.h5', 'r') as f:
    #     ds = f['t00000/s00/0/cells']
    #     print(ds.shape)
    # print()
    # res = [0.08, 0.08, 0.1]
    # with h5py.File(seg_path, 'r') as f:
    #     g = f['t00000/s00']
    #     for name, obj in g.items():
    #         print(name, ":", obj['cells'].shape, res)
    #         res = [re * 2 for re in res]
    # return

    # closest scale level is 3
    scale = 3
    stomach_id = 23
    with h5py.File(seg_path, 'r') as f:
        ds = f['t00000/s00/%i/cells' % scale]
        target = (ds[:] == stomach_id).astype('uint8') * 255

    out_dir = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/registration/9.9.9/EM_files_for_registration'
    out_path = os.path.join(out_dir, 'Stomach_forRegistration')
    resolution = [0.64, 0.64, 0.8]
    if as_tif:
        save_tif(target, out_path, resolution)
    else:
        save_bdv(target, out_path, resolution)


if __name__ == '__main__':
    make_stomach_target_prospr()
    make_stomach_em_target()
