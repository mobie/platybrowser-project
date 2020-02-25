import imageio
import numpy as np
from pybdv import make_bdv
from pybdv.util import get_scale_factors
from pybdv.metadata import get_resolution


def update_shell():
    p_tiff = '../../EM-Prospr/shell_seg.tiff'
    p_res_n5 = '../data/rawdata/sbem-6dpf-1-whole-segmented-resin.n5'
    p_res_xml = '../data/rawdata/sbem-6dpf-1-whole-segmented-resin.xml'

    scale_factors = get_scale_factors(p_res_n5, 0)[1:]
    resolution = get_resolution(p_res_xml, 0)

    scale_factors = [[2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]

    print(scale_factors)
    print(resolution)

    print("Load tiff ...")
    shell = np.asarray(imageio.volread(p_tiff))
    print(shell.shape)

    print("Write bdv")
    out_path = 'sbem-6dpf-1-whole-segmented-shell.n5'

    make_bdv(shell, out_path, downscale_factors=scale_factors, resolution=resolution, unit='micrometer',
             n_threads=8, chunks=(96,) * 3, convert_dtype=False)


def update_resin():
    p_tiff = '../../EM-Prospr/resin_seg.tiff'
    p_res_n5 = '../data/rawdata/sbem-6dpf-1-whole-segmented-resin.n5'
    p_res_xml = '../data/rawdata/sbem-6dpf-1-whole-segmented-resin.xml'

    scale_factors = get_scale_factors(p_res_n5, 0)[1:]
    resolution = get_resolution(p_res_xml, 0)

    scale_factors = [[2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]

    print(scale_factors)
    print(resolution)

    print("Load tiff ...")
    shell = np.asarray(imageio.volread(p_tiff))
    print(shell.shape)

    print("Write bdv")
    out_path = 'sbem-6dpf-1-whole-segmented-resin.n5'

    make_bdv(shell, out_path, downscale_factors=scale_factors, resolution=resolution, unit='micrometer',
             n_threads=8, chunks=(96,) * 3, convert_dtype=False)


if __name__ == '__main__':
    # FIXME something with this shell segmentation is off
    # print("Shell")
    # update_shell()
    print("Resin")
    update_resin()
