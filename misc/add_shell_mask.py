import h5py
from pybdv.converter import make_bdv


def add_shell_mask():
    in_path = '/g/arendt/EM_6dpf_segmentation/EM-Prospr/em-segmented-shell.h5'
    in_key = 't00000/s00/0/cells'

    with h5py.File(in_path, 'r') as f:
        ds = f[in_key]
        shell = ds[:]
    print(shell.shape)

    scale_factors = 4 * [[2, 2, 2]]
    resolution = [0.4, 0.32, 0.32]

    chunks = (96,) * 3
    out_path = '../data/rawdata/sbem-6dpf-1-whole-segmented-shell.n5'
    make_bdv(shell, out_path, scale_factors, resolution=resolution, unit='micrometer',
             n_threads=8, convert_dtype=False, chunks=chunks)


if __name__ == '__main__':
    add_shell_mask()
