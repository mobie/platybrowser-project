import z5py
import nifty.tools as nt

PATH = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'
KEY = 'volumes/paintera/'


def get_blocking(scale, block_shape):
    g = z5py.File(PATH)[KEY]
    ds = g['data/s%i' % scale]
    shape = ds.shape
    blocking = nt.blocking([0, 0, 0], shape, block_shape)
    return blocking


def check_subdivision(scale, block_shape):
    blocking = get_blocking(scale, block_shape)


if __name__ == '__main__':
    pass
