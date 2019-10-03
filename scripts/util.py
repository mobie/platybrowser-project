import h5py  # TODO exchange with elf.io.open_file


def add_max_id(path, key):
    with h5py.File(path) as f:
        ds = f[key]
        data = ds[:]
        max_id = int(data.max())
        ds.attrs['maxId'] = max_id
