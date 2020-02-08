from elf.io import open_file


def add_max_id(path, key, max_id=None):
    with open_file(path) as f:
        ds = f[key]
        if max_id is None:
            data = ds[:]
            max_id = int(data.max())
        ds.attrs['maxId'] = max_id


def read_resolution(paintera_path, paintera_key, to_um=True):
    with open_file(paintera_path, 'r') as f:
        g = f[paintera_key]
        attrs = g['data'].attrs
        if 'resolution' not in attrs:
            raise ValueError("Invalid paintera container")
        resolution = attrs['resolution']
    # convert to z,y,x from java's xyz conventin
    resolution = resolution[::-1]
    # convert from nm to um
    if to_um:
        resolution = [float(res) / 1000. for res in resolution]
    return resolution
