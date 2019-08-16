import os
from .xml_utils import get_h5_path_from_xml
from .sources import get_privates, get_image_names, get_segmentation_names


def add_to_bdv_config(name, path, bdv_config, relative_paths, ref_dir):

    # make sure that the h5path linked in the xml exists
    h5path = get_h5_path_from_xml(path, return_absolute_path=True)
    if not os.path.exists(h5path):
        msg = 'Path to h5-file in xml does not exist - %s, %s' % (path, h5path)
        raise RuntimeError(msg)

    if relative_paths:
        path = os.path.relpath(path, ref_dir)
    bdv_config[name] = path
    return bdv_config


def make_bdv_server_file(folder, out_path, relative_paths=True):
    """ Make the bigserver config file for a given release.
    """
    privates = get_privates()
    image_names = get_image_names()
    seg_names = get_segmentation_names()
    ref_dir = os.path.split(out_path)[0]

    bdv_config = {}
    for name in image_names:
        if name in privates:
            continue
        path = os.path.join(folder, 'images', '%s.xml' % name)
        bdv_config = add_to_bdv_config(name, path, bdv_config,
                                       relative_paths, ref_dir)

    for name in seg_names:
        if name in privates:
            continue
        path = os.path.join(folder, 'segmentations', '%s.xml' % name)
        bdv_config = add_to_bdv_config(name, path, bdv_config,
                                       relative_paths, ref_dir)

    with open(out_path, 'w') as f:
        for name, path in bdv_config.items():
            line = '%s\t%s\n' % (name, path)
            f.write(line)
