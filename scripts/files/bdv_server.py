import os
from .xml_utils import get_h5_path_from_xml


# TODO enable filtering for files by some patter.
# e.g. if we don't want to expose the fib dataset to the public yet
def make_bdv_server_file(folders, out_path):
    """ Make the bigserver config file from
    all xmls in folders.
    """
    file_list = {}
    for folder in folders:
        files = os.listdir(folder)
        for ff in files:
            path = os.path.join(folder, ff)

            # only add xmls
            ext = os.path.splitext(path)[1]
            if ext != '.xml':
                continue

            # make sure that the h5path linked in the xml exists
            h5path = get_h5_path_from_xml(path, return_absolute_path=True)
            if not os.path.exists(h5path):
                msg = 'Path to h5-file in xml does not exist - %s, %s' % (path,
                                                                          h5path)
                return RuntimeError(msg)

            name = os.path.splitext(ff)[0]
            file_list[name] = path

    with open(out_path, 'w') as f:
        for name, path in file_list.items():
            line = '%s\t%s\n' % (name, path)
            f.write(line)
