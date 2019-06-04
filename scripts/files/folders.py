import os


def make_folder_structure(root):
    # make all sub-folders
    os.makedirs(os.path.join(root, 'tables'))
    os.makedirs(os.path.join(root, 'images'))
    os.makedirs(os.path.join(root, 'segmentations'))
    os.makedirs(os.path.join(root, 'misc'))
