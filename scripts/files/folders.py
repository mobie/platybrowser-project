import os


def make_folder_structure(root):
    # make all sub-folders
    os.makedirs(os.path.join(root, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(root, 'segmentations'), exist_ok=True)
    os.makedirs(os.path.join(root, 'misc'), exist_ok=True)
