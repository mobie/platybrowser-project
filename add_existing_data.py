import os
from glob import glob
from scripts.sources import add_source, add_image, add_segmentation


def add_sources():
    # add em source
    add_source('sbem', '6dpf')
    # add prospr source
    add_source('prospr', '6dpf')


def add_images():
    base_folder = './data/0.2.1/images'
    # add all prospr images
    prospr_prefix = 'prospr-6dpf-1-whole'
    prospr_ims = glob.glob(os.path.join(base_folder, 'prospr-6dpf-1-whole-*'))
    for impath in prospr_ims:
        name = os.path.split(impath)[1]
        name, ext = os.path.splitext()
        if ext != '.xml':
            continue
        name = name[(len(prospr_prefix) + 1):]
        add_image(impath, prospr_prefix, name, copy_data=False)


def add_segmentations():
    pass


def add_existing_data():
    """ Add existing data to the json files that keep track of
        sources, image data and segmentations.
    """
    add_sources()
    add_images()
    add_segmentations()


if __name__ == '__main__':
    add_existing_data()
