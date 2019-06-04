from .to_bdv import to_bdv


def export_segmentation(paintera_path, paintera_key, folder, new_folder, name, resolution):
    """ Export a segmentation from paintera project to bdv file and
    compute segment lut for previous segmentation.

    Arguments:
        paintera_path: path to the paintera project corresponding to the new segmentation
        paintera_key: key to the paintera project corresponding to the new segmentation
        folder: folder for old segmentation
        new_folder: folder for new segmentation
        name: name of segmentation
        resolution: resolution [z, y, x] in micrometer
    """
    # export segmentation from paintera

    # compute mapping to old segmentation

    # convert to bdv
