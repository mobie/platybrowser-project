from datetime import datetime

#TODO - uncomment this part
# this is a task called by multiple processes,
# so we need to restrict the number of threads used by numpy
#from cluster_tools.utils.numpy_utils import set_numpy_threads
#set_numpy_threads(1)
import numpy as np

import h5py
import pandas as pd
from skimage.measure import regionprops, marching_cubes_lewiner, mesh_surface_area
from skimage.transform import resize
from skimage.util import pad


def log(msg):
    print("%s: %s" % (str(datetime.now()), msg))


# get shape of full data & downsampling factor
def get_scale_factor(path, key_full, key, resolution):
    with h5py.File(path, 'r') as f:
        full_shape = f[key_full].shape
        shape = f[key].shape

    # scale factor for downsampling
    scale_factor = [res * (fs / sh)
                    for res, fs, sh in zip(resolution,
                                           full_shape,
                                           shape)]
    return scale_factor


# changed to shape_pixel_size, is the plan to go to n_pixels in future?
def filter_table(table, min_size, max_size):
    if max_size is None:
        table = table.loc[table['shape_pixelsize'] >= min_size, :]
    else:
        criteria = np.logical_and(table['shape_pixelsize'] > min_size, table['shape_pixelsize'] < max_size)
        table = table.loc[criteria, :]
    return table


def filter_table_from_mapping(table, mapping_path):
    # read in numpy array of mapping of cells to nuclei - first column cell id, second nucleus id
    mapping = np.genfromtxt(mapping_path, skip_header=1, delimiter='\t')[:, :2].astype('uint64')
    # remove zero labels from this table too, if exist
    mapping = mapping[np.logical_and(mapping[:, 0] != 0,
                                     mapping[:, 1] != 0)]
    table = table.loc[np.isin(table['label_id'], mapping[:, 0]), :]
    return table


def load_data(ds, row, scale):
    # compute the bounding box from the row information
    mins = [row.bb_min_z, row.bb_min_y, row.bb_min_x]
    maxs = [row.bb_max_z, row.bb_max_y, row.bb_max_x]
    mins = [int(mi / sca) for mi, sca in zip(mins, scale)]
    maxs = [int(ma / sca) + 1 for ma, sca in zip(maxs, scale)]
    bb = tuple(slice(mi, ma) for mi, ma in zip(mins, maxs))
    # load the data from the bounding box
    return ds[bb]


def morphology_row_features(mask, scale):

    # Calculate stats from skimage
    ski_morph = regionprops(mask.astype('uint8'))

    # volume in pixels
    volume_in_pix = ski_morph[0]['area']

    # extent
    extent = ski_morph[0]['extent']

    # The mesh calculation below fails if an edge of the segmentation is right up against the
    # edge of the volume - gives an open, rather than a closed surface
    # Pad by a few pixels to avoid this
    mask = pad(mask, 10, mode='constant')

    # surface area of mesh around object (other ways to calculate better?)
    verts, faces, normals, values = marching_cubes_lewiner(mask, spacing=tuple(scale))
    surface_area = mesh_surface_area(verts, faces)

    # volume in microns
    volume_in_microns = np.prod(scale)*volume_in_pix

    # sphericity (as in morpholibj)
    # Should run from zero to one
    sphericity = (36*np.pi*(float(volume_in_microns)**2))/(float(surface_area)**3)

    return [volume_in_microns, extent, surface_area, sphericity]


def intensity_row_features(raw, mask):
    intensity_vals_in_mask = raw[mask]
    # mean and stdev - use float64 to avoid silent overflow errors
    mean_intensity = np.mean(intensity_vals_in_mask, dtype=np.float64)
    st_dev = np.std(intensity_vals_in_mask, dtype=np.float64)
    return mean_intensity, st_dev


# compute morphology (and intensity features) for label range
def morphology_features_for_label_range(table, ds, ds_raw,
                                        scale_factor_seg, scale_factor_raw,
                                        label_begin, label_end):
    label_range = np.logical_and(table['label_id'] >= label_begin, table['label_id'] < label_end)
    sub_table = table.loc[label_range, :]
    stats = []
    for row in sub_table.itertuples(index=False):
        label_id = int(row.label_id)

        # load the segmentation data from the bounding box corresponding
        # to this row
        seg = load_data(ds, row, scale_factor_seg)

        # compute the segmentation mask and check that we have
        # foreground in the mask
        seg_mask = seg == label_id
        if seg_mask.sum() == 0:
            # if the seg mask is empty, we simply skip this label-id
            continue

        # compute the morphology features from the segmentation mask
        result = [float(label_id)] + morphology_row_features(seg_mask, scale_factor_seg)

        # compute the intensiry features from raw data and segmentation mask
        if ds_raw is not None:
            raw = load_data(ds_raw, row, scale_factor_raw)
            # resize the segmentation mask if it does not fit the raw data
            if seg_mask.shape != raw.shape:
                seg_mask = resize(seg_mask, raw.shape,
                                  order=0, mode='reflect',
                                  anti_aliasing=True, preserve_range=True).astype('bool')
            result += intensity_row_features(raw, seg_mask)
        stats.append(result)
    return stats


def compute_morphology_features(table, segmentation_path, raw_path,
                                seg_key, raw_key,
                                scale_factor_seg, scale_factor_raw,
                                label_starts, label_stops):

    if raw_path != '':
        assert raw_key is not None and scale_factor_raw is not None
        f_raw = h5py.File(raw_path, 'r')
        ds_raw = f_raw[raw_key]
    else:
        f_raw = ds_raw = None

    with h5py.File(segmentation_path, 'r') as f:
        ds = f[seg_key]

        stats = []
        for label_a, label_b in zip(label_starts, label_stops):
            log("Computing features from label-id %i to %i" % (label_a, label_b))
            stats.extend(morphology_features_for_label_range(table, ds, ds_raw,
                                                             scale_factor_seg, scale_factor_raw,
                                                             label_a, label_b))
    if f_raw is not None:
        f_raw.close()

    # convert to pandas table and add column names
    stats = pd.DataFrame(stats)
    columns = ['label_id',
               'shape_volume_in_microns', 'shape_extent', 'shape_surface_area', 'shape_sphericity']
    if raw_path != '':
        columns += ['intensity_mean', 'intensity_st_dev']
    stats.columns = columns
    return stats


def morphology_impl(segmentation_path, raw_path, table, mapping_path,
                    min_size, max_size,
                    resolution, raw_scale, seg_scale,
                    label_starts, label_stops):
    """ Compute morphology features for a segmentation.

    Can compute features for multiple label ranges. If you want to
    compute features for the full label range, pass
    'label_starts=[0]' and 'label_stops=[number_of_labels]'

    Arguments:
        segmentation_path [str] - path to segmentation stored as h5.
        raw_path [str] - path to raw data stored as h5.
            Pass 'None' if you don't want to compute features based on raw data.
        table [pd.DataFrame] - table with default attributes
            (sizes, center of mass and bounding boxes) for segmentation
        mapping_path [str] - path to - path to nucleus id mapping.
            Pass 'None' if not relevant.
        min_size [int] - minimal size for objects used in calcualtion
        max_size [int] - maximum size for objects used in calcualtion
        resolution [listlike] - resolution in nanometer.
            Must be given in [Z, Y, X].
        raw_scale [int] - scale level of the raw data
        seg_scale [int] - scale level of the segmentation
        label_starts [listlike] - list with label start positions
        label_stops [listlike] - list with label stop positions
    """

    # keys to segmentation and raw data for the different scales
    seg_key_full = 't00000/s00/0/cells'
    seg_key = 't00000/s00/%i/cells' % seg_scale
    raw_key_full = 't00000/s00/0/cells'
    raw_key = 't00000/s00/%i/cells' % raw_scale

    # get scale factor for the segmentation
    scale_factor_seg = get_scale_factor(segmentation_path, seg_key_full, seg_key, resolution)

    # get scale factor for raw data (if it's given)
    if raw_path != '':
        log("Have raw path; compute intensity features")
        # NOTE for now we can hard-code the resolution for the raw data here,
        # but we might need to change this if we get additional dataset(s)
        raw_resolution = [0.025, 0.01, 0.01]
        scale_factor_raw = get_scale_factor(raw_path, raw_key_full, raw_key, raw_resolution)
    else:
        log("Don't have raw path; do not compute intensity features")
        raw_resolution = scale_factor_raw = None

    # remove zero label if it exists
    table = table.loc[table['label_id'] != 0, :]

    # if we have a mappin, only keep objects in the mapping
    # (i.e cells that have assigned nuclei)
    if mapping_path != '':
        log("Have mapping path %s" % mapping_path)
        table = filter_table_from_mapping(table, mapping_path)
        log("Number of labels after filter with mapping: %i" % table.shape[0])
    # filter by size
    table = filter_table(table, min_size, max_size)
    log("Number of labels after size filte: %i" % table.shape[0])

    log("Computing morphology features")
    stats = compute_morphology_features(table, segmentation_path, raw_path,
                                        seg_key, raw_key, scale_factor_seg, scale_factor_raw,
                                        label_starts, label_stops)
    return stats


if __name__ == '__main__':
    pass
