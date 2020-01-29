from datetime import datetime

# this is a task called by multiple processes,
# so we need to restrict the number of threads used by numpy
from cluster_tools.utils.numpy_utils import set_numpy_threads
set_numpy_threads(1)

import numpy as np

import vigra
import h5py
import pandas as pd
from skimage.measure import regionprops, marching_cubes_lewiner, mesh_surface_area
from skimage.util import pad
from scipy.ndimage.morphology import distance_transform_edt
from mahotas.features import haralick
from skimage.morphology import label, remove_small_objects


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


def filter_table(table, min_size, max_size):
    if max_size is None:
        table = table.loc[table['n_pixels'] >= min_size, :]
    else:
        criteria = np.logical_and(table['n_pixels'] > min_size, table['n_pixels'] < max_size)
        table = table.loc[criteria, :]
    return table


# some cell segmentations have a sensible total pixel size but very large bounding boxes i.e. they are small spots
# distributed over a large region, not one cell > filter for these cases by capping the bounding box size
def filter_table_bb(table, max_bb):
    total_bb = (table['bb_max_z'] - table['bb_min_z']) * (table['bb_max_y'] - table['bb_min_y']) * (
            table['bb_max_x'] - table['bb_min_x'])

    table = table.loc[total_bb < max_bb, :]

    return table


def filter_table_from_mapping(table, mapping_path):
    # read in numpy array of mapping of cells to nuclei - first column cell id, second nucleus id
    mapping = pd.read_csv(mapping_path, sep='\t')

    # remove zero labels from this table too, if exist
    mapping = mapping.loc[np.logical_and(mapping.iloc[:, 0] != 0,
                                         mapping.iloc[:, 1] != 0), :]
    table = table.loc[np.isin(table['label_id'], mapping['label_id']), :]

    # add a column for the 'nucleus_id' of the mapped nucleus (use this later to exclude the area covered
    # by the nucleus)
    table = table.join(mapping.set_index('label_id'), on='label_id', how='left')

    return table


# regions here are regions to exclude
def filter_table_region(table, region_path, regions=('empty', 'yolk', 'neuropil', 'cuticle')):
    region_mapping = pd.read_csv(region_path, sep='\t')

    # remove zero label if it exists
    region_mapping = region_mapping.loc[region_mapping['label_id'] != 0, :]

    for region in regions:
        region_mapping = region_mapping.loc[region_mapping[region] == 0, :]

    table = table.loc[np.isin(table['label_id'], region_mapping['label_id']), :]

    return table


def run_all_filters(table, min_size, max_size, max_bb, mapping_path, region_mapping_path):
    # remove zero label if present
    table = table.loc[table['label_id'] != 0, :]

    # filter to only keep cells with assigned nuclei
    if mapping_path is not None:
        log("Have mapping path %s" % mapping_path)
        table = filter_table_from_mapping(table, mapping_path)
        log("Number of labels after filter with mapping: %i" % table.shape[0])

    # filter to exclude certain regions
    if region_mapping_path is not None:
        log("Have region mapping path %s" % region_mapping_path)
        table = filter_table_region(table, region_mapping_path)
        log("Number of labels after region filter: %i" % table.shape[0])

    # filter by size of object (no. pixels)
    if min_size is not None or max_size is not None:
        table = filter_table(table, min_size, max_size)
        log("Number of labels after size filter: %i" % table.shape[0])

    # filter by bounding box size
    if max_bb is not None:
        table = filter_table_bb(table, max_bb)
        log("Number of labels after bounding box size filter %i" % table.shape[0])

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


def generate_column_names(raw_path, chromatin_path, exclude_path):
    columns = ['label_id']
    morph_columns = ['shape_volume_in_microns', 'shape_extent', 'shape_equiv_diameter',
                     'shape_major_axis', 'shape_minor_axis', 'shape_surface_area', 'shape_sphericity',
                     'shape_max_radius']

    columns += morph_columns

    if raw_path is not None:
        intensity_columns = ['intensity_mean', 'intensity_st_dev', 'intensity_median', 'intensity_iqr',
                             'intensity_total']
        texture_columns = ['texture_hara%s' % x for x in range(1, 14)]

        columns += intensity_columns

        if exclude_path is None:
            # radial intensity columns
            for val in [25, 50, 75, 100]:
                columns += ['%s_%s' % (var, val) for var in intensity_columns]

        columns += texture_columns

    if chromatin_path is not None:
        edt_columns = ['shape_edt_mean', 'shape_edt_stdev', 'shape_edt_median', 'shape_edt_iqr']
        edt_columns += ['shape_percent_%s' % var for var in [25, 50, 75, 100]]

        for phase in ['_het', '_eu']:
            columns += [var + phase for var in morph_columns]
            columns += [var + phase for var in edt_columns]

            if raw_path is not None:
                columns += [var + phase for var in intensity_columns]
                columns += [var + phase for var in texture_columns]

    return columns


def morphology_row_features(mask, scale):
    # Calculate stats from skimage
    ski_morph = regionprops(mask.astype('uint8'))

    volume_in_pix = ski_morph[0]['area']
    volume_in_microns = np.prod(scale) * volume_in_pix
    extent = ski_morph[0]['extent']
    equiv_diameter = ski_morph[0]['equivalent_diameter']
    major_axis = ski_morph[0]['major_axis_length']
    minor_axis = ski_morph[0]['minor_axis_length']

    # The mesh calculation below fails if an edge of the segmentation is right up against the
    # edge of the volume - gives an open, rather than a closed surface
    # Pad by a few pixels to avoid this
    mask = pad(mask, 10, mode='constant')

    # surface area of mesh around object (other ways to calculate better?)
    verts, faces, normals, values = marching_cubes_lewiner(mask, spacing=tuple(scale))
    surface_area = mesh_surface_area(verts, faces)

    # sphericity (as in morpholibj)
    # Should run from zero to one
    sphericity = (36 * np.pi * (float(volume_in_microns) ** 2)) / (float(surface_area) ** 3)

    # max radius = max distance from pixel to outside
    edt = distance_transform_edt(mask, sampling=scale, return_distances=True)
    max_radius = np.max(edt)

    return (volume_in_microns, extent, equiv_diameter, major_axis,
            minor_axis, surface_area, sphericity, max_radius)


def intensity_row_features(raw, mask):
    intensity_vals_in_mask = raw[mask]
    # mean and stdev - use float64 to avoid silent overflow errors
    mean_intensity = np.mean(intensity_vals_in_mask, dtype=np.float64)
    st_dev = np.std(intensity_vals_in_mask, dtype=np.float64)
    median_intensity = np.median(intensity_vals_in_mask)

    quartile_75, quartile_25 = np.percentile(intensity_vals_in_mask, [75, 25])
    interquartile_range_intensity = quartile_75 - quartile_25

    total = np.sum(intensity_vals_in_mask, dtype=np.float64)

    return mean_intensity, st_dev, median_intensity, interquartile_range_intensity, total


def radial_intensity_row_features(raw, mask, scale, stops=(0.0, 0.25, 0.5, 0.75, 1.0)):
    result = ()

    edt = distance_transform_edt(mask, sampling=scale, return_distances=True)
    edt = edt / np.max(edt)

    bottoms = stops[0:len(stops) - 1]
    tops = stops[1:]

    radial_masks = [np.logical_and(edt > b, edt <= t) for b, t in zip(bottoms, tops)]

    for m in radial_masks:
        result += intensity_row_features(raw, m)

    return result


def texture_row_features(raw, mask):
    # errors if there are small, isolated spots (because I'm using ignore zeros as true)
    # so here remove components that are < 10 pixels
    # may still error in some cases
    labelled = label(mask)
    if len(np.unique(labelled)) > 2:
        labelled = remove_small_objects(labelled, min_size=10)
        mask = labelled != 0
        mask = mask.astype('uint8')

    # set regions outside mask to zero
    raw_copy = raw.copy()
    raw_copy[mask == 0] = 0

    try:
        hara = haralick(raw_copy, ignore_zeros=True, return_mean=True, distance=2)

    except ValueError:
        log('Texture computation failed - can happen when using ignore_zeros')
        hara = (0.,) * 13

    return tuple(hara)


def radial_distribution(edt, mask, stops=(0.0, 0.25, 0.5, 0.75, 1.0)):
    result = ()

    bottoms = stops[0:len(stops) - 1]
    tops = stops[1:]

    radial_masks = [np.logical_and(edt > b, edt <= t) for b, t in zip(bottoms, tops)]

    for m in radial_masks:
        result += (np.sum(mask[m]) / np.sum(m),)

    return result


def chromatin_row_features(chromatin, edt, raw, scale_chromatin):
    result = ()

    result += morphology_row_features(chromatin, scale_chromatin)

    # edt stats, dropping the total value
    result += intensity_row_features(edt, chromatin)[:-1]
    result += radial_distribution(edt, chromatin)

    if raw is not None:
        # resize the chromatin masks if not same size as raw
        chromatin_type = chromatin.dtype
        if chromatin.shape != raw.shape:
            chromatin = chromatin.astype('float32')
            chromatin = vigra.sampling.resize(chromatin, shape=raw.shape, order=0)
            chromatin = chromatin.astype(chromatin_type)

        result += intensity_row_features(raw, chromatin)
        result += texture_row_features(raw, chromatin)

    return result


# compute morphology (and intensity features) for label range
def morphology_features_for_label_range(table, ds, ds_raw,
                                        ds_chromatin,
                                        ds_exclude,
                                        scale_factor_seg, scale_factor_raw,
                                        scale_factor_chromatin,
                                        scale_factor_exclude,
                                        label_begin, label_end):
    label_range = np.logical_and(table['label_id'] >= label_begin, table['label_id'] < label_end)
    sub_table = table.loc[label_range, :]
    stats = []
    for row in sub_table.itertuples(index=False):
        log(str(row.label_id))
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
        result = (float(label_id),) + morphology_row_features(seg_mask, scale_factor_seg)

        if ds_exclude is not None:
            exclude = load_data(ds_exclude, row, scale_factor_exclude)
            exclude_type = exclude.dtype

            # resize to fit seg
            if exclude.shape != seg_mask.shape:
                exclude = exclude.astype('float32')
                exclude = vigra.sampling.resize(exclude, shape=seg_mask.shape, order=0)
                exclude = exclude.astype(exclude_type)

            # binary for correct nucleus
            exclude = exclude == int(row.nucleus_id)

            # remove nucleus area form seg_mask
            seg_mask[exclude] = False

        # compute the intensity features from raw data and segmentation mask
        if ds_raw is not None:
            raw = load_data(ds_raw, row, scale_factor_raw)

            # resize the segmentation mask if it does not fit the raw data
            seg_mask_type = seg_mask.dtype

            if seg_mask.shape != raw.shape:
                seg_mask = seg_mask.astype('float32')
                seg_mask = vigra.sampling.resize(seg_mask, shape=raw.shape, order=0)
                seg_mask = seg_mask.astype(seg_mask_type)

            result += intensity_row_features(raw, seg_mask)
            # doesn't make sense to run the radial intensity if the nucleus area is being excluded, as the euclidean
            # distance transform then gives distance from the outside & nuclear surface - hard to interpret
            if ds_exclude is None:
                result += radial_intensity_row_features(raw, seg_mask, scale_factor_raw)
            result += texture_row_features(raw, seg_mask)

        if ds_chromatin is not None:
            chromatin = load_data(ds_chromatin, row, scale_factor_chromatin)

            # set to 1 (heterochromatin), 2 (euchromatin)
            heterochromatin = chromatin == label_id + 12000
            euchromatin = chromatin == label_id

            # skip if no chromatin segmentation
            total_heterochromatin = heterochromatin.sum()
            total_euchromatin = euchromatin.sum()
            if total_heterochromatin == 0 and total_euchromatin.sum() == 0:
                continue

            # euclidean distance transform for whole nucleus, normalised to run from 0 to 1
            whole_nucleus = np.logical_or(heterochromatin, euchromatin)
            edt = distance_transform_edt(whole_nucleus, sampling=scale_factor_chromatin, return_distances=True)
            edt = edt / np.max(edt)

            if ds_raw is None:
                raw = None

            if total_heterochromatin != 0:
                result += chromatin_row_features(heterochromatin, edt, raw, scale_factor_chromatin)
            else:
                result += (0.,) * 36

            if total_euchromatin != 0:
                result += chromatin_row_features(euchromatin, edt, raw, scale_factor_chromatin)
            else:
                result += (0.,) * 36

        stats.append(result)
    return stats


def morphology_impl_nucleus(nucleus_segmentation_path, raw_path, chromatin_path,
                            table,
                            min_size, max_size,
                            max_bb,
                            nucleus_resolution, chromatin_resolution,
                            nucleus_seg_scale, raw_scale, chromatin_scale,
                            label_starts, label_stops):
    """ Compute morphology features for nucleus segmentation.

       Can compute features for multiple label ranges. If you want to
       compute features for the full label range, pass
       'label_starts=[0]' and 'label_stops=[number_of_labels]'

       Arguments:
           nucleus_segmentation_path [str] - path to nucleus segmentation data stored as h5.
           raw_path [str] - path to raw data stored as h5.
               Pass 'None' if you don't want to compute features based on raw data.
           chromatin_path [str] - path to chromatin segmentation data stored as h5.
           table [pd.DataFrame] - table with default attributes
               (sizes, center of mass and bounding boxes) for segmentation
           min_size [int] - minimal size for objects used in calculation
           max_size [int] - maximum size for objects used in calculation
           max_bb [int] - maximum total volume of bounding box for objects used in calculation
           nucleus_resolution [listlike] - resolution in nanometer.
               Must be given in [Z, Y, X].
           chromatin_resolution [listlike] - resolution in nanometer.
               Must be given in [Z, Y, X].
           nucleus_seg_scale [int] - scale level of the segmentation.
           raw_scale [int] - scale level of the raw data
           chromatin_scale [int] - scale level of the segmentation.
           label_starts [listlike] - list with label start positions
           label_stops [listlike] - list with label stop positions
       """

    # keys for the different scales
    nucleus_seg_key_full = 't00000/s00/0/cells'
    nucleus_seg_key = 't00000/s00/%i/cells' % nucleus_seg_scale
    raw_key_full = 't00000/s00/0/cells'
    raw_key = 't00000/s00/%i/cells' % raw_scale
    chromatin_key_full = 't00000/s00/0/cells'
    chromatin_key = 't00000/s00/%i/cells' % chromatin_scale

    # filter table
    table = run_all_filters(table, min_size, max_size, max_bb, None, None)

    # get scale factors
    if raw_path is not None:
        log("Have raw path; compute intensity features")
        # NOTE for now we can hard-code the resolution for the raw data here,
        # but we might need to change this if we get additional dataset(s)
        raw_resolution = [0.025, 0.01, 0.01]
        scale_factor_raw = get_scale_factor(raw_path, raw_key_full, raw_key, raw_resolution)
        f_raw = h5py.File(raw_path, 'r')
        ds_raw = f_raw[raw_key]
    else:
        log("Don't have raw path; do not compute intensity features")
        scale_factor_raw = f_raw = ds_raw = None

    if chromatin_path is not None:
        log("Have chromatin path; compute chromatin features")
        scale_factor_chromatin = get_scale_factor(chromatin_path, chromatin_key_full, chromatin_key,
                                                  chromatin_resolution)
        f_chromatin = h5py.File(chromatin_path, 'r')
        ds_chromatin = f_chromatin[chromatin_key]
    else:
        log("Don't have chromatin path; do not compute chromatin features")
        scale_factor_chromatin = f_chromatin = ds_chromatin = None

    log("Computing morphology features")
    scale_factor_nucleus_seg = get_scale_factor(nucleus_segmentation_path, nucleus_seg_key_full, nucleus_seg_key,
                                                nucleus_resolution)
    with h5py.File(nucleus_segmentation_path, 'r') as f:
        ds = f[nucleus_seg_key]

        stats = []
        for label_a, label_b in zip(label_starts, label_stops):
            log("Computing features from label-id %i to %i" % (label_a, label_b))
            stats.extend(morphology_features_for_label_range(table, ds, ds_raw,
                                                             ds_chromatin,
                                                             None,
                                                             scale_factor_nucleus_seg, scale_factor_raw,
                                                             scale_factor_chromatin,
                                                             None,
                                                             label_a, label_b))

    for var in f_raw, f_chromatin:
        if var is not None:
            var.close()

    # convert to pandas table and add column names
    stats = pd.DataFrame(stats)
    stats.columns = generate_column_names(raw_path, chromatin_path, None)

    return stats


def morphology_impl_cell(cell_segmentation_path, raw_path,
                         nucleus_segmentation_path,
                         table, mapping_path,
                         region_mapping_path,
                         min_size, max_size,
                         max_bb,
                         cell_resolution, nucleus_resolution,
                         cell_seg_scale, raw_scale, nucleus_seg_scale,
                         label_starts, label_stops):
    """ Compute morphology features for cell segmentation.

       Can compute features for multiple label ranges. If you want to
       compute features for the full label range, pass
       'label_starts=[0]' and 'label_stops=[number_of_labels]'

       Arguments:
           cell_segmentation_path [str] - path to cell segmentation stored as h5.
           raw_path [str] - path to raw data stored as h5.
               Pass 'None' if you don't want to compute features based on raw data.
           nucleus_segmentation_path [str] - path to nucleus segmentation data stored as h5.
               Pass 'None' if you don't want to exclude the region covered by the nucleus from intensity &
               texture calculations.
           table [pd.DataFrame] - table with default attributes
               (sizes, center of mass and bounding boxes) for segmentation
           mapping_path [str] - path to - path to nucleus id mapping.
               Pass 'None' if not relevant.
           region_mapping_path [str] - path to - path to cellid to region mapping
               Pass 'None' if not relevant
           min_size [int] - minimal size for objects used in calculation
           max_size [int] - maximum size for objects used in calculation
           max_bb [int] - maximum total volume of bounding box for objects used in calculation
           cell_resolution [listlike] - resolution in nanometer.
               Must be given in [Z, Y, X].
           nucleus_resolution [listlike] - resolution in nanometer.
               Must be given in [Z, Y, X].
           cell_seg_scale [int] - scale level of the segmentation
           raw_scale [int] - scale level of the raw data
           nucleus_seg_scale [int] - scale level of the segmentation.
           label_starts [listlike] - list with label start positions
           label_stops [listlike] - list with label stop positions
       """

    # keys for the different scales
    cell_seg_key_full = 't00000/s00/0/cells'
    cell_seg_key = 't00000/s00/%i/cells' % cell_seg_scale
    raw_key_full = 't00000/s00/0/cells'
    raw_key = 't00000/s00/%i/cells' % raw_scale
    nucleus_seg_key_full = 't00000/s00/0/cells'
    nucleus_seg_key = 't00000/s00/%i/cells' % nucleus_seg_scale

    # filter table
    table = run_all_filters(table, min_size, max_size, max_bb, mapping_path, region_mapping_path)

    # get scale factors
    if raw_path is not None:
        log("Have raw path; compute intensity features")
        # NOTE for now we can hard-code the resolution for the raw data here,
        # but we might need to change this if we get additional dataset(s)
        raw_resolution = [0.025, 0.01, 0.01]
        scale_factor_raw = get_scale_factor(raw_path, raw_key_full, raw_key, raw_resolution)
        f_raw = h5py.File(raw_path, 'r')
        ds_raw = f_raw[raw_key]
    else:
        log("Don't have raw path; do not compute intensity features")
        scale_factor_raw = f_raw = ds_raw = None

    if nucleus_segmentation_path is not None:
        log("Have nucleus path; exclude nucleus for intensity measures")
        scale_factor_nucleus = get_scale_factor(nucleus_segmentation_path, nucleus_seg_key_full, nucleus_seg_key,
                                                nucleus_resolution)
        f_nucleus = h5py.File(nucleus_segmentation_path, 'r')
        ds_nucleus = f_nucleus[nucleus_seg_key]
    else:
        log("Don't have exclude path; don't exclude nucleus area for intensity measures")
        scale_factor_nucleus = f_nucleus = ds_nucleus = None

    log("Computing morphology features")
    scale_factor_cell_seg = get_scale_factor(cell_segmentation_path, cell_seg_key_full, cell_seg_key, cell_resolution)
    with h5py.File(cell_segmentation_path, 'r') as f:
        ds = f[cell_seg_key]

        stats = []
        for label_a, label_b in zip(label_starts, label_stops):
            log("Computing features from label-id %i to %i" % (label_a, label_b))
            stats.extend(morphology_features_for_label_range(table, ds, ds_raw,
                                                             None,
                                                             ds_nucleus,
                                                             scale_factor_cell_seg, scale_factor_raw,
                                                             None,
                                                             scale_factor_nucleus,
                                                             label_a, label_b))

    for var in f_raw, f_nucleus:
        if var is not None:
            var.close()

    # convert to pandas table and add column names
    stats = pd.DataFrame(stats)
    stats.columns = generate_column_names(raw_path, None, nucleus_segmentation_path)

    return stats


if __name__ == '__main__':
    pass
