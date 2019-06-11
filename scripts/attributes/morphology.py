# calculate morphology stats for cells / nuclei

import pandas as pd
import h5py
import numpy as np
from skimage.measure import regionprops, marching_cubes_lewiner, mesh_surface_area
from skimage.transform import resize
from skimage.util import pad


def calculate_slice(scale, mins, maxs):
    """
    Takes a scale (get this from xml file of object), and list of min and
    max coordinates in microns [min_coord_x_microns, min_coord_y_microns,
    min_coord_z_microns, max_coord_x_microns, max_coord_y_microns,
    max_coord_z_microns].
    Return a slice to use to extract
    that bounding box from the original image.
    """

    # convert min and max in microns to pixel ranges
    mins = [int(mi / sca) for mi, sca in zip(mins, scale)]
    maxs = [int(ma / sca) + 1 for ma, sca in zip(maxs, scale)]

    return tuple(slice(mi, ma) for mi, ma in zip(mins, maxs))


def calculate_stats_binary(mask, scale):
    """
    Calculate morphology statistics from the binary mask of an object
    1 in object, 0 outside.
    mask - numpy array for image, should be binary, 1 in object, 0 outside
    scale - list of form [x, y, z] stating scale in microns of image
    """
    # Calculate stats from skimage
    ski_morph = regionprops(mask)

    # volume in pixels
    volume_in_pix = ski_morph[0]['area']

    # extent
    extent = ski_morph[0]['extent']

    # The mesh calculation below fails if an edge of the segmentation is right up against the
    # edge of the volume - gives an open, rather than a closed surface
    # Pad by a few pixels to avoid this
    mask = pad(mask, 10, mode='constant')

    # surface area of mesh around object (other ways to calculate better?)
    # spacing z, y , x to match mask dimensions
    verts, faces, normals, values = marching_cubes_lewiner(mask, spacing=(scale[2], scale[1], scale[0]))
    surface_area = mesh_surface_area(verts, faces)

    # volume in microns
    volume_in_microns = scale[0]*scale[1]*scale[2]*volume_in_pix

    # sphericity (as in morpholibj)
    # Should run from zero to one
    sphericity = (36*np.pi*(float(volume_in_microns)**2))/(float(surface_area)**3)

    return volume_in_microns, extent, surface_area, sphericity


def calculate_stats_intensity(raw, mask):
    """
    Calculate intensity stats on the raw EM data for the region covered by the
    provided binary mask. Mask and raw must have the same dimensions!

    raw - numpy array of raw image data
    mask - numpy array of binary mask, 1 in object, 0 outside
    """

    intensity_vals_in_mask = raw[mask == 1]

    # mean and stdev - use float64 to avoid silent overflow errors
    mean_intensity = np.mean(intensity_vals_in_mask, dtype=np.float64)
    st_dev = np.std(intensity_vals_in_mask, dtype=np.float64)

    return mean_intensity, st_dev


# TODO - possibility that downscaled version
# of cell segmentation may not include some of the smallest cells in the table -
# will fail if this happens.
def calculate_row_stats(row, seg_path, input_type, morphology, intensity):
    """
    Calculate all morphology stats for one row of the table
    row - named tuple of values from that row
    input_type - 'nucleus' or 'cell'
    morphology - whether to calculate stats based on shape of segmentation: volume, extent, surface area, sphericity
    intensity - whether to calculate stats based on raw data covered by segmentation: mean_intensity, st_dev_intensity
    """
    if row.label_id % 100 == 0:
        print('Processing ' + str(row.label_id))

    # tuple to hold result
    result = (row.label_id,)

    if input_type == 'nucleus':
        full_seg_scale = [0.1, 0.08, 0.08]
        # use full res of segmentation - 0.08, 0.08, 0.1
        seg_level = 0
        # Use the /t00000/s00/3/cells for raw data - gives pixel size similar to the nuclear segmentation
        raw_level = 3

    elif input_type == 'cell':
        full_seg_scale = [0.025, 0.02, 0.02]
        # look at 4x downsampled segmentation - close to 0.08, 0.08, 0.1
        seg_level = 2

    bb_min = [row.bb_min_z, row.bb_min_y, row.bb_min_x]
    bb_max = [row.bb_max_z, row.bb_max_y, row.bb_max_x]
    with h5py.File(seg_path, 'r') as f:

        # get shape of full data & downsampled
        full_data_shape = f['t00000/s00/0/cells'].shape
        down_data_shape = f['t00000/s00/' + str(seg_level) + '/cells'].shape

        # scale for downsampled
        down_scale = [full_scale * (full_shape / down_shape)
                      for full_scale, full_shape, down_shape in zip(full_seg_scale,
                                                                    full_data_shape,
                                                                    down_data_shape)]
        # slice for seg file
        seg_slice = calculate_slice(down_scale, bb_min, bb_max)

        # get specified layer in bdv file
        data = f['t00000/s00/' + str(seg_level) + '/cells']
        img_array = data[seg_slice]

    # make into a binary mask
    binary = img_array == int(row.label_id)
    binary = binary.astype('uint8')

    # we need to skip for empty labels
    n_fg = binary.sum()
    if n_fg == 0:
        print("Skipping label", row.label_id, "due to empty segmentation mask")
        n_stats = 0
        if morphology:
            n_stats += 4
        if intensity:
            n_stats += 2
        dummy_result = (0.,) * n_stats
        result = result + dummy_result
        return result

    img_array = None

    # calculate morphology stats straight from segmentation
    if morphology:
        result = result + calculate_stats_binary(binary, down_scale)

    # calculate intensity statistics
    if intensity:

        # scale for full resolution raw data
        full_raw_scale = [0.025, 0.01, 0.01]

        # FIXME don't hardcode this path here
        with h5py.File('/g/arendt/EM_6dpf_segmentation/EM-Prospr/em-raw-full-res.h5', 'r') as f:

            # get shape of full data & downsampled
            full_data_shape = f['t00000/s00/0/cells'].shape
            down_data_shape = f['t00000/s00/' + str(raw_level) + '/cells'].shape

            # scale for downsampled
            down_scale = [full_scale * (full_shape / down_shape)
                          for full_scale, full_shape, down_shape in zip(full_raw_scale,
                                                                        full_data_shape,
                                                                        down_data_shape)]

            # slice for raw file
            raw_slice = calculate_slice(down_scale, bb_min, bb_max)

            # get specified layer in bdv file
            data = f['t00000/s00/' + str(raw_level) + '/cells']
            img_array = data[raw_slice]

        if img_array.shape != binary.shape:

            # rescale the binary to match the raw, using nearest neighbour interpolation (order = 0)
            binary = resize(binary, img_array.shape, order=0, mode='reflect', anti_aliasing=True, preserve_range=True)
            binary = binary.astype('uint8')

        # Calculate statistics on raw data
        result = result + calculate_stats_intensity(img_array, binary)

    return result


def calculate_morphology_stats(table, seg_path, input_type, morphology=True, intensity=True):
    '''
    Calculate statistics based on EM & segmentation.
    table - input table (pandas dataframe)
    input_type - 'nucleus' or 'cell'
    morphology - whether to calculate stats based on shape of segmentation:
        volume_in_microns, extent, surface area, sphericity
    intensity - whether to calculate stats based on raw data covered by segmentation: mean_intensity, st_dev_intensity
    '''

    # size cutoffs (at the moment just browsed through segmentation and chose sensible values)
    # not very stringent, still some rather large / small things included
    if input_type == 'nucleus':
        table = table.loc[table['n_pixels'] >= 18313.0, :]

    elif input_type == 'cell':
        criteria = np.logical_and(table['n_pixels'] > 88741.0, table['n_pixels'] < 600000000.0)
        table = table.loc[criteria, :]

    stats = [calculate_row_stats(row, seg_path, input_type, morphology, intensity)
             for row in table.itertuples(index=False)]

    # convert back to pandas dataframe
    stats = pd.DataFrame(stats)

    # names for columns
    columns = ['label_id']

    if morphology:
        columns = columns + ['shape_volume_in_microns', 'shape_extent', 'shape_surface_area', 'shape_sphericity']

    if intensity:
        columns = columns + ['intensity_mean', 'intensity_st_dev']

    # set column names
    stats.columns = columns

    return stats


def load_cell_nucleus_mapping(cell_nuc_mapping_path):
    # read in numpy array of mapping of cells to nuclei - first column cell id, second nucleus id
    cell_nucleus_mapping = np.genfromtxt(cell_nuc_mapping_path, skip_header=1, delimiter='\t')[:, :2]
    cell_nucleus_mapping = cell_nucleus_mapping.astype('uint64')
    # remove zero labels from this table too, if exist
    cell_nucleus_mapping = cell_nucleus_mapping[np.logical_and(cell_nucleus_mapping[:, 0] != 0,
                                                               cell_nucleus_mapping[:, 1] != 0)]
    return cell_nucleus_mapping


# TODO wrap this in a luigi task / cluster tools task, so we have caching and can move computation
# to the cluster
def write_morphology_nuclei(seg_path, table_in_path, table_out_path):
    """
    Write csv files of morphology stats for both the nucleus and cell segmentation

    seg_path - string, file path to nucleus segmentation
    cell_table_path - string, file path to cell table
    table_in_path - string, file path to nucleus table
    table_out_path - string, file path to save new nucleus table
    """

    nuclei_table = pd.read_csv(table_in_path, sep='\t')

    # remove zero label if it exists
    nuclei_table = nuclei_table.loc[nuclei_table['label_id'] != 0, :]

    # calculate stats for both tables and save results to csv
    result_nuclei = calculate_morphology_stats(nuclei_table, seg_path, 'nucleus',
                                               morphology=True, intensity=True)
    result_nuclei.to_csv(table_out_path, index=False, sep='\t')


# TODO wrap this in a luigi task / cluster tools task, so we have caching and can move computation
# to the cluster
def write_morphology_cells(seg_path, table_in_path,
                           cell_nuc_mapping_path, table_out_path):
    """
    Write csv files of morphology stats for both the nucleus and cell segmentation

    seg_path - string, file path to cell segmentation
    table_in_path - string, file path to cell table
    cell_nuc_mapping_path - string, file path to numpy array mapping cells to nuclei
        (first column cell id, second nucleus id)
    table_out_path - string, file path to save new cell table
    """

    cell_table = pd.read_csv(table_in_path, sep='\t')

    # remove zero label if it exists
    cell_table = cell_table.loc[cell_table['label_id'] != 0, :]
    cell_nucleus_mapping = load_cell_nucleus_mapping(cell_nuc_mapping_path)

    # only keep cells in the cell_nuc_mapping (i.e those that have assigned nuclei)
    cell_table = cell_table.loc[np.isin(cell_table['label_id'], cell_nucleus_mapping[:, 0]), :]

    result_cells = calculate_morphology_stats(cell_table, seg_path, 'cell', morphology=True, intensity=False)
    result_cells.to_csv(table_out_path, index=False, sep='\t')
