#! /g/schwab/Kimberly/Programs/anaconda3/bin/python
# Only works with a linux installation of ilastik atm - as it needs ./run_ilastik.sh

import subprocess
import h5py
import numpy as np
import skimage.morphology
import pandas as pd
import skimage
import os
import vigra.sampling
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calculate_slice(scale, minmax, addBorder):
    """
    Calculate slice to extract given bounding box in microns.

    Args:
        scale [listlike] - resolution given as [x, y, z]
        minmax [listlike] - list of min and max coordinates of bounding box in microns
            [min_coord_x_microns, min_coord_y_microns, min_coord_z_microns, max_coord_x_microns, max_coord_y_microns,
            max_coord_z_microns]
        addBorder [bool] - optionally add a 10 pixel border on each axis
    """

    # flip order - so now z, y, x
    mins = minmax[0:3][::-1]
    maxs = minmax[3:][::-1]
    scale = scale[::-1]

    mins = [int(mi / sca) for mi, sca in zip(mins, scale)]
    maxs = [int(ma / sca) + 1 for ma, sca in zip(maxs, scale)]

    if addBorder:
        # decrease the mins by 10 pixels
        mins = [(mi - 10) for mi in mins]
        # increase the max by 10 pixels
        maxs = [(ma + 10) for ma in maxs]

    return tuple(slice(mi, ma) for mi, ma in zip(mins, maxs))


def big_bounding_box(table, threshold):
    """
    Returns label_ids of nuclei with total bounding box volume in px
    above the threshold

    Args:
        table [pd.Dataframe] - table of nucleus statistics
        threshold [int] - total number of pixels in volume
    """
    vol = (table['bb_max_z'] - table['bb_min_z']) * (table['bb_max_y'] - table['bb_min_y']) * (
            table['bb_max_x'] - table['bb_min_x'])
    criteria = vol > threshold

    return table['label_id'][criteria]


def write_h5_files(table, folder, raw_seg_path):
    """
    Writes individual h5 file for each row in the table, equal to the bounding box of that object
    + a 10 pixel border on all dimensions

    Args:
        table [pd.Dataframe] - table of nucleus statistics
        folder [str] - a temporary folder to write files to
        raw_seg_path [str] - path to the raw segmentation .h5
    """

    for row in table.itertuples(index=False):

        # min max coordinates in microns for segmentation
        minmax_seg = [row.bb_min_x, row.bb_min_y,
                      row.bb_min_z, row.bb_max_x, row.bb_max_y,
                      row.bb_max_z]

        # raw scale (from xml) for 2x downsampled
        raw_scale = [0.02, 0.02, 0.025]

        # slice for raw file
        raw_slice = calculate_slice(raw_scale, minmax_seg, addBorder=True)
        with h5py.File(raw_seg_path, 'r') as f:
            # get 2x downsampled nuclei
            data = f['t00000/s00/1/cells']
            img_array = data[raw_slice]

        # write h5 file for nucleus
        result_path = folder + os.sep + str(row.label_id) + '.h5'
        with h5py.File(result_path, 'a') as f:

            # check dataset is bigger than 64x64x64
            if img_array.shape[0] >= 64 and img_array.shape[1] >= 64 and img_array.shape[2] >= 64:
                chunks = (64, 64, 64)
            else:
                chunks = img_array.shape

            dataset = f.create_dataset('dataset', chunks=chunks, compression='gzip', shape=img_array.shape,
                                       dtype=img_array.dtype)
            f['dataset'][:] = img_array


def get_label_id_from_file(file):
    """
    get label id from a file path
    """
    label_id = file.split(os.sep)[-1]
    label_id = label_id.split('.')[0]
    label_id = int(label_id)

    return label_id


def process_ilastik_output(table, ilastik_file, nucleus_seg_path, final_output):
    """
    Processes output h5 files form ilastik,  doing an opening / closing to clean up the
    segmentation, change label ids so (euchromatin == nucleus_id & heterochromatin == 12000 + nucleus_id) and use
    nucleus segmentation as a mask to set background to 0. Then writes to the main results file / deletes ilastik
    file.

    Args:
        table [pd.Dataframe] - table of nucleus statistics
        ilastik_file [str] - path to ilastik output file .h5
        nucleus_seg_path [str] - path to nuclear segmentation
        final_output [str] - path to the main output file .h5

    """

    # Get label id of nucleus from file name
    label_id = get_label_id_from_file(ilastik_file)

    # select correct row of table
    select = table['label_id'] == label_id

    # minmax of bounding box for that nucleus
    minmax_seg = [table.loc[select, 'bb_min_x'], table.loc[select, 'bb_min_y'],
                  table.loc[select, 'bb_min_z'], table.loc[select, 'bb_max_x'], table.loc[select, 'bb_max_y'],
                  table.loc[select, 'bb_max_z']]
    minmax_seg = [x.iloc[0] for x in minmax_seg]

    # read out ilastik result
    print('Processing Ilastik result...' + str(label_id))
    with h5py.File(ilastik_file, 'r') as f:
        dataset = f['exported_data']
        data = dataset[:]

    # reads in as zyxc, drop the c channel
    data = data[:, :, :, 0]

    # Convert from 1/2 label to 0/1 - now heterochromatin is 0 and euchromatin is 1
    data[data == 1] = 0
    data[data == 2] = 1

    # Then do opening / closing
    data = skimage.morphology.binary_opening(data)
    data = skimage.morphology.binary_closing(data)

    # remove the extra 10 pixels border around the nucleus
    data = data[10:data.shape[0] - 10, 10:data.shape[1] - 10, 10:data.shape[2] - 10]
    data = data.astype('uint16')

    # change to implicit mapping to nuclei
    # heterochromatin = nucleus id
    # euchromatin = 12000 + nucleus id
    data[data == 1] = label_id
    data[data == 0] = 12000 + label_id

    # segmentation file scale
    seg_scale = [0.08, 0.08, 0.1]

    # slice for segmentation file
    seg_slice = calculate_slice(seg_scale, minmax_seg, False)

    # open the nuclear segmentation for correct nucleus
    with h5py.File(nucleus_seg_path, 'r') as f:
        # get full-res dataset
        dataset = f['t00000/s00/0/cells']
        img_array = dataset[seg_slice]

    # binarise so 1 in the relevant nucleus, 0 outside
    img_array[img_array != label_id] = 0
    img_array[img_array == label_id] = 1

    # use the vigra resize here, seems much more memory efficient
    img_array = img_array.astype('float32')
    img_array = vigra.sampling.resize(img_array, shape=data.shape, order=0)
    img_array = img_array.astype('uint8')

    # set pixels outside the nucleus segmentation to 0
    data[img_array == 0] = 0
    img_array = None

    # raw scale (from xml) for 2x downsampled
    raw_scale = [0.02, 0.02, 0.025]

    # slice for raw file
    raw_slice = calculate_slice(raw_scale, minmax_seg, addBorder=False)

    # write to the main h5 file
    with h5py.File(final_output, 'r+') as f:
        result = f['dataset']

        # read in part covered by the nuclear bounding box
        result_data = result[raw_slice]

        # Set the part covered by the nuclear segmentation to the new values
        result_data[data != 0] = data[data != 0]

        # write it back
        result[raw_slice] = result_data

    # remove temporary segmentation file once write is successful
    os.remove(ilastik_file)


def ilastik_nuclei_prediction(nuclei_table, nucleus_seg_path, ilastik_project, ilastik_directory, tmp_input, tmp_output,
                              final_output, raw, chunk_size=3000, cores=32, memory=254000):
    """
    Processes a table of nuclei, predicting each with the specified ilastik project

    Args:
        nuclei_table [str] - path to table of nucleus statistics
        nucleus_seg_path [str] - path to nuclear segmentation
        ilastik_project [str] - path to ilastik project
        ilastik_directory [str] - directory of your ilastik installation
        tmp_input [str] - path to folder where temporary files for input to ilastik can be written
        tmp_output [str] - path to folder where temporary files for output from ilastik can be written
            final_output [str] - path to final output h5
        raw [str] - path to the raw data h5
        chunk_size [int] - number of nuclei to predict in each batch (need enough space to write this number
            of nuclei to tmp_input, and their prediction results to tmp_output)[doing it in batches like
            this is faster than starting up ilastik for every single nucleus]
        cores [int] - max number of cores for ilastik
        memory [int] - max amount of memory for ilastik (in MB) - set slightly below max, as this isn't strictly obeyed

    """

    logger.info('Producing segmentation of chromatin of nuclei')
    logger.info('INPUT FILE - nuclei table: %s', nuclei_table)
    logger.info('INPUT FILE - nucleus segmentation: %s', nucleus_seg_path)
    logger.info('INPUT FILE - ilastik project: %s', ilastik_project)
    logger.info('INPUT FILE - raw SBEM data: %s', raw)
    logger.info(
        'OUTPUT FILE - h5 file of segmentation, euchromatin label = nucleus label from table, heterochromatin label = '
        '12000 + nucleus label: %s',
        final_output)
    logger.info('PARAMETER - ilastik installation directory %s', ilastik_directory)
    logger.info('PARAMETER - chunksize, number of nuclei to process in each batch %s', chunk_size)
    logger.info('PARAMETER - cores for ilastik %s', cores)
    logger.info('PARAMETER - memory for ilastik %s', memory)

    table = pd.read_csv(nuclei_table, sep='\t')
    # remove zero label if exists
    table = table.loc[table['label_id'] != 0, :]

    # largest nuclei can cause problems - figure out which these are
    # threshold chosen from previous runs at 256 gigs of memory
    biggest_nuclei = big_bounding_box(table, threshold=100000)

    # produce result h5 with same shape as 2x downsampled raw data (this is what the ilastik project
    # was done on)
    with h5py.File(final_output, 'a') as f:
        # create a dataset the same size as constantin's cell
        # segmentation but all 0s
        # chunk / compression options are the same as constantin
        # uses in his bdv converter script
        dataset = f.create_dataset('dataset', chunks=(64, 64, 64), compression='gzip', shape=(11416, 12958, 13750),
                                   dtype='uint16')

    # set up chunks of chunkszie from start of table to end
    nrow = table.shape[0]
    start_in = list(range(0, nrow, chunk_size))
    if start_in[-1] == nrow:
        start_in.pop()
    end_in = [val + 1 for val in start_in[1:]]
    end_in.append(nrow + 1)

    # process each chunk of the table
    for start, end in zip(start_in, end_in):

        cut_table = table.iloc[start:end, :]

        print('writing h5 input files')

        # write nuclei to h5 files to input to ilastik
        write_h5_files(cut_table, tmp_input, raw)

        # settings to constrain memory  / number of cores that ilastik uses
        os.environ['LAZYFLOW_THREADS'] = str(cores)
        os.environ['LAZYFLOW_TOTAL_RAM_MB'] = str(memory)

        os.chdir(ilastik_directory)

        # get full path to each input h5 file
        input_files = os.listdir(tmp_input)
        input_files = [os.path.join(tmp_input, file) for file in input_files]

        print('running ilastik...')

        # run ilastik
        print(subprocess.check_output(
            ['./run_ilastik.sh', '--headless', '--project=' + ilastik_project, '--export_source=Simple Segmentation',
             '--output_filename_format=' + tmp_output + '/{nickname}.h5'] + input_files))

        print('removing temporary input files...')

        # remove temporary h5 files
        for file in input_files:
            os.remove(file)

        # temporary folder where output from ilastik (chromatin prediction) is saved
        output_files = os.listdir(tmp_output)
        output_files = [os.path.join(tmp_output, file) for file in output_files]

        # Can sometimes accidentally pick up temporary files like Thumbs on Windows
        # filter for just those that end with h5
        to_keep = [file.endswith('h5') for file in output_files]
        output_files = np.array(output_files)[np.array(to_keep)]

        # process each segmentation, doing some opening / closing then save to the main .h5 file
        for file in output_files:

            # check if this is one of the biggest nuclei, if so - skip it
            # the next stage is very memory hungry
            label_id = get_label_id_from_file(file)
            if np.isin(label_id, biggest_nuclei):
                logger.info(
                    'Skipping writing ilastik file for nucleus with id %s, too large - process this separately with '
                    'function process_ilastik_output',
                    label_id)
                continue

            process_ilastik_output(cut_table, file, nucleus_seg_path, final_output)


if __name__ == '__main__':
    nuclei_table = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.6.0/tables/sbem-6dpf-1-whole-segmented-nuclei-labels/default.csv'
    nucleus_seg_path = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.0.0/segmentations/sbem-6dpf-1-whole-segmented-nuclei-labels.h5'
    ilastik_project = '/g/schwab/Kimberly/Projects/SBEM_analysis/Data/Derived/Ilastik_classification/Intensity_corrected_runs/nuclei_classification.ilp'
    ilastik_directory = "/g/schwab/Kimberly/Programs/ilastik-1.3.2post1-Linux"
    tmp_input = '/g/schwab/Kimberly/Projects/SBEM_analysis/Data/Derived/Ilastik_classification/ilastik_temp_input'
    tmp_output = '/g/schwab/Kimberly/Projects/SBEM_analysis/Data/Derived/Ilastik_classification/ilastik_temp_output'
    final_output = '/g/schwab/Kimberly/Projects/SBEM_analysis/Data/Derived/Ilastik_classification/Intensity_corrected_runs/chromatin_prediction.h5'
    raw = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/analysis/em-raw-wholecorrected.h5'

    # in general run on cluster - 256GB ram, 32 cores
    ilastik_nuclei_prediction(nuclei_table, nucleus_seg_path, ilastik_project, ilastik_directory, tmp_input, tmp_output,
                              final_output, raw, chunk_size=3000, cores=32, memory=254000)

    # all nuclei should be predicted with these settings but the largest can't undergo the final
    # processing to write the ilastik output h5 file to the main file (memory errors)
    # for these process them separately with function process_ilastik_output using more memory
    # 8 cores, 384 GB memory

    # A few label ids near the end failed >> these were tiny fragments of usually 6 pixels or less > could easily add
    # in a filter for the minimum number of pixels to avoid these causing errors
