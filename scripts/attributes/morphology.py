#calculate morphology stats for cells / nuclei

import pandas as pd
import h5py
import numpy as np
from skimage.measure import regionprops, marching_cubes_lewiner, mesh_surface_area
from skimage.transform import resize
from skimage.io import imsave
from skimage.util import pad

def calculate_slice (scale, minmax):
    """
    Takes a scale (get this from xml file of object), and list of min and 
    max coordinates in microns [min_coord_x_microns, min_coord_y_microns,
    min_coord_z_microns, max_coord_x_microns, max_coord_y_microns, 
    max_coord_z_microns].
    Return a slice to use to extract
    that bounding box from the original image.
    """
    #Get mins and flip order - so now z, y, x
    mins = minmax[0:3][::-1]
    #Get maxes and flip order - so now z, y, x
    maxs = minmax[3:][::-1]
    
    #flip order of the scale
    scale = scale[::-1]
    
    #convert min and max in microns to pixel ranges
    mins = [int(mi / sca) for mi, sca in zip(mins, scale)]
    maxs = [int(ma / sca) + 1 for ma, sca in zip(maxs, scale)]
    
    return( tuple(slice(mi, ma) for mi, ma in zip(mins, maxs)) )
    
  
def calculate_stats_binary (mask, scale):
    """
    Calculate morphology statistics from the binary mask of an object
    1 in object, 0 outside. 
    mask - numpy array for image, should be binary, 1 in object, 0 outside
    scale - list of form [x, y, z] stating scale in microns of image
    """
    #Calculate stats from skimage
    ski_morph = regionprops(mask)
    
    #volume in pixels
    volume_in_pix = ski_morph[0]['area']
    
    #extent
    extent = ski_morph[0]['extent']
    
    #The mesh calculation below fails if an edge of the segmentation is right up against the
    #edge of the volume - gives an open, rather than a closed surface
    #Pad by a few pixels to avoid this
    mask = pad(mask, 10, mode = 'constant')
    
    #surface area of mesh around object (other ways to calculate better?)
    #spacing z, y , x to match mask dimensions
    verts, faces, normals, values = marching_cubes_lewiner(mask, spacing = (scale[2], scale[1] , scale[0]))
    surface_area = mesh_surface_area(verts, faces)
    
    #volume in microns
    volume_in_microns = scale[0]*scale[1]*scale[2]*volume_in_pix
    
    #sphericity (as in morpholibj)
    #Should run from zero to one
    sphericity = (36*np.pi*(float(volume_in_microns)**2))/(float(surface_area)**3)
    
    return (volume_in_microns, extent, surface_area, sphericity)


def calculate_stats_intensity (raw, mask):
    """
    Calculate intensity stats on the raw EM data for the region covered by the
    provided binary mask. Mask and raw must have the same dimensions!
    
    raw - numpy array of raw image data
    mask - numpy array of binary mask, 1 in object, 0 outside
    """
    
    intensity_vals_in_mask = raw[mask == 1]
    
    #mean and stdev - use float64 to avoid silent overflow errors
    mean_intensity = np.mean(intensity_vals_in_mask, dtype = np.float64)
    st_dev = np.std(intensity_vals_in_mask, dtype = np.float64)
    
    return (mean_intensity, st_dev)


#TODO - possibility that downscaled version of cell segmentation may not include some of the smallest cells in the table -
#will fail if this happens. 
def calculate_row_stats (row, seg_path, input_type, morphology, intensity):
    """
    Calculate all morphology stats for one row of the table
    row - named tuple of values from that row
    input_type - 'nucleus' or 'cell'
    morphology - whether to calculate stats based on shape of segmentation: volume, extent, surface area, sphericity
    intensity - whether to calculate stats based on raw data covered by segmentation: mean_intensity, st_dev_intensity
    """
    if row.label_id % 100 == 0:
        print('Processing ' + str(row.label_id))
    
    #tuple to hold result
    result = (row.label_id,)
    
    if input_type == 'nucleus':
        full_seg_scale = [0.08, 0.08, 0.1]
        #use full res of segmentation - 0.08, 0.08, 0.1
        seg_level = 0
        #Use the /t00000/s00/3/cells for raw data - gives pixel size similar to the nuclear segmentation
        raw_level = 3
        
    elif input_type == 'cell':
        full_seg_scale = [0.02, 0.02, 0.025]
        #look at 4x downsampled segmentation - close to 0.08, 0.08, 0.1
        seg_level = 2
    
    #min max bounding box in microns for segmentation
    minmax_seg = [row.bb_min_x, row.bb_min_y, row.bb_min_z, row.bb_max_x, row.bb_max_y, row.bb_max_z]
    
    with h5py.File(seg_path, 'r') as f:
        
        #get shape of full data & downsampled
        full_data_shape = f['t00000/s00/0/cells'].shape
        down_data_shape = f['t00000/s00/' + str(seg_level) + '/cells'].shape
            
        #scale for downsampled
        down_scale = [full_seg_scale[0]*(full_data_shape[2]/down_data_shape[2]),
                      full_seg_scale[1]*(full_data_shape[1]/down_data_shape[1]),
                      full_seg_scale[2]*(full_data_shape[0]/down_data_shape[0])]
            
        #slice for seg file
        seg_slice = calculate_slice(down_scale, minmax_seg)
            
        #get specified layer in bdv file
        data = f['t00000/s00/' + str(seg_level) + '/cells']
        img_array = data[seg_slice]
        
     
    #make into a binary mask
    binary = img_array == int(row.label_id)
    binary = binary.astype('uint8')
    
    img_array = None
    
    #calculate morphology stats straight from segmentation
    if morphology == True:

        result = result + calculate_stats_binary(binary, down_scale)
    
    #calculate intensity statistics
    if intensity == True:
        
        #scale for full resolution raw data
        full_raw_scale = [0.01, 0.01, 0.025]
            
        with h5py.File('/g/arendt/EM_6dpf_segmentation/EM-Prospr/em-raw-full-res.h5', 'r') as f:
            
            #get shape of full data & downsampled
            full_data_shape = f['t00000/s00/0/cells'].shape
            down_data_shape = f['t00000/s00/' + str(raw_level) + '/cells'].shape
            
            #scale for donwampled
            down_scale = [full_raw_scale[0]*(full_data_shape[2]/down_data_shape[2]),
                          full_raw_scale[1]*(full_data_shape[1]/down_data_shape[1]),
                          full_raw_scale[2]*(full_data_shape[0]/down_data_shape[0])]
            
            #slice for raw file
            raw_slice = calculate_slice(down_scale, minmax_seg)
            
            #get specified layer in bdv file
            data = f['t00000/s00/' + str(raw_level) + '/cells']
            img_array = data[raw_slice]
            
        
        if img_array.shape != binary.shape:

            #rescale the binary to match the raw, using nearest neighbour interpolation (order = 0)
            binary = resize(binary, img_array.shape, order = 0, mode = 'reflect', anti_aliasing = True, preserve_range = True)
            binary = binary.astype('uint8')
        
        
        #Calculate statistics on raw data
        result = result + calculate_stats_intensity(img_array, binary)
    
    
    return (result)

    
def calculate_morphology_stats(table, seg_path, input_type, morphology = True, intensity = True):
    '''
    Calculate statistics based on EM & segmentation.
    table - input table (pandas dataframe) 
    input_type - 'nucleus' or 'cell'
    morphology - whether to calculate stats based on shape of segmentation: volume_in_microns, extent, surface area, sphericity
    intensity - whether to calculate stats based on raw data covered by segmentation: mean_intensity, st_dev_intensity
    '''
    
    #size cutoffs (at the moment just browsed through segmentation and chose sensible values)
    #not very stringent, still some rather large / small things included
    if input_type == 'nucleus':
        table = table.loc[table['shape_pixelsize'] >= 18313.0, :]
        
    elif input_type == 'cell':
        criteria = np.logical_and(table['shape_pixelsize'] > 88741.0, table['shape_pixelsize'] < 600000000.0)
        table = table.loc[criteria, :]
    
    stats = [calculate_row_stats(row, seg_path, input_type, morphology, intensity) for row in table.itertuples(index=False)]
    
    #convert back to pandas dataframe
    stats = pd.DataFrame(stats)
    
    #names for columns
    columns = ['label_id']
    
    if morphology == True:
        columns = columns + ['shape_volume_in_microns', 'shape_extent', 'shape_surface_area', 'shape_sphericity']
        
    if intensity == True:
        columns = columns + ['intensity_mean', 'intensity_st_dev']
    
    #set column names
    stats.columns = columns
    
    return (stats)


def write_morphology_table (cell_seg_path, nucleus_seg_path, cell_table_path, nucleus_table_path, cell_nuc_mapping_path, nucleus_out_path, cell_out_path):
    """
    Write csv files of morphology stats for both the nucleus and cell segmentation
    
    cell_seg_path - string, file path to cell segmentation
    nucleus_seg_path - string, file path to nucleus segmentation
    cell_table_path - string, file path to cell table
    nucleus_table_path - string, file path to nucleus table
    cell_nuc_mapping_path - string, file path to numpy array mapping cells to nuclei (first column cell id, second nucleus id)
    nucleus_out_path - string, file path to save new nucleus table
    cell_out_path - string, file path to save new cell table
    """
    
    cell_table = pd.read_csv(cell_table_path, sep = '\t')
    
    nuclei_table = pd.read_csv(nucleus_table_path, sep = '\t')

    #remove zero label form both tables if it exists
    nuclei_table = nuclei_table.loc[nuclei_table['label_id'] != 0, :]
    cell_table = cell_table.loc[cell_table['label_id'] != 0, :]
    
    #read in numpy array of mapping of cells to nuclei - first column cell id, second nucleus id
    cell_nucleus_mapping = np.load(cell_nuc_mapping_path)
    #remove zero labels from this table too, if exist
    cell_nucleus_mapping = cell_nucleus_mapping[np.logical_and(cell_nucleus_mapping[:,0] != 0, cell_nucleus_mapping[:, 1] != 0)]
    
    #only keep cells in the cell_nuc_mapping (i.e those that have assigned nuclei)
    cell_table = cell_table.loc[np.isin(cell_table['label_id'], cell_nucleus_mapping[:, 0]), :]
    
    #calculate stats for both tables and save results to csv
    result_nuclei = calculate_morphology_stats(nuclei_table, nucleus_seg_path, 'nucleus', morphology = True, intensity = True)
    result_nuclei.to_csv(nucleus_out_path, index = False, sep = '\t')
    
    result_cells = calculate_morphology_stats(cell_table, cell_seg_path, 'cell', morphology = True, intensity = False)
    result_cells.to_csv(cell_out_path, index = False, sep = '\t')
    
    
    
    
    
    
    
    
