import pandas as pd
import numpy as np

# get xyz coordinates of nuclei in SBEM
nuc_path = 'W:/EM_6dpf_segmentation/platy-browser-data/data/1.0.1/tables/sbem-6dpf-1-whole-segmented-nuclei/default.csv'
nuc_table = pd.read_csv(nuc_path, sep='\t')
nuc_xyz = nuc_table[['anchor_x', 'anchor_y', 'anchor_z']]
# round to nearest int
nuc_xyz = np.round(nuc_xyz)
# write in format for transformix
with open('Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\Midline\\nuc_to_transform-1-0-1.txt', 'w') as f:
    f.write('point\n')
    f.write(str(nuc_xyz.shape[0]) + '\n')
    np.savetxt(f, nuc_xyz.values, fmt='%d')

# same for cells
cell_path = 'W:/EM_6dpf_segmentation/platy-browser-data/data/1.0.1/tables/sbem-6dpf-1-whole-segmented-cells/default.csv'
cell_table = pd.read_csv(cell_path, sep='\t')
cell_xyz = cell_table[['anchor_x', 'anchor_y', 'anchor_z']]
# round to nearest int
cell_xyz = np.round(cell_xyz)
# write in format for transformix
with open('Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\Midline\\cells_to_transform-1-0-1.txt', 'w') as f:
    f.write('point\n')
    f.write(str(cell_xyz.shape[0]) + '\n')
    np.savetxt(f, cell_xyz.values, fmt='%d')

# run through transformix from command-line (need an elastix installation - http://elastix.isi.uu.nl/)
# on windows:
# C:\Users\meechan\Documents\elastix-4.9.0-win64\transformix -def Z:\Kimberly\Projects\SBEM_analysis\Data\Derived\Midline\nuc_to_transform-1-0-1.txt -out Z:\Kimberly\Projects\SBEM_analysis\Data\Derived\Midline\1-0-1-nuc -tp Z:\Kimberly\Projects\SBEM_analysis\src\sbem_analysis\paper_code\files_for_midline_xyz\TransformParameters.BSpline10.9.9.9.txt
# C:\Users\meechan\Documents\elastix-4.9.0-win64\transformix -def Z:\Kimberly\Projects\SBEM_analysis\Data\Derived\Midline\cells_to_transform-1-0-1.txt -out Z:\Kimberly\Projects\SBEM_analysis\Data\Derived\Midline\1-0-1-cell -tp Z:\Kimberly\Projects\SBEM_analysis\src\sbem_analysis\paper_code\files_for_midline_xyz\TransformParameters.BSpline10.9.9.9.txt

# ------------------------------------------

# read in result and add to table
res_table = np.loadtxt('Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\Midline\\1-0-1-nuc\\outputpoints.txt',
                       delimiter=';', dtype='str')
res_column = res_table[:, 4]
result = [val.split('[')[-1] for val in res_column]
result = [val.split(']')[0] for val in result]
result = [val.split(' ')[1:4] for val in result]
res = []
for val in result:
    res.append([float(v) for v in val])
res_df = np.array(res)

# add as column
nuc_table['new_x'] = res_df[:, 0]
nuc_table['new_y'] = res_df[:, 1]
nuc_table['new_z'] = res_df[:, 2]
nuc_table.to_csv('Z:\\Kimberly\\Projects\\SBEM_analysis\\src\\sbem_analysis\\paper_code\\files_for_midline_xyz\\prospr_space_nuclei_points_1_0_1.csv', index=False, sep='\t')

# same for cells
res_table = np.loadtxt('Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\Midline\\1-0-1-cell\\outputpoints.txt',
                       delimiter=';', dtype='str')
res_column = res_table[:, 4]
result = [val.split('[')[-1] for val in res_column]
result = [val.split(']')[0] for val in result]
result = [val.split(' ')[1:4] for val in result]
res = []
for val in result:
    res.append([float(v) for v in val])
res_df = np.array(res)

# add as column
cell_table['new_x'] = res_df[:, 0]
cell_table['new_y'] = res_df[:, 1]
cell_table['new_z'] = res_df[:, 2]

cell_table.to_csv('Z:\\Kimberly\\Projects\\SBEM_analysis\\src\\sbem_analysis\\paper_code\\files_for_midline_xyz\\prospr_space_cells_points_1_0_1.csv', index=False, sep='\t')
