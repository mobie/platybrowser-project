# Requirements:
#
# mobie-utils-python:
# - conda create -c conda-forge mobie_utils
#
# platybrowser project:
# - git clone https://github.com/mobie/platybrowser-project.git
# - git checkout -b adding_data

import mobie
import mobie.metadata
import h5py
import tifffile

image_name = 'em-raw-low-res-test-3'  # name in MoBIE; MUST be unique within the whole project!
view = mobie.metadata.get_default_view("image", image_name, menu_name="sbem")

print("adding image...")

mobie.add_image(
    input_path='/g/cba/exchange/em-raw-low-res-dxy1_28um-dz1_6um.tif',
    input_key='',  # the input is a single tif image, so we leave input_key blank
    file_format='ome.zarr',
    root='/g/cba/exchange/platybrowser-project/data',
    dataset_name='1.0.1',
    image_name=image_name,
    resolution=(1.6, 1.28, 1.28),
    chunks=(96, 96, 96),
    scale_factors= 2 * [[2, 2, 2]]
    unit='micrometer',
    view=view,
    max_jobs=4,
    target='slurm',
)

print("done!")


#########################

from skimage import io, measure
import pandas as pd

# Load the 3D label mask image
image = io.imread('/Users/tischer/Desktop/platy-nuclei.tif')  # Make sure to provide the correct path to your file

# Measure the properties of the labeled regions
properties = measure.regionprops_table(image, properties=('label', 'centroid'))

# Convert the results to a DataFrame
df = pd.DataFrame(properties)

# Rename the columns to match the desired output
df.rename(columns={'label': 'spot_id', 'centroid-0': 'z', 'centroid-1': 'y', 'centroid-2': 'x'}, inplace=True)

# Apply the scaling to the coordinates
df['x'] *= 0.32  # Scaling for x
df['y'] *= 0.32  # Scaling for y
df['z'] *= 0.4   # Scaling for z

df.to_csv('centroids.tsv', sep='\t', index=False)


#########################


import mobie.metadata

source_name = "nuclei_centers"
view = mobie.metadata.get_default_view(source_type="spots", source_name="nuclei_centers", menu_name="spots", )

print("Adding spots...")

mobie.add_spots(
    input_table="/Users/tischer/Documents/platybrowser-project/misc/centroids.tsv",
    root="/Users/tischer/Documents/platybrowser-project/data",
    dataset_name="1.0.1",
    spot_name=source_name,
    menu_name="spots",
    unit="micrometer",
    view=view,
)

print("done!")
