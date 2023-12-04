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

image_name = 'em-raw-low-res-test'  # name in MoBIE; MUST be unique within the whole project!
view = mobie.metadata.get_default_view("image", image_name, menu_name="sbem")

print("adding image...")

#image_data = tifffile.imread('/g/cba/exchange/em-raw-low-res-dxy1_28um-dz1_6um.tif')
#with h5py.File("/g/cba/exchange/tmp.h5", "a") as f:
#    f.create_dataset("data", data=image_data, compression="gzip")

scale_factors = 2 * [[2, 2, 2]]

mobie.add_image(
    input_path='/g/cba/exchange/tmp.h5',
    input_key='data',  # the input is a single tif image, so we leave input_key blank
    file_format='ome.zarr',
    root='/g/cba/exchange/platybrowser-project/data',
    dataset_name='1.0.1',
    image_name=image_name,
    resolution=(1.6, 1.28, 1.28),
    chunks=(96, 96, 96),
    scale_factors=scale_factors,
    unit='micrometer',
    view=view,
    target='slurm',
)

print("done!")
