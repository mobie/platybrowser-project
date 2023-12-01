# Requirements:
#
# # mobie-utils-python
# - git clone https://github.com/mobie/mobie-utils-python
# - cd mobie-utils-python
# - conda env create -f environment.yaml
# - conda activate mobie
# - cd mobie-utils-python
# - pip install -e .
#
#
# # platybrowser-project
# - git clone https://github.com/mobie/platybrowser-project.git
# - git checkout -b add_my_data
#

import mobie
import mobie.metadata

mobie_project_folder = "/Users/tischer/Documents/platybrowser-project/data"
dataset_name = "1.0.1"
input_image_file = "/Users/tischer/Documents/platybrowser-project/misc/em-raw-low-res-dxy1_28um-dz1_6um.tif"

image_name = 'em-raw-low-res-test'  # name in MoBIE; MUST be unique within the whole project!
unit = 'micrometer'
resolution = (1.6, 1.28, 1.28)  # zyx
chunks = (96, 96, 96)
scale_factors = 1 * [[2, 2, 2]]  # resolution pyramid
print("scale factors:", scale_factors)

view = mobie.metadata.get_default_view("image", image_name, menu_name="sbem")

print("adding image...")

mobie.add_image(
    input_path=input_image_file,
    input_key='',  # the input is a single tif image, so we leave input_key blank
    file_format="ome.zarr",
    root=mobie_project_folder,
    dataset_name=dataset_name,
    image_name=image_name,
    resolution=resolution,
    chunks=chunks,
    scale_factors=scale_factors,
    unit=unit,
    view=view
)

print("done!")

