# platy-browser-tables

Data and data-generation for the [platybrowser](https://github.com/embl-cba/fiji-plugin-platyBrowser).


## Data storage

Image data (only links for the image volumes) and derived data for all versions are stored in the folder `data`.
We follow a versioning scheme inspired by [semantic versioning](https://semver.org/), hence all version
numbers are given as `MAJOR.MINOR.PATCH`.

- `PATCH` is increased if the derived data is update, e.g. due to corrections in some segmentation or new attributes in some table. This is usually triggered automatically (see section below).
- `MINOR` is increased if new derived data is added, e.g. a new segmentation for some structure or a new attribute table. This needs to be done manually.
- `MAJOR` is increased if new image / raw data is added, e.g. a new animal registered to the atlas or new genes. This needs to be done manually.

For a given version `X.Y.Z`, the data is stored in the directory `/data/X.Y.Z/` with subfolders:

- `images`: Raw image or gene expression data. Contains bigdata-viewer xml files with absolute links to h5 files on the embl server.
- `misc`: Miscellanous data.
- `segmentations`: Segmentation volumes derived from the image data. Only xml files.
- `tables`: CSV tables with attributes derived from image data and segmentations.


### Table storage

Derived attributes are stored in csv tables. Tables must be associated with a segmentation file `segmentations/segmentation-name.xml`
All tables associated with a given segmentation must be stored in the sub-directory `tables/segmentation-name`.
If this directory exists, it must at least contain the file `default.csv` with spatial attributes of the segmentation objects , which are necessary for the platybrowser table functionality.

TODO think about linking to previous table version. Simplest option would be to replace the directory `tables/segmentation-name` with a
text file that contains the absolute path to the previous table version.


## Data generation

In addition to the data, the scripts for generating the derived data are also collected here.
`scripts/segmentation` contains the scripts to generate the derived segmentations with automated segmentation approaches.
The other derived data can be generated for new segmentation versions with the script `update_platy_browser.py`;
`make_initial_version.py` was used to generate the initial data in `/data/0.0.0`.


## Installation

TODO
