# platy-browser-data

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

### File naming

Xml / hdf5 filenames must adhere to the following naming scheme, in order to clearly identify the origin of the data:
the names must be prefixed by the header `MODALITY-STAGE-ID-REGION`, where
- `MODALITY` is a shorthand for the imaging modality used to obtain the data, e.g. `sbem` for serial blockface electron microscopy.
- `STAGE` is a shorthand for the develpmental stage, e.g. `6dpf` for six day post ferilisation.
- `ID` is a number that distinguishes individual animals of a given modality and stage or distinguishes different set-ups for averaging based modalities like prospr.
- `REGION` is a shorthand for the region of the animal covered by the data, e.g. `parapod` for the parapodium or `whole` for the whole animal.

Currently, the data contains the three modalities
- `sbem-6dpf-1-whole`
- `prospr-6dpf-1-whole`
- `fibsem-6dpf-1-parapod`

### Table storage

Derived attributes are stored in csv tables. Tables must be associated with a segmentation file `segmentations/segmentation-name.xml`
All tables associated with a given segmentation must be stored in the sub-directory `tables/segmentation-name`.
If this directory exists, it must at least contain the file `default.csv` with spatial attributes of the segmentation objects , which are necessary for the platybrowser table functionality.

If tables do not change between versions, they can be represented as soft-links to the old version.


## Data generation

In addition to the data, the scripts for generating the derived data are also collected here.
`scripts/segmentation` contains the scripts to generate the derived segmentations with automated segmentation approaches.
The other derived data can be generated for new segmentation versions with the script `update_platy_browser.py`;
`make_initial_version.py` was used to generate the initial data in `/data/0.0.0`.


## Usage

TODO


## Installation

TODO


## BigDataServer

TODO
