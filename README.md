# platy-browser-data

Data and data-generation for the [platybrowser](https://github.com/embl-cba/fiji-plugin-platyBrowser).


## Data storage

Image data and derived data for all versions are stored in the folder `data`.
We follow a versioning scheme inspired by [semantic versioning](https://semver.org/), hence all version
numbers are given as `MAJOR.MINOR.PATCH`.

- `PATCH` is increased if the derived data is update, e.g. due to corrections in some segmentation or new attributes in some table. It is increased by `update_patch.py`.
- `MINOR` is increased if new derived data is added, e.g. a new segmentation for some structure or a new attribute table. It is increased by `update_minor.py`.
- `MAJOR` is increased if new image / raw data is added, e.g. a new animal registered to the atlas or new genes. It is increased by `update_major.py`.

For a given version `X.Y.Z`, the data is stored in the directory `/data/X.Y.Z/` with subfolders:

- `images`: Raw image or gene expression data. Contains bigdata-viewer xml and hdf5 files. The hdf5 files are not under version control.
- `misc`: Miscellanous data.
- `segmentations`: Segmentation volumes derived from the image data.
- `tables`: CSV tables with attributes derived from image data and segmentations.

### File naming

Xml / hdf5 filenames must adhere to the following naming scheme in order to clearly identify the origin of the data:
the names must be prefixed by the header `MODALITY-STAGE-ID-REGION`, where
- `MODALITY` is a shorthand for the imaging modality used to obtain the data, e.g. `sbem` for serial blockface electron microscopy.
- `STAGE` is a shorthand for the develpmental stage, e.g. `6dpf` for six day post ferilisation.
- `ID` is a number that distinguishes individual animals of a given modality and stage or distinguishes different set-ups for averaging based modalities like prospr.
- `REGION` is a shorthand for the region of the animal covered by the data, e.g. `parapod` for the parapodium or `whole` for the whole animal.

Currently, the data contains the three modalities
- `sbem-6dpf-1-whole`
- `prospr-6dpf-1-whole`

### Table storage

Derived attributes are stored in csv tables. Tables must be associated with a segmentation file `segmentations/segmentation-name.xml`
All tables associated with a given segmentation must be stored in the sub-directory `tables/segmentation-name`.
If this directory exists, it must at least contain the file `default.csv` with spatial attributes of the segmentation objects , which are necessary for the platybrowser table functionality.

If tables do not change between versions, they can be stored as soft-links to the old version.


## Usage

We provide three scripts to update the respective release digit:
- `update_patch.py`: Create new version folder and update derived data.
- `update_minor.py`: Create new version folder and add derived data.
- `update_major.py`: Create new version folder and add primary data. 

All three scripts take the path to a json file as argument. The json needs to encode which data to update/add
according to the following specification: TODO describe.
See some example json files in `example_updates/`.

For now, we do not add any files to version control automatically, so after calling one of the update
scripts, you must add all new files yourself and then make a release via `git tag -a X.Y.Z -m "DESCRIPTION"`.

In addition, the script `make_dev_folder.py` can be used to create a development folder. It copies the most
recent release folder into a folder prefixed with  dev-`, that will not be put under version control.


## Installation

The data is currently hosted on the arendt EMBL share, where a conda environment with all necessary dependencies is
available. This environment is used by default.

It can be installed elsewhere using the `environment.yaml` file we provide:
```
conda env create -f environment.yaml
```


## BigDataServer

TODO


## Data generation

In addition to the data, the scripts for generating the derived data are also collected here.
`scripts/segmentation` contains the scripts to generate the derived segmentations with automated segmentation approaches.
`deprecated/make_initial_version.py` was used to generate the initial data in `/data/0.0.0`.
