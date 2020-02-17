# platybrowser-backend

This repository contains the data and scripts for data generation for the [platybrowser](https://github.com/platybrowser/platybrowser-fiji), a viewer for multi-model big image data. The resource at hand is for the 6 day old Platynereis larva, for details see [Whole-body integration of gene expression and single-cell morphology](TODO link), but the viewer can also be used for other data-sets given in the same storage layout.

## Data storage

Image meta-data and derived data is stored in the folder `data`. In order to deal with changes in the data, we follow a versioning scheme inspired by [semantic versioning](https://semver.org/). Version numbers are given as `MAJOR.MINOR.PATCH` where

- `PATCH` is increased if the derived data is updated, e.g. due to corrections in some segmentation or new attributes in some table.
- `MINOR` is increased if new derived data is added, e.g. a new segmentation or a new attribute table is added.
- `MAJOR` is increased if a new modality is added, e.g. data from a different imaging source or a different individual.

For a given version `X.Y.Z`, the data is stored in the directory `/data/X.Y.Z/` which contains the subfolders

- `images`: Contains meta-data for all images in bigdata-viewer xml fromat. The actual image data (stored either as hdf5 or n5) is not under version control and can either be read from the local file system (subfolder `local`) or a remote object store (subfolder `remote`). In addition, the `images` folder contains a dictionary mapping image names to default settings in `images.json`.
- `misc`: Miscellanous data.
- `tables`: CSV tables with additional data derived from the image data.

### File naming

The image names must adhere to the following naming scheme in order to be identifiable:
the names must be prefixed by the header `MODALITY-STAGE-ID-REGION`, where
- `MODALITY` is a shorthand for the imaging modality used to obtain the data, e.g. `sbem` for serial blockface electron microscopy.
- `STAGE` is a shorthand for the develpmental stage, e.g. `6dpf` for six day post ferilisation.
- `ID` is a number that distinguishes individual animals of a given modality and stage or distinguishes different set-ups for averaging based modalities.
- `REGION` is a shorthand for the region covered by the data, e.g. `parapod` for the parapodium or `whole` for the whole animal.

Currently, the data contains the two modalities
- `sbem-6dpf-1-whole`
- `prospr-6dpf-1-whole`

### Table storage

Derived attributes are stored in csv tables, which must be associated with specific image data.
The tables associated with a given image name must be stored in the sub-directory `tables/image-name`.
If this directory exists, it must at least contain the file `default.csv` with spatial attributes of the objects in the image. If tables do not change between versions, they can be stored as soft-links to the old version.

### Version updates

We provide three scripts to update the respective release type:
- `update_patch.py`: Create new version folder and update derived data.
- `update_minor.py`: Create new version folder and add new image data or derived data.
- `update_major.py`: Create new version folder and add new modality. 
All three scripts take the path to a json file as argument, which encodes the data to update or add.

For `update_patch`, the json must contain a dictonary with the two keys `segmentations` and `tables`.
Each key needs to map to a list that contains (valid) segmentation names. For names listed in `segmentations`,
the segmentation AND corresponding tables will be updated. For `tables`, only the tables will be updated.
The following example would trigger segmentation and table update for the cell segmentation and a table update for the nucleus segmentation:
```
{"segmentations": ["sbem-6dpf-1-whole-segmented-cells-labels"],
 "tables": ["sbem-6dpf-1-whole-segmented-nuclei-labels"]}
```

`TODO update this part`
For `update_minor`, the json needs to contain a list of dictionaries. Each dictionary corresponds to new
data to add to the platy browser. There are three valid types of data, each with different required and optional fields:
- `images`: New image data. Required fields are `source`, `name` and `input_path`. `source` refers to the primary data the image data is associated with, see [naming scheme](https://git.embl.de/tischer/platy-browser-tables#file-naming). `name` specifies the name this image data will have, excluding the naming scheme prefix. `input_path` is the path to the data to add, needs to be in bdv hdf5 format. The field `is_private` is optional. If it is `true`, the data will not be exposed in
  the public big data server.
- `static segmentations`: New (static) segmentation data. The required fields are `source`, `name` and `segmentation_path` (corresponding to `input_path` in `images`). The fields `table_path_dict` and `is_private` are optional. `table_dict_path` specifies tables associated with the segmentation as a dictionary `{"table_name1": "/path/to/table1.csv", ...}`. If given, one of the table names must be `default`.
- `dynamic segmentations`: New (dynamic) segmentation data. The required fields are `source`, `name`, `paintera_project` and `resolution`. `paintera_project` specifies path and key of a n5 container storing paintera corrections for this segmentation. `resolution` is the segmentation's resolution in micrometer. The fields `table_update_function` and `is_private` are optional. `table_update_function` can be specified to register a function to generate tables for this segmentation. The function must be
  importable from `scripts.attributes`.
The following example would add a new prospr gene to the images and a new static and dynamic segmentation derived from the em data:
```
[{"source": "prospr-6dpf-1-whole", "name": "new-gene-MED", "input_path": "/path/to/new-gene-data.xml"}
 {"source": "sbem-6dpf-1-whole", "name": "new-static-segmentation", "segmentation_path": "/path/to/new-segmentation-data.xml",
  "table_path_dict": {"default": "/path/to/default-table.csv", "custom": "/path/to/custom-table.csv"}},
 {"source": "sbem-6dpf-1-whole", "name": "new-dynamic-segmentation", "paintera_project": ["/path/to/dynamic-segmentation.n5", "/paintera/project"],
  "table_update_function": "new_update_function"}]
```

For `update_major`, the json needs to contain a dictionary. The dictionary keys correpond to new primary sources (cf. [naming scheme](https://git.embl.de/tischer/platy-browser-tables#file-naming))'
to add to the platy browser. Each key needs to map to a list of data entries. The specification of these entries corresponds to `update_minor`, except that the field `source` is not necessary.
The following example would add a new primary data source (FIBSEM) and add the corresponding raw data as private data:
```
{"fib-6dpf-1-whole": [{"name": "raw", "input_path": "/path/to/fib-raw.xml", "is_private": "true"}]}
```
`TODO end update`

See `example_updates/` for some example json update files.

We do not add any files to version control automatically. So after calling one of the update
scripts, you must add all new files yourself and then make a release via `git tag -a X.Y.Z -m "DESCRIPTION"`.

The script `update_registration.py` can be used to update registered data with a new transformation.
It creates a new patch version folder and updates all relevant data.


## Scripts

This repository also contains scripts that were used to generate most of the data for [Whole-body integration of gene expression and single-cell morphology](TODO link). `mmpb` contains a small python library that bundles most of this functionality as well as helper functions for the version updates.

### Segmentation

The folder `segmentation` contains the scripts used to generate segmentations for cells, nuclei and other tissue derived from the EM data with automated segmentation approaches.

### Registration

The folder `registration` contains the transformations for different registration versions as well as the scripts
to generate the transformations for a given registration. version You can use the script `registration/apply_registration.py` to apply a registration transformation to a new input file.

### Installation

We provide conda environments to run the python scripts. In order to install the main environment used to run the segmentation scripts and perform version updates, run
```bash
conda env create -f software/mmpb_environment.yaml
conda activate platybrowser
python setup.py install
```
To run the network training or prediction scripts a different environment is necessary, which can be installed via
```bash
conda env create -f software/train_environment.yaml
conda activate platybrowser-train
python setup.py install
```
