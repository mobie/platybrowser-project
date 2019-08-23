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
according to the following specifications:

For `update_patch`, the json needs to contain a dictonary with the two keys `segmentations` and `tables`.
Each key needs to map to a list that contains (valid) names of segmentations. For names listed in `segmentations`,
the segmentation AND corresponding tables will be updated. For `tables`, only the tables will be updated.
The following example would trigger segmentation and table update for the cell segmentation and a table update for the nucleus segmentation:
```
{"segmentations": ["sbem-6dpf-1-whole-segmented-cells-labels"],
 "tables": ["sbem-6dpf-1-whole-segmented-nuclei-labels"]}
```

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

See `example_updates/` for additional json update files.

For now, we do not add any files to version control automatically. So after calling one of the update
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

The platy browser can be served with [BigDataViewerServer](https://github.com/bigdataviewer/bigdataviewer-server).
On the EMBL server, you can start it from one of the version foldes misc directories:

```
cd data/X.Y.Z/misc
java -jar /g/cba/exchange/bigdataserver/bigdataviewer-server-2.1.2-jar-with-dependencies.jar -d bdv_server.txt
```


## Data generation

In addition to the data, the scripts for generating the derived data are also collected here.
`scripts/segmentation` contains the scripts to generate the derived segmentations with automated segmentation approaches.
`deprecated/make_initial_version.py` was used to generate the initial data in `/data/0.0.0`.
