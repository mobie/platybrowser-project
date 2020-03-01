# PlatyBrowser

This repository contains the data and the scripts for data generation for the PlatyBrowser, a resource for exploring a full EM volume of a 6 day old Platynereis larva combined with a gene expression atlas and tissue, cellular and ultra-structure segmentations.
For details, see [Whole-body integration of gene expression and single-cell morphology](https://www.biorxiv.org/content/10.1101/2020.02.26.961037v1).
It is implemented using the [MultiModalBrowser](https://github.com/platybrowser/mmb-fiji) (MMB), a tool for exploring multi-modal big image data.

## Data storage

Image meta-data and derived data is stored in the folder `data`. In order to deal with changes to this data, we follow a versioning scheme inspired by [semantic versioning](https://semver.org/). Version numbers are given as `MAJOR.MINOR.PATCH` where

- `PATCH` is increased if the derived data is updated, e.g. due to corrections in a segmentation or new attributes in a table.
- `MINOR` is increased if new derived data is added, e.g. a new segmentation or a new table is added.
- `MAJOR` is increased if a new modality is added, e.g. data from a different imaging source or a different specimen.

For a given version `X.Y.Z`, the data is stored in the directory `data/X.Y.Z` which contains the following subfolders:

- `images`: Contains meta-data for all images in bigdata-viewer xml format. The actual image data (stored either as hdf5 or n5) is not under version control and can either be read from the local file system (subfolder `local`) or a remote object store (subfolder `remote`). In addition, the `images` folder contains a dictionary mapping image names to viewer and storage settings in `images.json`.
- `misc`: Contains miscellanous data.
- `tables`: Contains csv tables with additional data derived from the image data.

### File naming

Image names must be prefixed by the header `MODALITY-STAGE-ID-REGION`, where
- `MODALITY` is a shorthand for the imaging modality used to obtain the data, e.g. `sbem` for serial blockface electron microscopy.
- `STAGE` is a shorthand for the develpmental stage, e.g. `6dpf` for six days post fertilisation.
- `ID` is a number that distinguishes individual animals of a given modality and stage or distinguishes different set-ups for averaging based modalities.
- `REGION` is a shorthand for the region covered by the data, e.g. `parapod` for the parapodium or `whole` for the whole animal.

### Table storage

Derived attributes are stored in csv tables, which must be associated with specific image data.
The tables associated with a given image name must be stored in the sub-directory `tables/image-name`.
If this directory exists, it must at least contain the file `default.csv` with spatial attributes of the objects in the image. If tables do not change between versions, they can be stored as relative soft-links to the old version.

### Version updates

We provide three scripts to update the respective release types:
- `update_patch.py`: Create new version folder and update derived data.
- `update_minor.py`: Create new version folder and add new image data or derived data.
- `update_major.py`: Create new version folder and add new modality. 
All three scripts take the path to a json file as argument, which encodes the data to update or to add.

For `update_patch.py` the json must contain a dictonary with the two keys `segmentations` and `tables`
where each key maps to a list containing existing segmentation names. For names listed in `segmentations`,
the segmentation AND corresponding tables (if present) will be updated. For `tables`, only the tables will be updated.
The following example would trigger an update of the segmentation and tables for the cell segmentation and a table update for the nucleus segmentation:
```
{"segmentations": ["sbem-6dpf-1-whole-segmented-cells"],
 "tables": ["sbem-6dpf-1-whole-segmented-nuclei"]}
```

For `update_minor.py` and `update_major.py`, the json must contain a dictionary mapping the names of new image data to their source files and viewer settings.
See `example_updates/` for some example json update files.

In addition, `update_registration.py` can be used to update data undergoing registration with a new registration transformation. It creates a new patch version folder and updates all relevant data.

We do not add any files to version control automatically. So after calling one of the update
scripts, add the new version folder to git and make a release via `git tag -a X.Y.Z -m "DESCRIPTION"`.


## Scripts

This repository also contains scripts that were used to generate most of the data for [Whole-body integration of gene expression and single-cell morphology](https://www.biorxiv.org/content/10.1101/2020.02.26.961037v1). `mmpb` contains a small python library that bundles most of this functionality as well as helper functions for the version updates.

### Segmentation

The folder `segmentation` contains the scripts used to generate segmentations for cells, nuclei and other tissue derived from the EM data with automated segmentation approaches.

### Registration

The folder `registration` contains the transformations for different registration versions as well as the scripts
to generate the transformations for a given version. You can use the script `registration/apply_registration.py` to apply a registration transformation to a new input file.

### Analysis

The folder `analysis` contains several scripts used for further data analyss, most notabbly cluster analysis based gene expression and cellular morphology.

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

## Citation

If you use this resource, please cite [Whole-body integration of gene expression and single-cell morphology](https://www.biorxiv.org/content/10.1101/2020.02.26.961037v1).
If you use the segmentation or registration functionality, please also include the appropriate citations, see [segmentation/README.md](https://github.com/platybrowser/platybrowser-backend/blob/master/segmentation/README.md) 
or [registration/README.md](https://github.com/platybrowser/platybrowser-backend/blob/master/registration/README.md) for details. For the initial gene expression atlas generated by ProSPr, please cite [Whole-organism cellular gene-expression atlas reveals conserved cell types in the ventral nerve cord of Platynereis dumerilii](https://www.pnas.org/content/114/23/5878.short).


## Contributing data

If you want to contribute data to this resource, please raise an issue about this in this repository or contact TODO.
