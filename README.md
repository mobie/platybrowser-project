[![DOI](https://zenodo.org/badge/238514327.svg)](https://zenodo.org/badge/latestdoi/238514327)

# PlatyBrowser

This repository contains the data and the scripts for data generation for the PlatyBrowser, a resource for exploring a full EM volume of a 6 day old Platynereis larva combined with a gene expression atlas and tissue, cellular and ultra-structure segmentations.
For details, see [Whole-body integration of gene expression and single-cell morphology](https://www.biorxiv.org/content/10.1101/2020.02.26.961037v1).
It is implemented using [MoBIE](https://github.com/mobie/mobie), a platform for exploring and sharing multi-modal big image data.

**If you are new to MoBIE:**

- Please install the [Fiji plugin](https://github.com/mobie/mobie-viewer-fiji?tab=readme-ov-file#install)
- Please enter this Github repo URL (https://github.com/cyrilcros/platybrowser-project-2025) as your MoBIE project

The default Platybrowser MoBIE dataset is https://github.com/mobie/platybrowser-project. What you have here is currently a fork of this repository.
**Just clicking *Open Platybrowser* in Mobie won't work until we merge, when we publish.**

## Data storage - edited August 6th 2025

Image meta-data and derived data is stored in the folder `data`. In order to deal with changes to this data, we previously followed a versioning scheme (see below). A new version would be a new MoBIE dataset. As of August 2025, we are not versioning the data explicitly but instead relying on Git.
In this fork:

* the `1.0.1` segmentation is now associated to the main 6dpf EM dataset, named `platybrowser_6dpf`
* we progressively remove the other datasets as part of clean up efforts
* we adopt the [MoBIE project specification](https://mobie.github.io/specs/mobie_spec.html)
* we keep for now the `local` data specification but it won't work unless you use specific repo with the data. I would need to migrate it.
* I rename the `images/remote` part to `images/bdv-n5-s3`, update those `sources` and leave the current S3 bucket fully untouched.

As in MoBIE, we have the following:
- `images`: Contains meta-data for all images in bigdata-viewer xml format. The actual image data (stored either as hdf5 or n5) is not under version control and can either be read from the local file system (subfolder `local`, would have to change per #1) or a remote object store (subfolder `bdv-n5-s3`, formerly `remote`). In addition, the `images` folder contains a dictionary mapping image names to viewer and storage settings in `images.json`.
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

### Version updates - edited August 6th 2025

Previously, the previous incarnation of Platybrowser was versioned using version numbers for datasets. You would add a new version folder to git and make a release via `git tag -a X.Y.Z -m "DESCRIPTION"`. 

The logic was, quoting Constantin Pape:
- `Z` is increased if the derived data is updated, e.g. due to corrections in a segmentation or new attributes in a table.
- `Y` is increased if new derived data is added, e.g. a new segmentation or a new table is added.
- `X` is increased if a new modality is added, e.g. data from a different imaging source or a different specimen.

The problem we have is that it is a bit of a special use case that was developped in parallel with [MoBIE](https://github.com/mobie/mobie), and a lot
of the scripts that were being used this way are outdated. Because there is already a concept of version control in MoBIE, there is a duplication of efforts
when trying to also version control datasets.

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

If you want to contribute data to this resource, please raise an issue about this in this repository or contact us at mmplatyb@gmail.com.
