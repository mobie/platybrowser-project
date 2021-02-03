# UNet for platynereis cell membrane prediction

3D UNet for predicting cell membranes in SBEM volume of *Platynereis dumerilii* larva.
The [3d U-Net](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49) was trained with [long-range affinity loss](https://arxiv.org/abs/1706.00120).
Its predictions, are further used for instance segmentation via the [Lifted Multicut workflow](https://www.frontiersin.org/articles/10.3389/fcomp.2019.00006/full), including priors from the nucleus segmentation.

If you use any of the segmentation functionality provided, please cite the [main publication](https://www.biorxiv.org/content/10.1101/2020.02.26.961037v1) and the appropriate methods. 

Training data and weights for the 3d U-Nets are available on zenodo:
[Training Data](https://zenodo.org/record/3675220/files/membrane.zip?download=1), [Weights](https://zenodo.org/record/3675288/files/cilia.nn?download=1)
