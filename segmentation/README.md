# Segmentation Methods

Different methods were used to segment the structures of interest in the EM-Volume:
- cells: Cell membranes were segmented with a [3d U-Net](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49) trained with [long-range affinity loss](https://arxiv.org/abs/1706.00120). Based on these predictions, instance segmentation was performed via [Lifted Multicut workflow](https://www.frontiersin.org/articles/10.3389/fcomp.2019.00006/full), including priors from the nucleus segmentation.
- chromatin: [Ilastik](https://www.nature.com/articles/s41592-019-0582-9) pixel classification was used, restricted to the segmented nuclei.
- cilia: Cilia boundaries were segmented with a [3d U-Net](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49) trained with [long-range affinity loss](https://arxiv.org/abs/1706.00120). Based on these predictions, instance segmentation was performed via [Mutex Watershed](http://openaccess.thecvf.com/content_ECCV_2018/html/Steffen_Wolf_The_Mutex_Watershed_ECCV_2018_paper.html) and [Block-wise Multicut](http://openaccess.thecvf.com/content_ICCV_2017_workshops/w1/html/Pape_Solving_Large_Multicut_ICCV_2017_paper.html).
- cuticle: Cuticle boundaries were segmented with a [3d U-Net](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49) trained with [long-range affinity loss](https://arxiv.org/abs/1706.00120). Based on these predictions, instance segmentation was performed via [Mutex Watershed](http://openaccess.thecvf.com/content_ECCV_2018/html/Steffen_Wolf_The_Mutex_Watershed_ECCV_2018_paper.html) and [Block-wise Multicut](http://openaccess.thecvf.com/content_ICCV_2017_workshops/w1/html/Pape_Solving_Large_Multicut_ICCV_2017_paper.html).
- ganglia: The ganglia were segmented by manually selecting the ids of segmented cells.
- nuclei: Nuclear membranes were segmented with a [3d U-Net](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49) trained with [long-range affinity loss](https://arxiv.org/abs/1706.00120). Based on these predictions, instance segmentation was performed via [Mutex Watershed](http://openaccess.thecvf.com/content_ECCV_2018/html/Steffen_Wolf_The_Mutex_Watershed_ECCV_2018_paper.html) and [Block-wise Multicut](http://openaccess.thecvf.com/content_ICCV_2017_workshops/w1/html/Pape_Solving_Large_Multicut_ICCV_2017_paper.html).
- tissue: Tissue and regions were segmented using the  [Ilastik carving workflow](https://www.nature.com/articles/s41592-019-0582-9)

If you use any of the segmentation functionality provided, please cite the [main publication]() AND the appropriate methods. 
For most of these methods, the scalable implementations in [cluster tools](https://github.com/constantinpape/cluster_tools) were used.

Training data and weights for the 3d U-Nets are available on zenodo:
- cells: [Training Data](), [Weights]()
- cilia: [Training Data](), [Weights]()
- chromatin: [Training Data](), [Weights]()
- nuclei: [Training Data](), [Weights]()

TODO ilastik projects for chromatin and tissue/regions on zenodo?
