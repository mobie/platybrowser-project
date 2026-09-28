# Platynereis dumerilii Multimodal Atlas

## Introduction
This dataset presents a multimodal, cellular-resolution atlas of the complete body of the marine annelid *Platynereis dumerilii* at 6-days post-fertilization (dpf). It serves as a comprehensive framework that systematically integrates single-cell morphological ultrastructure with whole-body gene expression maps.

## How It Was Generated
The dataset was generated through a multi-scale combination of high-resolution electron microscopy and spatial transcriptomics:
* **Electron Microscopy:** A complete 6-dpf worm was imaged using serial block-face scanning electron microscopy (SBEM). This produced a continuous 3D volume consisting of 11,416 planar images with a resolution of 10 nm in the x/y axes and 25 nm section thickness.
* **Gene Expression Atlas:** An existing signal probability mapping (ProSPr) gene expression atlas was expanded to include 201 genes, prioritizing transcription factors and neural effectors.
* **Integration:** The multimodal integration was achieved by registering the low-resolution ProSPr atlas to the high-resolution SBEM volume using stepwise, automated image registration. This alignment yielded a registration accuracy of less than one cell diameter.

## What Was Segmented
Researchers utilized a combination of 3D U-net neural networks and automated agglomeration algorithms to segment various features from the multi-terabyte SBEM volume:
* **Cellular Structures:** Every individual cell boundary and nucleus within the whole body was segmented.
* **Subnuclear Features:** The nuclei were further segmented based on image contrast to differentiate between the active euchromatin and the highly scattering heterochromatin/nucleoli phases.
* **Tissues and Organelles:** Specific anatomical structures were semantically segmented at a targeted scale, including longitudinal muscles, neuropil, individual ciliary bundles in the nephridia, and the animal's cuticle.

## Cell and Nucleus Count
* The automated cellular segmentation successfully identified **11,402 individual cells with nuclei**.
* The extracted nuclei ranged in size from 33.6 to 147.5 µm³, while the full cells ranged from 59.8 to 1,224.6 µm³.

## Quantitative Dataset Metrics
| Metric | Value |
| :--- | :--- |
| **Dataset Size** | 2.5 TB (comprising 11,416 planar images stitched from over 200,000 tiles) |
| **Image Resolution** | 10 x 10 nm² per pixel with a 25 nm section thickness |
| **Extracted Features** | 140 distinct morphometric descriptors calculated for every cell and nucleus |
| **Mapped Genes** | 201 genes from the ProSPr atlas (including 78 transcription factors) |
| **Traced Neurons** | 384 neurons manually traced, covering 36.94 mm of neurite length |

## Advanced Algorithms and Segmentation Techniques
The processing of this multi-terabyte volume required specialized machine-learning pipelines to handle different aspects of the cellular anatomy:
* **Algorithmic Framework:** The researchers utilized a 3D U-net neural network coupled with the Mutex Watershed algorithm for highly accurate nuclear segmentation (99.0% agreement with expert annotations). 
* **Top-Down Constraints:** To map the full cellular boundaries (achieving 90.3% accuracy), they employed a Lifted Multicut framework. This system uniquely used the previously segmented nuclei as top-down constraints to prevent the algorithm from falsely merging adjacent cells.
* **Virtual Cells (VCs):** Registering a lower-resolution gene atlas onto high-resolution electron microscopy can cause overlapping errors due to biological variability. To solve this, the team generated 12,393 "Virtual Cells"—spatially coherent units of homogeneous gene expression—that allowed them to more accurately assign genetic profiles to the physical 3D cell structures. 

## Project Interest and Significance
The overarching interest of this project lies in bridging the gap between an organism's genotype and its physical cellular phenotype on a whole-body scale:
* **Genotype-Phenotype Correlation:** By linking a cell's morphometric descriptors—such as shape, volume, and chromatin topography—directly to its gene expression profile, the project establishes how genetic decoding drives functional cellular specialization.
* **Chromatin as a Proxy for Gene Activation:** The detailed segmentation of the nuclei revealed that cells with highly active genes (like ciliated or digestive cells) have larger nuclei and a greater heterochromatin surface area. This physically reflects the unpacking of DNA required for active transcription.
* **Identifying Bilateral Symmetry:** The morphometric data—specifically the intensity and texture of a cell's chromatin—was so highly detailed that the researchers could use it computationally to identify a cell's exact bilateral twin on the opposite side of the worm's body.
* **Redefining Tissues:** The dataset allows for the clustering of cells into genetically defined groups, demonstrating that coherent gene expression naturally aligns with morphological tissue boundaries and specific ganglionic nuclei.
* **Evolutionary Insights and Brain Architecture:** Mapping neuronal tracings alongside gene expression has unlocked new evolutionary perspectives. The dataset uncovered that the Platynereis head is a mixture of segmentally iterated parts and highly unique regions. Researchers discovered sensory-neurosecretory properties in the annelid's associative mushroom bodies, which remarkably share molecular anatomy with the vertebrate telencephalon. Detailed molecular mapping identified a specific proliferative region that expresses a unique combination of transcription factors, drawing strong evolutionary parallels to the development of interneurons in the vertebrate brain.
* **Open Access Tool:** To make this vast resource accessible, the project provides an open-source Fiji plugin called the "PlatyBrowser". This tool allows researchers worldwide to interactively explore, visualize, and analyze the terabyte-sized multimodal big image data remotely. 

## 2025 Paper Additions

### Single-cell sequencing identifies individuated cell types

Single-nuclei RNA-seq was performed across 6 different larval batches (14 10X Genomics libraries), obtaining 92,136 nuclei after quality control and doublet removal. This coverage is approximately 10 times the average number of cells in a *Platynereis* larva at 6 dpf, sufficient to sample all bilateral cellular pairs.

Using Seurat (resolution 0.8), an initial 65 clusters were obtained, but these did not resolve major cell classes (e.g. muscle and neurons clustering together). A bootstrapped neighbor-joining tree supported 13 clades for further analysis, with 11 "no clade" standalone clusters. Clades were assigned broad labels based on marker genes.

Each well-supported clade was iteratively subclustered and manually inspected for unique expression profiles (transcription factors and effector genes). Clusters lacking specific expression profiles were excluded as developmental precursors or insufficiently resolved cell states. This yielded a curated atlas of **268 genetically individuated cell types** representing **31,121 differentiated cells**. Each cell type received a unique identifier recording its refinement history (e.g. `clade11sub3subsub8`). A neighbor-joining tree from these curated cell types groups them by shared differentiation programmes rather than developmental proximity.

### Mapping cell types to the EM volume using HCR-FISH

To go beyond the 205 genes in ProSPr and spatially map all scRNAseq-defined cell types, a new pipeline with enhanced sensitivity was established based on **HCR-FISH**. Over 240 volumes were processed with fully automated registration to the EM volume, without any manual input. The registered expression is visualized in the new edition of PlatyBrowser.

Registration accuracy was tested on genes expressed in morphologically distinct structures (e.g. r-opsin in adult eyes, st-mhc in muscle cells). Registration is highly consistent between replicates, enabling single-cell accuracy without needing to average over many samples (as was done for ProSPr). The enhanced sensitivity of HCR-FISH also enabled the addition of previously undetectable genes.

### scLocator: mapping scRNAseq cell types to EM cells

The **scLocator** algorithm probabilistically matches scRNAseq-defined cell types to segmented EM cells using registered marker gene expression. For every EM cell, a partial gene expression vector is defined from marker genes. For every scRNAseq cell type, both a complete expression profile and a marker-gene-only vector are defined. These are probabilistically matched without enforcing assumptions about the similarity of the gene expression spaces between scRNAseq and spatial marker signals.

Each mapped cell type is then curated manually, taking into account bilateral symmetry and known staining artefacts. Automatically located and curated cell types are visualized in the new PlatyBrowser (as seen in the `2025-paper-cell-type-predictions` UI selection group).

### Cellular differentiation programmes

Gene co-expression analysis across the curated cell types revealed **42 gene clusters** with distinct expression patterns. By swapping the raw count matrix to cluster genes across cell types, and in parallel using weighted gene co-expression network analysis, the team identified large gene sets active consistently across cell types belonging to the same clade. Most of these sets encode effector genes associated with cellular morphology, physiology, and architecture — representing cellular differentiation programmes that establish clade-specific structure-function.

These programmes were analysed by inspecting specifically expressed genes and the functional cellular modules they encode: receptor and messenger systems, cytoskeletal elements, organelle features, physiological pathways, membrane characteristics, and junctional components. Using scLocator, these programmes were linked to the subcellular morphology of the 6 dpf worm, identifying **eight major cellular differentiation programmes**:

1. Innate immune cells
2. Epidermis
3. Gut
4. Coelomic support cells
5. Sarcomeric musculature
6. Glia
7. Glands
8. Neurons

### Cell type naming convention and mapping to MoBIE nuclei

Subtypes follow a hierarchical naming convention derived from the iterative subclustering process: `cladeXsubYsubsubZ`. The final `subsub` division may not exist (e.g. `clade11sub48` has no `subsub`). A clade of leftover outliers is named `noclade`, producing names like `nocladesub1`.

scLocator maps these cell types to individual **nuclei** (by `label_id` in MoBIE). The mappings are stored as tables in `data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-segmented-nuclei/`:

| Table | Content |
|-------|---------|
| `broad_types_cluster_probability.tsv` | Probabilities for the 8 broad differentiation programmes plus individual programme columns |
| `detailed_cell_types_cluster_probability.tsv` | Probabilities for the fine-grained cell types (e.g. `clade11sub48`, `nocladesub1`) |

Both tables share these conventions:
- **Columns**: one per programme/cell type (probability ∈ [0, 1]), plus three special columns:
  - `zero` — probability that the cell has no detectable signal
  - `autofluorescence` — probability that the cell expresses all genes due to autofluorescence
  - `most probable cluster` — the maximum a posteriori (MAP) assignment, i.e. the programme or cell type with the highest probability for that nucleus
- **Key**: `label_id` matching the nuclei `default.tsv`

In the PlatyBrowser viewer, these tables power views in the `2025-paper-cell-type-predictions` group — for example, `colorByColumn: "most probable cluster"` with `lut: "glasbey"` displays the MAP assignment, while `colorByColumn: "clade11sub48"` with `lut: "viridisZeroTransparent"` shows the probability of a specific subtype across all nuclei.
