log <- file(snakemake@log[[1]], open="wt")
sink(log)
sink(log, type="message")

library(pheatmap)
library(tidyverse)

#saving as here - https://stackoverflow.com/questions/43051525/how-to-draw-pheatmap-plot-to-screen-and-also-save-to-file
save_pheatmap_pdf <- function(x, filename, width=30, height=20) {
  pdf(filename, width=width, height=height)
  grid::grid.newpage()
  grid::grid.draw(x$gtable)
  dev.off()
}

# heatmap of morphology stats where named gene is expressed
# name of gene, morphology dataframe, gene dataframe, path to directory to save result to
make_heatmap <- function(gene_name, morph, genes, path) {

    gene_col <- genes[gene_name]
    # if no expression, skip
    if (sum(gene_col) == 0) {
      return()
    }

    # only keep rows in morphology table where gene is expressed
    cut_morph <- morph[gene_col == 1,]
    
    # if only one row, need to set clustering to false
    if (sum(gene_col == 1) == 1) {
      cluster_heat <- pheatmap(cut_morph, fontsize=8, treeheight_col = 0, treeheight_row = 0, cluster_rows=FALSE, cluster_cols=FALSE)
      save_pheatmap_pdf(cluster_heat, file.path(path, paste0(gene_name, '_heatmap.pdf')))
    
    } else {
      cluster_heat <- pheatmap(cut_morph, fontsize=8, treeheight_col = 0, treeheight_row = 0)
      save_pheatmap_pdf(cluster_heat, file.path(path, paste0(gene_name, '_heatmap.pdf')))
    }
    }
    
    
if (!dir.exists(snakemake@output$out_dir)) {
  dir.create(snakemake@output$out_dir)
}

morph_names <- read.csv(snakemake@input$morph_names, as.is=TRUE, sep='\t')
morph <- read.csv(snakemake@input$morph, as.is=TRUE, sep='\t', header=FALSE)
genes <- read.csv(snakemake@input$genes, as.is=TRUE, sep='\t')

just_genes <- genes[,-which(names(genes) %in% c('label_id_cell', 'label_id_nucleus'))]
names(morph) <- names(morph_names)

# heatmap based on morphology for each gene
path <- snakemake@output$out_dir
for (gene in names(just_genes)) {
  make_heatmap(gene, morph, just_genes, path)
}