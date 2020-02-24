log <- file(snakemake@log[[1]], open="wt")
sink(log)
sink(log, type="message")

library(pheatmap)
library(tidyverse)

# saving as here - https://stackoverflow.com/questions/43051525/how-to-draw-pheatmap-plot-to-screen-and-also-save-to-file
save_pheatmap_pdf <- function(x, filename, width=30, height=20) {
  pdf(filename, width=width, height=height)
  grid::grid.newpage()
  grid::grid.draw(x$gtable)
  dev.off()
}

if (!dir.exists(snakemake@output$heatmap_dir)) {
  dir.create(snakemake@output$heatmap_dir)
}

clusters <- read.csv(snakemake@input$clusters, as.is=TRUE, sep='\t')
genes <- read.csv(snakemake@input$genes, as.is=TRUE, sep='\t')

just_genes <- genes[,-which(names(genes) %in% c('label_id_cell', 'label_id_nucleus'))]
just_genes['clusters'] <- clusters[,ncol(clusters)]

# median heatmap
med_genes <- just_genes %>% group_by(clusters) %>% summarise_all(funs(median))
median_heat <- pheatmap(med_genes[,2:length(med_genes)], labels_row = med_genes$clusters, fontsize = 8)
save_pheatmap_pdf(median_heat, file.path(snakemake@output$heatmap_dir, 'median_bycluster.pdf'))
# save table too so can look through later
write.table(med_genes, file.path(snakemake@output$heatmap_dir, 'median_bycluster.csv'), sep='\t', row.names=FALSE)

# mean heatmap
mea_genes <- just_genes %>% group_by(clusters) %>% summarise_all(funs(mean))
mean_heat <- pheatmap(mea_genes[,2:length(mea_genes)], labels_row = mea_genes$clusters, fontsize = 8)
save_pheatmap_pdf(mean_heat, file.path(snakemake@output$heatmap_dir, 'mean_bycluster.pdf'))
# save table too so can look through later
write.table(mea_genes, file.path(snakemake@output$heatmap_dir, 'mean_bycluster.csv'), sep='\t', row.names=FALSE)

# heatmap for each cluster individually
just_genes <- genes[,-which(names(genes) %in% c('label_id_cell', 'label_id_nucleus'))]
for (cluster in unique(clusters[,ncol(clusters)])){
  cut_genes <- just_genes[clusters[,ncol(clusters)] == cluster,]
  # remove genes with zero expression to reduce clutter in heatmap
  cut_genes <- cut_genes[,colSums(cut_genes) != 0]
  cluster_heat <- pheatmap(cut_genes, fontsize=8, treeheight_col = 0, treeheight_row = 0)
  save_pheatmap_pdf(cluster_heat, file.path(snakemake@output$heatmap_dir, paste0(cluster, '_heatmap.pdf')))
}
  