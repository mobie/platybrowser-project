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
morph_names <- read.csv(snakemake@input$morph_names, as.is=TRUE, sep='\t')
morph <- read.csv(snakemake@input$morph, as.is=TRUE, sep='\t', header=FALSE)
names(morph) <- names(morph_names)
morph['clusters'] <- clusters[,ncol(clusters)]

# median heatmap
med <- morph %>% group_by(clusters) %>% summarise_all(funs(median))
median_heat <- pheatmap(med[,2:length(med)], labels_row = med$clusters, fontsize = 8)
save_pheatmap_pdf(median_heat, file.path(snakemake@output$heatmap_dir, 'median_bycluster.pdf'))
# save table too so can look through later
write.table(med, file.path(snakemake@output$heatmap_dir, 'median_bycluster.csv'), sep='\t', row.names=FALSE)

# mean heatmap
mea <- morph %>% group_by(clusters) %>% summarise_all(funs(mean))
mean_heat <- pheatmap(mea[,2:length(mea)], labels_row = mea$clusters, fontsize = 8)
save_pheatmap_pdf(mean_heat, file.path(snakemake@output$heatmap_dir, 'mean_bycluster.pdf'))
# save table too so can look through later
write.table(mea, file.path(snakemake@output$heatmap_dir, 'mean_bycluster.csv'), sep='\t', row.names=FALSE)

# heatmap for each cluster individually
morph <- read.csv(snakemake@input$morph, as.is=TRUE, sep='\t', header=FALSE)
names(morph) <- names(morph_names)
for (cluster in unique(clusters[,ncol(clusters)])){
  cut_morph <- morph[clusters[,ncol(clusters)] == cluster,]
  cluster_heat <- pheatmap(cut_morph, fontsize=8, treeheight_col = 0, treeheight_row = 0)
  save_pheatmap_pdf(cluster_heat, file.path(snakemake@output$heatmap_dir, paste0(cluster, '_heatmap.pdf')))
}
  