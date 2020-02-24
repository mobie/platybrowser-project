log <- file(snakemake@log[[1]], open="wt")
sink(log)
sink(log, type="message")

library(pheatmap)
library(tidyverse)

# heatmap pdf - https://stackoverflow.com/questions/43051525/how-to-draw-pheatmap-plot-to-screen-and-also-save-to-file
save_pheatmap_pdf <- function(x, filename, width=30, height=20) {
  pdf(filename, width=width, height=height)
  grid::grid.newpage()
  grid::grid.draw(x$gtable)
  dev.off()
}

if (!dir.exists(snakemake@output$heatmap_dir)) {
  dir.create(snakemake@output$heatmap_dir)
}

morph <- read.csv(snakemake@input$morph, as.is=TRUE, sep='\t')
cut_morph <- morph[,-which(names(morph) %in% c('label_id'))]

# median heatmap
med_morph <- cut_morph %>% group_by(clusters) %>% summarise_all(funs(median))
median_heat <- pheatmap(med_morph[,2:length(med_morph)], labels_row = med_morph$clusters, fontsize = 8)

save_pheatmap_pdf(median_heat, file.path(snakemake@output$heatmap_dir, 'median_bycluster.pdf'))
write.table(med_morph, file.path(snakemake@output$heatmap_dir, 'median_bycluster.csv'), sep='\t', row.names=FALSE)

# mean heatmap
mea_morph <- cut_morph %>% group_by(clusters) %>% summarise_all(funs(mean))
mean_heat <- pheatmap(mea_morph[,2:length(mea_morph)], labels_row = mea_morph$clusters, fontsize = 8)

save_pheatmap_pdf(mean_heat, file.path(snakemake@output$heatmap_dir, 'mean_bycluster.pdf'))
# save table too so can look through later
write.table(mea_morph, file.path(snakemake@output$heatmap_dir, 'mean_bycluster.csv'), sep='\t', row.names=FALSE)


# heatmap for each cluster individually
just_morph <- morph[,-which(names(morph) %in% c('label_id', 'clusters'))]
for (cluster in unique(morph$clusters)){
  cut_morph <- just_morph[morph$clusters == cluster,]
  cluster_heat <- pheatmap(cut_morph, fontsize=8, treeheight_col = 0, treeheight_row = 0)
  save_pheatmap_pdf(cluster_heat, file.path(snakemake@output$heatmap_dir, paste0(cluster, '_heatmap.pdf')))
}
