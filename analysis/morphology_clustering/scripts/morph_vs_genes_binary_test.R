log <- file(snakemake@log[[1]], open="wt")
sink(log)
sink(log, type="message")

library(pheatmap)
library(tidyverse)
library(parallel)

inner_loop <- function (g, m) {
  print(g[1])

  # if gene all 0s or all 1s, just set to NAs, as wilcox test cannot be preformed
  if (length(unique(g)) != 2) {
    return(rep(NA, ncol(m)))
  } else {
    sapply(m, function(y) {
      data = data.frame('gene' = g, 'morph' = y)
      return(wilcox.test(morph~gene, data=data)$p.value)
    })

  }

}

no_cores <- detectCores()
cl <- makeCluster(no_cores, outfile=snakemake@log[[1]])

morph <- read.csv(snakemake@input$morph, as.is=TRUE, sep='\t')
gene <- read.csv(snakemake@input$gene, as.is=TRUE, sep='\t')
morph <- morph[,-which(names(morph) %in% c('label_id_cell', 'label_id_nucleus'))]
gene <- gene[,-which(names(gene) %in% c('label_id_cell', 'label_id_nucleus'))]

# perform a wilcox test
result <- parLapply(cl, gene, inner_loop, morph)

result_df <- as.data.frame(result)
write.table(result_df, snakemake@output$wilcox, sep='\t', row.names=FALSE)

# p.adjust for all those not nan - adjust for multiple testing with bonferroni
result_adj <- lapply(result_df, function (x) {
  if (sum(is.na(x)) > 0) {
    return(rep(NA, nrow(result_df)))
  } else {
    return(p.adjust(x, method='bonferroni'))
  }
})

result_adj_df <- as.data.frame(result_adj)
write.table(result_adj_df, snakemake@output$wilcox_bon, sep='\t', row.names=FALSE)

stopCluster(cl)