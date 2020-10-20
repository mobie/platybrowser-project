library(tidyverse)

points <- read.table(
  'Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\plot_values_1_0_1.csv', sep='\t', header=TRUE)

# cut to first 100 points
points <- points[points$x <= 100,]

pointsize = 3

ggplot(data=points) + geom_point(aes(x=x, y=y_all, col = "all"), size=pointsize) + geom_line(aes(x=x, y=y_all, col = "all")) + 
  geom_point(aes(x=x, y=y_cell, col = "cell"), size=pointsize) + geom_line(aes(x=x, y=y_cell, col = "cell")) +
  geom_point(aes(x=x, y=y_nuc, col= "nucleus"), size=pointsize) + geom_line(aes(x=x, y=y_nuc, col= "nucleus")) +
  geom_point(aes(x=x, y=y_chrom, col = "chromatin"), size=pointsize) + geom_line(aes(x=x, y=y_chrom, col = "chromatin")) +
  
  geom_point(aes(x=x, y=y_random), size=pointsize) + geom_line(aes(x=x, y=y_random, col='randomised')) +
  scale_color_manual(values = c(all="#E69F00", cell="#56B4E9", nucleus="#009E73", chromatin="#CC79A7", randomised='black')) + 
  xlab('First neighbour that meets\nbilateral criteria') + ylab('Fraction of all cells') + ylim(0, 1) +
  theme_bw() + scale_x_continuous(limits=c(1, 100)) + theme(text=element_text(size=35)) + 
  theme(legend.position=c(0.75,0.83), legend.title = element_blank())

ggsave('Z:\\Kimberly\\Projects\\SBEM_analysis\\Data\\Derived\\snakemake_morphology\\bilateral_pairs\\pair_comparison_1_0_1_big_font.png', width = 8, height = 10, dpi=300)
