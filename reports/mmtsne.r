library(R.matlab)
library(mmtsne)
library(tidyverse)
library(ggrepel)

set.seed(3141)

wa <- readMat('./data/wa1000.mat')
symwa <- p2sp(wa$P)
labels <- unlist(wa$words)
maps <- mmtsneP(symwa, no_maps=10, max_iter=3000)

idx <- sample.int(n=1000, size=100, replace=FALSE)

plots = list()

for(map in c(1:10)){
  coords <- maps$Y[,,map]
  x <-  coords[idx,1]
  y <- coords[idx,2]
  pts <- data_frame(x,y)
  
  plots[[map]] = ggplot(pts, aes(x,y, label=labels[idx])) +
    geom_text_repel() +
    geom_point(color='red')
}

pdf("plots.pdf")
for (i in 1:10) {
  print(plots[[i]])
}
dev.off()
