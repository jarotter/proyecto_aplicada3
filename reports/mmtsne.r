library(R.matlab)
library(mmtsne)

wa <- readMat('./data/wa1000.mat')
symwa <- p2sp(wa$P)

maps <- mmtsneP(symwa, no_maps=10)
