# Should work out of the box with basic R language installed
co <- read.table("collinear_distances.txt")
non <- read.table("non_collinear_distances.txt")
w <- wilcox.test(co$V1, non$V1)
write.table(data.frame(pval=w$p.value, W=w$statistic), row.names=FALSE, quote=FALSE, sep="\t", "collinear_wilcox_pvals.txt")
