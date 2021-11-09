# Shoulkd work with a base R language install
pvals <- read.table("fdr_input_pvals.txt")
fdrs <- p.adjust(pvals$V1, method="fdr")
write.table(fdrs, row.names=FALSE, col.names=FALSE, "fdr_output_fdrs.txt")
