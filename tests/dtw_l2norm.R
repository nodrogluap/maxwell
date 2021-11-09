# Run this script with "Rscript dtw_l2_norm.R"
# To set up the environment, download conda (https://github.com/ucvm/synergy/wiki/Using-Conda), and then do conda install r-dtwclust
set.seed(42)
library(dtwclust)
# Random 10 values
subject <- round(runif(10, min=400, max=1000))
# Random adjustment to those values
query <- subject + round(runif(10, min=-30, max=30))
# L2 norm
res2 <- dtw2(query, subject, step.pattern=symmetric1)
write.table(query, col.names=FALSE, row.names=FALSE, sep="\n", quote=FALSE, "dtw_l2_norm_query.txt")
write.table(subject, col.names=FALSE, row.names=FALSE, sep="\n", quote=FALSE, "dtw_l2_norm_subject.txt")
write.table(res2$distance**2, col.names=FALSE, row.names=FALSE, "dtw_l2_norm_squared_distance.txt")
