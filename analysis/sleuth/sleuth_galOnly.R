"""
Galactose-only linear model:
~Replicate + condition
Simplest model, compute galactose response for each strain separately.
"""

# set the number of available cores to 6
options(mc.cores = 6L)
# load sleuth
library("sleuth")

base_dir <- "/Users/porfirio/lab/yeastEP/RNAseqFUSTAF"
sample_id <- dir(file.path(base_dir, '/data/kal_quant'))
kal_dirs <- sapply(sample_id, function(id) file.path(base_dir, 'data/kal_quant', id))
s2c_all <- read.csv(file.path(base_dir, "/data", "info.csv"), header = TRUE, stringsAsFactors=FALSE)
s2c_all <- dplyr::mutate(s2c_all, path = kal_dirs)
# use a single genotype
for (geno in unique(s2c_all['genotype'])[[1]]){
    s2c <- s2c_all[s2c_all['genotype']==geno,]
    # make sleuth object
    so <- sleuth_prep(s2c, ~Replicate + condition,
                      extra_bootstrap_summary = TRUE)

    # fit the full model
    so <- sleuth_fit(so)
    # fit a reduced model, no replicate effect
    so <- sleuth_fit(so, ~condition, 'norep')
    # compute likelihood ratio
    so <- sleuth_lrt(so, 'norep', 'full')
    # compute wald test
    so <- sleuth_wt(so, which_beta = 'conditiongal')
    # save
    write.csv(sleuth_results(so, 'conditiongal'),
            file=file.path(base_dir, sprintf('sleuth/gal_only/sleuth_%s.csv', geno)),
            quote=FALSE, row.names=FALSE)}
