"""
Main linear model:
~Replicate + genotype + condition + genotype*condition
Batch effects and experimental grouping by genotype and condition with interaction.
"""

# set the number of available cores to 6
options(mc.cores = 6L)
# load sleuth
library("sleuth")

base_dir <- "/Users/porfirio/lab/yeastEP/RNAseqFUSTAF"
sample_id <- dir(file.path(base_dir, '/data/kal_quant'))
kal_dirs <- sapply(sample_id, function(id) file.path(base_dir, 'data/kal_quant', id))
s2c <- read.csv(file.path(base_dir, "/data", "info.csv"), header = TRUE, stringsAsFactors=FALSE)
s2c <- dplyr::mutate(s2c, path = kal_dirs)
# !!must manually make sure s2c and kal_dirs are consistent before proceeding

# make sleuth object
so <- sleuth_prep(s2c, ~Replicate + genotype + condition + genotype*condition,
                  extra_bootstrap_summary = TRUE)

# fit the full model
so <- sleuth_fit(so)
# fit a reduced model, no replicate effect
so <- sleuth_fit(so, ~genotype + condition, 'norep')
# compute likelihood ratio
so <- sleuth_lrt(so, 'norep', 'full')

# fit a reduced model, no interactions
so <- sleuth_fit(so, ~Replicate + genotype + condition, 'reduced')
# compute likelihood ratio
so <- sleuth_lrt(so, 'reduced', 'full')

# compute wald test on each beta and save
betas <- colnames(design_matrix(so))[-1]
for (b in betas){
    # compute wald test
    so <- sleuth_wt(so, which_beta = b)
    # save
    write.csv(sleuth_results(so, b),
            file=file.path(base_dir, sprintf('sleuth/main/sleuth_%s.csv', b)),
            quote=FALSE, row.names=FALSE)}

# save sleuth object
saveRDS(so, file=file.path(base_dir, 'sleuth/main/so.rds'))

## launch EDA shiny app
#sleuth_live(so)
#
## to load sleuth object
#library("sleuth")
#base_dir <- "BASE_DIR"
#so <- readRDS(file.path(base_dir, 'so.rds'))
