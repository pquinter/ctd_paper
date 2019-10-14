"""
CTD linear model:
~Replicate + condition + CTD + CTD*condition,

Groups are:
genotype    CTD
    TL47   full
    yQC7  short
   yQC15  short
   yQC16  short
"""

# set the number of available cores to 6
options(mc.cores = 6L)
# load sleuth
library("sleuth")
base_dir <- "/Users/porfirio/lab/yeastEP/RNAseqFUSTAF"
sample_id <- dir(file.path(base_dir, '/data/kal_quant'))
kal_dirs <- sapply(sample_id, function(id) file.path(base_dir, 'data/kal_quant', id))
s2c <- read.csv(file.path(base_dir, "data", "info.csv"), header = TRUE, stringsAsFactors=FALSE)
s2c <- dplyr::mutate(s2c, path = kal_dirs)
# !!must manually make sure s2c and kal_dirs are consistent before proceeding
# remove all WT
#s2c = s2c[!grepl("TL47", s2c$genotype),]

# make sleuth object
# group yQC15, yQC16 and yQC7
so <- sleuth_prep(s2c, ~Replicate + condition + CTD + CTD*condition,
                  extra_bootstrap_summary = TRUE)

# fit the full model
so <- sleuth_fit(so)

# compute wald test on each beta and save
betas <- colnames(design_matrix(so))[-1]
for (b in betas){
    # compute wald test
    so <- sleuth_wt(so, which_beta = b)
    # save
    write.csv(sleuth_results(so, b),
            file=file.path(base_dir, sprintf('sleuth/CTDgroup/sleuth_%s.csv', b)),
            quote=FALSE, row.names=FALSE)}

# save sleuth object
saveRDS(so, file=file.path(base_dir, 'sleuth/CTDgroup/so.rds'))
