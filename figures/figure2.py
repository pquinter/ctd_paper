from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from utils import ecdftools, plot
from joblib import Parallel, delayed
import multiprocessing

n_jobs = multiprocessing.cpu_count()

###############################################################################
# Growth curves for strains with 26, 14, 12 and 10 CTDr
###############################################################################

od = pd.read_csv('./data/growthOD_merged_tidy.csv')
strains = ['TL47', 'yQC5','yQC6','yQC62', 'yQC63', 'yQC64']
# make short palette
ctdr = [26,14,12,10,9,8]
colors_ctd, patches_ctd = plot.get_palette(ctdr)
fig, ax = plt.subplots(figsize=(14, 8))
ax = plot.growth_curve(od, strains=strains, high=1.5, ax=ax, colors=colors_ctd)
ax.set(xlim=(-200, 1000), ylim=(0.15, 1.6), xticks=(np.arange(0, 1100, 200)))
plt.tight_layout()
plt.savefig('./figures/output/Fig2_growth_.svg')

###############################################################################
# Doubling times for strains with 26, 14, 12 and 10 CTDr
###############################################################################

growth = pd.read_csv('./data/growthrates_merged.csv')
fig, ax = plt.subplots(figsize=(10,8))
ax = plot.stripplot_errbars('inverse max df', 'strain', 'stdev', strains,
                    growth, ax=ax, colors=[colors_ctd[c] for c in colors_ctd])
ax.set(xlabel='Doubling Time (min)', ylabel='CTD repeats')
# assign intelligible strain names
strain_labels = ['26','14','12','10','9','8']
plt.yticks(plt.yticks()[0], strain_labels)
plt.tight_layout()
plt.savefig('./figures/output/Fig2_growthrates.png')

###############################################################################
# RPB1/RPB3 nuclear fluorescence
###############################################################################
rpb1fluor = pd.read_csv('./data/mScarRPB1_052019_mean.csv')
fig, ax = plt.subplots(figsize=(14, 8))
ax = plot.ecdfbystrain('mean_intensity_nuc', rpb1fluor, ax=ax, formal=True,
                                    line_kws = {'alpha':0.8,'rasterized':True})
ax.set(xlabel='RPB1 Nuclear Fluorescence (a.u.)', ylabel='ECDF', xlim=(230, 900))
plt.tight_layout()
plt.savefig('./figures/output/Fig2_rpb1fluor.svg')

rpb3fluor = pd.read_csv('./data/mScarRPB3_022020_mean.csv')
rpb3fluor = rpb3fluor[~(rpb3fluor.strain.isin([15,16]))]
fig, ax = plt.subplots(figsize=(10, 8))
ax = plot.ecdfbystrain('mean_intensity_nuc', rpb3fluor, ax=ax, formal=True,
                                    line_kws = {'alpha':0.8,'rasterized':True})
ax.set(xlabel='RPB3 Nuclear Fluorescence (a.u.)', ylabel='ECDF', xlim=(120, 600))
plt.tight_layout()
plt.savefig('./figures/output/Fig2_rpb3fluor.svg')

###############################################################################
# RNA-seq
###############################################################################

# get sleuth data
sleuth = pd.read_csv('./data/sleuth_all.csv').dropna(axis=0)
# get unique coefficient id, as they occur in multiple models
sleuth['cid'] = sleuth['coef'] + sleuth['model']
# get q-value cumulative distribution by model and coefficient
qvalecdf_series = Parallel(n_jobs=n_jobs)(delayed(ecdftools.compute_qvalecdf)(model, coef, _df)
                       for (model, coef), _df in sleuth.groupby(['model','coef']))
qvalecdf = pd.concat(qvalecdf_series)
# get back cid
qvalecdf['cid'] = qvalecdf['coef'] + qvalecdf['model']

###############################################################################
# Heatmap of q-value distribution for galactose, truncation and interaction coef
###############################################################################

coefs = ['galmain','yQC7main','yQC7:galmain']
coef_names = ['Galactose','Truncation','Interaction']
ax = plot.qvalecdf_hmap(qvalecdf, coefs, coef_names=coef_names)
plt.savefig('./figures/output/Fig2_RNAseqHmap.svg')

###############################################################################
# Lollipop plot of number of transcripts at q-value<0.1
###############################################################################

ax = plot.coef_stemplot(qvalecdf, coefs, qval_thresh=0.1, coef_names=coef_names)
plt.savefig('./figures/output/Fig2_RNAseqLolipop.svg')

###############################################################################
# Scatter plot of interaction vs galactose coefficients
###############################################################################
plt.ioff()
ax = plot.scatter_coef(sleuth, 'galmain', 'yQC7:galmain', alpha=0.2)
# Get GAL4 gene targets
gal4targets = pd.read_csv('./data/gal4_targetgenes_ORegAnno20160119.csv')
gal4targets = sleuth[sleuth.target_id.isin(gal4targets.Gene_ID.values)]
# plot on top with different color
plot.scatter_coef(gal4targets, 'galmain', 'yQC7:galmain', auto_ref=False,
                                        color='#f85f68', alpha=0.8, ax=ax)
ax.set(ylabel='log Fold-change\n(Interaction)', xlabel='log Fold-change\n(Galactose)')
plt.savefig('./figures/output/Fig2_RNAseqScatter.svg')
