from matplotlib import pyplot as plt
import pandas as pd
from utils import plot
import matplotlib.patches as mpatches
from utils import ecdftools, plot
from joblib import Parallel, delayed
import multiprocessing

n_jobs = multiprocessing.cpu_count()
colors = {'genotype':{'TL47':'#1daea7', 'yQC7':'#7a324c', 'yQC15':'#da6363','yQC16':'#e59191'},
            'condition':{'ctrl':'#1a2224', 'gal':'#e45f9f'},
            'rep':{'rep1':'#009473', 'rep2':'#fc606b', 'rep3':'#3498DB'}}
patches = {l:[mpatches.Patch(color=colors[l][j], label=j)
            for j in colors[l]] for l in ('genotype', 'condition','rep')}

###############################################################################
# PCA plot
###############################################################################
pca_df = pd.read_csv('./data/RNAseq_pca.csv')
# dataframe with explained variance
pca_meta = pd.read_csv('/Users/porfirio/lab/yeastEP/figures_paper/data/RNAseq_pca_varexpl.csv')

fig, ax = plt.subplots(figsize=(10,8))
# plot by condition (+- galactose)
pca_df.groupby('condition').apply(lambda x:\
    ax.scatter(x['pc1_tpm'], x['pc2_tpm'], color=colors['condition'][x.name],
    s=150, rasterized=True, alpha=1))
# plot by genotype on top
pca_df.groupby('genotype').apply(lambda x:\
    ax.scatter(x['pc1_tpm'], x['pc2_tpm'], color=colors['genotype'][x.name],
    s=100, rasterized=True, edgecolors='w'))
ax.set(xlabel='PC1 ({0:.0f}% variance)'.format(pca_meta.explained_variance_tpm[0]*100),
    ylabel='PC2 ({0:.0f}% variance)'.format(pca_meta.explained_variance_tpm[1]*100))
plt.legend(handles=patches['genotype'])
plt.tight_layout()
plt.savefig('./figures/output/FigS5_PCA.svg')

###############################################################################
# Alternative model: grouping FUS and TAF
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
# Lollipop plot of number of transcripts at q-value<0.1
###############################################################################
coefs = ['yQC15main','yQC16main','LDLDLDgroup']
coef_names = ['FUS fusion (Full model)', 'TAF fusion (Full model)', 'LCD fusion (LCD model)']
colors = ['#da6363', '#da6363','#326976']

fig, ax = plt.subplots(figsize=(22,4))
ax = plot.coef_stemplot(qvalecdf, coefs, qval_thresh=0.1, coef_names=coef_names,
        ax=ax, orient='h',color=colors)
plt.savefig('./figures/output/FigS5_RNAseqLolipop_LCD.svg')

###############################################################################
# Scatter plot of interaction vs galactose coefficients
###############################################################################
fig, ax = plt.subplots(figsize=(10,8))
ax = plot.scatter_coef(sleuth, 'yQC15main', 'yQC16main', ax=ax)
ax.set(ylabel='TAF fusion',xlabel='FUS fusion')
plt.savefig('./figures/output/FigS5_RNAseqScatter_FUSvTAF.svg')

fig, ax = plt.subplots(figsize=(10,8))
ax = plot.scatter_coef(sleuth, 'LDshortLDgroup', 'LDLDLDgroup', ax=ax)
ax.set(ylabel='Truncation + LCD fusion',xlabel='Truncation')
plt.savefig('./figures/output/FigS5_RNAseqScatter_LDshortLD.svg')
