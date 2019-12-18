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
colorby = 'rep'
pca_df.groupby(colorby).apply(lambda x:\
    ax.scatter(x['pc1_tpm'], x['pc2_tpm'], color=colors[colorby][x.name],
    s=200, rasterized=True, alpha=1))
# plot by genotype on top
pca_df.groupby('genotype').apply(lambda x:\
    ax.scatter(x['pc1_tpm'], x['pc2_tpm'], color=colors['genotype'][x.name],
    s=100, rasterized=True, edgecolors='w'))
ax.set(xlabel='PC1 ({0:.0f}% variance)'.format(pca_meta.explained_variance_tpm[0]*100),
    ylabel='PC2 ({0:.0f}% variance)'.format(pca_meta.explained_variance_tpm[1]*100))
plt.legend(handles=patches['genotype']+patches['rep'])
plt.tight_layout()
plt.savefig('./figures/output/FigS5_PCA_rep.svg')

###############################################################################
# Scatter plot of interaction vs galactose coefficients
###############################################################################

plt.ioff()

# coefficients and labels
coefs = ['yQC7:galmain','yQC15:galmain','yQC16:galmain']
coef_labels = ['Truncation','FUS fusion','TAF fusion']
# GAL4 gene targets
gal4targets = pd.read_csv('./data/gal4_targetgenes_ORegAnno20160119.csv')
gal4targets = sleuth[sleuth.target_id.isin(gal4targets.Gene_ID.values)]
for coef, clabel in zip(coefs, coef_labels):
    fig, ax = plt.subplots(figsize=(10,8))
    ax = plot.scatter_coef(sleuth, 'galmain', coef, ax=ax)
    # plot gal4 targets on top in orange
    plot.scatter_coef(gal4targets, 'galmain', coef, color='#f85f68', ax=ax,
                                    alpha=0.8, auto_ref=False)
    ax.set(ylabel='log Fold-change\n({} Interaction)'.format(clabel), xlabel='log Fold-change\n(Galactose)')# xlim=(-6.5, 3), ylim=(-3, 7))
    plt.savefig('./figures/output/Fig4_RNAseqScatter_Int_{}.svg'.format(clabel.split(' ')[0]))

###############################################################################
# Volcano plots 
###############################################################################

# get sleuth data
sleuth = pd.read_csv('./data/sleuth_all.csv').dropna(axis=0)
# get unique coefficient id, as they occur in multiple models
sleuth['cid'] = sleuth['coef'] + sleuth['model']

## Galactose only coefficients

plt.ioff()
coefs = ['TL47gal_only','yQC7gal_only','yQC15gal_only','yQC16gal_only']
coef_labels_gal = ['Wild-type','Truncation','FUS fusion','TAF fusion']
fig, axes = plt.subplots(1,4, figsize=(32,8), sharey=False, sharex=True)
[plot.volcano_plot(coef, sleuth, ax, s=5, alpha=0.1) for coef, ax in zip(coefs, axes.ravel())]
[ax.set(ylim=(-0.1, 10.1), xlim=(-6,6), xlabel='log Fold-change\n(Galactose, {})'.format(clabel))\
                            for ax,clabel in zip(axes.ravel(), coef_labels_gal)]
plt.savefig('./figures/output/FigS5_VolcanoRnaseq_GalOnly.svg')

## Galactose from LCD fusion
coefs = ['yQC15main','yQC16main','LDLDLDgroup']
coef_labels_lcd = ['FUS fusion','TAF fusion', 'LCD fusion']
fig, axes = plt.subplots(1,3, figsize=(24,8), sharey=False, sharex=True)
[plot.volcano_plot(coef, sleuth, ax, s=5, alpha=0.1) for coef, ax in zip(coefs, axes.ravel())]
[ax.set(ylim=(-0.1, 10.1), xlim=(-6,6), xlabel='log Fold-change\n({})'.format(clabel))\
                            for ax,clabel in zip(axes.ravel(), coef_labels_lcd)]
plt.savefig('./figures/output/FigS5_VolcanoRnaseq_LCDgroup.svg')

###############################################################################
# Alternative model: grouping FUS and TAF
###############################################################################

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
ax.set(ylabel='log Fold-change\n(TAF fusion)',xlabel='log Fold-change\n(FUS fusion)')
plt.savefig('./figures/output/FigS5_RNAseqScatter_FUSvTAF.svg')

fig, ax = plt.subplots(figsize=(10,8))
ax = plot.scatter_coef(sleuth, 'LDshortLDgroup', 'LDLDLDgroup', ax=ax)
ax.set(ylabel='log Fold-change\n(Truncation + LCD fusion)',xlabel='log Fold-change\n(Truncation)')
plt.savefig('./figures/output/FigS5_RNAseqScatter_LDshortLD.svg')
