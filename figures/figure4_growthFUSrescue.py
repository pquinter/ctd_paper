from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from utils import ecdftools, plot
import seaborn as sns
import multiprocessing
###############################################################################
# Doubling times for FUS and TAF15 rescued strains
###############################################################################

n_jobs = multiprocessing.cpu_count()
growth = pd.read_csv('./data/growthrates_merged.csv')
strains = ['TL47', 'yQC62','yQC16','yQC15', 'yQC63', 'yQC27', 'yQC64', 'yQC28', 'yQC29','yQC56']
rescued = ['yQC15','yQC16','yQC27','yQC28','yQC29','yQC56']
# colors for truncated and fused strains
colors = ['#326976', '#da6363']
colors_resc = [colors[int(strain in rescued)] for strain in strains]

fig, ax = plt.subplots(figsize=(14,8))
ax = plot.stripplot_errbars('inverse max df', 'strain', 'stdev', strains,
                    growth, ax=ax, colors=colors_resc)
ax.set(xlabel='Doubling Time (min)', ylabel='CTD repeats')
# assign intelligible strain names
strain_labels =\
['26','10','10-TAF','10-FUS','9','9-FUS','8','8-FUS','7-FUS','6-FUS']
plt.yticks(plt.yticks()[0], strain_labels)
plt.xlim(90, 300)
plt.tight_layout()
plt.savefig('./figures/output/Fig4_growthrates_FUS.svg')

###############################################################################
# Heatmap of q-value distributions
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

# turn off interactive mode because plot doesn't fit
plt.ioff()
fig, axes = plt.subplots(2, figsize=(20, 6))
coefs = (['yQC7main','yQC15main','yQC16main'], ['yQC7:galmain', 'yQC15:galmain','yQC16:galmain'])
coef_names = (['Truncation','FUS fusion','TAF fusion'])
for i, ax in enumerate(axes):
    ax = plot.qvalecdf_hmap(qvalecdf, coefs[i], coef_names=coef_names, ax=ax)
axes[0].set(xlabel='', xticks=[], ylabel='')
axes[1].set(ylabel='Galactose\nInteraction')
# rotate axis back
plt.xticks(rotation=90)
plt.subplots_adjust(hspace=0.02)
plt.savefig('./figures/output/Fig4_RNAseqHmap.svg')

###############################################################################
# Lollipop plot of number of transcripts at q-value<0.1
###############################################################################

plt.ion()
# color by LCD fusion as in growth rate plot
colors = ['#326976', '#da6363', '#da6363']*2
ax = plot.coef_stemplot(qvalecdf, coefs[0]+coefs[1], qval_thresh=0.1, coef_names=coef_names*2, color=colors)
ax.set(yticks=np.arange(0, 2500, 500))
plt.savefig('./figures/output/Fig4_RNAseqLolipop.svg')

###############################################################################
# Volcano plots of interactions
###############################################################################

# coefficients and labels
coefs = ['yQC7:galmain','yQC15:galmain','yQC16:galmain']
coef_labels = ['Truncation','FUS fusion','TAF fusion']

plt.ioff()
fig, axes = plt.subplots(1,3, figsize=(22,9), sharey=False, sharex=True)
[plot.volcano_plot(coef, sleuth, ax, s=5, alpha=0.1) for coef, ax in zip(coefs, axes.ravel())]
[ax.set(ylim=(-0.1, 10.1), xlabel='log Fold-change\n({} Interaction)'.format(clabel))\
                            for ax,clabel in zip(axes.ravel(), coef_labels)]
plt.savefig('./figures/output/Fig4_VolcanoRnaseq_Int.svg')

###############################################################################
# Scatter plot of galactose only coefficients
###############################################################################

coefs = ['yQC7gal_only','yQC15gal_only','yQC16gal_only']
for coef, clabel in zip(coefs, coef_labels):
    fig, ax = plt.subplots(figsize=(9,9))
    ax = plot.scatter_coef(sleuth, 'TL47gal_only', coef, ax=ax)
    # plot gal4 targets on top in orange
    plot.scatter_coef(gal4targets, 'TL47gal_only', coef, color='#f85f68', ax=ax,
                                                alpha=0.5, auto_ref=False)
    ax.set(ylabel='log Fold-change\n(Galactose, {})'.format(clabel), xlabel='log Fold-change\n(Galactose, Wild-type)')
    plt.savefig('./figures/output/Fig4_RNAseqScatter_GalOnly_{}.svg'.format(clabel.split(' ')[0]))

###############################################################################
# Fraction of active cells from smFISH
###############################################################################

freq_df = pd.read_csv('./data/smFISH_GAL10_GAL3_FracActive.csv')
order = ['TL47','yQC62','yQC16','yQC15']
colors = {s:c for s,c in zip(order,['#326976', '#98b4ba','#823b3b','#da6363'])}

for gene, _freq_df in freq_df.groupby('gene'):
    fig, ax = plt.subplots(figsize=(9,8))
    sns.stripplot(y='strain', x='frac_active', data=_freq_df,
        ax=ax, alpha=0.3, order=order, palette=colors, size=10)
    sns.pointplot(y='strain', x='frac_active', data=_freq_df,
        order=order, ax=ax, alpha=1, palette=colors, size=20, join=False, ci=99)
    # below two lines to draw points over strip plot
    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100, label="")
    plt.legend([], frameon=False)
    ax.set(xlabel='Active cells fraction', ylabel='CTD repeats', xticks=np.arange(0,1.1, 0.2))
    strain_labels = ['26','10','10-TAF','10-FUS']
    plt.yticks(plt.yticks()[0], strain_labels)
    plt.tight_layout()
    plt.title(gene.upper())
    plt.savefig('./figures/output/Fig4_FracActive_{}.svg'.format(gene))

###############################################################################
# TS intensity ECDF from smFISH
###############################################################################
parts = pd.read_csv('./data/smFISH_GAL10_GAL3_TS.csv')
parts = parts[parts.strain.isin(['TL47','yQC62','yQC15','yQC16'])]
# scale intensity: it's arbitrary units, no need to have extremely long numbers
parts['mass'] = parts.groupby(['date','gene']).mass.transform(lambda x: (x-np.min(x))/(np.max(x)-np.min(x)))
parts = parts[parts.date==9132019]
for gene, _parts in parts.groupby('gene'):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax = plot.ecdfbystrain('mass', _parts, groupby='strain', colors=colors, ax=ax, formal=True, line_kws={'alpha':1, 'rasterized':True})
    ax.set(xlabel='TS Fluorescence (a.u.)', ylabel='ECDF', title=gene.upper(),
            xlim=((-0.05,0.8) if gene=='gal3' else (-0.05, 0.6)))
#    [ax.axvline(np.median(vals.mass), color=colors[strain], ls='--', alpha=0.8, ymin=0.04)\
#            for strain, vals in _parts.groupby('strain')]
    plt.tight_layout()
    plt.savefig('./figures/output/Fig4_ECDF_{}.svg'.format(gene))
