from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from joblib import Parallel, delayed
from utils import plot

from ast import literal_eval

###############################################################################

aacount = pd.read_csv('./data/fustafctd_aacount.csv')
# get order to sort by frequency in CTD
order = aacount[aacount.protein=='CTD'].sort_values('freq', ascending=False).aminoacid.values
colors = {'CTD':'#326976', 'FUS':'#da6363','TAF15':'#7a324c'}
fig, ax = plt.subplots(figsize=(15,8))
sns.stripplot(x='aminoacid', y='freq', hue='protein', data=aacount, ax=ax,
        palette=colors, s=15, alpha=0.5, order=order)
ax.set(xlabel='Amino acid', ylabel='Frequency')
plt.legend(handles=plot.get_patches(colors), ncol=3)
plt.tight_layout()
plt.savefig('./figures/output/FigS4_aaComp.svg')

###############################################################################
# Disorder probabilities in FUS and TAF15LCDs
###############################################################################

fustaf_disorder = pd.read_csv('./data/fus_taf_mobidb.csv')
# convert to probs back to array (for some reason they are stored as strings)
fustaf_disorder['p'] = fustaf_disorder.p.apply(literal_eval)
# get yeast CTD data as well
rpb1data = pd.read_csv('./data/rpb1_seq_props.csv')
# convert to probs back to array (for some reason they are stored as strings)
rpb1data['p'] = rpb1data.p.apply(literal_eval)
yCTD = rpb1data.loc[rpb1data.species_long.str.contains('cerevisiae')]

# Get disorder probabilities
dis_probs = fustaf_disorder.p.apply(pd.Series).values
# add yeast CTD, starts in amino acid
yCTD = np.array(yCTD.p.values[0][1534:] + [np.nan]*15)
dis_probs = pd.DataFrame(np.vstack((dis_probs, yCTD)))

fig, ax = plt.subplots(1, figsize=(15, 8))
ax.plot(dis_probs.iloc[0].values, '-', c=colors['FUS'], lw=3, alpha=0.6, label='FUS LCD')
ax.plot(dis_probs.iloc[1].values, '-', c=colors['TAF15'], lw=3, alpha=0.6, label='TAF15 LCD')
ax.plot(dis_probs.iloc[2].values, '-', c=colors['CTD'], lw=3, alpha=0.6, label='CTD')
# plot 0.5 threshold
ax.axhline(0.5, ls='--', c='k')
ax.set(ylim=(0,1.05), ylabel='Disorder Probability', xlabel='Residue Position', yticks=np.arange(0,1.1, 0.25))
plt.tight_layout()
plt.savefig('./figures/output/FigS4_FUSTAFdisorder.svg')

###############################################################################
# Binding rates (slopes) re-plotted from Kwon et al (2013) Cell
###############################################################################

rates = pd.read_csv('./data/kwon2013_BindingRates_lin_regress_leastsq.csv')
order = ['WT','1C','2E','3E','4E','5D','7','9']
rates['_'] = 0

fig, ax = plt.subplots(figsize=(15,8))
sns.stripplot(x='strain', y='beta', order=order, data=rates, ax=ax,
        color='#326976', s=15, alpha=0.3, rasterized=True)
ax.set(xlabel='FUS variant', ylabel=r'Droplet Binding Rate')
# assign intelligible strain names
strain_labels = ['WT','1C','2E','3E','4E','5D','7A','S2']
plt.xticks(plt.xticks()[0], strain_labels)
plt.tight_layout()
plt.savefig('./figures/output/FigS4_Kwon2013_DropletBinding.svg')

###############################################################################
# RPB3 nuclear fluorescence
###############################################################################

rpb3fluor = pd.read_csv('./data/mScarRPB3_022020_mean.csv')
rpb3fluor = rpb3fluor[~(rpb3fluor.strain.isin([6]))]
colors, patches = plot.get_palette(['26','10'])
colors['10_TAF'] = '#823b3b'
colors['10_FUS'] = '#da6363'
patches = plot.get_patches(colors)
fig, ax = plt.subplots(figsize=(15, 8))
ax = plot.ecdfbystrain('mean_intensity_nuc', rpb3fluor, ax=ax, formal=True,
    line_kws = {'alpha':0.8,'rasterized':True}, colors=colors, patches=patches)
ax.set(xlabel='RPB3 Nuclear Fluorescence (a.u.)', ylabel='ECDF', xlim=(120, 600))
plt.tight_layout()
plt.savefig('./figures/output/FigS4_rpb3fluor.svg')
