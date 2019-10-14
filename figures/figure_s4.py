from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from joblib import Parallel, delayed
from utils import plot


###############################################################################

aacount = pd.read_csv('./data/fustafctd_aacount.csv')
# get order to sort by frequency in CTD
order = aacount[aacount.protein=='CTD'].sort_values('freq', ascending=False).aminoacid.values
colors = {'CTD':'#326976', 'FUS':'#da6363','TAF15':'#7a324c'}
fig, ax = plt.subplots(figsize=(14,8))
sns.stripplot(x='aminoacid', y='freq', hue='protein', data=aacount, ax=ax,
        palette=colors, s=15, alpha=0.5, order=order)
ax.set(xlabel='Amino acid', ylabel='Frequency')
plt.legend(handles=plot.get_patches(colors), ncol=3)
plt.tight_layout()
plt.savefig('./figures/output/FigS4_aaComp.svg')

###############################################################################

###############################################################################
# Binding rates (slopes) re-plotted from Kwon et al (2013) Cell
###############################################################################

rates = pd.read_csv('./data/kwon2013_BindingRates_lin_regress_leastsq.csv')
order = ['WT','1C','2E','3E','4E','5D','7','9']
rates['_'] = 0

fig, ax = plt.subplots(figsize=(14,8))
sns.stripplot(x='strain', y='beta', order=order, data=rates, ax=ax,
        color='#326976', s=15, alpha=0.3, rasterized=True)
ax.set(xlabel='FUS variant', ylabel=r'Droplet Binding Rate')
# assign intelligible strain names
strain_labels = ['WT','1C','2E','3E','4E','5D','7A','S2']
plt.xticks(plt.xticks()[0], strain_labels)
plt.tight_layout()
plt.savefig('./figures/output/FigS4_Kwon2013_DropletBinding.svg')
