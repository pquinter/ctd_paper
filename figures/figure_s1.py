from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from utils import plot

################################################################################
# Amino acid frequency in CTD sequences
################################################################################

rpb1aacount = pd.read_csv('./data/rpb1_aacounts.csv')
# compute amino acid frequency from count and length
freq = pd.melt(rpb1aacount.apply(lambda x: x/x.ldlen, axis=1).drop(['ldlen'], axis=1), var_name='aminoacid', value_name='freq')
# get order to sort by average frequency
order = freq.groupby('aminoacid').mean().reset_index().sort_values('freq', ascending=False).aminoacid.values
plt.ioff()
fig, ax = plt.subplots(figsize=(32,8))
sns.boxplot(x='aminoacid', y='freq', data=freq, ax=ax, notch=True,
        color='#326976', order=order)
# horizontal line at expected percentage based on uniform distribution of aa
plt.axhline(1/20, color='r', alpha=0.3, ls='dashed')
ax.set(xlabel='Amino acid', ylabel='Frequency')
plt.tight_layout()
plt.savefig('./figures/output/FigS1_aaComp.svg')

################################################################################
# Distributions of aromaticity, net charge and GRAVY
################################################################################

rpb1props = pd.read_csv('./data/rpb1_chem_props.csv')
for prop, name in zip(('arom','gravy','net_charge'),('Aromaticity',' Grand Average of Hydropathy','Net Charge')):
    fig, ax = plt.subplots(figsize=(10,8))
    plot.ecdf_ci(prop, rpb1props, ax=ax)
    ax.set(ylabel='ECDF', xlabel=name)
    plt.tight_layout()
    plt.savefig('./figures/output/FigS1_LD{}.svg'.format(prop))
