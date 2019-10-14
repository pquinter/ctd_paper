import pandas as pd
import numpy as np
from ast import literal_eval

import seaborn as sns
from matplotlib import pyplot as plt
from utils import plot

###############################################################################
# Heatmap of disorder probabilities in RPB1 homologs
###############################################################################

rpb1data = pd.read_csv('./data/rpb1_seq_props.csv')
# convert to probs back to array (for some reason they are stored as strings)
rpb1data['p'] = rpb1data.p.apply(literal_eval)
# get genus for stats
rpb1data['genus'] = rpb1data.species.str.split(' ', expand=True)[0]

# Get disorder probabilities
dis_probs = rpb1data.p.apply(pd.Series).replace(np.nan, 0)
# Sort by protein length
sort_ix = rpb1data.sort_values('prot_len').index
dis_probs = dis_probs.loc[sort_ix]
dis_probs.index = np.arange(len(dis_probs))

# turn off interactive mode, matrix is too big
plt.ioff()
# Plot heatmap
fig = plt.figure(figsize=(18, 14))
sns.heatmap(dis_probs, cmap=plt.cm.viridis, rasterized=True, 
    yticklabels=0, xticklabels=500)
plt.xlabel('Residue Position')
plt.tight_layout()
plt.savefig('./figures/output/Fig1_hmap.svg')

###############################################################################
# Fig 1B: ECDF of RPB1 protein lengths
###############################################################################

# interactive mode back on
plt.ion()
ax = plot.plot_ecdf(rpb1data.prot_len.values)
ax.set(xlabel='Protein Length (amino acids)', ylabel='ECDF', yticks=(0,0.5,1.0))
plt.subplots_adjust(top=0.92, bottom=0.231, left=0.137, right=0.96)
plt.savefig('./figures/output/Fig1_protlenECDF.svg')

###############################################################################
# Fig 1C: Protein length vs number of disordered amino acids
###############################################################################

ax = plot.scatter('prot_len','disorder_aa', rpb1data)
ax.set(xlabel='Protein Length (amino acids)', ylabel='Disordered Amino Acids')
plt.subplots_adjust(top=0.92, bottom=0.231, left=0.137, right=0.96)
plt.savefig('./figures/output/Fig1_protlenDis.svg')

###############################################################################
# Fig 1D: Gene density vs number of disordered amino acids
###############################################################################
ax = plot.scatter('genes_per_Mb','disorder_aa', rpb1data)
ax.set(xlabel='Gene Density (genes/Mb)', ylabel='Disordered Amino Acids')
plt.subplots_adjust(top=0.92, bottom=0.231, left=0.137, right=0.96)
plt.savefig('./figures/output/Fig1_GeneDensDis.svg')

