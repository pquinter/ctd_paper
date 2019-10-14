from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from utils import ecdftools, plot

###############################################################################
# Doubling times for FUS and TAF15 mutant rescued strains
###############################################################################

growth = pd.read_csv('./data/growthrates_merged.csv')
growth = growth.drop_duplicates('strain', keep='last')
strains = ['yQC15','yQC40', 'yQC41', 'yQC42', 'yQC43', 'yQC44', 'yQC45', 'yQC46', 'yQC16','yQC48', 'yQC49', 'yQC62']
# colors for WT and mutant LCD
colors =  ['#da6363']+ ['#7a324c']*7 + ['#da6363']+ ['#7a324c']*2 + ['#326976']

#strains = ['TL47','yQC5','yQC6','yQC62','yQC15'] + ['yQC4'+str(i) for i in np.arange(0, 10)] + ['yQC16']
fig, ax = plt.subplots(figsize=(14,10))
ax = plot.stripplot_errbars('inverse max df', 'strain', 'stdev', strains,
                    growth, ax=ax, colors=colors)
ax.set(xlabel='Doubling Time (min)', ylabel='')
# assign intelligible strain names
strain_labels =\
['WT','1C','2E','3E','4E','5D','7A','S2','WT','2H','3K','10CTDr']
plt.yticks(plt.yticks()[0], strain_labels)
plt.tight_layout()
plt.savefig('./figures/output/Fig5_growthrates_FUS.svg')

###############################################################################
# FUS droplet binding rates vs growth rate
###############################################################################

brate_fus = pd.read_csv('./data/FUSbrates_Kwon2013_lin_regress_leastsq.csv')
brate_fus = brate_fus.groupby('strain').mean().reset_index()
brate_fus.strain = brate_fus.strain.replace({'7':'7A','9':'S2'})
# rename strains
fus_strains = ['yQC15','yQC40', 'yQC41', 'yQC42', 'yQC43', 'yQC44', 'yQC45', 'yQC46']
fus_labels = ['WT','1C','2E','3E','4E','5D','7A','S2']
strain2label = {x:y for x,y in zip(fus_strains, fus_labels)}
growth_brate = growth[growth.strain.isin(fus_strains)].copy()
growth_brate['strain'] = growth_brate.strain.map(strain2label)
brate_fus = pd.merge(growth_brate, brate_fus, on='strain')
# colors for WT and mutant LCD
fig, ax = plt.subplots(figsize=(14,10))
plot.scatter('inverse max df', 'beta', brate_fus, scatter_kws={'s':150, 'alpha':0.8}, ax=ax, color=colors[:8])
ax.errorbar(brate_fus['inverse max df'], brate_fus['beta'], xerr=brate_fus['stdev'], color=colors[:8], linestyle='None', elinewidth=3, alpha=0.8)
ax.set(xlabel='Doubling Time (min)', ylabel='FUS Hydrogel\nBinding Rate (h$^{-1}$a.u.)', xticks=np.arange(118, 136, 4))
ax.annotate('pearson_r={:.2f}'.format(brate_fus[['inverse max df', 'beta']].corr().values[1,0]), (125, 1.5))
[ax.annotate(xy, l) for xy, l in\
    zip(brate_fus.strain.values, brate_fus[['inverse max df','beta']].values+0.02)]
plt.tight_layout()
plt.savefig('./figures/output/Fig5_growthvbrate.svg')

###############################################################################
# Self-recruitment Boxplot of number of spots per cell
###############################################################################

selfrecruit = pd.read_csv('./data/selfrecruit.csv')
# plot order
order = ['yQC21', 'TL47pQC121', 'TL47pQC116',  #control, 10r and 13r CTD
        'TL47pQC99', 'TL47pQC1192E', 'TL47pQC1195D', 'TL47pQC119S2', #FUS and mutants
        'TL47pQC115', 'TL47pQC1202H','TL47pQC1203K'] #TAF and mutants
labels = ['PP7-GFP (-)','10CTDr','13CTDr','WT', '2E','5D','S2','WT','2H','3K']
plt.ioff()
ax = plot.selfrecruit_boxp(selfrecruit, order=order, labels=labels,
        median='TL47pQC116', boxkwargs={'height':10, 'aspect':2.5})
plt.savefig('./figures/output/Fig5_selfrecruit.svg')

###############################################################################
# Self-recruitment TS intensities
###############################################################################

selfrecruit_ts = pd.read_csv('./data/selfrecruit_TS.csv')
plt.ioff()
#(7.5, 20.5) dimensions for smaller figure
fig, axes = plt.subplots(len(order), figsize=(9, 24.5), sharex=True)
for strain, ax in zip(order, axes):
    _df = selfrecruit_ts[selfrecruit_ts.strain==strain]
    color = plot.selfr_pal[strain]
    plot.ecdf_ci('mass_norm', _df, ax=ax,
            color=color, formal=1, label=strain, line_kws={'alpha':1})
    ax.axvline(_df.mass_norm.median(), color=color, ls='--', alpha=0.8)
ax.set(xlim=(0,50), xticks=(np.arange(0, 55, 10)))
[ax.annotate(l, (30, 0.1), fontsize=25) for ax, l in zip(axes, labels)]
axes[4].set(ylabel='ECDF')
axes[-1].set(xlabel='Normalized Fluorescence')
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
plt.savefig('./figures/output/Fig5_selfrecruit_TS.svg')

###############################################################################
# Number of spots vs droplet binding rate (two few points, not very useful)
###############################################################################

# rename self recruit df strains
strain2label = {x:y for x,y in zip(order, labels)}
selfrecruit['strain'] = selfrecruit.strain.map(strain2label)
selfrecruit_mean = selfrecruit[selfrecruit.num_spots==1].groupby('strain').mean().reset_index()
# merge with droplet binding rates
brate_fus = pd.merge(selfrecruit_mean, brate_fus, on='strain')
plot.scatter('frac_cells','beta',brate_fus)
