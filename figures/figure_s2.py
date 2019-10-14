from matplotlib import pyplot as plt
import pandas as pd
from utils import plot
import numpy as np
from skimage import io
import seaborn as sns
from matplotlib import cm
import pickle
import sys

###############################################################################
# 2D feature plots with GPC decision surface
###############################################################################

clf_scaler_path = './data/pp7_GPClassifier.p'
# load classifier and scaler
try:
    with open(clf_scaler_path, 'rb') as f:
        clf = pickle.load(f)
        scaler = pickle.load(f)
except FileNotFoundError:
    print("""
    Pickled classifier file to generate Figure S2 not found, skipping.
    """)
    sys.exit()

pp7data = pd.read_csv('./data/pp7movs_yQC21-23.csv')

# Training#####################################################################
training_set = pd.read_csv('./data/pp7_trainingset.csv')
# scale features
scaled_lbld = scaler.transform(training_set[['corrwideal','mass_norm']].values)
axes = plot.plot2dDecisionFunc(clf, scaled_lbld, training_set.manual_label, figsize=(16, 12))
axes[1,0].set(xlim=(-2,4), ylim=(-2,6))
axes[0,0].set(xlim=(-2,4), ylim=(0,0.7))
axes[1,1].set(xlim=(-2,6), ylim=(0,0.7))
# save as png; SVG is messed up
plt.savefig('./figures/output/FigS2_Training2d.png')

# Classification###############################################################
scaled_lbld = scaler.transform(pp7data[['corrwideal','mass_norm']].values)
axes = plot.plot2dDecisionFunc(clf, scaled_lbld, pp7data.GPCprob>0.5, figsize=(16, 12))
axes[1,1].set(ylim=(0,1.1), xlim=(-1,6))
axes[0,0].set(xlim=(-2,4))
axes[1,0].set(xlim=(-2,4), ylim=(-1,6))
plt.savefig('./figures/output/FigS2_Data2dGPC.png')

# F-1 score by Prob. threshold ################################################
colors = {True:'#326976', False:'#da6363'}
prob_f1 = pd.read_csv('./data/pp7_classif_f1scoreProb.csv')
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(prob_f1.prob_thresh, prob_f1.f1score_False, color=colors[False])
ax.plot(prob_f1.prob_thresh, prob_f1.f1score_True, color=colors[True])
ax.set(ylabel='F1 score', xlabel='GPC Probability threshold')
ax.axvline(0.5, ls='--', c='k', alpha=0.5)
plt.legend(handles=plot.get_patches(colors))
plt.tight_layout()
plt.savefig('./figures/output/FigS2_f1score_prob.svg')


###############################################################################
# Nuclear fluorescence by strain
###############################################################################

fig, ax = plt.subplots(figsize=(10, 8))
plot.ecdfbystrain('nuc_fluor',
        pp7data[pp7data.laser_power==150].drop_duplicates(['roi','mov_name']),
        ax=ax, formal=True, line_kws={'alpha':1, 'rasterized':True})
ax.set(xlabel='Nuclear Fluorescence (a.u.)', ylabel='ECDF')
plt.tight_layout()
plt.savefig('./figures/output/FigS2_pp7Nucfluor.svg')

###############################################################################
# Correlation between nuclear fluorescence and burst fluorescence
###############################################################################
fluor_metrics = (('raw_mass','Raw Fluorescence (a.u.)'),('mass_norm','Normalized Fluorescence'))
for fluor, fluor_name in fluor_metrics:
    fig, ax = plt.subplots(figsize=(10, 8))
    pp7data.plot(x='nuc_fluor', y=fluor, kind='hexbin', colormap='viridis', ax=ax, rasterized=True)
    ax.set(ylabel=fluor_name, xlabel='Nuclear fluorescence (a.u.)', xticks=np.arange(150, 350, 50))
    plt.tight_layout()
    plt.savefig('./figures/output/FigS2_pp7{}vNucfluor.svg'.format(fluor))

###############################################################################
# Normalization of burst fluorescence for strains yQC21, yQC22 and yQC23
###############################################################################

lp_alpha = {100:[0.2, 0.5], 150: [0.5, 1]}
for fluor, fluor_name in fluor_metrics:
    fig, ax = plt.subplots(figsize=(10, 8))
    for lp, df in pp7data.groupby('laser_power'):
        ax = plot.ecdfbystrain(fluor, df, ax=ax, formal=True,
                ci_alpha=lp_alpha[lp][0], med_alpha=lp_alpha[lp][0],
                line_kws={'alpha':lp_alpha[lp][1], 'rasterized':True})
    ax.set(xlabel=fluor_name, ylabel='ECDF')
    plt.tight_layout()
    plt.savefig('./figures/output/FigS2_pp7{}.svg'.format(fluor))
