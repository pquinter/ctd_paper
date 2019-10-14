from matplotlib import pyplot as plt
import pandas as pd
from utils import plot
import numpy as np
from skimage import io
import seaborn as sns
from matplotlib import cm

pp7data = pd.read_csv('./data/pp7movs_yQC21-23.csv')

###############################################################################
# Transcription traces
###############################################################################

traces = pp7data[(pp7data.frame>0)&(pp7data.frame<125)].pivot_table(index=['strain','mov_name','roi'], columns='frame', values='mass_norm')
traces = traces.fillna(np.nanmin(traces))
# convert frame numbers to time in minutes
traces.columns = [int(i) for i in (np.arange(0, 20*traces.shape[1], 20)/60)]

# plot example trace
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(traces.iloc[8].values, color='#326976', lw=3)
ax.set(ylabel='Intensity', yticks=[], xlabel='Time', xticks=[])
plt.tight_layout()
plt.savefig('./figures/output/example_trace.svg')

# heatmap
# set kwarg cbar==True to get colorbar
plt.ioff()
fig, axes = plt.subplots(3,1, figsize=(10,13), sharex=True)
for ax, (strain, _traces) in zip(axes, traces.groupby('strain')):
    sns.heatmap(_traces, ax=ax, cmap='afmhot', xticklabels=10, cbar=False,
            vmin=np.min(traces.values), vmax=np.max(traces.values),
            cbar_kws={'label': 'Normalized Fluorescence'})
    plt.xticks(rotation=60)
    ax.set(xlabel='', ylabel=plot.CTDr_dict[strain], yticks=[])
ax.set(xlabel='Time (min)')
plt.tight_layout()
plt.subplots_adjust(hspace=0.01)
plt.savefig('./figures/output/Fig3_pp7traces_afmhot.png')

###############################################################################
# Distribution of burst fluorescence for strains yQC21, yQC22 and yQC23
###############################################################################

parts_max = pp7data.groupby(['particle','mov_name','strain','CTDr']).mass_norm.max().reset_index()
fig, ax = plt.subplots(figsize=(10, 8))
ax = plot.ecdfbystrain('mass_norm', parts_max, ax=ax, formal=True, line_kws={'alpha':1, 'rasterized':True})
ax.set(xlabel='Normalized Fluorescence', ylabel='ECDF', xlim=(5.5, 10.5))
ax.set_xticks(np.arange(6, 11))
plt.tight_layout()
plt.savefig('./figures/output/Fig3_pp7movfluor.svg')

###############################################################################
# Distribution of interburst times for strains yQC21, yQC22 and yQC23
###############################################################################

interpeak = pd.read_csv('./data/pp7_interpeaktime.csv')
fig, ax = plt.subplots(figsize=(10, 8))
ax = plot.ecdfbystrain('time', interpeak, ax=ax, formal=True, line_kws={'alpha':1, 'rasterized':True})
ax.set(xlabel='Inter-burst time (s)', ylabel='ECDF', xlim=(-100, 2200))
plt.tight_layout()
plt.savefig('./figures/output/Fig3_pp7interburst.svg')

###############################################################################
# Fraction of active cells from pp7 snapshots
###############################################################################

freq_df = pd.read_csv('./data/pp7_frac_active_cells.csv')

# manually determined intensity threshold
thresh = 7
colors = cm.viridis(np.linspace(0,1, len(freq_df['time_postinduction'].unique())))
order = ['yQC21','yQC22','yQC23','yQC62','yQC63']

fig, ax = plt.subplots(figsize=(10,8))
sns.stripplot(y='strain', x='frac_active', data=freq_df[freq_df.thresh==thresh],
    ax=ax, alpha=0.5, hue=freq_df[freq_df.thresh==thresh].time_postinduction,
    order=order, palette=colors, size=10)
sns.pointplot(y='strain', x='frac_active', data=freq_df[freq_df.thresh==thresh],
    order=order, ax=ax, alpha=1, color='#326976', size=20, join=False, ci=99)
# below two lines to draw points over strip plot
plt.setp(ax.lines, zorder=100)
plt.setp(ax.collections, zorder=100, label="")
plt.legend([], frameon=False)
ax.set(xlabel='Active cells fraction', ylabel='CTD repeats')
strain_labels = ['26','14','12','10','9']
plt.yticks(plt.yticks()[0], strain_labels)
plt.tight_layout()
plt.savefig('./figures/output/Fig3_pp7fracactive.svg')

###############################################################################
# TS fluorescence from pp7 snapshots
###############################################################################

parts = pd.read_csv('./data/pp7_snapshot_parts.csv')
# filter by strain, manually determined intensity  threshold and GPC prob
order = [26,14,12,10,9]
parts = parts[(parts.mass_norm>=7)&(parts.corrwideal>=0.5)&(parts.CTDr.isin(order))]
# turn off interactive mode because plot does not fit on screen
plt.ioff()
fig, axes = plt.subplots(len(order), figsize=(10, 24))
for CTDr, ax in zip(order, axes):
    _df = parts[parts.CTDr==CTDr]
    color = plot.colors_ctd[CTDr]
    plot.ecdf_ci('mass_norm', _df, ax=ax,
            color=color, formal=1, label=CTDr, line_kws={'alpha':1})
    ax.axvline(_df.mass_norm.median(),
            color=color, ls='--', alpha=0.8)
    ax.set(yticks=(0, 0.5, 1), xlim=(5,25), xticks=(np.arange(5, 26, 5)))
    ax.annotate('{} CTD repeats'.format(CTDr), (15, 0.3), fontsize=35)
axes[2].set(ylabel='ECDF')
axes[-1].set(xlabel='Normalized Fluorescence')
#axes[2].set(ylabel='ECDF', xlim=(5,25), xticks=(np.arange(5, 26, 5)))
plt.tight_layout()
plt.savefig('./figures/output/Fig3_pp7snapshotfluor.svg')

###############################################################################
# Sample image of PP7 cell for diagram
###############################################################################
mov = io.imread('./data/pp7sample_02132019_yQC21_150u10%int480_1.tif')
# frame number and coordinates for cell with active and inactive TS
frame_no_active, frame_no_inactive = 38, 35
coords = (slice(287, 357, None), slice(563, 649, None))# zoomed in manually and obtained with im_utils.zoom2roi func
for frame_no, active in zip((frame_no_active, frame_no_inactive), ('active','inactive')):
    im = mov[frame_no][coords]
    io.imsave('./figures/output/yQC21pp7movsample_{}.tif'.format(active), im)
# burn 1um scale bar in fiji, ~14 px long
