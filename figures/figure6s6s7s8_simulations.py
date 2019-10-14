import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import multiprocessing

from utils import gillespie_quant as quant
from utils import plot

# number of jobs for parallel processing with joblib
n_jobs = multiprocessing.cpu_count()
burstprops = pd.read_csv('./data/gillespie_burstprops.csv', comment='#')
try:
    burstprops_paramarr = pd.read_csv('./data/gillespie_burstprops_paramarray.csv', comment='#')
except FileNotFoundError:
    paramarraynotfound=1
    pass
samples = pd.read_csv('./data/gillespie_samples.csv', comment='#')
outdir = './figures/output/'

###############################################################################
## Traces
###############################################################################

# Single trace for each CTD
ctdlens = samples.ctd.unique()[1::2]
colors = {'PIC':'#588370','pol':'#ff7373','pol_p':'#994545'}
for model, samples_plot in samples[(samples.ctd.isin(ctdlens))&(samples.run==500)].groupby('model'):
    n_ctd = samples_plot.ctd.unique().shape[0]
    fig, axes = plt.subplots(n_ctd, sharex=True, sharey=True, figsize=(17, 5.5))
    for ax, (ctd, _samples) in zip(axes, samples_plot.groupby('ctd')):
        plot.traceplot(_samples, ax=ax, colors=colors, active_alpha=0.3)
        # make space for labels to the left inside the plot
        ax.set(xlim=(-5, 50), xticks=np.arange(0,51,10), ylim=(0,22), yticks=(0,15))
    ax.legend(loc=0, fontsize=12)
    sns.despine()
    axes[n_ctd//2].set_ylabel('Number of molecules')
    ax.set(xlabel='Time')
    plt.tight_layout()
    plt.savefig(outdir+'Fig6_gillespie_traceplot_{}.svg'.format(model))

# All traces as heatmap
# pivot to get individual traces for PIC, pol and pol_p
plt.ioff()
traces = samples.pivot_table(index=['model','ctd','run'], columns='time', values=['PIC','pol','pol_p'])
fig, axes = plot.traceplot_hmap(traces, figsize=(19,16.5), cmap='viridis')
plt.subplots_adjust(hspace=0.02, wspace=0.02)
plt.savefig(outdir+'FigS6_GillespieSampleHeatmap.svg')

##############################################################################
## Inter-burst duration
##############################################################################

groupby = ['model','ctd','run']
# get interburst time, filter out bursts of size<1!
inter_burst = Parallel(n_jobs=n_jobs)(delayed(quant.get_interb_time)
    (groupedby, data, cols=groupby) for groupedby, data in\
    tqdm(burstprops[burstprops.burst_size>0].groupby(groupby)))
inter_burst = pd.concat(inter_burst, ignore_index=True)

# Summary
ax = plot.medplot(inter_burst, 'inter_burst')
ax.set(ylabel='Inter-burst Duration', xlabel='CTD Length', yticks=np.arange(1, 3, 0.5))
plt.savefig(outdir+'Fig6_GillespieInterBurst.svg')

# ECDF
axes = plot.ecdfplot(inter_burst[inter_burst.inter_burst>5], 'inter_burst')# save=outdir+'inter_ecdf.pdf')
[ax.set(xlim=(0,12), xlabel='Inter-Burst Duration') for ax in axes]
plt.savefig(outdir+'FigS7_GillespieECDFinterburst.svg')

if paramarraynotfound:
    print("""
    Stochastic simulations data with parameter array not found, skipping related figures.
    """)
    pass
else:
    # get interburst time for parameter array
    groupby = ['model','ctd','var_p','var_p_val','run']
    inter_burst_paramarr = Parallel(n_jobs=n_jobs)(delayed(quant.get_interb_time)
        (groupedby, data, cols=groupby) for groupedby, data in\
        tqdm(burstprops_paramarr[burstprops_paramarr.burst_size>0].groupby(groupby)))
    inter_burst_paramarr = pd.concat(inter_burst_paramarr, ignore_index=True)
    # Inter burst times resulting from parameter exploration as heatmap
    fig, axes = plt.subplots(2, figsize=(15,20))
    for ax, (title, group) in zip(axes, inter_burst_paramarr.groupby('model')):
        plot.hmap_paramarr(group, 'inter_burst', ['var_p','var_p_val','ctd'], ax=ax, title=title, vmax=20, annot='all', ytick_symbol=False)
    [ax.set(xlabel='CTD Length') for ax in axes]
    plt.tight_layout()
    plt.savefig(outdir+'FigS8_interburst_paramarr.svg')

###############################################################################
# Fraction of active cells
###############################################################################
# get number of actively transcribing cells at each condition, i.e. pol_p>0
# discard first 10 timepoints to reach steady state
active_cells_frac_mes = samples[samples.time>10].groupby(['ctd','model','time']).apply(lambda x: np.sum(x.pol_p>0)/len(x)).reset_index(name='active_cells_frac')
ax = plot.medplot(active_cells_frac_mes, 'active_cells_frac', plotf='point', groupby=['model','ctd','time'])
ax.set(ylabel='Active Cells Fraction', xlabel='CTD Length', yticks=np.arange(0.4, 1.1, 0.1))
plt.savefig(outdir+'Fig6_GillespieActiveCellFrac.svg')

##############################################################################
## Burst size
##############################################################################

# Summary
ax = plot.medplot(burstprops[burstprops.burst_size>0], 'burst_size')
ax.set(ylabel='Burst Size (mRNAs)', xlabel='CTD Length', yticks=np.arange(0, 30, 5))
plt.savefig(outdir+'Fig6_GillespieBurstSize.svg')

# ECDF
axes = plot.ecdfplot(burstprops[burstprops.burst_size>0], 'burst_size')#, save=outdir+'bs_ecdf.pdf')
[ax.set(xlim=(0,100), xlabel='Burst Size (mRNAs)') for ax in axes]
plt.savefig(outdir+'FigS7_GillespieECDFBurstSize.svg')

##############################################################################
## Burst size with parameter array
##############################################################################

if paramarraynotfound: pass
else:
    # params to plot
    plotval, params2plot = 'log_burst_size', ['phi','epsilon']
    # Burst sizes resulting from parameter exploration as heatmap
    bprops = burstprops_paramarr[(burstprops_paramarr.var_p.isin(params2plot))&(burstprops_paramarr.burst_size>0)].copy()
    # plot log otherwise hard to see; make sure data are numeric for log!
    bprops['log_burst_size'] = bprops.burst_size.apply(lambda x: np.log10(int(x)))
    fig, axes = plt.subplots(3, figsize=(17,7))
    for ax, (title, group) in zip(axes, bprops.groupby(['model','var_p'])):
        plot.hmap_paramarr(group, plotval, ['var_p','var_p_val','ctd'],
                vmin=bprops[plotval].min(), vmax=bprops[plotval].max(),
                ax=ax, title=title)
    [ax.set(xlabel='') for ax in axes[:2]]
    ax.set(xlabel='CTD Length')
    plt.savefig(outdir+'Fig6_GillespieBurstSizePhiEps.svg')

    fig, axes = plt.subplots(2, figsize=(15,20))
    for ax, (title, group) in zip(axes, burstprops_paramarr[burstprops_paramarr.burst_size>0].groupby('model')):
        plot.hmap_paramarr(group, 'burst_size', ['var_p','var_p_val','ctd'], ax=ax, title=title, vmax=20, annot='all')
    [ax.set(xlabel='CTD Length') for ax in axes]
    plt.tight_layout()
    plt.savefig(outdir+'FigS8_burstsize_paramarr.svg')

###############################################################################
## QQ plots
###############################################################################

# get quantiles for burst size
quants_bsize = Parallel(n_jobs=n_jobs)(delayed(quant.get_quants)
        (model, ctd, group, 'burst_size', (0.5,), 'geom')\
        for (model, ctd), group in tqdm(burstprops[burstprops.burst_size>0].groupby(['model','ctd'])))
quants_bsize = pd.concat(quants_bsize, ignore_index=True)

# get quantiles for between bursts interval
quants_interb = Parallel(n_jobs=n_jobs)(delayed(quant.get_quants)
        (model, ctd, group, 'inter_burst', (1,), 'expon')\
        for (model, ctd), group in tqdm(inter_burst.groupby(['model','ctd'])))
quants_interb = pd.concat(quants_interb, ignore_index=True)

axes = plot.qqplot(quants_bsize, save=outdir+'FigS7_GillespieQQplotBurstSize.svg')
axes = plot.qqplot(quants_interb, save=outdir+'FigS7_GillespieQQplotInterburst.svg')

###############################################################################
## Burst duration and fail
###############################################################################

# Burst duration
burstprops['burst_dur'] = burstprops.end_burst - burstprops.start_burst

# Mean burst size vs duration
df_med = burstprops.groupby(['model','ctd'])[['burst_dur','burst_size']].apply(np.mean).reset_index()

# Connected scatter plot
fig, ax = plt.subplots(figsize=(10,8))
df_med.groupby('model').apply(lambda x: ax.scatter(x.burst_size, x.burst_dur, 
    color=plot.colors_model[x.name], s=50))
df_med.groupby('model').apply(lambda x: ax.plot(x.burst_size, x.burst_dur, 
    color=plot.colors_model[x.name], ls='--', alpha=0.3))
ax.set(ylabel='Burst Duration', xlabel='Burst size (mRNAs)', ylim=(0.3, 1.8))
plt.legend(handles=plot.patches_model, loc='lower right', fontsize=20)
plt.tight_layout()
plt.savefig(outdir+'FigS8_GillespieBurstDurSize.svg')

# Unsuccessful bursts: ON state that dissasembled before burst
failfunc = lambda x: np.sum(x<1)/len(x)
ax = plot.medplot(burstprops, 'burst_size', summf=failfunc, figsize=(10,8))
ax.set(xlabel='CTD Length', ylabel='Failed Bursts Fraction')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(outdir+'FigS8_GillespieFailedBurstFrac.svg')

###############################################################################
# Compare fraction of active cells to burst size and frequency
###############################################################################

# summarize burst size, frequency and active cells fraction
feats = ('burst_size','inter_burst','active_cells_frac')
for feat, df in zip(feats, (burstprops, inter_burst, active_cells_frac_mes)):
    summ_df = df.groupby(['ctd','model'])[feat].mean().reset_index(name=feat)
    try: frac_size_inter = pd.merge(frac_size_inter, summ_df, on=['ctd','model'])
    except NameError: frac_size_inter = summ_df

fig, axes = plt.subplots(1,3, figsize=(30,8))
frac_size_inter.groupby('model').apply(lambda x: axes[0].scatter(x.inter_burst.values, x.active_cells_frac.values,
            color=plot.colors_model[x.name], alpha=0.3, s=100))
frac_size_inter.groupby('model').apply(lambda x: axes[0].plot(x.inter_burst.values, x.active_cells_frac.values,
            color=plot.colors_model[x.name], alpha=0.3, ls='--'))
frac_size_inter.groupby('model').apply(lambda x: axes[1].scatter(x.burst_size.values, x.active_cells_frac.values,
            color=plot.colors_model[x.name], alpha=0.3, s=100))
frac_size_inter.groupby('model').apply(lambda x: axes[1].plot(x.burst_size.values, x.active_cells_frac.values,
            color=plot.colors_model[x.name], ls='--', alpha=0.3))
axes[2].scatter(frac_size_inter.burst_size.values, frac_size_inter.inter_burst.values,
            c=frac_size_inter.active_cells_frac.values, cmap='viridis', alpha=1, s=100)
frac_size_inter.groupby('model').apply(lambda x: axes[2].scatter(x.burst_size.values, x.inter_burst.values,
            edgecolors=plot.colors_model[x.name], facecolor='None', lw=2, s=130))
frac_size_inter.groupby('model').apply(lambda x: axes[2].plot(x.burst_size.values, x.inter_burst.values,
            color=plot.colors_model[x.name], ls='--', alpha=0.3))

axes[1].legend(handles=plot.patches_model, fontsize=12, loc=4)
axes[0].set(xlabel='Inter-burst Duration', ylabel='Active Cells Fraction', yticks=np.arange(0.4,1.01,.2))
axes[1].set(xlabel='Burst Size (mRNAs)', ylabel='Active Cells Fraction', yticks=np.arange(0.4,1.01,.2))
axes[2].set(xlabel='Burst Size (mRNAs)', ylabel='Inter-burst Duration')

# colorbar
_fracact = frac_size_inter.active_cells_frac.unique()
norm = mpl.colors.Normalize(vmin=_fracact.min(), vmax=_fracact.max())
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm._A = []
cb = plt.colorbar(sm, label=r'Active Cells Fraction')
plt.tight_layout()
plt.savefig(outdir+'FigS8_frac_active_bs_inter.svg')
