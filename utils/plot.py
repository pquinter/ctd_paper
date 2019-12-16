"""
Generate plots
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib import gridspec
from utils import ecdftools
import seaborn as sns
import corner

# style and colors ===========================================================

# matplotlib style
rc= {'axes.edgecolor': '0.3', 'axes.labelcolor': '0.3', 'text.color':'0.3',
    'axes.spines.top': 'False','axes.spines.right': 'False',
    'xtick.color': '0.3', 'ytick.color': '0.3', 'font.size':'35',
    'savefig.bbox':'tight', 'savefig.transparent':'True', 'savefig.dpi':'500'}
for param in rc:
    mpl.rcParams[param] = rc[param]

# scatter style
scatter_kws = {"s": 50, 'alpha':0.3,'rasterized':True}
# line style
line_kws = {'alpha':0.3,'rasterized':True}
CTDr_dict = {'yQC21':26,'yQC22':14,'yQC23':12}

def get_patches(color_dict):
    """
    Generate patches for figure legend
    """
    patches = [mpatches.Patch(color=color_dict[l], label=l) for l in color_dict]
    return patches

def get_palette(groups, palette='GnBu_d'):
    """
    Generate dict of color palette and corresponding legend patches
    Using seaborn color palettes
    """
    color_dict = {g:c for g,c in zip(groups, sns.color_palette(palette, len(groups)))}
    patches = get_patches(color_dict)
    return color_dict, patches

# CTD color palette
ctdr = ['26','14','12','10','9','8']
ctdr = [26,14,12,10,9,8]
colors_ctd, patches_ctd = get_palette(ctdr)

# Self recruitment palette
# composite of two cubehelix palettes, one for FUS and one for TAF variants

selfr_pal = {var:c for (var,c) in zip(
    ['TL47pQC99', 'TL47pQC1192E', 'TL47pQC1195D', 'TL47pQC119S2', #FUS variants
    'TL47pQC115', 'TL47pQC1202H','TL47pQC1203K'], #TAF variants
    sns.diverging_palette(220, 20, n=10))}
selfr_pal.update({var:c for (var,c) in zip(\
                ['yQC21', 'TL47pQC121', 'TL47pQC116'],#control, 13r and 10r CTD
                ['#4D4D4D','#DA84A6','#8A236E'])})

# Colors for Gillespie simulations
ctd_len = np.linspace(0.1, 0.9, 9)
labels_ctd = ['{0:0.1f}'.format(c) for c in ctd_len]
colors_ctdgil, patches_ctdgil = get_palette(ctd_len, palette='viridis')

colors = ('#D52A7C','#4E8DCE', '#587081')
models = ['manypol','onepol','manypol_FUS']
colors_model = {m:c for m,c in zip(models, colors)}
patches_model = [mpatches.Patch(color=colors_model[l], label=l) for l in colors_model]


# scatter functions ===========================================================

def scatter(x, y, df, ax=None, color='#326976', scatter_kws=scatter_kws):
    """
    Convenience function to plot scatter from dataframe with default mpl style
    """
    if ax is None: fig, ax = plt.subplots(figsize=(14,7))
    ax.scatter(df[x].values, df[y].values, color=color, **scatter_kws)
    return ax

def selfrecruit_boxp(df, x='num_spots', y='frac_cells', hue='strain', 
    order=None, labels=None,
    median=None, xlabel='Number of Spots', ylabel='Fraction of Cells',
    palette=selfr_pal, boxkwargs={'height':12, 'aspect':18/10},
    median_kwargs={'linestyle':'--', 'linewidth':2, 'alpha':0.3}):
    """
    Boxplot of number of spots by cell fraction for self-recruitment assay
    """
    fig = sns.catplot(x=x, y=y, hue=hue, kind='box', legend=False,
            palette=palette, hue_order=order, data=df,
            **boxkwargs)

    # Plot median
    # easier to use boxplot func for horizontal line, hide everything but median
    if median:
        sns.boxplot(x=x, y=y, whis=0, fliersize=0, linewidth=0,
            data=df[df.strain==median], color=palette[median],
            medianprops=median_kwargs, boxprops={'alpha':0}, ax=fig.ax)
    fig.ax.set(xlabel=xlabel, ylabel=ylabel,
            xlim=(-0.5, 5.5), xticks=np.arange(0,6))
    fig.ax.set_xticklabels(np.arange(1,7))

    if labels and order:
        patches = [mpatches.Patch(color=palette[l], label=name) for l,name in zip(order, labels)]
    else:
        patches = [mpatches.Patch(color=palette[l], label=l) for l in palette]
    fig.ax.legend(handles=patches, ncol=len(patches))
    return fig.ax

# ECDF functions ==============================================================

def plot_ecdf(arr, formal=0, ax=None, label='', color='#326976', scatter_kws=scatter_kws, line_kws=line_kws):#alpha=0.3, formal=0, label='', ylabel='ECDF', xlabel='', color='b', title='', rasterized=True, lw=None):
    """
    Convenience function to plot ecdf with default mpl style
    """
    if ax==None: fig, ax = plt.subplots(figsize=(14,7))
    if formal:
        ax.plot(*ecdftools.ecdf(arr, conventional=formal), label=label, color=color, **line_kws)
    else:
        ax.scatter(*ecdftools.ecdf(arr, conventional=formal), label=label, color=color, **scatter_kws)
    return ax


def ecdf_ci(x, df, ax=None, no_bs=1000, ci=99, plot_median=True, color='#326976',
        ci_alpha=0.3, med_alpha=0.8, label='', **ecdfkwargs):
    """
    Plot ECDF with bootstrapped confidence interval
    """
    if ax is None: fig, ax = plt.subplots()

    if ci:
        # generate bootstrap samples
        bs_samples = ecdftools.bs_samples(df, x, no_bs)
        # compute ecdfs from bs samples
        bs_ecdf = ecdftools.ecdfs_par(bs_samples)
        # get ECDF confidence intervals
        quants, ci_high, ci_low = ecdftools.ecdf_ci(bs_ecdf, ci=ci)
        # plot intervals
        ax.fill_betweenx(quants, ci_high, ci_low, color=color,
                alpha=ci_alpha, rasterized=True)

    if plot_median:
        # get median and bootstrapped CI if available
        median = np.median(df[x])
        try:
            # get median index
            med_ix = np.argmin((np.abs(quants - 0.5)))
            err = np.array([[median-ci_low[med_ix]],   # lower error
                            [ci_high[med_ix]-median]]) # higher error
        except NameError: err = None
        ax.errorbar(median, 1.05, xerr=err, color=color, fmt='.',
                markersize=20, alpha=med_alpha, elinewidth=3, rasterized=True)

    plot_ecdf(df[x], ax=ax, color=color, label=label, **ecdfkwargs)
    return ax

def ecdfbystrain(x, df, groupby='CTDr', ax=None, strains='all', plot_median=True,
        ci=99, no_bs=1000, ci_alpha=0.3, med_alpha=0.8, colors=colors_ctd, patches=patches_ctd,
        **ecdfkwargs):
    """
    Plot ECDFs by strain
    x: str
        name of column to plot
    """
    if ax is None: fig, ax = plt.subplots()
    if strains=='all': strains = df[groupby].unique()

    for s, group in df[df[groupby].isin(strains)].groupby(groupby):
        ecdf_ci(x, group, ax=ax, no_bs=no_bs, ci=ci, plot_median=plot_median,
                color=colors[s], ci_alpha=ci_alpha, med_alpha=med_alpha, 
                label=s, **ecdfkwargs)

    # filter legend patches
    patches = [p for p in patches if p.get_label() in [str(s) for s in strains]]
    ax.legend(handles=patches)
    return ax

# RNAseq ===============================================================

def qvalecdf_hmap(df, coefs, coef_names=None, ax=None):
    """
    Plot cumulative distribution of q-values as a heatmap

    df: DataFrame
        with q-value thresholds, number of transcripts and coefficients
        must have column names: ['cid','qval_thresh','no_transcripts']
    coefs: iterable
        list of coefficients to plot
    """
    # select coefficients
    df_sel = df[df.cid.isin(coefs)]
    # pivot data into heatmap format
    df_hmap = df_sel.pivot(index='cid', columns='qval_thresh', values='no_transcripts')
    # sort
    sorter= dict(zip(coefs,range(len(coefs))))
    df_hmap['sorter'] = df_hmap.index.map(sorter)
    df_hmap.sort_values('sorter', inplace=True)

    # plot selected
    if ax is None: fig, ax = plt.subplots(figsize=(22,2*len(coefs)))
    sns.heatmap(df_hmap, yticklabels=df_hmap.index, xticklabels=99, ax=ax,
            cmap='viridis', rasterized=True, cbar_kws={'label':'Transcripts'})

    # format xticks
    x_format = ax.xaxis.get_major_formatter()
    # make q-value axis str to avoid extremely long floats
    x_format.seq = ["{:0.1f}".format(float(s)) for s in x_format.seq]
    ax.xaxis.set_major_formatter(x_format)
    # horizontal xticks
    plt.xticks(rotation=0)
    ax.set(xlabel='q-value threshold', ylabel='Coefficient')
    # hide blue strip: no genes after cum fraction>1.0!
    ax.set_xlim(0, 999)
    plt.tight_layout()
    # assign intelligible coefficient names if provided
    if coef_names is not None: ax.set_yticklabels(coef_names)
    return ax

def coef_stemplot(df, coefs, coef_names=None, qval_thresh=0.1, color='#326976', 
        orient='v', ax=None):
    """
    Plot number of transcripts at q-value thresh per coefficient
    """
    # select coefficients
    df_sel = df[df.cid.isin(coefs)]
    # get number of transcripts at q-value threshold for each coefficient
    transcripts = df_sel[df_sel.qval_thresh<qval_thresh].sort_values('qval_thresh').groupby('cid').tail(1)
    # sort
    sorter= dict(zip(coefs,range(len(coefs))))
    transcripts['sorter'] = transcripts.cid.map(sorter)
    transcripts.sort_values('sorter', inplace=True)
    # plot
    if ax is None and orient=='v': fig, ax = plt.subplots(figsize=(len(coefs)*2, 8))
    elif ax is None and orient=='h': fig, ax = plt.subplots(figsize=(12,len(coefs)*2))
    if orient=='v':
        ax.vlines(transcripts.cid, 0, transcripts.no_transcripts, colors=color)
        ax.scatter(transcripts.cid, transcripts.no_transcripts, s=150, color=color)
        ax.set(ylabel=r'Transcripts at q<{0:0.1f}'.format(qval_thresh), ylim=0)
        ax.margins(0.1)
        plt.xticks(rotation=60)
        if coef_names is not None: ax.set_xticklabels(coef_names)
    elif orient=='h':
        # sort again to have up-down order
        if isinstance(color, list): color=color[::-1]
        transcripts.sort_values('sorter', ascending=False, inplace=True)
        ax.hlines(transcripts.cid, 0, transcripts.no_transcripts, colors=color)
        ax.scatter(transcripts.no_transcripts, transcripts.cid, s=150, color=color)
        ax.set(xlabel=r'Transcripts at q<{0:0.1f}'.format(qval_thresh), xlim=0)
        ax.margins(0.1)
        if coef_names is not None: ax.set_yticklabels(coef_names[::-1])
    plt.tight_layout()
    return ax

def scatter_coef(df, x_coef, y_coef, ax=None, auto_ref=True, alpha=0.2,
    color='#326976', color_autoref='#99003d',qvaloffset=1e-100, **scatter_kwargs):
    """
    Scatter plot of coefficients
    df: DataFrame
        Must contain columns ['cid', 'target_id','b']
    x_coef, y_coef: str
        name of coefficients to plot
    auto_ref: bool
        whether to plot diagonal of x_coef vs x_coef with appropriate sign from correlation
    qvaloffset: float
        small number to add qval to prevent infinity from np.log(0)
    """

    if ax is None: fig, ax = plt.subplots(figsize=(12,10))

    # split and merge to make sure they are ordered
    xx = df[df.cid==x_coef]
    yy = df[df.cid==y_coef]
    merge = pd.merge(xx, yy, on='target_id')

    # scatter with size inversely proportional to log of q-value
    ax.scatter(merge.b_x.values, merge.b_y.values,
                s=-np.log(merge.qval_y.values+qvaloffset),
                rasterized=True, alpha=alpha, c=color, **scatter_kwargs)
    if auto_ref:
        # get correlation and use sign to plot reference
        corr = np.corrcoef(merge.b_x.values, merge.b_y.values)[0,1]
        print('Pearson correlation = {0:.2f}'.format(corr))
        ax.plot(xx.b.values, np.sign(corr) * xx.b.values, '--',
            s=1, rasterized=True, alpha=0.5, c=color_autoref)
        ax.legend(['Pearson correlation = {0:.2f}'.format(corr)], fontsize=20)
    ax.axhline(0, ls='--', alpha=0.1, color='k')
    ax.axvline(0, ls='--', alpha=0.1, color='k')
    ax.set(xlabel=x_coef, ylabel=y_coef)
    plt.tight_layout()
    return ax

def volcano_plot(coef, df, ax=None, thresh=0.1,
                        colors=('#326976','#99003d'), alpha=0.1):
    """
    Volcano plot, where y-axis is log10 q-value and
    x-axis is beta, approximately log fold-change

    coef: beta coefficient to plot
    df: DataFrame containing beta and q-value by transcript
        Must contain columns ['b','qval']
    thresh: q-value threshold below which to make points red
    ax: matplotlib axis
    """

    if ax is None: fig, ax = plt.subplots(figsize=(6,8))
    df = df.loc[sleuth.coef==coef]
    # split into significant and nonsignificant
    nonsig = df.loc[df.qval>thresh]
    sig = df.loc[df.qval<=thresh]
    for data, c in zip((nonsig, sig), colors):
        ax.scatter(data.b.values, -np.log10(data.qval.values), alpha=alpha, s=3, c=c)
    ax.set(title=coef, xlabel='Beta', ylabel=r'-log$_{10}$(q-value)')
    plt.tight_layout()
    return ax

# Growth curves ===============================================================
def filtOD(x, low=0.15, high=0.8):
    """
    get exponential region of growth curve, between 'low' and 'high' ODs
    """
    od = x.od.values
    time = x.Time.values
    # mask values in between allowed OD range
    mask = (od>low) & (od<high)
    if np.sum(mask)==0: return 0,0
    # create masked array
    masked_od = np.ma.array(od, mask=np.logical_not(mask))
    # get longest contigous array
    contigs = np.ma.flatnotmasked_contiguous(masked_od)
    # measure array lengths
    contigs_len = [s.stop - s.start for s in contigs]
    # get longest
    contig_ind = contigs[np.argmax(contigs_len)]
    od = od[contig_ind].astype(float)
    time = time[contig_ind].astype(int)
    # subtract initial time
    time -= time[0]
    return time, od

def growth_curve(curve_tidy, strains='all', groupby='CTDr', plot='range',
        colors=colors_ctd, alpha=0.8, range_alpha=0.3, low=0.15, high=0.8,
        ax=None, figsize=(12,8)):
    """
    Plot growth curves by strain
    """
    if strains=='all': strains = curve_tidy.strain.unique()
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    # filter strains
    curve_tidy = curve_tidy[curve_tidy.strain.isin(strains)]
    for strain, curve in curve_tidy.groupby(groupby):
        # filter OD to specified range
        curves_filt = [filtOD(x[1], low=low, high=high) for x in curve.groupby('well')]

        # plot every replicate
        if plot=='all':
            [ax.plot(_time, _curve, alpha=alpha, rasterized=True, label=strain,
                color=colors[strain]) for (_time, _curve) in curves_filt]

        # plot mean and replicate range only
        elif plot=='range':
            time, curves_filt = curves_filt[0][0], [c[1] for c in curves_filt]
            # get time of shortest curve
            time = time[:min([len(c) for c in curves_filt])]
            # get mean and range
            odlow = [np.min(_ods) for _ods in zip(*curves_filt)]
            odhigh = [np.max(_ods) for _ods in zip(*curves_filt)]
            odmean = [np.mean(_ods) for _ods in zip(*curves_filt)]
            ax.fill_between(time, odhigh, odlow, alpha=range_alpha,
                    rasterized=True, color=colors[strain])
            ax.plot(time, odmean, alpha=alpha, rasterized=True, label=strain,
                    color=colors[strain])

    patches = [mpatches.Patch(color=colors[s], label=s) for s in curve_tidy[groupby].unique()]
    ax.legend(handles=patches, title=groupby)
    ax.set(xlabel='Time (min)', ylabel='OD$_{600}$')
    plt.tight_layout()

    return ax

def _stripplot_errbars(strip_data, i, ax, jlim=0.1, yerr=None, xerr=None, **kwargs):
    """ Hack to plot errorbars with jitter """
    from scipy import stats
    jitterer = stats.uniform(-jlim, jlim * 2).rvs
    cat_pos = np.ones(strip_data.size) * i
    cat_pos += jitterer(len(strip_data))
    ax.errorbar(strip_data, cat_pos, yerr, xerr, **kwargs)

def stripplot_errbars(x, y, err, order, df, errax='xerr', colors='#326976', 
        ax=None, plot_kws={'fmt':'o', 'alpha':0.8, 'ms':8, 'elinewidth':2,'rasterized':True}):
    """
    Plot horizontal stripplot with custom errorbars for each point from dataframe
    """
    if ax is None: fig, ax = plt.subplots()
    if isinstance(colors, str): colors=[colors]*len(order)
    # create sorting index; needed because of duplicate labels in groups
    sort_ix = {strain:ix for ix, strain in enumerate(order)}
    # plot with seaborn to add labels and hide with alpha; do NOT rasterize,
    # otherwise cannot save as vector graphics
    sns.stripplot(x=x, y=y, data=df, order=order, alpha=0, ax=ax)
    # plot with by group with errorbars
    for strain, group in df.groupby(y):
        # Get xvalue from sorting index, skip if not included
        try: yval = sort_ix[strain]
        except KeyError: continue
        _stripplot_errbars(group[x].values, yval, ax, color=colors[yval],
                                xerr=group[err], **plot_kws)
    return ax

# Spot classification ===========================================================

def plot2dDecisionFunc(clf, xs, ys, colors=('#326976','#da6363'), labels=(True, False),
        xlabel='Correlation with Ideal Spot', ylabel='Intensity',
        plot_data_contour=True, plot_scatter=False,
        scatter_alpha=0.1, figsize=(12, 10)):
    """
    Plot decision surface of classifier with 2D points on top
    """
    # transpose data if necessary
    if len(xs)>len(xs.T): xs = xs.T

    # make grid
    xx, yy = np.meshgrid(
            np.linspace(np.min(xs[0]), np.max(xs[0]), 100),
            np.linspace(np.min(xs[1]), np.max(xs[1]), 100))
    # get decision function
    if hasattr(clf, 'decision_function'):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    z = z.reshape(xx.shape)

    # put things in dataframe
    data = pd.DataFrame()
    data[xlabel] = xs[0]
    data[ylabel] = xs[1]
    data['y'] = ys

    colors = dict(zip(labels, colors))
    # make the base figure for corner plot
    ndim = len(xs)
    fig, axes = plt.subplots(ndim, ndim, figsize=figsize)

    if not plot_data_contour: axes[1,0].clear()

    if hasattr(clf, 'decision_function'):
        # plot decision boundary
        axes[1,0].contour(xx, yy, z, levels=[0], linewidths=2, colors='#FFC300')
    else:
        # or probability distribution
        cs = axes[1,0].contourf(xx, yy, z, cmap='viridis')
    handles = [mpatches.Patch(color=colors[l], label=l) for l in labels]
    axes[0,1].legend(handles=handles, loc='lower left')

    # plot data with corner
    data.groupby('y').apply(lambda x: corner.corner(x, color=colors[x.name], 
        hist_kwargs={'density':True}, fig=fig, rasterized=True))
    # plot data on top
    if plot_scatter:
        data.groupby('y').apply(lambda x: axes[1,0].scatter(x[xlabel],
        x[ylabel], alpha=scatter_alpha, color=colors[x.name], rasterized=True))

    # add colorbar to countourf. Must be done after corner, or it will complain
    if hasattr(clf, 'predict_proba'):
        fig.colorbar(cs, ax=axes[1,0], ticks=np.linspace(0,1,5))
    plt.tight_layout()
    return axes

# Stochastic simulations ======================================================

def qqplot(quants, groupby=['model','ctd'], annotate='r', figsize=(30,8), save=False):
    """
    Make a QQ-plot from dataframe with theoretical and empirical quantiles
    Number of axes correspond to first element of `groupby`,
    each with second element's number of plots
    Annotate with statistic on legend, next to second `groupby` element
    """
    n_groups = len(quants.groupby(groupby[0]))
    fig, axes = plt.subplots(1, n_groups, sharex=True, figsize=figsize)
    axes_dict = {m:ax for m, ax in zip(quants[groupby[0]].unique(), axes)}
    # plot quantiles
    quants.groupby(groupby).apply(lambda x: axes_dict[x.name[0]].scatter(\
        x.theor.values, x.data.values,
        color=colors_ctdgil[x.name[1]], alpha=0.2, rasterized=True))
    # plot fit and label with correlation coefficient
    quants.groupby(groupby).apply(lambda x: axes_dict[x.name[0]].plot(\
        x.theor.values, x.slope.values*x.theor.values+x.intercept.values,
        label='CTD:{0:.1f}, r={1}'.format(x.ctd.iloc[0], str(x[annotate].iloc[0])[:4]),
        color=colors_ctdgil[x.name[1]], alpha=0.2, rasterized=True))
    # add title
    quants.groupby(groupby).apply(lambda x: axes_dict[x.name[0]].set_title(\
        x.name[0], fontsize=12, loc='left'))
    [ax.legend(fontsize=15) for ax in axes]
    [ax.set(xlabel='Theoretical quantiles', ylabel='Empirical quantiles') for ax in axes]
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')
    return axes

def ecdfplot(df, value, groupby=['model','ctd'], alpha=1, figsize=(30,8), save=False):
    """
    ECDF plot of two-level grouped dataframe
    """
    n_groups = len(df.groupby(groupby[0]))
    fig, axes = plt.subplots(1,n_groups, figsize=figsize)
    axes_dict = {m:ax for m, ax in zip(df[groupby[0]].unique(), axes)}
    df.groupby(groupby).apply(lambda x: plot_ecdf(x[value].values,
        formal=1, ax=axes_dict[x.name[0]],
        color=colors_ctdgil[x.name[1]]))
    [ax.set(ylabel='ECDF', xlabel=value) for ax in axes]
    [axes_dict[n].set_title(n, fontsize=12, loc='left') for n in axes_dict]
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')
    return axes

def medplot(df, value, x='ctd', plotf='point', hue='model', summf=np.mean,
        groupby=['model','ctd','run'], linealpha=0.5,
        ax=None, figsize=(14,7.5), palette=colors_model, save=False, **kwargs):
    """
    Summary plot of two-level grouped dataframe
    summf: function for summary statistic, also used in pointplot
    plotf: str. `box` or `point` plot function to use
    """
    df_med = df.groupby(groupby)[value].apply(summf).reset_index()
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    if plotf=='point':
        g = sns.pointplot(x=x, y=value, hue=hue,
                data=df_med, palette=palette, ax=ax, ci=99, linestyles='--', **kwargs)
        plt.setp(g.lines, alpha=linealpha)
    elif plotf=='box':
        sns.boxplot(x=x, y=value, hue=hue,
                data=df_med, palette=palette, ax=ax, **kwargs)
    ax.legend(handles=patches_model)
    if x=='ctd':
        ax.set_xticklabels(['{0:0.1f}'.format(x) for x in df.ctd.unique()])
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')
    return ax

def traceplot_hmap(traces, values=['PIC','pol','pol_p'], cmap='viridis', figsize=(12,18), save=False):
    """
    Plot all sample traces as a heatmap
    """
    # make a CTD column for visualization
    ctd_col = np.array([traces.loc['manypol','pol'].reset_index().ctd.values]*3).T
    # get common max value for each type of plot
    vmax = {p:traces[p].values.max() for p in values}

    # plot all traces
    fig, axes = plt.subplots(3, len(values), sharey=False, sharex=False, figsize=figsize)
    for (_model, _traces), _axes in zip(traces.groupby('model'), axes):
        [sns.heatmap(np.hstack((ctd_col*vmax[p], _traces[p].values)),
            vmin=0, vmax=vmax[p], ax=ax, cmap=cmap, cbar_kws={'shrink': 0.4},
            yticklabels=[], xticklabels=[], rasterized=True) for p, ax in zip(values, _axes)]
    [ax.set_title(p) for p, ax in zip(values, axes[0])]
    [ax.set_ylabel(p) for p, ax in zip(traces.index.levels[0], axes[:,0])]
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    return fig, axes

def traceplot(trace, values=('pol','pol_p'), time=50, annotate=False, ax=None, legend=False, colors={'PIC':'#ffb90f', 'pol':'#e53939','pol_p':'#5c6994'}, active_alpha=0.5, alpha=0.5):
    """
    Plot a single trace and highlight active state (PIC)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.set(xlabel='Time', ylabel='Molecules')

    time_points = np.linspace(0, time, trace.time.values.shape[0]) # time samples
    for i, pol in enumerate(values):
        ax.scatter(time_points, trace[pol].values, c=colors[pol], alpha=alpha, zorder=100)
        ax.plot(time_points, trace[pol].values, c=colors[pol], label=values[i], zorder=100)

    # plot highlight formed PIC
    (lefts,) = np.where(np.diff(trace['PIC'].values)==1)
    (rights,) = np.where(np.diff(trace['PIC'].values)==-1)
    if len(rights)<len(lefts): rights = np.concatenate((rights, np.array([trace.shape[0]-1])))
    for l,r in zip(lefts, rights):
        ax.axvspan(time_points[l], time_points[r], alpha=active_alpha, color=colors['PIC'], zorder=0)
    if annotate:
        ax.annotate(' ctd\n{0:.2f}'.format(annotate), (ax.get_xlim()[0], ax.get_ylim()[1]), fontsize=12)
    if legend:
        patches = [mpatches.Patch(color=colors[l], label=l) for l in values]
        plt.legend(handles=patches, fontsize=10)
        plt.tight_layout()
    return ax

def hmap_paramarr(df, value, groupby, x='ctd', param_col='var_p_val',
        summf=np.nanmean, ax=None, title='', annot=False, ytick_symbol=True,
        **hmap_kwargs):
    """
    Plot summary of df with varying parameters as heatmap
    df: dataframe
    value: value to plot
    x: x-axis
    param_col: y-axis, varying parameter
    summf: function to compute summary statistic
    """

    df = df.groupby(groupby)[value].apply(summf).reset_index()
    hmap = df.pivot_table(index=groupby[:-1], columns=x, values=value)

    ctd_len = df[x].unique()
    # make a param column for visualization
    param_vals = hmap.index.get_level_values(param_col)
    n_params = np.unique(param_vals).shape[0]
    pdict = {p:v for p,v in zip(param_vals, np.linspace(0, 1, n_params))}
    # use specified vmax or get it from data
    try: maxval = hmap_kwargs['vmax']
    except KeyError: maxval = np.nanmax(hmap.values)
    param_viz = np.array([pdict[p] for p in param_vals]) * maxval

    # add to the left of heatmap
    hmap.insert(0, 0, param_viz)
    # fill nans with 0
    hmap = hmap.fillna(0)

    if ax is None: fig, ax = plt.subplots()

    # make short yticklabels, actual value is now indicated by col in hmap
    _yticklabels = hmap.index.get_level_values(0)
    # get center position for unique ylabels
    yticks, ytcounts = np.unique(_yticklabels, return_counts=True)
    # make array of empty lables and fill in only center position
    yticklabels = [list(np.full(t, '')) for t in ytcounts]
    for pos, yt, yset in zip(ytcounts//2, yticks, yticklabels):
        # turn into letter symbol
        if ytick_symbol: yset[pos] = r'$\{}$'.format(yt)
        else: yset[pos] = yt
    yticklabels = [y for ys in yticklabels for y in ys]

    # make annotation array with only params annotated, unless asked for all
    annot_arr = np.full_like(hmap, '', dtype='U10')
    annot_arr[:,0] = np.array(['{0:0.1f}'.format(p) for p in param_vals])
    if annot=='all':
        annot_arr = hmap.values.copy()
        annot_arr[:,0] = np.array([param_vals])

    sns.heatmap(hmap, yticklabels=yticklabels,
            annot=annot_arr, fmt='s' if annot is False else '.2f',
            annot_kws={'fontsize':15, 'ha': 'left'},
            ax=ax, cmap='viridis', cbar_kws={'label':value}, **hmap_kwargs, rasterized=True)
    ax.set_xticklabels(['p']+['{0:0.1f}'.format(x) for x in ctd_len], rotation=0)
    ax.set_title(title, fontsize=12, loc='left')
    ax.set(ylabel='')
    ax.tick_params(axis='y', rotation=0)
    return ax
