from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import cm
from utils import plot

outdir = './figures/output/'
bs_act_summ = pd.read_csv('./data/gillespie_bsize_obsvtrue.csv')

colors = cm.magma(np.linspace(0,1, len(bs_act_summ.var_p_val.unique())))
colors = {delta:c for delta,c in zip(bs_act_summ.var_p_val.unique(), colors)}

# 4 ############################################################################
# Figure S7 subplot
# Compare error in burst size estimate with active cells fraction

# color edge by ctd length
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(bs_act_summ.active_cells_frac.values, bs_act_summ.bs_dev.values, alpha=1, cmap='viridis',
            c=bs_act_summ.ctd.values, s=130)
bs_act_summ.groupby('var_p_val').apply(lambda x:ax.plot(x.active_cells_frac, x.bs_dev, ls='--', c=colors[x.name], lw=2))
ax.set(xlabel='Active Cells Fraction', ylabel='Burst size error\n(Observed - True)')
ax.axhline(0, ls='--', c='k')
plt.tight_layout()
plt.savefig(outdir+'FigS8_activecells_errorbsize_manypol.svg')
