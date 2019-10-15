"""
Explore influence of elongation rate (delta) in difference between observed and True burst size.
"""

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import quant, plot

samples = pd.read_csv('./data/gillespie_samples_manypol_vardelta.csv', comment='#')
burstprops = pd.read_csv('./data/gillespie_burstprops_manypol_vardelta.csv', comment='#')

# pivot to get individual traces for PIC, pol and pol_p
traces = samples.pivot_table(index=['model','ctd','run','var_p_val'], columns='time', values=['pol_p'])

# get apparent burst size
burstprops_app =  Parallel(n_jobs=12)(delayed(quant.get_app_bs)
                                  (_model, _ctd, _run, _delta, tr)
        for (_model, _ctd, _run, _delta), tr in tqdm(traces.iterrows()))
burstprops_app = pd.concat((burstprops_app), ignore_index=True)

# Fraction of active cells
active_cells_frac_mes = samples[samples.time>10].groupby(['ctd','var_p_val','time']).apply(lambda x: np.sum(x.pol_p>0)/len(x)).reset_index(name='active_cells_frac')

# Get difference between True and apparent burst size
bs_med = burstprops.groupby(['var_p_val','ctd','run'])['burst_size'].apply(np.mean).reset_index()
appbs_med = burstprops_app.groupby(['var_p_val','ctd','run'])['app_bs'].apply(np.mean).reset_index()
bs_med = pd.merge(bs_med, appbs_med, on=['var_p_val','ctd','run'])
bs_dev = bs_med.groupby(['var_p_val','ctd'])[['burst_size','app_bs']].apply(np.mean).reset_index()
bs_dev['bs_dev'] = bs_dev.app_bs - bs_dev.burst_size

# Merge frac active cells with burst size deviation
bsdev_summ = bs_dev.groupby(['var_p_val','ctd'])[['bs_dev']].mean().reset_index()
actcells_summ = active_cells_frac_mes.groupby(['var_p_val','ctd'])[['active_cells_frac']].mean().reset_index()
bs_act_summ = pd.merge(bsdev_summ, actcells_summ, on=['var_p_val','ctd'])
bs_act_summ.to_csv('./data/gillespie_bsize_obsvtrue.csv', index=False)
burstprops_app.to_csv('./data/gillespie_burstpropsapp.csv', index=False)
