import pandas as pd
import numpy as np
import glob
from matplotlib import pyplot as plt
import seaborn as sns
import re

part_dir = '../output/pipeline_snapshots/particles/parts.csv'
parts = pd.read_csv(part_dir)

# read segmentation data
seg_dir = '../output/pipeline_snapshots/segmentation'
seg_dir = glob.glob(seg_dir+'/*csv')
seg_df = []
for sdir in seg_dir:
    _df = pd.read_csv(sdir)
    _df['mov_name'] = sdir.split('/')[-1][:-4]
    seg_df.append(_df)
seg_df = pd.concat(seg_df, ignore_index=True)

# get fraction of active cells per field of view per fluor thresh
freq_df = pd.DataFrame()
thresh_arr = np.arange(4,20)
for thresh in thresh_arr:
    filt_ix = (parts.mass_norm>thresh)&(parts.corrwideal>=0.5)
    part_count = parts[filt_ix].groupby(['time_postinduction','mov_name','strain']).count().x.reset_index()
    part_count.columns = ['time_postinduction','mov_name','strain','no_TS']
    cell_count = seg_df.groupby('mov_name').count().x_cell.reset_index()
    cell_count.columns = ['mov_name','no_cells']
    _freq_df = pd.merge(part_count, cell_count, on='mov_name')
    _freq_df['frac_active'] = _freq_df.no_TS / _freq_df.no_cells
    _freq_df['thresh'] = thresh
    freq_df = freq_df.append(_freq_df).reset_index(drop=True)
freq_df.to_csv('/Users/porfirio/lab/yeastEP/figures_paper/data/pp7_frac_active_cells.csv', index=False)
