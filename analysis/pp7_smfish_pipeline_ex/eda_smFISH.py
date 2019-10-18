import re

import corner
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from utils import particle

part_dir = '../output/pipeline_smfish/particles/parts_filtered.csv'

seg_dir = '../output/pipeline_smfish/segmentation/082019'
part_dir = '../output/pipeline_smfish/particles/parts_filtered_smFISHgal082019_gal10.csv'
parts = pd.read_csv(part_dir)
parts['gene'] = 'gal10'
part_dir = '../output/pipeline_smfish/particles/parts_filtered_smFISHgal082019_gal3.csv'
parts_gal3 = pd.read_csv(part_dir)
parts_gal3['gene'] = 'gal3'
parts = pd.concat((parts, parts_gal3), ignore_index=True)
parts['date'] = parts.mov_name.apply(lambda x: x.split('_')[0])

# get segmentation data
seg_dir = glob.glob(seg_dir+'/*csv')
seg_df = []
for sdir in seg_dir:
    _df = pd.read_csv(sdir)
    _df['mov_name'] = sdir.split('/')[-1][:-4]
    seg_df.append(_df)
seg_df = pd.concat(seg_df, ignore_index=True)
parts.strain = parts.strain.str.replace('yQC627','yQC62')
# gene manually-determined intensity thresholds by circling spots on images
thresh = {('gal3','08152019'):700,('gal10','08152019'):100,
          ('gal3','09132019'):1200,('gal10','09132019'):500}
# fiter particles for each gene
parts = parts[(parts.corrwideal>=0.4)&(parts.cell>0)]
# assign TS or mRNA label based on threshold
parts['label'] = 'mRNA'
parts.loc[parts.apply(lambda x: x.mass>thresh[(x.gene,x.date)], axis=1), 'label'] = 'TS'

# get cell and TS counts
cell_count = seg_df.groupby('mov_name').count().centroid_cell.reset_index()
cell_count.columns = ['mov_name','no_cells']
part_count = parts[parts.label=='TS'].drop_duplicates(['cell','mov_name','gene']).groupby(['date','gene','mov_name','strain']).count().x.reset_index()
part_count.columns = ['date','gene','mov_name','strain','no_TS']
freq_df = pd.merge(part_count, cell_count, on='mov_name')
freq_df['frac_active'] = freq_df.no_TS / freq_df.no_cells
parts[parts.label=='TS'].to_csv('/Users/porfirio/lab/yeastEP/figures_paper/data/smFISH_GAL10_GAL3_TS.csv', index=False)
freq_df.to_csv('/Users/porfirio/lab/yeastEP/figures_paper/data/smFISH_GAL10_GAL3_FracActive.csv', index=False)
