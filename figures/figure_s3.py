from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from utils import plot

###############################################################################
# plot mean aligned boolean traces
###############################################################################
aligned_traces = pd.read_csv('./data/aligned_traces_tidy.csv')
# add repeat number for color palette
aligned_traces['ctdr'] = aligned_traces.strain.map(plot.CTDr_dict)
fig, ax = plt.subplots(figsize=(10,8))
sns.lineplot(x='time', y='spot', hue='ctdr', data=aligned_traces, ax=ax, palette=plot.colors_ctd)
ax.set(xlabel='Time (min)', ylabel='Mean Transcription\nSpot Presence', xticks=np.arange(0, 50, 10))
plt.tight_layout()
plt.savefig('./figures/output/FigS3_alignedBooltraces.svg')
