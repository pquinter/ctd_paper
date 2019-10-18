"""
Compute correlation of particle image patches with an ideal spot (gaussian blurred single light point source)
"""
import glob
import pickle

import numpy as np
import pandas as pd
from utils import image, particle

part_dir = '../output/pipeline/particles/parts_allframesimputed.csv'
spots_dir = '../output/pipeline/spot_images/imputed_rawonly'

# Load particle image patches
parts = pd.read_csv(part_dir)
pids_all, rawims_all, bpims_all = particle.load_patches(spots_dir, bp=False)

# compute correlation with ideal spot #########################################
parts.drop('corrwideal', axis=1, inplace=True)
win9x9 = np.s_[:,3:12,3:12]
corrs = image.corr_widealspot(rawims_all[win9x9], wsize=9, PSFwidth=4.2)
parts = pd.merge(parts, pd.DataFrame({'corrwideal':corrs, 'pid':pids_all}), on='pid')

parts.to_csv(part_dir, index=False)
