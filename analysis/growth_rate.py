"""
Tidy OD600 96-well plate reader data and compute doubling time using Peter Swain's code
"""
import pandas as pd
import numpy as np

import os
import sys
import time

from joblib import Parallel, delayed
from tqdm import tqdm
from utils import growth
import multiprocessing

outdir = './data/'
indir = './data/raw/'
fname = 'od600'

dt = 15 # time interval between OD600 reads
odmin, odmax = 0.25, 1.4 # OD range to use for fit
n_jobs=multiprocessing.cpu_count()

# put data in tidy dataframe
data_path = indir+'01112019_Formatted.txt'
layout_path = indir+'01122019_PlateLayout.csv'
curve_tidy = growth.tidy_growthcurve(data_path, layout_path)
# analyze OD traces in parallel grouped by line
fit = Parallel(n_jobs=n_jobs)(delayed(growth.fitderiv_par)(strain, od, dt=dt)
        for strain, od in tqdm(curve_tidy[(curve_tidy.od>odmin)&(curve_tidy.od<odmax)].groupby(['strain','line'])))
fit_df = growth.fitderiv2df(fit)

curve_tidy.to_csv(outdir+'{}_tidydata.csv'.format(fname), index=False)
fit_df.to_csv(outdir+'{}_growthrates.csv'.format(fname), index=False)
