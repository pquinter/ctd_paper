"""
Fetch particle image patches from images or moveis for characterization and classification
"""
import glob
import multiprocessing

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from utils import particle

part_dir = '../output/pipeline/particles/parts_filtered.csv'
data_dir = '../data/2019_galinducedt0/*tif'
spots_path = '../output/pipeline/spot_images/imputed_rawonly'
# Number of cores for parallel processing.
# Each iteration needs a ton of memory, beware of increasing number of cores
n_cores = multiprocessing.cpu_count()//2

movpath_list = glob.glob(data_dir)
parts = pd.read_csv(part_dir)
# Process movies in parallel
Parallel(n_jobs=n_cores)(delayed(particle.get_patches)(mov_path, spots_path, parts, get_bp=False)
                    for mov_path in tqdm(movpath_list))
