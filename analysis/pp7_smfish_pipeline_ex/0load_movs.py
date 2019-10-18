"""
Batch maximum intensity z-projection of movies
"""
import glob
from os import listdir

from joblib import Parallel, delayed
from tqdm import tqdm
from utils import image

datadir = '/Volumes/PQdata/data/PP7/2019/unloaded/'
savedir = '../data/2019_galinducedt0/'
dirs_toload = [datadir + d + '/' for d in listdir(datadir) if 'DS' not in d]
mov_names = [glob.glob(d + '*.nd')[0].split('/')[-1][:-3] for d in dirs_toload]
proj = Parallel(n_jobs=6)(
    delayed(image.load_zproject_STKcollection)(mov_dir + '*STK', savedir + name + '.tif')
    for mov_dir, name in tqdm(zip(dirs_toload, mov_names)))
