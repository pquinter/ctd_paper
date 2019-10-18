"""
Segment movies using time projection
"""
import glob

from joblib import Parallel, delayed
from tqdm import tqdm

from utils import image

data_dir = '../data/2019_galinducedt0/'
proj_dir = '../output/pipeline/time_proj/'
seg_dir = '../output/pipeline/segmentation/'
movpath_list = glob.glob(data_dir + '*tif')

# Project movies (just returns None if projected movie exists)
Parallel(n_jobs=12)(delayed(image.proj_mov)(movpath, proj_dir)
                    for movpath in tqdm(movpath_list))
# Get path to projected movies
movproj_plist = glob.glob(proj_dir + '*tif')
movproj_plist = [p for p in movproj_plist if 'ref' not in p]
# Segment cells (skips already segmented movies)
segmentation = Parallel(n_jobs=12)(
    delayed(image.segment_image)(movproj_path, seg_dir)
    for movproj_path in tqdm(movproj_plist))
