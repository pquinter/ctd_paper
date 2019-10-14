"""
Detect and wrangle 2D peak data from smFISH and PP7 images
"""
import pandas as pd
import numpy as np
from skimage import io

from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

import trackpy as tp
import re
from utils import image
import os
import glob
import pickle
import warnings

n_jobs = multiprocessing.cpu_count()
##############################################################################
# particle detection
##############################################################################

def locate_par(frame_no, frame, diameter, **kwargs):
    """ Parallelizable particle locate function from trackpy """
    part_df = tp.locate(frame, diameter, **kwargs)
    part_df['frame'] = frame_no
    return part_df

def locate_batch_par(mov, diameter=3, n_jobs=n_jobs, **kwargs):
    """ Parallelized batch particle locate with trackpy """
    parts = Parallel(n_jobs=n_jobs)(delayed(locate_par)(i, frame, diameter, **kwargs)
            for i, frame in tqdm(enumerate(mov), desc='Detecting peaks'))
    parts = pd.concat(parts).reset_index(drop=True)
    return parts

def group_getroi(df, mask, n_jobs=n_jobs):
    """ Parallelized get ROI from mask for grouped dataframe """

    def getroi(group, mask):
        """ Get roi value from single or movie of masks """
        # single mask
        if mask.ndim==2:
            return group.apply(lambda coords: mask[int(coords.y), int(coords.x)], axis=1)
        # or movie of masks
        elif mask.ndim==3:
            return group.apply(lambda coords: mask[int(coords.frame)][int(coords.y), int(coords.x)], axis=1)
        else: raise ValueError('Mask of wrong shape: {}'.format(mask.shape))

    app_list = Parallel(n_jobs=n_jobs)(delayed(getroi)(group, mask)
        for name, group in tqdm(df.groupby('frame'), desc='masking dataframe'))

    return pd.concat(app_list).values

def locate_batch(mov_dir, movie=True):

    ################################################################################
    # Load data
    ################################################################################

    mov = io.imread(mov_dir)
    mov_name = re.search(r'.+/(.+)(?:\.tif)$', mov_dir).group(1)
    laser_power = int(re.search(r'(\d{3})u.*?480', mov_name).group(1))

    # get ROIs
    if movie:
        markers_dir = '../output/pipeline/segmentation/{}.tif'.format(mov_name)
        markers = io.imread(markers_dir)
    else:
        markers_dir = '../output/pipeline_snapshots/segmentation/{}.tif'.format(mov_name)
        markers = io.imread(markers_dir)
        # make a one movie frame with image for everything else to work
        mov = np.stack([mov])
    rois = image.markers2rois(markers)

    ################################################################################
    # Detect transcription sites and assign cell label
    ################################################################################

    # identify transcription particles in parallel
    locate_kwargs = {'minmass':25, 'characterize':True, 'noise_size':1, 'smoothing_size':5}
    parts = locate_batch_par(mov, diameter=3, **locate_kwargs)
    # make markers and get median fluor with manually selected ROIs for each frame
    markers, nuc_fluor = image.refine_markers(rois, mov)
    # assign roi number to each spot; parallelized is much faster
    parts['roi'] = group_getroi(parts, markers)
    # and metadata
    parts['laser_power'] = laser_power
    parts['mov_name'] = mov_name
    # add median nuclear fluorescence
    parts = pd.merge(parts, nuc_fluor, on=['roi','frame'])
    # raw mass to nuclear fluorescence ratio
    parts['mass_norm'] = parts.raw_mass.values / parts.nuc_fluor.values
    parts['strain'] = parts.mov_name.apply(lambda x: re.search(r'_(((yQC)|(TL)).+?)_', x).group(1))

    return parts

def locate_batch_smfish(im_path, seg_dir='../output/pipeline_smfish/segmentation/', movie=True):

    ################################################################################
    # Load data
    ################################################################################

    im = io.imread(im_path)
    mov_name = re.search(r'.+/(.+)(?:\.tif)$', im_path).group(1)
    markers_dir = '{}/{}.tif'.format(seg_dir, mov_name)
    markers = io.imread(markers_dir)
    #seg_dir = '../output/pipeline_smfish/segmentation/{}.csv'.format(mov_name)
    #seg_df = pd.read_csv(seg_dir)

    im = io.imread(im_path)
# have to hardcode channels below, for some reason kwargs dont work with joblib
#    fish = im[:,:,0] # smFISH channel
#    autof = im[:,:,1] # autofluorescent channel for cell segmentation
#    dapi = im[:,:,2] # Dapi/nuclear channel

    #fish = im[1] # smFISH gal3 channel
    fish = im[0] # smFISH gal10 channel
    autof = im[3] # autofluorescent channel for cell segmentation
    dapi = im[2] # Dapi/nuclear channel



    ################################################################################
    # Detect transcription sites and assign cell label
    ################################################################################

    # identify transcription particles in parallel
    locate_kwargs = {'minmass':25, 'characterize':True, 'noise_size':1, 'smoothing_size':5}
    parts = tp.locate(fish, diameter=3, **locate_kwargs)
    # assign roi number to each spot; parallelized is much faster
    parts['cell'] = parts.apply(lambda coords: markers[0][int(coords.y), int(coords.x)], axis=1)
    parts['nucleus'] = parts.apply(lambda coords: markers[1][int(coords.y), int(coords.x)], axis=1)
    parts['mov_name'] = mov_name
    parts['strain'] = parts.mov_name.apply(lambda x: re.search(r'_(((yQC)|(TL)|(hC)|(wC)).+?)_', x).group(1))
    # add frame for everything else to work
    parts['frame'] = 0
    # add pid
    parts['particle'] = parts.index
    parts['pid'] = parts['mov_name']+'_'+parts['particle'].apply(str)+'_'+parts['frame'].apply(str)

    return parts


def cleanup_track(parts, traj_len_thresh=1, part_per_frame=1, remove_hotpixels=10):
    # make sure df belongs to single movie
    assert len(parts.mov_name.unique())==1
    # remove peaks outside cells
    parts_filt = parts[parts.roi>0]
    # remove hot pixels
    if remove_hotpixels:
        parts_filt = parts_filt[parts_filt.mass<parts_filt.mass.mean()*remove_hotpixels]
    # keep two brightest spots per ROI per frame; 'tail' method allows to keep top N
    parts_filt = parts_filt.sort_values('mass').groupby(['roi','frame']).tail(part_per_frame+1).reset_index(drop=True)
    # link particles with search range of 5 px and memory of 1 frame
    parts_filt = tp.link_df(parts_filt, 5, memory=1)
    # compute trajectory length for each particle
    traj_len = parts_filt.groupby('particle').count().reset_index()[['particle','x']]
    traj_len.columns = ['particle','traj_len']
    parts_filt = pd.merge(parts_filt, traj_len, on='particle')
    # remove trajectories too long to be true
    parts_filt = parts_filt[parts_filt.traj_len<parts_filt.traj_len.mean()*10]
    # filter spurious spots by trajectory length; faster than tp.filter_stubs
    parts_filt = parts_filt[parts_filt.traj_len>traj_len_thresh]
    # keep only one spot per ROI per frame: brightest or with longest traj
    parts_filt = parts_filt.sort_values(['mass','traj_len']).groupby(['roi','frame']).tail(part_per_frame).reset_index(drop=True)
    # assign unique particle id
    parts_filt['pid'] = parts_filt['mov_name']+'_'+parts_filt['particle'].apply(str)+'_'+parts_filt['frame'].apply(str)
    return parts_filt

##############################################################################
# particle image patches retrieval
##############################################################################
def get_bbox(center, size=9, im=None, return_im=True, pad=2, mark_center=False,
        size_z=None):
    """
    Get square bounding box around center

    Arguments
    ---------
    center: tuple
        x, y coordinates
    size: int
        size of the bounding box in pixels.
        Must be an odd number, or it will be round up.
    im: 2D array
        image to extract window from
    return_im: bool
        whether to return the bbox image or just coordinates
    pad: int
        size of padding around returned image
    mark_center: bool
        whether to draw a cross to mark image center

    Returns
    ---------
    im_bbox or bbox: array or numpy slice
        image bounding box

    """
    x, y = center
    x, y = int(x), int(y)
    # get bbox coordinates
    # if can't get full bbox, get partial and avoid negative values
    s = size//2
    ys = np.max((0, y-s))
    xs = np.max((0, x-s))
    bbox = np.s_[ys:y+s+1, xs:x+s+1]
    if return_im:
        # get bbox image
        im_bbox = im[bbox].copy()
        if mark_center:
            im_bbox[s-1:s+2,s] = np.min(im_bbox)
            im_bbox[s,s-1:s+2] = np.min(im_bbox)
        if pad:
            im_bbox = np.pad(im_bbox, pad, 'constant', constant_values=np.min(im_bbox))
        return im_bbox
    else: return bbox

def get_bbox3d(center, size=9, im=None, return_im=True, pad=2, mark_center=False, size_z=None):
    """
    Get square bounding box around center

    Arguments
    ---------
    center: tuple
        x, y coordinates
    size: int
        size of the bounding box in pixels in xy dimensions.
        Get all z-stacks by default.
    im: 2D array
        image to extract window from
    return_im: bool
        whether to return the bbox image or just coordinates
    size_z: 'Full' or None
        If 'Full', get all z planes. Otherwise get same number as xy size.

    Returns
    ---------
    im_bbox or bbox: array or numpy slice
        image bounding box

    """
    x, y, z = center
    x, y, z = int(x), int(y), int(z)
    # get bbox coordinates
    s = size//2
    if size_z=='Full':
        bbox = np.s_[:,y-s:y+s+1, x-s:x+s+1]
    else:
        bbox = np.s_[z-s:z+s+1,y-s:y+s+1, x-s:x+s+1]
    if return_im:
        # get bbox image
        im_bbox = im[bbox].copy()
        if pad:
            im_bbox = np.pad(im_bbox, pad, 'constant', constant_values=np.min(im_bbox))
        return im_bbox
    else: return bbox

def get_batch_bbox(bbox_df, ims_dict, size=9, movie=False,
        pad=0, mark_center=0, im3d=False, size_z=None, coords_col=None,
        frame_col='frame', imname_col='imname'):
    """
    Get square bounding boxes in batch around center coords from dataframe

    Arguments
    ---------
    bbox_df: DataFrame
        df with object coordinates and corresponding image name.
        Must contain columns ['x','y','imname'] and 'frame' for movie
        If only single movie is specified, 'imname' col is not used.
    ims_dict: dictionary of images or movie
        dict of images. Keys must be the same as `imname`s in df
        can also be single movie array
    size: int
        xy size of bounding box to get from image around object coordinates.
        If 3d, size in z is also this, unless Full is specified
    movie: bool
        True if ims_dict contains movies
    pad: int
        Padding to add to each image.
    mark_center: bool
        whether to mark center of each image
    im3d: bool
        whether image is 3D or not
    size_z: 'Full' or None
        If 'Full', get all z planes. Otherwise get same number as xy size.
    coords_col: list, str or None
        name of column(s) containing coordinates

    Returns
    ---------
    ims_array: stack or list
        stack of images if possible. If different sized arrays, returns list.

    """

    if im3d:
        _getbboxfunc = get_bbox3d
        coords = ['x','y','z']
    else:
        _getbboxfunc = get_bbox
        coords = ['x','y']
    if coords_col:
        coords = coords_col
    if movie:
        if type(ims_dict) is not dict:
            warnings.warn("""
                            No image dictinary provided.
                            Assuming all images come from single movie.""")
            ims_df = bbox_df.apply(lambda x: [_getbboxfunc(x[coords], size,
                    ims_dict[int(x[frame_col])], mark_center=mark_center, pad=pad,
                    size_z=size_z)], axis=1)
        else:
            ims_df = bbox_df.apply(lambda x: [_getbboxfunc(x[coords], size,
                    ims_dict[x[imname_col]][int(x[frame_col])], mark_center=mark_center,
                    pad=pad, size_z=size_z)], axis=1)
    else:
        ims_df = bbox_df.apply(lambda x: [_getbboxfunc(x[coords], size,
                ims_dict[x[imname_col]], mark_center=mark_center, pad=pad, size_z=size_z)], axis=1)
    try:
        return np.stack([i[0] for i in ims_df])
    except ValueError:
        warnings.warn('Encountered images of different Z depths. Returning list of arrays.')
        return [i[0] for i in ims_df]

def get_patches(mov_path, patch_dir, parts, movie=True, get_bp=True,
        n_jobs=n_jobs):
    """ Load movie, smooth and retrieve spot patches from raw and smooth """
    # default params
    radius, noise_size, smoothing_size, threshold, im_size = 1.5, 1, 5, 1, 15
    # check if already analyzed this movie
    print('Analyzing {0}'.format(mov_path))
    mov_name = mov_path.split('/')[-1][:-4] # load movie to memory
    fname_ims = '{}/{}_15x15spots.p'.format(patch_dir, mov_name)
    if os.path.isfile(fname_ims):
        print('already analyzed this one, moving on...')
        return None
    raw_mov = io.imread(mov_path)
    if not movie: raw_mov = np.stack([raw_mov])
    if movie=='smfish':
        #raw_mov = np.stack([raw_mov[:,:,0]])
        #raw_mov = np.stack([raw_mov[1]]) #gal3
        raw_mov = np.stack([raw_mov[0]]) # gal10
    if get_bp:
        bp_mov = np.stack(Parallel(n_jobs=n_jobs)(delayed(tp.bandpass)(f, noise_size, smoothing_size, threshold)
                for f in tqdm(raw_mov, desc='applying bandpass filter')))
    # get particles of current movie
    print('retrieving spot images...')
    _parts = parts[parts.mov_name==mov_name]
    if len(_parts)<1: return None
    raw_ims = get_batch_bbox(_parts, raw_mov, size=im_size, movie=True)
    # delete movies from memory! (assign None) otherwise computer breaks...
    raw_mov = None
    if get_bp:
        bp_ims = get_batch_bbox(_parts, bp_mov, size=im_size, movie=True)
        bp_mov = None
    else: bp_ims = None
    print('pickling...')
    with open(fname_ims, 'wb') as f:
        pickle.dump(_parts.pid.values, f)
        pickle.dump(raw_ims, f)
        pickle.dump(bp_ims, f)
    raw_ims, bp_ims = None, None
    print('done')

##############################################################################
# particle annotation and classification
##############################################################################

def load_patches(spots_dir, bp=True, shape='auto', filter_trunc=True):
    """
    Load spot image patches
    spots_dir: str
        directory containing all pickled spot images by movie
    bp: boolean
        whether pickled spots includes bandpassed image set
    shape: tuple, auto
        tuple with shape of image patch to check for
        auto: infer most common shape from images
    filter_trunc: bool
        whether to filter out images truncated images
    """
    if '.p' not in spots_dir:
        spots_path = glob.glob(spots_dir+'/*.p')
    else: spots_path = [spots_dir]
    pids_all, rawims_all, bpims_all = [],[],[]
    for _path in spots_path:
        mov_name = _path.split('/')[-1][:-13]
        try:
            with open(_path, 'rb') as f:
                pids_all.extend(pickle.load(f))
                rawims_all.extend(pickle.load(f))
                if bp: bpims_all.extend(pickle.load(f))
        except EOFError:
            pids_all, rawims_all, bpims_all = [],[],[]
            with open(_path, 'rb') as f:
                rawims_all.extend(pickle.load(f))
                if bp: bpims_all.extend(pickle.load(f))
    if filter_trunc:
        if shape=='auto':
            # get most common (median) image shape
            shape_all = np.array([im.shape[-2:] for im in rawims_all])
            shape = tuple([np.median(shape_all[:,i]).astype(int) for i in range(shape_all.ndim)])
        # filter out all truncated images
        is_full = [im.shape==shape for im in rawims_all]
        if bp:
            rawims_all, bpims_all, pids_all = [np.array(arr)[is_full]\
                                for arr in (rawims_all, bpims_all, pids_all)]
        else:
            rawims_all, pids_all = [np.array(arr)[is_full]\
                                for arr in (rawims_all, pids_all)]
    # Turn them into concatenated arrays
    try:
        rawims_all, bpims_all = [np.stack(a) for a in (rawims_all, bpims_all)]
    except ValueError: pass
    return pids_all, rawims_all, bpims_all

def get_manual_labels(sampled_frames, mov_name, mov_labeled, parts, brush_val=0, verbose=True):
    """
    Get which spots were labeled as True in sample movies edited in Fiji
    True spots have a paintbrush on top

    Arguments
    ---------

    sampled_frames: DataFrame
        contains movie name and original frame number; index is sample movie frame
    mov_name: str
    mov_labeled: array
    parts: DataFrame
        whole particle dataframe to add labels to
    brush_val: int
        Fiji paintbrush value used to label spots (black or 0 is easy to detect)
    verbose: bool
        whether to print how many spots were labeled as True

    Returns
    ---------
    _parts: DataFrame
        copy of slice of parts DataFrame with manual labels
    """
    # get frame numbers from sample movie and corresponding original movie
    sample_fno = sampled_frames.index.values
    orig_fno = sampled_frames.frame.values
    # get relevant particle data
    _parts = parts[(parts.mov_name==mov_name)&(parts.frame.isin(orig_fno))].copy()
    # map sampled frames to original movie frame number
    rep_values = dict(zip(orig_fno, sample_fno))
    _parts['sframe'] = _parts.frame.map(rep_values)
    # get intensity value from labeled movie
    _parts['int_value'] = _parts.apply(lambda coords:\
            mov_labeled[int(coords.sframe), int(coords.y), int(coords.x)], axis=1)
    # get label (spots were labeled in Fiji, by drawing a paintbrush)
    _parts['manual_label'] = _parts.int_value.values == brush_val
    if verbose:
        man_count = _parts.manual_label.sum()
        print("For Movie {0} {1} were labeled".format(mov_name, man_count))
    del _parts['int_value']
    return _parts

def get_cix(start, stop, step):
    """ Generator of two by two indices to iterate in chunks of size `step` """
    return tqdm(zip(np.arange(start, stop, step), np.arange(start+step, stop+step, step)))

def predict_prob(df, feat, clf_scaler_path, chunk_size=1e4,
        n_jobs=n_jobs):
    """
    Classify spots from dataframe

    Arguments
    ---------
    df: dataframe
        Must contain columns `feat`
    feat: list
        features to be classified (e.g. mass and correlation_with_idealspot)
    clf_scaler_path: str
        path to pickled classifier and scaler objects
    chunk_size: int
        predict probability of slices this size
    n_jobs: int
        number of jobs for parallel processing

    Returns
    -------
    prob_pred: array
        predicted probabilities of being True
    """

    with open(clf_scaler_path, 'rb') as f:
        clf = pickle.load(f)
        scaler = pickle.load(f)
    # scale features like training data
    feat_scaled = scaler.transform(df[feat].values)
    # predict probability in chunks, otherwise fails
    # Defualt predicted label is just thresholded by >=50% prob, don't do that
    chunk_size = int(chunk_size)
    prob_pred = Parallel(n_jobs=n_jobs)(delayed(clf.predict_proba)(feat_scaled[t_:_t])
            for t_, _t in tqdm(get_cix(0, len(feat_scaled), chunk_size)))
    prob_pred = np.concatenate(prob_pred)
    return prob_pred[:,1] # prob of being True, first col is 1-prob

def impute_coords(coords_df, cols=['mov_name','roi','particle','frame','x','y']):
    """
    Add and fill in missing frames with coordinates of last observed particle
    with a forward fill.
    If frame 0 is absent, back fill that one first, then forward fill all.

    Arguments
    ---------
    coords_df: dataframe
        Most likely df from groupby object. Must contain specified columns.
    cols: list
        columns to preserve and fill if NaNs present

    Returns
    -------
    coords_df: dataframe
        with rows including every frame in the range [0-max(coords_df.frame)]
    """
    if 'frame' not in cols: cols+='frame'
    coords_df = coords_df[cols]
    # make dataframe with complete number of frames
    allframes = pd.DataFrame(np.arange(0, coords_df.frame.max()+1, dtype=int), columns=['frame'])
    # add to dataframe, fill with nans when frame is not present
    coords_df = pd.merge(coords_df, allframes, on='frame', how='outer')
    coords_df = coords_df.sort_values('frame').reset_index(drop=True)
    # fill first frame if missing to be able to do forward fill
    if any(coords_df.loc[coords_df.frame==0, cols].isnull()):
        coords_df.loc[coords_df.frame==0, cols] =\
                coords_df.fillna(method='bfill').loc[coords_df.frame==0, cols]
    # fill the rest
    coords_df = coords_df.fillna(method='ffill')
    return coords_df
