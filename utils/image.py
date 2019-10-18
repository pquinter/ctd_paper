"""
Process fluorescence images
"""

import pandas as pd
import numpy as np
import scipy
import re

import skimage
from skimage import io
import skimage.filters
import skimage.segmentation
import scipy.ndimage
from skimage.external.tifffile import TiffFile

import trackpy as tp
from tqdm import tqdm
import datetime
import pickle
import os
import warnings
import glob

from joblib import Parallel, delayed
import multiprocessing

from utils import particle

###############################################################################
# utilities
###############################################################################

def binary_mask(radius, ndim):
    """
    Elliptical mask in a rectangular array
    """
    points = [np.arange(-rad, rad + 1) for rad in radius]
    if len(radius) > 1:
        coords = np.array(np.meshgrid(*points, indexing="ij"))
    else:
        coords = np.array([points[0]])
    r = [(coord/rad)**2 for (coord, rad) in zip(coords, radius)]
    return sum(r) <= 1

def compute_mass(mask, im):
    """
    Compute sum of masked image intensity
    Return NaN if mask is not of the same shape
    """
    try: return np.sum(mask*im)
    except ValueError: return np.nan

def im_block(ims, cols, norm=True, sort=False):
    """
    Construct block of images

    Arguments
    ---------
    ims: array or iterable of arrays
        images to concatenate in block
    cols: int
        number of columns in image block
    norm: bool
        whether to normalize/scale each image
    sort: False or function
        whether to sort images with output value of function before making block

    Returns
    ---------
    block: array

    """
    if not all(ims[0].shape==im.shape for im in ims):
        warnings.warn('There are frames of different shapes. Resized with black pixels.')
        max_h = max([im.shape[0] for im in ims])
        max_w = max([im.shape[1] for im in ims])
        ims = [resize_frame(im, max_h, max_w) for im in ims]
    ims = np.stack(ims)
    if sort:
        if 'axis' in sort.__code__.co_varnames:# numpy function like max, min, mean
            ims = ims[np.argsort(sort(sort(ims, axis=1), axis=1))]
        else: # corr_widealspot and the likes working on image batches
            ims = ims[np.argsort(sort(ims))]
    if norm:
        ims = normalize(ims)
    # make image block
    nrows = int(ims.shape[0]/cols)
    xdim, ydim = ims.shape[1:]
    block = []
    for c in np.arange(0, cols*nrows, cols):
        block.append(np.hstack(ims[c:c+cols]))
    block = np.vstack(block)
    return block

def z_project(stack, project='max', mindim=2):
    """
    Z-project stack based on maximum value.

    Arguments
    ---------
    stack: array_like.
        input image stack
    project: str
        which value to project: maximum (max), minimum (min)
    mindim: int
        minimum number of dimensions in stack to project, otherwise return stack

    Returns
    ---------
    z_im: z-projection of image
    """

    if stack.ndim<=mindim:
        return stack
    if project == 'max':
        z_im = np.maximum.reduce([z for z in stack])
    if project == 'min':
        z_im = np.minimum.reduce([z for z in stack])
    return z_im

def remove_cs(im, perc=1e-4, tol=3, wsize=10):
    """
    Replace hot pixels or cosmic rays with window median value

    Arguments
    ---------
    im: array
    perc, tol: float
        Percentile and tolerance. Pixels larger than (100-perc)*tol are replaced.
    wsize: int
        size of window

    Returns
    ---------
    frame: array
        image with hot pixels replaced by median value of a 5x5 square
    """
    if im.ndim>3:
        return np.array([remove_cs(i) for i in im])
    frame = im.copy()
    s = wsize//2
    max_allowed = np.percentile(frame, 100-perc)*tol
    # identify where hot pixels are
    cosmic_ix = np.where(frame>max_allowed)
    # change those values to median values of a wsize*wsize window
    for (x,y) in zip(*cosmic_ix):
        frame[x,y] = np.median(frame[x-s:x+s, y-s:y+s])
    return frame

def corr_widealspot(ims, wsize=None, PSFwidth=4.2, n_jobs=multiprocessing.cpu_count()):
    """
    Compute correlation of set of image patches to an ideal spot:
    single point source blurred with gaussian of PSF width
    Useful to filter spurious peaks in low signal to noise ratio images

    Arguments
    ---------
    wsize: int
        size of the window around spot
    PSFwidth: float
        width of the point spread function

    Returns
    ---------
    corrs: array
        correlations with ideal spot
    """
    if wsize is None:
        wsize = ims.shape[-1]
    # Create ideal spot
    idealspot = np.full((wsize,wsize), 0) # zero filled wsize*wsize array
    idealspot[wsize//2,wsize//2] = 1 # single light point source at the center
    idealspot = skimage.filters.gaussian(idealspot, sigma=PSFwidth) # PSF width blur
    # pearson corr on projected im is best, assuming im is centered at potential
    # peak. This is usually eq to max of normalized cross-correlation.
    # Also tried spearman and 3d stacks, slower and not worth it.
    if ims.ndim>2:
        corrs = Parallel(n_jobs=n_jobs)(delayed(np.corrcoef)(idealspot.ravel(), im.ravel())
                           for im in tqdm(ims))
        # retrieve relevant correlation coefficient from matrix
        corrs = np.array([c[1][0] for c in corrs])
        #corrs = np.array([np.corrcoef(idealspot.ravel(), im.ravel())[1][0] for im in ims])
        return corrs
    else:
        return np.corrcoef(idealspot.ravel(), ims.ravel())[1][0]

def regionprops2df(regionprops, props = ('label','area','bbox',
    'intensity_image', 'mean_intensity','max_intensity','min_intensity')):
    """
    Convert list of region properties to dataframe

    Arguments
    ---------
    regionprops: list of skimage.measure._regionprops objects
    props: list of str, properties to store

    Returns
    ---------
    Pandas DataFrame with region properties
    """
    if not isinstance(regionprops, list): regionprops = [regionprops]
    return pd.DataFrame([[r[p] for p in props] for r in regionprops],columns=props)

###############################################################################
# segmentation
###############################################################################

def mask_image(im, im_thresh=None, min_size=15, block_size=101, selem=skimage.morphology.disk(15),
        clear_border=True):
    """
    Create a binary mask to segment nuclei using adaptive threshold.
    Useful to find nuclei of varying intensities.
    Remove small objects, fill holes and perform binary opening (erosion
    followed by a dilation. Opening can remove small bright spots (i.e. “salt”)
    and connect small dark cracks. This tends to “open” up (dark) gaps between
    (bright) features.)

    Arguments
    ---------
    im: array_like
        input image
    thresh: array_like, optional
        thresholded image
    min_size: float or int
        minimum size of objects to retain
    block_size: odd int
        block size for adaptive threshold, must be provided if im_thresh is None

    Returns
    ---------
    im_thresh: array_like
        thresholded binary image 
    """
    if im_thresh is None:
        im_thresh = im>skimage.filters.threshold_local(im, block_size)
    im_thresh = skimage.morphology.remove_small_objects(im_thresh, min_size=min_size)
    im_thresh = ndimage.morphology.binary_fill_holes(im_thresh, morphology.disk(1.8))
    im_thresh = skimage.morphology.binary_opening(im_thresh, selem=selem)
    im_thresh = skimage.morphology.remove_small_objects(im_thresh, min_size=min_size)
    if clear_border:
        im_thresh = skimage.segmentation.clear_border(im_thresh)
    return im_thresh

def label_sizesel(im, im_mask, maxint_lim, minor_ax_lim, 
        major_ax_lim, area_lim, watershed=False):
    """
    Create and label markers from image mask, 
    filter by area and compute region properties

    Arguments
    ---------
    im: array_like
        input image
    im_mask: boolean array
        image mask
    watershed: boolean
        whether to perform watershed on markers
    feature_lim: iterable of two
        minimum and maximum bounds for each feature, inclusive

    Returns
    ---------
    markers: array_like
        labeled image, where each object has unique value and background is 0
    nuclei: list of region props objects
        list of region properties of each labeled object
    """
    markers = skimage.morphology.label(im_mask)
    if watershed:
        # harsh erosion to get basins for watershed
        im_mask_eroded = skimage.measure.label(\
                skimage.morphology.binary_erosion(im_mask,
                selem=skimage.morphology.diamond(8)))
        # watershed transform using eroded cells as basins
        markers = skimage.morphology.watershed(markers,
                im_mask_eroded, mask=im_mask)
    nuclei = skimage.measure.regionprops(markers, im, coordinates='xy')
    # get only markers within area bounds, above intensity thersh and 
    # not oversaturated
    all_labels = np.unique(markers)
    sel_labels = [n.label for n in nuclei if \
                    minor_ax_lim[0] <= n.minor_axis_length <= minor_ax_lim[1]
                    and major_ax_lim[0] <= n.major_axis_length <= major_ax_lim[1] \
                    and area_lim[0] <= n.area <= area_lim[1] \
                    and maxint_lim[0] <= n.max_intensity <= maxint_lim[1]]
    rem_labels = [l for l in all_labels if l not in sel_labels]
    # remove unselected markers
    for l in rem_labels:
        markers[np.where(np.isclose(markers,l))] = 0

    nuclei = [n for n in nuclei if n.label in sel_labels]

    return markers, nuclei

def segment_from_seeds(im, seed_markers, mask_params, dilate=False):
    """
    Segment cells by reconstructing from nuclei markers using watershed

    Arguments
    ---------
    im: array
        image to segment
    seed_markers: array
        integer labeled image of seeds to expand from for watershed
    mask_params: tuple of 3
        min_size, block_size, disk_size for mask_image func
        min size also used to filter markers
    dilate: bool
        whether to perform morphological dilation.
        Useful to keep stuff close to edge

    Returns
    ---------
    markers, seed_markers: array
        only objects with seeds and seeds with objects
    """
    min_size, block_size, disk_size = mask_params
    mask = mask_image(im, min_size=min_size, block_size=block_size,
        selem=skimage.morphology.disk(disk_size))
    if dilate:
    # enlarge mask to keep particles close to edge. Doing this before watershed
    # prevents invasion into other cells and is faster, smart
        mask = skimage.morphology.binary_dilation(mask,
                                            selem=skimage.morphology.disk(10))
    markers = skimage.measure.label(mask)
    # watershed transform using nuclei as basins, also removes cells wo nucleus
    markers = skimage.morphology.watershed(markers,
            seed_markers, mask=mask)

    # remove markers of less than min size
    regionprops = skimage.measure.regionprops(markers)
    regionprops = regionprops2df(regionprops, props=('label', 'area'))
    # remove candidates smaller than min size
    for flabel in regionprops[regionprops.area<min_size].label.values:
        markers[np.where(np.isclose(markers, flabel))] = 0

    # ensure use of same labels for nuclei
    seed_mask = seed_markers>0
    seed_markers  =  seed_mask * markers

    return markers, seed_markers

###############################################################################
# RPB1 nuclear fluorescence Pipeline
###############################################################################

def get_cell_quantiles(im_path, cell_channel=0, nuc_channel=1, quant=(0.5, 0.9), strain_patt=r'.+?_(.+?)_.+'):
    """
    Segment two channel image, one nuclear and one cell channel

    Arguments

    im_path: str
    cell_channel, nuc_channel: int

    Returns

    quant: DataFrame
        with fluorescence and area of segmented cells in each channel
    """

    im_name = re.search(r'.+/(.+)(?:\.tif)$', im_path).group(1)
    strain = re.search(strain_patt, im_name).group(1)

    im = io.imread(im_path).T
    cell_im = im[cell_channel]
    nuc_im = im[nuc_channel]

    # mask cells using adaptive threshold
    mask_cell = mask_image(cell_im, min_size=100, block_size=101,
                selem=skimage.morphology.disk(5), clear_border=True)

    # filter by size and label cells
    maxint_lim, minor_ax_lim, major_ax_lim, area_lim = (100,2e4), (50, 5000), (50, 5000), (100, 10000)
    markers_cell, cell_props = label_sizesel(cell_im, mask_cell,
            maxint_lim, minor_ax_lim, major_ax_lim, area_lim, watershed=True)

    # get individual cell masks from region props
    cell_bbox_mask = [((slice(c.bbox[0], c.bbox[2]),
            slice(c.bbox[1], c.bbox[3])), c.filled_image) for c in cell_props]

    # get cell image in each channel
    cell_ims_cc = [cell_im[bbox]*mask for bbox, mask in cell_bbox_mask] # cell channel
    cell_ims_nc = [nuc_im[bbox]*mask for bbox, mask in cell_bbox_mask] # cell channel

    # get quantiles of each cell
    quant_df = pd.DataFrame()
    for q in quant:
        quant_df['{}_quant_cc'.format(q)] = [np.quantile(c.ravel(), q) for c in cell_ims_cc]
        quant_df['{}_quant_nc'.format(q)] = [np.quantile(c.ravel(), q) for c in cell_ims_nc]

    quant_df['strain'] = strain
    quant_df['im_name'] = im_name
    quant_df['cell_area'] = [c.area for c in cell_props]

    return quant_df

###############################################################################
# PP7/smFISH Pipeline
###############################################################################
def proj_mov(mov_dir, savedir):
    mov_name = re.search(r'.+/(.+)(?:\.tif)$', mov_dir).group(1)
    saveto = '{0}{1}.tif'.format(savedir, mov_name)
    saveto_ref = '{0}{1}_ref.tif'.format(savedir, mov_name)
    # check if projected movie already exists
    if os.path.isfile(saveto):
        warnings.warn('{} projection exists, skipping.'.format(mov_name))
        return None
    mov = io.imread(mov_dir)
    mov_proj = skimage.filters.median(z_project(mov))
    mov_proj = mov_proj.copy()
    mov_proj = remove_cs(mov_proj, perc=0.001, tol=2)
    io.imsave(saveto, mov_proj)
    io.imsave(saveto_ref, mov[10]) # as a reference if needed

def segment_image(im_path, savedir=None, maxint_lim=(100,500),
        minor_ax_lim = (15,500), major_ax_lim=(20,500), area_lim=(1000,5000)):
    im_name = re.search(r'.+/(.+)(?:\.tif)$', im_path).group(1)
    # check if segmented movie already exists
    if savedir is not None:
        seg_im_path = savedir+im_name+'.tif'
        if os.path.isfile(seg_im_path):
            warnings.warn('{} segmentation data exists, skipping.'.format(im_name))
            return None
    im = io.imread(im_path)
    # mask projected image using adaptive threshold to find nuclei of min_size
    m_mask = mask_image(im, min_size=100, block_size=101,
                selem=skimage.morphology.disk(5), clear_border=True)
    # get nuclei markers with watershed transform, bound by area and intensity
    markers_proj, reg_props = label_sizesel(im, m_mask,
        maxint_lim, minor_ax_lim, major_ax_lim, area_lim, watershed=True)
    # convert nuclei region properties to dataframe
    reg_props = regionprops2df(reg_props, props=('label','centroid', 'area'))
    # expand centroid coordinates into x,y cols
    reg_props[['y_cell','x_cell']] = reg_props.centroid.apply(pd.Series)
    reg_props = reg_props.drop('centroid', axis=1)
    # enlarge nuclei markers to keep particles close to nuclear edge
    markers_proj = skimage.morphology.dilation(markers_proj,
            selem=skimage.morphology.disk(5))
    if savedir:
        io.imsave(seg_im_path, markers_proj)
        reg_props.to_csv(savedir+im_name+'.csv', index=False)
    return markers_proj, reg_props

def markers2rois(markers):
    """ Make ROI slice objects from markers """
    # get bounding boxes
    rois = [r.bbox for r in skimage.measure.regionprops(markers, coordinates='xy')]
    # convert to slice objects
    rois = [(slice(xy[0],xy[2]), slice(xy[1],xy[3])) for xy in rois]
    return rois

def load_zproject_STKcollection(load_pattern, savedir=None, n_jobs=6):
    """
    Load collection or single STK files and do maximum intensity projection

    Arguments
    ---------
    load_pattern: str
        pattern of file paths
    savedir: str
        directory to save projected images

    Returns
    ---------
    projected: nd array or np stack
        projected images

    """
    collection = io.ImageCollection(load_pattern, load_func=TiffFile)
    projected = Parallel(n_jobs=n_jobs)(delayed(z_project)(zseries.asarray())
                       for zseries in tqdm(collection))
    #projected = []
    #for zseries in collection:
    #    _im = zseries.asarray()
    #    if _im.ndim>2: _im = z_project(_im)
    #    projected.append(_im)
    if len(collection)>1:
        projected = np.stack(projected)
    else: projected = projected[0]
    if savedir:
        try:
            io.imsave(savedir, projected)
        except:
            warnings.warn("Could not save image. Make sure {} exists".format(savedir))
    return projected

def load_zproject_STKimcollection(load_pattern, savedir=None, n_jobs=6):
    """
    Load collection or single STK files and do maximum intensity projection

    Arguments
    ---------
    load_pattern: str
        pattern of file paths
    savedir: str
        directory to save projected images

    Returns
    ---------
    projected: nd array or np stack
        projected images

    """
    collection = io.ImageCollection(load_pattern, load_func=TiffFile)
    # get names of any images already processed
    strain_dir = re.search(r'(\d+_(:?yQC|TL).+?)_', collection[0].filename).group(1)
    strain_dir = savedir + strain_dir + '/'
    proj_extant = [im.split('/')[-1][:-4] for im in glob.glob(strain_dir+'*tif')]
    # filter out those
    [warnings.warn('{} projection exists, skipping.'.format(p)) for p in proj_extant]
    collection = [im for im in collection if im.filename[:-4] not in proj_extant]
    zproj = lambda imname, im: (imname, z_project(im))
    projected = Parallel(n_jobs=n_jobs)(delayed(zproj)
            (im.filename, im.asarray()) for im in tqdm(collection))
    if savedir:
        try: os.mkdir(strain_dir)
        except FileExistsError: pass
        [io.imsave(strain_dir+name[:-3]+'tif', im) for name, im in projected]
    return projected

def segment_image_smfish(im_path, measure_fluor=None, savedir=None):
    """ Function to process images in parallel with joblib """
    im_name = re.search(r'.+/(.+)(?:\.tif)$', im_path).group(1)
    # check if segmented movie already exists
    if savedir is not None:
        seg_im_path = savedir+im_name+'.tif'
        if os.path.isfile(seg_im_path):
            warnings.warn('{} segmentation data exists, skipping.'.format(im_name))
            return None
    im = io.imread(im_path)
    # For some reason joblib does not work if autof/dapi_channel is an arg,
    # hardcoded below works fine though
    #    autof = im[autof_channel] # autofluorescent channel for cell segmentation
    #    dapi = im[dapi_channel] # Dapi/nuclear channel
    autof = im[3] # autofluorescent channel for cell segmentation
    dapi = im[2] # Dapi/nuclear channel

    # segment nuclei, channel with best signal ================================
    # threshold image with yen filter
    im_thresh = dapi>skimage.filters.threshold_yen(dapi)
    # create mask and label
    nuc_seeds = mask_image(dapi, im_thresh=im_thresh, block_size=101,
                    min_size=100, selem=skimage.morphology.disk(5))
    nuc_seeds = skimage.measure.label(nuc_seeds)
    print('nuclei watershed reconstruction from centers...')
    nuclei_markers, center_markers = segment_from_seeds(dapi, nuc_seeds,
           (100, 101, 10), dilate=False)

    print('cell watershed reconstruction from nuclei...')
    cell_markers, nuclei_markers = segment_from_seeds(autof, nuc_seeds,
           (1000, 151, 15), dilate=True)

    markers = np.stack((cell_markers, nuclei_markers))

    # make nuclei centers dataframe ===========================================
    # get centroids (so all bbox are same size)
    nuc_regionprops = skimage.measure.regionprops(nuclei_markers, dapi)
    cell_regionprops = skimage.measure.regionprops(cell_markers, autof)
    nreg_props = regionprops2df(nuc_regionprops, props=('label', 'centroid', 'area'))
    creg_props = regionprops2df(cell_regionprops, props=('label', 'centroid', 'area'))
    reg_props = nreg_props.merge(creg_props, on='label', suffixes=('_nuc','_cell'))
    # measure fluorescence on channel if necessary (polydT images)
    if isinstance(measure_fluor, int):
        fluor_regprops = skimage.measure.regionprops(cell_markers, im[measure_fluor])
        freg_props = regionprops2df(fluor_regprops, props=('label', 'mean_intensity'))
        reg_props = reg_props.merge(freg_props, on='label')
    reg_props['im_name'] = im_name
    if savedir:
        io.imsave(seg_im_path, markers)
        reg_props.to_csv(savedir+im_name+'.csv', index=False)
    return markers, reg_props

def fluor_fromseg(im_path, seg_im_path, subtract_bg=True):
    """
    Get total cell and nuclear fluorescence for each channel from segmented images
    """

    def get_improps(markers, im, part):
        props = skimage.measure.regionprops(markers, im)
        props = regionprops2df(props, props=('label', 'area','intensity_image'))
        props['intensity_sum'] = props['intensity_image'].apply(np.sum)
        props = props.drop(columns=['intensity_image'])
        props.columns = [l.format(part) for l in\
                            ['label', 'area{}','intensity_sum{}']]
        return props

    im = io.imread(im_path)
    if subtract_bg:
        # remove hot pixels
        im = [remove_cs(_im, perc=0.001) for _im in im]
        im = [skimage.img_as_float(_im) for _im in im]
        # subtract background with strong gaussian blur
        im = [_im-skimage.filters.gaussian(_im, 50) for _im in im]
    seg_im = io.imread(seg_im_path)
    regprops = pd.DataFrame()
    for channel, _im in enumerate(im):
        _regprops = pd.merge(*[get_improps(_markers, _im, part)\
                for part, _markers in zip(['_cell','_nuc'], seg_im)])
        _regprops['channel'] = channel
        regprops = pd.concat((regprops, _regprops), ignore_index=True)
    regprops['im_name'] = re.search(r'.+/(.+)(?:\.tif)$', im_path).group(1)

    return regprops

##############################################################################
# Manual cell selection and segmentation
##############################################################################

def drawROIedge(roi, im, lw=2, fill_val='max'):
    """
    Draw edges around Region of Interest in image

    Arguments
    ---------
    roi: tuple of slice objects, as obtained from zoom2roi
    im: image that contains roi
    lw: int
        edge thickness to draw
    fill_val: int
        intensity value for edge to draw

    Returns
    --------
    _im: copy of image with edge around roi

    """
    _im = im.copy()
    # check if multiple rois
    if not isinstance(roi[0], slice):
        for r in roi:
            _im = drawROIedge(r, _im)
        return _im
    # get value to use for edge
    if fill_val=='max': fill_val = np.max(_im)
    # get start and end of rectangle
    x_start, x_end = roi[0].start, roi[0].stop
    y_start, y_end = roi[1].start, roi[1].stop
    # draw it
    _im[x_start:x_start+lw, y_start:y_end] = fill_val
    _im[x_end:x_end+lw, y_start:y_end] = fill_val
    _im[x_start:x_end, y_start:y_start+lw] = fill_val
    _im[x_start:x_end, y_end:y_end+lw] = fill_val
    return _im

def manual_roi_sel(mov, rois=None, cmap='viridis', plot_lim=5):
    """
    Manuallly crop multiple regions of interest (ROI)
    from max int projection of movie

    Arguments
    ---------
    mov: array like
        movie to select ROIs from
    rois: list, optional
        list of coordinates, to be updated with new selection

    Returns
    ---------
    roi_movs: array_like
        list of ROI movies
    rois: list
        updated list of ROI coordinates, can be used as frame[coords]

    """
    # max projection of movie to single frame
    mov_proj = skimage.filters.median(z_project(mov))
    mov_proj = mov_proj.copy()
    mov_proj = remove_cs(mov_proj, perc=0.001, tol=2)
    if rois: mov_proj = drawROIedge(rois, mov_proj)
    else: rois = []
    fig, ax = plt.subplots(1)
    ax.set(xticks=[], yticks=[])
    i=0
    while True:
        # make new figure to avoid excess memory consumption
        if i>plot_lim:
            plt.close('all')
            fig, ax = plt.subplots(1)
            ax.set(xticks=[], yticks=[])
        plt.tight_layout() # this fixes erratic drawing of ROI contour
        ax.set_title('zoom into roi, press Enter, then click again to add\nPress Enter to finish',
                fontdict={'fontsize':12})
        ax.imshow(mov_proj, cmap=cmap)
        _ = plt.ginput(10000, timeout=0, show_clicks=True)
        roi = zoom2roi(ax)
        # check if roi was selected, or is just the full image
        if roi[0].start == 0 and roi[1].stop == mov_proj.shape[0]-1:
            plt.close()
            break
        else:
            # mark selected roi and save
            mov_proj = drawROIedge(roi, mov_proj)
            rois.append(roi)
            # zoom out again
            plt.xlim(0, mov_proj.shape[1]-1)
            plt.ylim(mov_proj.shape[0]-1,0)
        i+=1
    roi_movs = [np.stack([f[r] for f in mov]) for r in rois]
    return roi_movs, rois

def refine_markers(rois, mov, n_jobs=multiprocessing.cpu_count()):
    """ Make movie mask with segmented regions only in ROIs """

    def mask_frame(f_number, rois, mov):
        """ Create intensity thresholded, limited to ROIs,
        labeled mask for frame """
        # background is 0, start labeled ROIs at 1
        mask = np.zeros_like(mov[0])
        labels_l, nuc_fluor_l = [], []
        for mask_val, _roi in enumerate(rois, start=1):
            _roi_im = mov[f_number][_roi]
            # intensity based segmentation inside ROI
            _roi_mask = mask_image(_roi_im, min_size=100, block_size=101,
                    selem=skimage.morphology.disk(5), clear_border=False)
            # get median nuclear intensity
            labels_l.append(mask_val)
            nuc_fluor_l.append(np.median(_roi_im[_roi_mask]))
            # dilate segmented cells to keep particles on the edge
            _roi_mask = skimage.morphology.dilation(_roi_mask,
                        selem=skimage.morphology.disk(5))
            # limit segmented cell to ROI
            mask[_roi] = mask_val * _roi_mask
        nuc_fluor_df = pd.DataFrame({'roi':labels_l, 'nuc_fluor':nuc_fluor_l})
        nuc_fluor_df['frame'] = f_number

        return (mask, nuc_fluor_df)

    mask_fluor_list = Parallel(n_jobs=n_jobs)(delayed(mask_frame)(f_number, rois, mov)
        for f_number in tqdm(range(len(mov)), desc='generating masks'))
    # unpack mask and nuclear intensity
    mask_mov = np.stack([mi[0] for mi in mask_fluor_list])
    nuc_fluor = pd.concat([mi[1] for mi in mask_fluor_list])

    return mask_mov, nuc_fluor

def get_nuc_fluor(mov_dir, parts, movie=True):
    mov = io.imread(mov_dir)
    if isinstance(parts,str): parts = pd.read_csv(parts)
    mov_name = re.search(r'.+/(.+)(?:\.tif)$', mov_dir).group(1)
    parts = parts[parts.mov_name==mov_name].reset_index(drop=True)
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

    # make markers and get median fluor with manually selected ROIs for each frame
    markers, nuc_fluor = refine_markers(rois, mov)
    nuc_fluor['mov_name'] = mov_name
    return nuc_fluor
