"""
Compute and bootstrap ECDFs
"""
import numpy as np
import pandas as pd
import numba
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

# get number of cores for parallel processing
no_cpus = multiprocessing.cpu_count()

@numba.jit(nopython=True)
def draw_bs_sample(data):
    """
    Draw a bootstrap sample from a 1D data set.
    """
    return np.random.choice(data, size=len(data))

def bs_samples(df, col, no_bs):
    """
    Generate bootstrap samples from 1D datasets in parallel
    """
    bs_samples = Parallel(n_jobs=no_cpus)(delayed(draw_bs_sample)
            (df[col].values) for _ in tqdm(range(no_bs)))
    return bs_samples

def ecdf(data, conventional=False, buff=0.1, min_x=None, max_x=None):
    """
    Computes the x and y values for an ECDF of a one-dimensional
    data set.

    Parameters
    ----------
    data : array_like
        Array of data to be plotted as an ECDF.
    conventional : bool, default False
        If True, generates x,y values for "conventional" ECDF, which
        give staircase style ECDF when plotted as plt.plot(x, y, '-').
        Otherwise, gives points x,y corresponding to the concave
        corners of the conventional ECDF, plotted as
        plt.plot(x, y, '.').
    buff : float, default 0.1
        How long the tails at y = 0 and y = 1 should extend as a
        fraction of the total range of the data. Ignored if
        `coneventional` is False.
    min_x : float, default -np.inf
        If min_x is greater than extent computed from `buff`, tail at
        y = 0 extends to min_x. Ignored if `coneventional` is False.
    max_x : float, default -np.inf
        If max_x is less than extent computed from `buff`, tail at
        y = 0 extends to max_x. Ignored if `coneventional` is False.

    Returns
    -------
    x : array_like, shape (n_data, )
        The x-values for plotting the ECDF.
    y : array_like, shape (n_data, )
        The y-values for plotting the ECDF.
    """

    # Get x and y values for data points
    x, y = np.sort(data), np.arange(1, len(data)+1) / len(data)

    if conventional:
        # Set defaults for min and max tails
        if min_x is None:
            min_x = -np.inf
        if max_x is None:
            max_x = np.inf

        # Set up output arrays
        x_conv = np.empty(2*(len(x) + 1))
        y_conv = np.empty(2*(len(x) + 1))

        # y-values for steps
        y_conv[:2] = 0
        y_conv[2::2] = y
        y_conv[3::2] = y

        # x- values for steps
        x_conv[0] = max(min_x, x[0] - (x[-1] - x[0])*buff)
        x_conv[1] = x[0]
        x_conv[2::2] = x
        x_conv[3:-1:2] = x[1:]
        x_conv[-1] = min(max_x, x[-1] + (x[-1] - x[0])*buff)

        return x_conv, y_conv

    return x, y

def ecdfs_par(data):
    """
    Compute ECDFs from data in parallel
    """
    ecdfs = Parallel(n_jobs=no_cpus)(delayed(ecdf)(sample)
                for sample in tqdm(data))
    return ecdfs

def ecdf_ci(bs_ecdf, ci=99):
    """
    Compute confidence interval for collection of bootstrapped ECDFs
    """
    # Get ECDF quantiles and values
    quants = bs_ecdf[0][1]
    bs_vals = np.stack([_bs[0] for _bs in bs_ecdf])
    # make sure these are cumulative quantiles, i.e. span range [0-1]
    if np.isclose(quants[0], 0.0, atol=1e-2) and np.isclose(quants[-1], 1.0, atol=1e-2):
        warnings.warn('quantiles are not exactly in [0-1] range: top:{}, bottom:{}'.format(quants[0], quants[-1]))
    # get CIs at each cumulative fraction
    ci_high = [np.percentile(_vals, ci) for _vals in bs_vals.T]
    ci_low = [np.percentile(_vals, 100-ci) for _vals in bs_vals.T]
    return quants, ci_high, ci_low

def compute_qvalecdf(model, coef, df):
    """
    Compute cumulative distribution of q-values in number of transcripts
    and store in dataframe with coefficient and model
    """
    no_transcripts, qval_thresh = [], []
    # iterate over 1000 steps in the q-value [0-1]
    for t in np.linspace(0, 1, 1000):
        # count how many transcripts are below qval thresh
        no_transcripts.append(np.sum((df.qval<=t)))
        qval_thresh.append(t)
    # store in dataframe
    qvalecdf = pd.DataFrame()
    qvalecdf['no_transcripts'] = no_transcripts
    qvalecdf['qval_thresh'] = qval_thresh
    qvalecdf['coef'] = coef
    qvalecdf['model'] = model
    return qvalecdf
