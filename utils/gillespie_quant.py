"""
Compute statistics from gillespie simulations
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.signal
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

n_jobs = multiprocessing.cpu_count()

def get_interb_time(groupedby, data, cols=['model','ctd','run']):
    """
    Compute inter burst time for parallel computation
    on groups from groupby object

    groupedby: iterable of names by which data is being grouped
    cols: iterable of column names

    inter_burst: dataframe with between bursts interval and groupedby cols

    """
    # get inter burst time: start of burst - end of previous burst
    inter_burst = data.start_burst.values[1:] - data.end_burst.values[:-1]
    # add interburst time column
    cols.append('inter_burst')
    inter_burst = pd.DataFrame({c:d for c, d in zip(cols, (*groupedby, inter_burst))})
    return inter_burst

def get_quants(model, ctd, data, value, params, dist='geom',
    cols=['theor','data','slope','intercept','r','ks','ks_pval','model','ctd']):
    """
    Get empirical and theoretical quantiles for QQ-plot
    """
    # Get raw data; cast to float for probplot
    values = data[value].values.astype(float)
    # Get sorted data, theor quantiles, fit for qq-plot and store in DF
    _quants, _fit = stats.probplot(values, dist=dist, sparams=params)
    # Kolmogorov–Smirnov test, param needs to be inferred for this to make sense
    _ks = stats.kstest(values, dist, params)
    _quants_df = pd.DataFrame({c:d for c,d in zip(cols, (*_quants, *_fit, *_ks))})
    # Add model and ctd column
    _quants_df = _quants_df.assign(**{c:d for c,d in zip(('model','ctd'), (model,ctd))})
    return _quants_df

def _bs_frac_active(t, samples, no_samples=100, groupby=['ctd','model']):
    """ Parallelizable compute fraction active cells """
    cell_sample = np.random.randint(0, samples.run.max(), no_samples)
    frac_active = samples[(samples.run.isin(cell_sample))&(samples.time==t)]\
        .groupby(groupby).apply(lambda x: np.sum(x.pol_p>0)/len(x))\
        .reset_index(name='active_cells_frac')
    return frac_active

def bs_frac_active(samples, no_bs=1000, t0=10, seed=42, n_jobs=n_jobs, **kwargs):
    """
    Parallel compute bootstrapped fraction of active cells
    samples: df
    no_bs: int
        number of bootstraps
    t0: int
        start time. Though does not make much difference setting>0
    """
    np.random.seed(seed)
    t_sample = np.random.randint(t0, samples.time.max(), no_bs)
    frac_active =  Parallel(n_jobs=n_jobs)(delayed(_bs_frac_active)
                (time, samples, **kwargs) for time in tqdm(t_sample))
    return pd.concat(frac_active, ignore_index=True)

def get_app_bs(model, ctd, run, delta, tr):
    """ Compute apparent burst size from traces """
    # get peak positions
    peaks = scipy.signal.find_peaks(tr, distance=3, prominence=0.5*np.std(tr))
    # get apparent burst size and save with metadata
    burstprops = pd.DataFrame()
    burstprops['app_bs'] = np.array([tr[p] for p in peaks[0]])
    burstprops['model'] = model
    burstprops['ctd'] = ctd
    burstprops['run'] = run
    burstprops['var_p_val'] = delta
    return burstprops
