"""
Convenience functions to wrangle OD600 growth data and estimate growth rates with Peter Swain's software
"""

import pandas as pd
import numpy as np
from utils.fitderivpackage103 import fitderiv

def get_CTDr(strain):
    """
    Map CTD repeat number to strain name
    """
    CTDr_dict = {'TL47':26, 'yQC21':26, 'yQC5':14, 'yQC6':12, 'yQC7':10,
            'yQC62':10, 'yQC63':9, 'yQC64':8, 'yQC15':10, 'yQC16':10}
    # split line number from unique strain ID
    if '.' in strain: strain = strain.split('.')[0]
    elif '_' in strain: strain = strain.split('_')[0]
    try: return CTDr_dict[strain]
    except KeyError: return 0

def tidy_growthcurve(data_path, layout_path, dt=15, temp=30):
    """
    Put growth curve data in tidy dataframe
    data_path, layout_path: str
        path to OD data and plate layout
    """
    curve = pd.read_csv(data_path, comment='#')
    layout = pd.read_csv(layout_path, comment='#', index_col='row')

    # check temperature is close to 30 at all time points
    mean_temp = pd.to_numeric(curve.dropna(how='any')['Temperature']).mean()
    assert np.isclose(mean_temp, temp, rtol=0.1)
    print('Mean temperature is {t:0.3f} C'.format(t=mean_temp))

    # make tidy
    curve_tidy = pd.melt(curve, id_vars=['Time', 'Temperature'], var_name='well', value_name='od')
    # get labels from plate layout
    labels = {well:layout[well[1:]].loc[well[0]] for well in curve_tidy['well']}
    curve_tidy['strain'] = curve_tidy.well.map(labels)
    # split strain and line number
    curve_tidy[['strain','line']] = curve_tidy.strain.str.split('.', expand=True)
    # fill in missing line numbers with 'x' to avoid groupby issues with 'None'
    curve_tidy['line'] = curve_tidy.line.fillna('x')
    # delete empty wells and times
    curve_tidy = curve_tidy.dropna(subset=['strain']).dropna(axis=0, how='all')
    # add number of CTDr
    curve_tidy['CTDr'] = curve_tidy.strain.map(get_CTDr)
    # add Time in minutes
    for well in curve_tidy.well.unique():
        curve_tidy.loc[curve_tidy.well==well, 'Time'] = np.arange(0, np.sum(curve_tidy.well==well)*dt, dt)
    curve_tidy = curve_tidy.dropna(subset=['od']).dropna(axis=0, how='all')
    return curve_tidy

def fitderiv_par(strain, od, dt=15):
    """ Convenient function for parallel calculation of growth rate """

    # reset time to drop nans
    for well in od.well.unique():
        od.loc[od.well==well, 'Time'] = np.arange(0, np.sum(od.well==well)*dt, dt)

    # turn OD replicates into array
    od = od.pivot(index='Time', columns='well', values='od').dropna()
    od = np.array(od.values, dtype='float64')

    # create time array
    t = np.arange(0, od.shape[0]*dt, dt)
    # compute doubling times
    q = fitderiv.fitderiv(t, od)
    return strain, q

def fitderiv2df(fitlist):
    """ Convert list of (strain, fit) pairs to dataframe """
    dict_list = []
    for strain, _dt in fitlist:
        dt_dict = _dt.ds
        # get back strain and label
        dt_dict['strain'] = strain[0]
        dt_dict['line'] = strain[1]
        # get std deviation
        dt_dict['stdev'] = np.sqrt(dt_dict['inverse max df var'])
        dict_list.append(dt_dict)
    return pd.DataFrame(dict_list)

