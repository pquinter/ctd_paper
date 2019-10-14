"""
Stochastic simulation of transcription model

Columns of states are: PIC, pol, pol_p
-PIC: preinitiation complex, proteins required to recruit first polymerase
      can only be removed when there is no polymerase bound
-pol: promoter bound unphosphorylated polymerase
-pol_p: phosphorylated transcribing polymerase, no longer promoter bound
**Notice phosphorylation is the only way to remove promoter bound polymerase

Reactions are:
Proteins(TF, Mediator) <> PIC <> PIC-PolII(n) > PolII_p(Transcribing) > PolII_p(free)
CTD length is (0,1) and promotes
          -Pol II binding to PIC
          -Pol II binding to PIC-PolII
But slows down transition to PolII_p (transcribing)
"""

import numpy as np
import pandas as pd
import numba
from tqdm import tqdm
from utils import gillespie_burst
import multiprocessing

# number of jobs for parallel processing; also sets number of simulations
# can set manually in the `Parameters` section
n_jobs = multiprocessing.cpu_count()

@numba.njit
def propensity(population, t, alpha, gamma, beta, epsilon, phi, delta):
    """
    Returns an array of propensities given a set of parameters
    and an array of populations, allowing for time (t) dependence.
    **Allow for multiple polymerases bound to PIC at a time if epsilon>0
    """
    # Unpack population
    # Preinitiation complex, promoter (PIC or CTD) bound Pol II, phospho PolII (transcribing)
    PIC, pol, pol_p = population

    return np.array([alpha * int(PIC<1), # Make a PIC, only one allowed
                     gamma * PIC * int(pol<1), # Remove PIC, only if there is no Pol II
                     beta * PIC * int(pol<1) + int(pol>0) * epsilon, # recruit Pol II, beta for PIC; epsilon for CTD-CTD interaction
                     phi * pol, # Remove Pol II by phosphorylation
                     delta * pol_p]) # Degrade phospho Pol II: finish transcript

state_update = np.array([[1, 0, 0],   # Make PIC
                          [-1, 0, 0],  # Degrade PIC
                          [0, 1, 0],   # Bind Pol II to PIC and PIC-PolII complex
                          [0, -1, 1],  # Remove promoter Pol II by phosphorylation, gain phospho pol II
                          [0, 0, -1]]) # Remove phospho Pol II

###############################################################################
# Parameters

n_ctd = 9 # number of CTD lengths
ctd_len = np.linspace(0.1, 0.9, n_ctd) # Array of CTD lengths; (0,1) for simplicity
alpha = 1  # PIC formation rate
gamma = 3 # PIC degradation rate
beta = 30  # Pol II binding rate, proportional to CTD length
epsilon = 10 # Pol II binding rate to PIC-PolII complex, via CTD-CTD interaction
phi = 100  # Phosphorylation rate, inversely proportional to CTD length
delta = 1  # Removal of phospho PolII , i.e. elongation rate; Only useful for visualizing traces

size, n_threads, time_res = 50, n_jobs, 101 # num of samples, jobs for parallel processing, and timepoints
time_points = np.linspace(0, 50, time_res) # time samples
population_0 = np.array([0, 0, 0], dtype=int) # initial conditions

###############################################################################
# Burst definition of two types, first is cannonical

# Indicates transition into, out of and increasing a burst, and burst state
# 1) Burst is during ON state: TFs->PIC->TFs. All models have bursts.
burst_kwargs = {'burst_in':0, 'burst_out':1, 'burst_gain':3, 'burst_state':0}
# 2) Burst is during PIC-Pol state: PIC->PIC-Polb->PIC, i.e. burst_out==gain
# No bursts in one pol model, hence not great for consistency with literature
#burst_kwargs = {'burst_in':2, 'burst_out':3, 'burst_gain':3, 'burst_state':1}

###############################################################################
# Models

# onepol: Only single polymerase is allowed to bind the PIC at a time
#   achieved by setting epsilon=0
# manypol: Multiple polymerases can bind at rate epsilon after initial
#   recruitment at rate beta
# manypol_FUS: Simulates rescue with fusion to a self-interacting protein
#   that keeps epsilon and phi constant
models = {'onepol':(alpha, gamma, beta, 0, phi, delta),
          'manypol':(alpha, gamma, beta, epsilon, phi, delta),
          'manypol_FUS':(alpha, gamma, beta, 5, 10, delta)}
###############################################################################

# DataFrame for gillespie samples and burst properties
cols = ['PIC','pol','pol_p','run','time', 'ctd', 'model']
samples = pd.DataFrame(columns=cols)
cols_burstp = ['start_burst','end_burst','burst_size','run','ctd','model','var_p','var_p_val']
burstprops = pd.DataFrame(columns=cols_burstp)

# Whether to explore a parameter array
explore_param = False

if explore_param:
    # build parameter array
    var_names = ('alpha','gamma','beta','epsilon','phi')
    arr = np.logspace(-1, 3, 5)
    param_arr = {vname:arr for vname in var_names}
else : param_arr = {'alpha':[alpha]} # fake parameter array to keep single alpha

for model in tqdm(models):
    for var_name in tqdm(param_arr):
        # in one polymerase model epsilon=0 always
        if var_name=='epsilon' and model=='onepol': continue

        for var_p in tqdm(param_arr[var_name], desc='param array: {}'.format(var_name)):
            for ctd in ctd_len:

                # Unpack model parameters
                alpha, gamma, beta, epsilon, phi, delta = models[model]
                # assign new value to variable parameter
                locals()[var_name] = var_p

                # Update CTD dependent parameters
                beta_ctd = beta * ctd       # PIC binding rate
                epsilon_ctd = epsilon * ctd # CTD-CTD binding rate
                phi_ctd = phi * (1-ctd)     # CTD phosphorylation rate
                phi_ctd = phi
                if model == 'manypol_FUS': # epsilon and phi stay constant
                    epsilon_ctd, phi_ctd = epsilon, phi

                args = (alpha, gamma, beta_ctd, epsilon_ctd, phi_ctd, delta)
                _samples, _burstprops = gillespie_burst.gillespie_ssa(propensity,
                                                    state_update,
                                                    population_0,
                                                    time_points,
                                                    size=size,
                                                    args=args,
                                                    **burst_kwargs,
                                                    n_threads=n_threads,
                                                    progress_bar=False)

                # Unpack burst properties
                # structure of _burstprops is:
                # n_threads * size * (start_burst, end_burst, burst_size)
                # unpack runs of burst properties
                _burstprops = [_props for _thread in _burstprops for _props in _thread]
                # get run number of each burst prop array and put together
                _burst_runs = [np.full(len(p[0]), i) for i,p in enumerate(_burstprops)]
                _burstprops = [p for i in range(len(_burstprops))\
                            for p in zip(*_burstprops[i], _burst_runs[i])]
                _burstprops = pd.DataFrame(_burstprops, columns=cols_burstp[:4])
                _burstprops['ctd'] = ctd
                _burstprops['model'] = model
                _burstprops['var_p'] = var_name
                _burstprops['var_p_val'] = var_p
                burstprops = pd.concat((burstprops, _burstprops), ignore_index=True)

                # Unpack samples into dataframe
                _samples = pd.concat([pd.DataFrame(s, columns=cols[:3]) for s in _samples])
                _samples['run'] = np.repeat(np.arange(size*n_threads), time_res)
                _samples['time'] = np.tile(np.arange(time_res), size*n_threads)
                _samples['ctd'] = ctd
                _samples['model'] = model
                _samples['var_p'] = var_name
                _samples['var_p_val'] = var_p

                samples = pd.concat((samples, _samples), ignore_index=True)

                # clear memory
                _samples, _burstprops = None, None

# for some reason numeric columns are not numeric...
for p in cols[:3]: samples[p] = pd.to_numeric(samples[p])
burstprops['burst_size'] = pd.to_numeric(burstprops['burst_size'])

# create metadata string with models and parameter values
metadata='# Parameters for Gillespie simulation\n'
for model_params in [[m]+list(models[m]) for m in models]:
    metadata +=\
"""# Model:{}
# alpha={}, gamma={}, beta={}, epsilon={}, phi={}, delta={}
""".format(*model_params)

# Save Gillespie samples and burst properties
df2save = ((samples, burstprops),('samples','burstprops'))
# write to file with parameter metadata
for df, name in zip(*df2save):
    with open('../output/simulation/gillespie_{}.csv'.format(name), 'w') as f:
        f.write(metadata)
        df.to_csv(f, index=False)
