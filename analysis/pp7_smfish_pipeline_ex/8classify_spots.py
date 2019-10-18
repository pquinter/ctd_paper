"""
Compute GPC probability of particle image patches
"""
import pandas as pd
from utils import particle

part_dir = '../output/pipeline/particles'
parts = pd.read_csv(part_dir+'parts.csv')
clf_scaler_path = '../output/pipeline/GPClassification/GPCclfRBF.p'
parts['GPCprob'] = particle.predict_prob(parts, ['corrwideal','mass_norm'], clf_scaler_path)
# get common particle pid (unique per particle but not per frame)
parts['cpid'] = parts['mov_name']+'_'+parts['particle'].apply(str)
parts.to_csv(part_dir+'parts_labeled.csv', index=False)
