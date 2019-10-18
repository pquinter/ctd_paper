"""
Train a Gaussian Process Classifier with manually labeled spots
"""
import pickle

import corner
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils import plot

###############################################################################
# Train a Gaussian Process Classifier (GPC)
###############################################################################
# get spot labels
save_dir = '../output/pipeline/GPClassification/'
parts_labeled = pd.read_csv(save_dir+'training/parts_labeled_trainingset.csv')

# scale mass and corrwideal for classification
scaler = StandardScaler().fit(parts_labeled[['corrwideal','mass_norm']].values)
scaled_lbld = scaler.transform(parts_labeled[['corrwideal','mass_norm']].values)
parts_labeled['mass_scaled'] = scaled_lbld.T[0]
parts_labeled['corr_scaled'] = scaled_lbld.T[1]

# split into training and test set
train_set, test_set = train_test_split(parts_labeled, random_state=42)
# get X and Y values
X_train, X_test = [s[['corr_scaled','mass_scaled']].values for s in (train_set, test_set)]
y_train, y_test = [s['manual_label'].values for s in (train_set, test_set)]

#clf_scaler_path = '../output/pipeline/GPClassification/GPCclfRBF.p'
#with open(clf_scaler_path, 'rb') as f:
#    clf = pickle.load(f)
#    scaler = pickle.load(f)

# train a gaussian process classifier with RBF kernel (Default)
clf = GaussianProcessClassifier(1.0 * RBF(1.0),  random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
plot2dDecisionFunc(clf, X_train, y_train, save=save_dir+'prob_surfaceGPC.pdf')
clf.score(X_test, y_test)
labels_pred = clf.predict_proba(X_test)[:,1]

# compute f1 score: harmonic mean between precision and recall
# see https://en.wikipedia.org/wiki/F1_score
prob_f1 = pd.DataFrame()
prob_f1['prob_thresh'] = np.linspace(0.1, 1, 90, endpoint=False)
f1score = np.array([metrics.precision_recall_fscore_support(y_test, labels_pred>thresh)[2]
        for thresh in prob_thresh])
prob_f1['f1score_False']= f1score[:,0]
prob_f1['f1score_True']= f1score[:,1]
prob_f1.to_csv(save_dir+'prob_f1score.csv', index=False)

fig, ax = plt.subplots()
ax.plot(prob_f1.prob_thresh, prob_f1.f1score_False, color='r')
ax.plot(prob_f1.prob_thresh, prob_f1.f1score_True, color='b')
ax.set(ylabel='F1 score', xlabel='Prob. threshold')
ax.axvline(0.5, ls='--', c='k', alpha=0.5)
plt.legend(['False spots','True spots'])
sns.despine()
plt.tight_layout()
plt.savefig(save_dir+'f1score_prob.pdf', bbox_inches='tight')

classif_report = metrics.classification_report(y_test, labels_pred>0.5)
with open(save_dir+'GPCclfRBF.p', 'wb') as f:
    pickle.dump(clf, f)
    pickle.dump(scaler, f)
with open(save_dir +'classif_report.txt', "w") as text_file:
    text_file.write(classif_report)
