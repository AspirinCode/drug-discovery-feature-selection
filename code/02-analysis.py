# suppress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFECV

# config
dataset_file = '../dataset/dataset.csv'
verbosity = 1

# read dataset
df = pandas.read_csv(dataset_file, index_col=0)
# get columns with nonzero variance
df = df.loc[:, df.var() > 0]
feature_names = list(df[df.columns.drop('Class')])

# split to data X and labels y
X = df[df.columns.drop('Class')].values.astype('float32')
y = df['Class'].values

# scale data
# Note: 
# for RBF StandardScaler produce 4% better accuracy than MinMaxScaler
# for Linear MinMaxScaler produce 3% better accuracy than StandardScaler
# scaler = StandardScaler()
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# train SVM 
# Note: already reach 0.991 (RBF) and 0.990 (Linear)
# Note: using all (unbalanced) data: 0.9967692289459841
svc = SVC(kernel='linear', cache_size=1000, max_iter=1000, verbose=verbosity)
scores = cross_val_score(svc, X, y, cv=10, n_jobs=-1, verbose=verbosity)
print('Mean {}'.format(np.mean(scores)))

