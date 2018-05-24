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
feature_mask_file = 'SVM_RFE_features_mask.json'
plot_img_file = 'SVM_RFE_chart.png'
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
svc = SVC(kernel='linear', cache_size=1000, max_iter=1000, verbose=verbosity)
# scores = cross_val_score(svc, X, y, cv=10, n_jobs=-1, verbose=verbosity)
# print('Mean {}'.format(np.mean(scores)))

# SVM-RE feature selection with cross-validation
# http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html
selector = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy', verbose=verbosity, n_jobs=-1)
selector.fit(X, y)
print('Optimal number of features: {}'.format(selector.n_features_))
print('Average accuracy: {}'.format(max(selector.grid_scores_)))
# Note: using full (unbalanced) dataset
# Optimal number of features: 383
# Average accuracy: 0.9966104289695464


# save features mask
with open(feature_mask_file, 'w') as f:
    sel_features = np.array(feature_names)[selector.support_]
    json.dump(sel_features.tolist(), f)

# plot
scores = selector.grid_scores_
plt.figure()
plt.title('SVM-RFE Feature Selection')
plt.xlabel('Number of features selected')
plt.ylabel('CV Accuracy')
plt.plot(range(1, len(scores) + 1), scores)
plt.savefig(plot_img_file)
plt.show()