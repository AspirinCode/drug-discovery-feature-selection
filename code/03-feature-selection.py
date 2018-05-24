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
from sklearn_genetic import GeneticSelectionCV

# config
dataset_file = '../dataset/dataset.csv'
feature_mask_file = 'WM_GA_SVM_features_mask.json'
plot_img_feat_file = 'WM_GA_SVM_feat_chart.png'
plot_img_acc_file = 'WM_GA_SVM_acc_chart.png'
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
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# GA feature selection
estimator = SVC(kernel='linear', cache_size=1000, max_iter=1000, verbose=verbosity)
selector = GeneticSelectionCV(estimator, n_population=20, n_generations=100, cv=2, caching=True, verbose=verbosity, n_jobs=-1)
selector = selector.fit(X, y)
count = (selector.support_ == True).sum()
print("Optimal number of features: {}".format(count))
max_tuples = selector.logbook_.select('max')
scores, num_feats = zip(*max_tuples)
print("Average best accuracy: {}".format(max(scores)))

# Note: with full dataset (unbalanced)
# Optimal number of features: 255
# Average best accuracy: 0.9962926604107819

# save features mask
with open(feature_mask_file, 'w') as f:
    sel_features = np.array(feature_names)[selector.support_]
    json.dump(sel_features.tolist(), f)

# plot accuracy per generation
plt.figure()
plt.title('Wrapper Method (GA) Feature Selection')
plt.xlabel('Generation')
plt.ylabel('CV Accuracy')
plt.plot(range(1, len(scores) + 1), scores)
plt.savefig(plot_img_acc_file)

# plot accuracy per set of features
plt.figure()
plt.title('Wrapper Method (GA) Feature Selection')
plt.xlabel('Generation')
plt.ylabel('Number of features')
plt.plot(range(1, len(num_feats) + 1), num_feats)
plt.savefig(plot_img_feat_file)
plt.show()
