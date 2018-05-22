import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFECV

dataset_file = '../dataset/dataset.csv'
verbosity = 1

df = pandas.read_csv(dataset_file, index_col=0)
# drop column with nonzero variance
df = df.loc[:, df.var() > 0]

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
# scores = cross_val_score(svc, X, y, cv=10, n_jobs=2, verbose=verbosity)
# print('Mean {}'.format(np.mean(scores)))

# SVM-RE feature selection with cross-validation
# http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy', verbose=1)
rfecv.fit(X, y)
print(rfecv.ranking_)
print("Optimal number of features : %d" % rfecv.n_features_)

'''
Ranking:
[ 1, 1, 6, 1, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 14, 1, 1, 1, 1, 1, 1, 1, 1, 1 10, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 27, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 11, 1, 1, 1, 1, 1, 1 33, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 31, 1, 1, 1, 1, 1 30, 1, 1 36, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 29, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9 34, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 22, 1, 1, 1, 1, 1 23, 1, 1 24 19, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 15, 1, 1, 1, 1, 1, 1 26, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 37, 1, 1, 1, 1, 1, 1, 1, 1 12, 1, 1, 1, 1, 1, 1, 1, 1 21, 1, 1, 1, 1, 1 20, 1, 1 25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 35, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1 32, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 38, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 4, 1, 1, 1, 1 17, 1, 1, 1, 1, 1, 1, 1, 1 18, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 28, 1, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 16, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Optimal number of features : 415
'''

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.savefig('SVM_RFECV.png')
plt.show()