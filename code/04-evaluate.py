# suppress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas
import numpy as np
import json
import os
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# config
train_file = '../dataset/dataset.csv'
test_file = '../dataset/dataset_test.csv'
feature_mask_files = [None, 'WM_GA_SVM_features_mask.json', 'SVM_RFE_features_mask.json']
verbosity = 1

# read dataset
df_train = pandas.read_csv(train_file, index_col=0)
df_test = pandas.read_csv(test_file, index_col=0)

for feature_mask_file in feature_mask_files:
    if feature_mask_file:
        # apply features mask        
        with open(feature_mask_file, 'r') as f:        
            feature_mask = json.load(f)
        
        # split to data X and labels y
        X_train = df_train[feature_mask].values.astype('float32')
        y_train = df_train['Class'].values
        X_test = df_test[feature_mask].values.astype('float32')
        y_test = df_test['Class'].values
    else:
        # no mask
        X_train = df_train.values.astype('float32')
        y_train = df_train['Class'].values
        X_test = df_test.values.astype('float32')
        y_test = df_test['Class'].values

    # scale data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # evaluate
    estimator = SVC(kernel='rbf', max_iter=1000, verbose=verbosity)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    # print result
    filename = os.path.basename(feature_mask_file) if feature_mask_file else 'No feature selection'
    features = ', '.join(feature_mask) if feature_mask_file else 'All'
    print('\n***\n')
    print('Feature mask filename: {}'.format(filename))
    print('Accuracy: {}'.format(score))
    print('Feature(s): {}'.format(features))