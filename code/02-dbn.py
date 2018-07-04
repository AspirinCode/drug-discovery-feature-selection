'''
Train and test DBN on pubchem hiv+decoy dataset

@author yohanes.gultom@gmail.com
'''

import numpy as np
import pandas
import matplotlib.pyplot as plt
import json
import os 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.classification import accuracy_score
from sklearn.externals import joblib
from dbn.tensorflow import SupervisedDBNClassification
# use "from dbn import SupervisedDBNClassification" for computations on CPU with numpy

# config
dataset_file = '../dataset/dataset.csv'
model_file = '02-model.pkl'
scaler_file = '02-scaler.pkl'
verbosity = 1

# read dataset
df = pandas.read_csv(dataset_file, index_col=0)
# get columns with nonzero variance
# df = df.loc[:, df.var() > 0]
# split to data X and labels y
X = df[df.columns.drop('Class')].values.astype('float32')
y = df['Class'].values

# scale data    
# Note: 
# for RBF StandardScaler produce 4% better accuracy than MinMaxScaler
# for Linear MinMaxScaler produce 3% better accuracy than StandardScaler
# check if scaler exist
if os.path.isfile(scaler_file):
    scaler = joblib.load(scaler_file) 
else:
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# check if model exist
if os.path.isfile(model_file):
    classifier = SupervisedDBNClassification.load(model_file)
else:
    # Training
    classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=100,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    # Save the model
    joblib.dump(scaler, scaler_file)
    classifier.save(model_file)    

# cross validate
scores = cross_val_score(classifier, X, y, cv=10, n_jobs=-1, verbose=verbosity)
print('Mean {}'.format(np.mean(scores)))
