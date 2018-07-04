'''
Train a DBN using pubchem hiv+decoy dataset and test it against herbaldb dataset

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
test_file = '../dataset/dataset_test.csv'
model_file = '03-model.pkl'
scaler_file = '03-scaler.pkl'
verbosity = 1

# check if models exist
if os.path.isfile(model_file) and os.path.isfile(scaler_file):
    # load models
    scaler = joblib.load(scaler_file) 
    classifier = SupervisedDBNClassification.load(model_file)

else:

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
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Training
    classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=100,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    # classifier.fit(X_train, y_train)
    classifier.fit(X, y)
    # Save the model
    joblib.dump(scaler, scaler_file)
    classifier.save(model_file)    

    # Test
    # Note: Accuracy: 0.996160 (full unbalanced dataset)
    # y_pred = classifier.predict(X_test)
    # print('Done.\nAccuracy (PubChem): %f' % accuracy_score(y_test, y_pred))


# Test HerbalDB
df_test = pandas.read_csv(test_file, index_col=0)
X_test = df_test[df_test.columns.drop('Class')].values.astype('float32')
y_test = df_test['Class'].values
X_test = scaler.transform(X_test)
y_pred = classifier.predict(X_test)
print('Done.\nAccuracy (HerbalDB): %f' % accuracy_score(y_test, y_pred))