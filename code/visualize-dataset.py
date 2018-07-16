"""
Visualize dataset using t-SNE with various perplexities

Result:
circles, perplexity=5 in 1.2e+02 sec
circles, perplexity=30 in 1.8e+02 sec
circles, perplexity=50 in 2.3e+02 sec
circles, perplexity=100 in 3.4e+02 sec

Execution time:
real    15m2.178s
user    14m8.398s
sys     0m30.222s

@author yohanes.gultom@gmail.com
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
import os
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from time import time

# config
dataset_file = '../dataset/dataset.csv'
n_components = 2
perplexities = [5, 30, 50, 100]
chart_filename_tpl = 'visualize-dataset_tsne_{}.png'

# check if display available
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

# read dataset
df = pandas.read_csv(dataset_file, index_col=0)
# get columns with nonzero variance
df = df.loc[:, df.var() > 0]

# split to data X and labels y
X = df[df.columns.drop('Class')].values.astype('float32')
y = df['Class'].values

# separate data by class
red = y == 0
green = y == 1

# scale data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# tsne
for i, perplexity in enumerate(perplexities):
    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='random', random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(X)
    t1 = time()
    print("t-SNE perplexity={} in {:.2g} sec".format(perplexity, t1 - t0))

    # plot
    fig, ax = plt.subplots() 
    ax.set_title("Perplexity=%d" % perplexity)
    ax.scatter(Y[red, 0], Y[red, 1], c="r")
    ax.scatter(Y[green, 0], Y[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
    filename = chart_filename_tpl.format(perplexity)
    plt.savefig(filename)
    print("chart saved in {}".format(filename)

plt.show()