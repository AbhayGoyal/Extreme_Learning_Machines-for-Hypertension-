# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

# Demo python notebook for sklearn elm and random_hidden_layer modules
#
# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD

# <codecell>

from time import time
from matplotlib.pyplot import *
from sklearn.model_selection import train_test_split
from sklearn.cluster import k_means
from sklearn.metrics import f1_score,precision_recall_fscore_support, confusion_matrix as score, accuracy_score
import numpy as np
import pandas as pd
from elmclassify import ELMClassifier, ELMRegressor, GenELMClassifier, GenELMRegressor
from random_layer import RandomLayer, MLPRandomLayer, RBFRandomLayer, GRBFRandomLayer
from math import sqrt
import csv
import matplotlib.pyplot as plt
import plotly.tools as tls
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import load_iris, load_digits, load_diabetes, make_regression

# <codecell>
""""
def make_toy():
    x = np.arange(0.25, 20, 0.1)
    y = x * np.cos(x) + 0.5 * sqrt(x) * np.random.randn(x.shape[0])
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return x, y


# <codecell>
"""
pdf = pd.read_csv('hype.csv')
#print(pdf.head())
#x = np.arange(0.25, 20, 0.1)
x = pdf[['sex','body weight','height','smoker','systolic blood preassure','diastolic blood preassure','max systolic blood preassure']]#,'heart failure']]
y = pdf[['HYPERTENSION(1,0)']]
#print(x.head())
#print(y.head())


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# <codecell>




# <codecell>

# RBFRandomLayer tests



# <codecell>




# <codecell>

elmc = ELMClassifier(n_hidden=500, activation_func='multiquadric')


# <codecell>

elmr = ELMRegressor(random_state=0, activation_func='gaussian', alpha=0.0)
elmr.fit(x, y)
print
elmr.score(x, y), elmr.score(x, y)
plot(x, y, x, elmr.predict(x))


# <codecell>


# <codecell>



# <codecell>

#rhl = RandomLayer(n_hidden=1000, alpha=1.0)
rhl = RBFRandomLayer(n_hidden=500, rbf_width=0.0001)
elmr = GenELMClassifier(hidden_layer=rhl)
elmr.fit(x_train, y_train)
predicted = elmr.predict(x_test)
#precision, recall, fscore, support = score(y_test, predicted)

print("RBF Random")
#print('precision: {}'.format(precision))
#print('recall: {}'.format(recall))
#print('fscore: {}'.format(fscore))
#print('support: {}'.format(support))
#print("RBF different rbf_width",elmr.score(y_test,elmr.predict(x_test)))



# <codecell>

nh = 20
(ctrs, _, _) = k_means(x_train, nh)
unit_rs = np.ones(nh)

#rhl = RBFRandomLayer(n_hidden=nh, activation_func='inv_multiquadric')
#rhl = RBFRandomLayer(n_hidden=nh, activation_func='gaussian')
#rhl = GRBFRandomLayer(n_hidden=nh, centers=ctrs, radii=unit_rs)
rhl = GRBFRandomLayer(n_hidden=nh, grbf_lambda=.0001, centers=ctrs)
elmr = GenELMClassifier(hidden_layer=rhl)
elmr.fit(x_train, y_train)
#Y_pred = elmr.predict(x_test)
print("MELM GRBF" ,elmr.score(x_train, y_train), elmr.score(x_test, y_test))
print("MELM GRBF confusion", elmr.confusion(x_test, y_test))
#print("accuracy GRBF", accuracy_score(y_test, Y_pred))

#plot(x, y, x, elmr.predict(xtoy))

# <codecell>

rbf_rhl = RBFRandomLayer(n_hidden=100, random_state=0, rbf_width=0.0001)
elmc_rbf = GenELMClassifier(hidden_layer=rbf_rhl)
elmc_rbf.fit(x_train, y_train)
y_pred_rbf = elmr.predict(x_test)

#print("F1 RBF rbf", f1_score(y_test, y_pred_rbf, average="macro"))
#print("accuracy RBF", accuracy_score(y_test, y_pred_rbf))
print('RBF GenELM Classifier',elmc_rbf.score(x_train, y_train), elmc_rbf.score(x_test, y_test))


def powtanh_xfer(activations, power=1.0):
    return pow(np.tanh(activations), power)


tanh_rhl = MLPRandomLayer(n_hidden=500, activation_func=powtanh_xfer, activation_args={'power': 3.0})
elmc_tanh = GenELMClassifier(hidden_layer=tanh_rhl)
elmc_tanh.fit(x_train, y_train)
#print("tanh score",elmc_tanh.score(x_train, y_train), elmc_tanh.score(x_test, y_test))

# <codecell>


# <codecell>

# <codecell>

# <codecell>




