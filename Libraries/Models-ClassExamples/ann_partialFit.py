"""
ann_partialFit.py, Created on Thu Feb 20 17:41:33 2020
@author:    crjones4
@info:      Code example created from Dr Creed Jones of Virginia Tech,
            lecture 22 - Neuroal Networks

            Artificial Neural Network code for a regression ann and output
            the 'loss function' over each epoch.
"""

from sklearn import neural_network as ann
from sklearn import metrics
from sklearn import preprocessing as preproc
import sklearn.model_selection as modelsel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pathName = "C:/Data/"
fileName = "ccpp.xlsx"
IDName = "ID"
targetName = "F"

dataFrame = pd.read_excel(pathName + fileName, sheet_name='all')
X = dataFrame.drop([IDName, targetName], axis=1).to_numpy()
scalerX = preproc.MinMaxScaler()
scalerX.fit(X)

X = scalerX.transform(X)
Y = dataFrame[targetName].to_numpy()

trainX, testX, trainY, testY = modelsel.train_test_split(X, Y,
                                                         test_size=0.2,
                                                         random_state=24061)
trainX, validX, trainY, validY = modelsel.train_test_split(trainX, trainY,
                                                           test_size=0.25,
                                                           random_state=24061)

# Solve the problem using an artificial neural network
hl = (150, 150) # was (15,15)
clf = ann.MLPRegressor(hidden_layer_sizes=hl,
activation='tanh', solver='adam', max_iter=10000)
trainingLoss = []
validationLoss = []
for epoch in range(1000):
    clf.partial_fit(trainX,trainY)
    trainingLoss.append(1-clf.score(trainX, trainY))
    validationLoss.append(1-clf.score(validX, validY))
    annPredY = clf.predict(testX)
    print("\n\rANN: MSE = %f" %
    metrics.mean_squared_error(testY, annPredY))