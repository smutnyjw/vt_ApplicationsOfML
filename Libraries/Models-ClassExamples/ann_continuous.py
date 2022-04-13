"""
ann_continuous.py, Created on Thu Feb 20 17:41:33 2020
@author:    crjones4
@info:      Code example created from Dr Creed Jones of Virginia Tech,
            lecture 22 - Neural Networks

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
trainX, testX, trainY, testY = \
modelsel.train_test_split(X, Y, test_size=0.3, random_state=24061)


# Solve the problem using an artificial neural network
hl = (15, 15)   # Two layers of 15nodes and 15nodes.
clf = ann.MLPRegressor(hidden_layer_sizes=hl,
                       activation='tanh',
                       solver='adam',
                       alpha=0.0001,
                       early_stopping=True,
                       max_iter=10000,
                       validation_fraction=0.42)
clf.fit(trainX,trainY)
annPredY = clf.predict(testX)
trainingLoss = np.asarray(clf.loss_curve_)
validationLoss = np.sqrt(1 - np.asarray(clf.validation_scores_))
factor = trainingLoss[1] / validationLoss[1]
validationLoss = validationLoss*factor
print("\n\rANN: MSE = %f" % metrics.mean_squared_error(testY, annPredY))


# create figure and axis objects with subplots()
xlabel = "epochs (hl=" + str(hl) + ")"
fig, ax = plt.subplots()
ax.plot(trainingLoss, color="blue")
ax.set_xlabel(xlabel,fontsize=10)
ax.set_ylabel("loss",color="blue",fontsize=10)
ax.plot(validationLoss,color="red")
ax.set_yscale('log')
plt.show()