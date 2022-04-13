"""
ann_classification.py, Created on Thu Feb 20 17:41:33 2020
@author:    crjones4
@info:      Code example created from Dr Creed Jones of Virginia Tech,
            lecture 22 - Neural Networks.

            Artificial Neural Network code for a classification ann and output
            the 'loss function' over each epoch.
"""

from sklearn import neural_network as ann
from sklearn import metrics
from sklearn import preprocessing as preproc
import sklearn.model_selection as modelsel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pathName = "C:/Data/Shuttle/"
fileName = "shuttle.xlsx"
targetName = "class"

dataFrame = pd.read_excel(pathName + fileName, sheet_name='all')
X = dataFrame.drop([targetName], axis=1).to_numpy()
scalerX = preproc.MinMaxScaler()
scalerX.fit(X)
X = scalerX.transform(X)
Y = dataFrame[targetName].to_numpy()

rseed = 98043
trainX, testX, trainY, testY = modelsel.train_test_split(X, Y, test_size=0.2, random_state=rseed)
trainX, validX, trainY, validY = modelsel.train_test_split(trainX, trainY, test_size=0.25, random_state=rseed)
regpenalty = 0.001
hl = (5, 5)


clf = ann.MLPClassifier(hidden_layer_sizes=hl, activation='tanh', solver='adam', random_state=rseed,
alpha=regpenalty)
trainingLoss = []
validationLoss = []
for epoch in range(100):
    clf.partial_fit(trainX,trainY,classes=np.unique(Y))
    trainingLoss.append(1-clf.score(trainX, trainY))
    validationLoss.append(1-clf.score(validX, validY))
annPredY = clf.predict(testX)
print("\n\rANN: %d mislabeled out of %d points"
% ((testY != annPredY).sum(), testX.shape[0]))
print(metrics.confusion_matrix(testY, annPredY))


# create figure and axis objects with subplots()
xlabel = "epochs (hl=" + str(hl) + ")"
fig,ax = plt.subplots()
ax.plot(trainingLoss, color="blue")
ax.set_xlabel(xlabel,fontsize=10)
ax.set_ylabel("loss",color="blue",fontsize=10)
ax.plot(validationLoss,color="red")
#ax.set_yscale('log')
plt.show()
