"""
logisticRegression.py, Created on Thu Feb 20 17:41:33 2020
@author:    crjones4
@info:      Code example created from Dr Creed Jones of Virginia Tech,
            lecture 18 - Logistic Regression
                Used for binary classification based on a 'decision threshold'.
"""

from sklearn import linear_model as linmod
from sklearn import metrics
from sklearn import preprocessing as preproc
import sklearn.model_selection as skmodelsel
import pandas as pd
import numpy as np

#################################################

ranseed = 10
pathName = "C:\\Data\\"
fileName = "generators.xlsx" # read from Excel file
targetName = "GOOD"
IDName = "ID"
doScale = True

#################################################

dataFrame = pd.read_excel(pathName + fileName, sheet_name='extended')

x = dataFrame.drop([IDName, targetName], axis=1).to_numpy()
y = dataFrame[targetName].to_numpy()

trainX, testX, trainY, testY = skmodelsel.train_test_split(
    x, y, test_size=0.3, random_state=ranseed)

if (doScale):
    scalerX = preproc.MinMaxScaler(feature_range=(-1, 1))
    scalerX.fit(trainX)
    trainX = scalerX.transform(trainX)

mlr = linmod.LogisticRegression(tol=1e-6) # creates the regressor object – note the lower tolerance!
mlr.fit(trainX, trainY)

Ypred = mlr.predict(trainX)
Ypredclass = 1*(Ypred > 0.5)
print("R2 = %f, MSE = %f, Classification Accuracy = %f" %
(metrics.r2_score(testY, Ypred), metrics.mean_squared_error(testY, Ypred), metrics.accuracy_score(testY, Ypredclass)))
print("W: ", np.append(np.array(mlr.intercept_), mlr.coef_))

poly = preproc.PolynomialFeatures(2) # object to generate polynomial basis functions
trainX = dataFrame.drop([IDName, targetName], axis=1).to_numpy()
bigTrainX = poly.fit_transform(trainX)
if (doScale):
    scalerX = preproc.MinMaxScaler(feature_range=(-1, 1))
    scalerX.fit(bigTrainX)
    bigTrainX = scalerX.transform(bigTrainX)

mlrf = linmod.LogisticRegression() # creates the regressor object
mlrf.fit(bigTrainX, trainY)

Ypred = mlrf.predict(bigTrainX)
Ypredclass = 1*(Ypred > 0.5)
print("R2 = %f, MSE = %f, Classification Accuracy = %f" %
    (metrics.r2_score(testY, Ypred), metrics.mean_squared_error(testY, Ypred), metrics.accuracy_score(testY, Ypredclass)))
print("W: ", np.append(np.array(mlr.intercept_), mlr.coef_))
