"""
ECE5984 SP20 Multivariate Linear Regression
Created on Thu Feb 20 17:41:33 2020
@author: crjones4
"""
from sklearn import preprocessing as preproc
from sklearn import linear_model as linmod
from sklearn import metrics
import pandas as pd
import numpy as np

#################################################

pathName = "C:\\Data\\"
fileName = "DublinRental.xlsx" # read from Excel file
targetName = "PRICE"
IDName = "ID"
catName = "ENERGY"

#################################################

dataFrame = pd.read_excel(pathName + fileName, sheet_name='train2') # with Energy binaries
trainX = pd.dataFrame.drop(['IDName', 'catName', 'targetName'],
                          axis=1).to_numpy()

trainY = pd.dataFrame[targetName].to_numpy()
mlr = linmod.LinearRegression() # creates the regressor object
mlr.fit(trainX, trainY)
print("R2 is %f" % mlr.score(trainX, trainY))
print("W = ", mlr.intercept_, mlr.coef_)

query = np.array([[600,6,1,0,0,20]])
print("prediction:", query, mlr.predict(query))
print(metrics.mean_squared_error(trainY, mlr.predict(trainX)))