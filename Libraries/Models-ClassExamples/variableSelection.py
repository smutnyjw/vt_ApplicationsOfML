"""
nearestNeighbor.py, Created on Thu Feb 20 17:41:33 2020
@author:    crjones4
@info:      Code example created from Dr Creed Jones of Virginia Tech,
            lecture 17 - Variable Selection
"""
import sklearn as sk
import sklearn.feature_selection as featsel
import sklearn.linear_model as sklinear_model
import sklearn.model_selection as skmodelsel
import sklearn.preprocessing as skpreproc
from sklearn.impute import SimpleImputer as imputer
from pandas import DataFrame as df
import numpy as np


def tryVariableSelection(pred, targ, sel, dir, labels):
    ranseed = 98043
    xtrain, xtest, ytrain, ytest = skmodelsel.train_test_split(
        pred, targ, test_size=0.3, random_state=ranseed)
    model = sklinear_model.LinearRegression()

    if sel == 'sequential':
        selector = featsel.SequentialFeatureSelector(
            model, direction=dir, n_features_to_select=6)
    elif sel == 'RFE':
        selector = featsel.RFE(model, step=1, n_features_to_select=6)
    elif sel == 'RFECV':
        selector = featsel.RFECV(model, step=1, cv=5)

    selector.fit(xtrain, ytrain)
    newxtrain = selector.transform(xtrain)
    newxtest = selector.transform(xtest)
    model.fit(newxtrain, ytrain)
    print("\nUsing: {0}".format(labels[selector.get_support() == True]))
    print("Method {0}: Training set R-sq={1:8.5f}, test set MSE={2:e}".format(
        dir,
        model.score(newxtrain, ytrain),
        sk.metrics.mean_squared_error(ytest, model.predict(newxtest))))

#################################################

ranseed = 1
featureLabels = []
targetLabel = []

#################################################

xf = df[featureLabels]
yf = df[targetLabel]

newpred = imputer.fit_transform(xf.to_numpy(np.float64))
scaler = skpreproc.MinMaxScaler()
normpred = scaler.fit_transform(newpred)
target = yf.to_numpy(np.float64)

xtrain, xtest, ytrain, ytest = skmodelsel.train_test_split(
    normpred, target, test_size=0.3, random_state=ranseed)
model = sklinear_model.LinearRegression()
xtraintrim = xtrain[:,0:6]
xtesttrim = xtest[:,0:6]
regr = model.fit(xtraintrim, ytrain)
print("\nUsing: {0}".format(featureLabels[0:6]))
print("First 6: Training set R-sq={0:8.5f}, test set MSE={1:e}".format(
    regr.score(xtraintrim,
    ytrain),
    sk.metrics.mean_squared_error(ytest, regr.predict(xtesttrim))))


tryVariableSelection(normpred, target, 'sequential', 'forward', featureLabels)
tryVariableSelection(normpred, target, 'sequential', 'backward', featureLabels)
tryVariableSelection(normpred, target, 'RFE', 'RFE', featureLabels)
tryVariableSelection(normpred, target, 'RFECV', 'RFECV', featureLabels)