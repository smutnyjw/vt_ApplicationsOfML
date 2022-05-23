'''
ann.py
@author:    John Smutny
@team:      James Ensminger, Ben Johnson, Anagha Mudki, John Smutny
@info:      Regression artificial neural network (ann) model to predict the
            future stock price of the Qualcomm semiconductor company.

            PENDING FUNCTIONALITY
            The ann model cycles through several model frameworks and then
            chooses the best architecture to maximize profit from a $1000
            investment 28 days prior to sale.
'''

from operator import mod
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification

from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import models.helperFunctions as hf

### Set Constants
#######################################

OUTPUT_FILES = True
ASCENDING_DATES = True

INCOME_TOLERANCE = 1.05
PREDICT_FUTURE_DAY = 28
INVESTMENT = 1000

pneHundredCalcs = 100

nEstimatorsValue = 100
criterionToUse = 'squared_error'
maxDepthToUse = 10
randomStateToUse = 0

#code based on ANN model to ensure it works with yaml
def SimpleEnsembleModel(
    trainTestSplit: tuple,
    nEstimators: int = nEstimatorsValue,
    criterionInUse: str = criterionToUse,
    maxDepthInUse: int = maxDepthToUse,
    randomStateInUse: int = randomStateToUse
):
    ''' 
    Ensemble model, is RandomForestRegressor, code and documentation found at 
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

    class sklearn.ensemble.RandomForestRegressor(n_estimators=100, 
    *, 
    criterion='squared_error', 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_features='auto', 
    max_leaf_nodes=None, 
    min_impurity_decrease=0.0, 
    bootstrap=True, 
    oob_score=False, 
    n_jobs=None, 
    random_state=None, 
    verbose=0, 
    warm_start=False, 
    ccp_alpha=0.0, 
    max_samples=None)

    :param trainX: pd.DataFrame containing training predictors
    :param testX: pd.DataFrame containing test predictors
    :param trainY: pd.DataFrame containing training target(s)
    :param testY: pd.DataFrame containing test target(s)
    '''
    trainX, testX, trainY, testY = trainTestSplit
    # Define RandomForestRegressor parameters
    model = RandomForestRegressor(n_estimators=nEstimators, criterion=criterionInUse, max_depth=maxDepthInUse, random_state=randomStateInUse)
   
    mseOneHundred = []
    lowestMSEModel = 99999
    lowestModel = model
    # Train and Evaluate the Ensamble
    for x in range(pneHundredCalcs):
        model.fit(trainX.to_numpy(), trainY.to_numpy())
        ensemblePredY = model.predict(testX)
        mseOneHundred.append(metrics.mean_squared_error(testY, ensemblePredY))
        if(lowestMSEModel > metrics.mean_squared_error(testY, ensemblePredY)):
            lowestMSEModel = metrics.mean_squared_error(testY, ensemblePredY)
            lowestModel = model

    print(f'{__name__} MSE AVERAGE = {FindAverage(mseOneHundred)}')
    return lowestModel


def FindAverage(list):
    return sum(list)/len(list)
