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

from sklearn import neural_network as ann
from sklearn import metrics
from sklearn import preprocessing as preproc
import sklearn.model_selection as modelsel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import helperFunctions as hf
import modelFunctions as mf


### Set Constants
#######################################

#INPUT_FILE = '../../data/QCOM_HistoricalData_5yr.csv'
INPUT_FILE = 'C:/Data/QCOM_HistoricalData_5yr.csv'

TEST_RATIO = 0.3
RANDOM_SEED = 10

IDName = "Date"
targetName = "Close/Last"

FINANCIAL_FEATURES = ['Close/Last', 'Open', 'High', 'Low']
FEATURES_TO_EXPAND = ['Close/Last', 'Volume']
NUM_PREV_DAYS_TO_TRACK = 3


### Main Processing
#######################################

# load data and add columns to expand data as necessary.
df_raw = pd.read_csv(INPUT_FILE)

df_edit = hf.removeDollarSign(df_raw, FINANCIAL_FEATURES)
df_appended = hf.appendPastData(df_raw,
                                NUM_PREV_DAYS_TO_TRACK,
                                FEATURES_TO_EXPAND)
df_appended.to_csv("test_appendPastData.csv")

# Normalize and separate data into Independent & Dependent Variables.
x = df_appended.drop([IDName, targetName], axis=1)
#print(x.columns)
y = df_appended[targetName]

mf.CreateSimpleBaggingModel(x,y)
#mf.CreateBaggingModel(x, y)