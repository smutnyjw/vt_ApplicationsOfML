#!/usr/bin/env python3

'''
@package    project1.linreg
@info       Perform sklearn LinearRegression and RidgeCV modeling on processed
            NOAA daily weather data. Currently using data for Charleston, SC
@author     bencjohn@vt.edu
@Repo       https://github.com/beenjohn/aml-5984/tree/main/project1
'''

# Python libraries
import sys

# External libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

# Constants
RANDOM_SEED = 919
DATA_FILE = 'processed/USW00013880-SINGLEDAY-DUB.csv' if len(sys.argv) < 2 else sys.argv[1]
TARGET = ['NEXTPRECIPAMT']

# Load the processed data frame
df = pd.read_csv(DATA_FILE)

# Drop the last row as it will not have a valid NEXTPRECIPAMT
df = df.drop(labels=(len(df.index)-1), axis=0)

# Generate the training and test data split
FEATURES = df.drop(columns=TARGET+[
    'id',
    'date',
    'event_count',
    'PREVDATE',
    'PREVEVENT_COUNT',
    'PRECIPFLAG',
    'NEXTPRECIPFLAG'
], axis=1).columns
targets = df[TARGET]
features = df[FEATURES]
print(features.describe())

X_train, X_test, Y_train, Y_test = train_test_split(
    features,
    targets,
    test_size=0.3,
    random_state=RANDOM_SEED,
)

# Scale the data
scaler = MinMaxScaler(feature_range=(-1, 1))
scalertrain = scaler.fit(X_train)
scalertest = scaler.fit(X_test)
xtrain = scalertrain.transform(X_train)
xtest = scalertest.transform(X_test)
ytrain = Y_train.to_numpy()
ytest = Y_test.to_numpy()

# pandas quick DQR
print(X_train.describe())
print(Y_train.describe())
print(pd.DataFrame(xtrain, columns=X_train.columns).describe())

# Run the LinearRegression test
print('\n\nLinearRegression:\n========================================')
lmodel = LinearRegression()
lmodel = lmodel.fit(xtrain, ytrain)
print(f'R2 Score:\t{lmodel.score(xtrain, ytrain)}')
print(f'W:\t\t{lmodel.intercept_}')
lt_predict = lmodel.predict(xtest)
lcoefficients = pd.concat(
    [
        pd.DataFrame(X_train.columns, columns=['Feature']),
        pd.DataFrame(np.transpose(lmodel.coef_), columns=['Coefficient'])
    ],
    axis = 1,
).sort_values(by=['Coefficient'])
print(f'Coefficients:\n{lcoefficients}')
r2 = lmodel.score(xtest, ytest)
print(f'Prediction Score: {r2}')
mse = mean_squared_error(ytest, lt_predict)
print(f'MSE:\t\t{mse}')

# Write results to CSV
coef = lcoefficients.to_dict()
values = {}
values['W_0'] = lmodel.intercept_[0]
for key in coef['Feature'].keys():
    values[f'W_{coef["Feature"][key].upper()}'] = coef['Coefficient'][key]
values['MSE'] = mse
values['R2'] = r2
results = {'Parameter': list(values.keys()), 'Value': list(values.values())}
results = pd.DataFrame(results, columns=results.keys())
results.to_csv('processed/linear_regression_two_day_results.csv', index_label=None)