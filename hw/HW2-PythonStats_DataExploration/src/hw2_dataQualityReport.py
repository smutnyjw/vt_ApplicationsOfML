#########################################################################
#   File:   hw_dataQualityReport.py
#   Name:   John Smutny
#   Course: ECE-5984: Applications of Machine Learning
#   Date:   02/15/2022
#   Description:
#       Use numpy to perform statistical analysis on a datasets.
#       Then use Panda DataFrames to create a Data Quality Report.
##########################################################################


import pandas
import numpy as np
from Libraries.DataExploration.DataQualityReport import DataQualityReport

#####################################
# Initial loading of data
filename = 'C:/Data/HeartDisease.xlsx'
df = pandas.read_excel(filename)  # read an Excel spreadsheet

#####################################
# Dissect the data into text labels, features, and the desired target variable.
print('File {0} is of size {1}'.format(filename, df.shape))
labels = df.columns
featureLabels = labels.drop('target').values  # get just the predictors
xFrame = df[featureLabels]
yFrame = df['target']  # and the target variable
predictors = xFrame.to_numpy(np.float64)  # convert them to numpy arrays
target = yFrame.to_numpy(np.float64)

#####################################
# Create an organized data set summary for the console using a data frame.
report = DataQualityReport()

for thisLabel in labels:  # for each column, report basic stats
    thisCol = df[thisLabel]
    report.addCol(thisLabel, thisCol)

print(report.to_string())


#####################################
# Print all reports to statistics report to excel
# 1) Statistics Report
outFilename = "C:/Data/HeartDisease-DataQualityReport.xlsx"
report.statsdf.to_excel(sheet_name='DataQualityReport',
                        excel_writer=outFilename)

# 2) Covariance Matrix (using .cov() dataframe function)
outFilename = "C:/Data/HeartDisease-CovarianceMatrix.xlsx"
df.cov().to_excel(sheet_name='CovarianceMatrix', excel_writer=outFilename)

# 3) Correlation Matrix (using .corr() dataframe function)
outFilename = "C:/Data/HeartDisease-CorrelationMatrix.xlsx"
df.corr().to_excel(sheet_name='CorrelationMatrix', excel_writer=outFilename)
