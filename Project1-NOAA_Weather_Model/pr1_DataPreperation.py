#########################################################################
#   File:   pr1_DataPreperation.py
#   Name:   John Smutny
#   Course: ECE-5984: Applications of Machine Learning
#   Date:   03/01/2022
#   Description:
#       For Project1; perform initial data preparation to get NOAA provided
#       data into a usable state to model.
#
#   Readme:
#   https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_station/readme-by_station.txt
##########################################################################

import pandas
import numpy as np
from Libraries.DataExploration.DataQualityReport import DataQualityReport

#####################################
# Initial loading of headerl-less data and adding a header
#   Date = yyyymmdd
#   OBS-Time = hhmm

header = ['ID', 'Date', 'Element', 'Value', 'MeasurementFlag',
          'QualityFlag', 'SourceFlag', 'OBS-Time']
# filename = 'C:/Data/USW00013880-Test.xlsx'
# df = pandas.read_excel(filename, sheet_name='TestData')
# df.columns = header

filename = 'C:/Data/USW00013880-Test.csv'
df = pandas.read_csv(filename, header=None)
df.columns = header

print(df[1:5])

##########################################################################
# TODO - Add logic to remove data points from the starting dataset.
# Based on desired quality, remove certain data points.
# Actions List:
# 1) ...
# 2) ...


##########################################################################
# Create a new dataframe that will have one column per ELEMENT designation.
#   Every entry into the new dataframe is a day and columns of every event
#   that day.

# 1) Create final Dataframe header, define features and target variables.
baseColumns = ['Date']
elementTypes = df.Element.unique()
targetVariables = ['PRECIPFLAG', 'PRECIPAMT', 'NEXTDAYPRECIPFLAG',
                   'NEXTDAYPRECIPAMT']

newHeader = baseColumns +elementTypes.tolist() + targetVariables
print("New Headers:")
print(newHeader)

# 2) Isolate a single day's total events
daysMeasured = df.Date.unique()
isoDaysEvents = df.loc[df['Date'] == daysMeasured[0]]
print("Example isoDaysEvents:")
print(isoDaysEvents)

# 3) Summarize the day's events into a single entry
#       Create a blank entry
keys = newHeader
blankValues = [0]*len(newHeader)
entry = dict(zip(keys, blankValues))
print("Blank Entry pre-processing")
print(entry)

#       Fill out a day's summary from each datapoint found in the original
#       dataset.

#   Loop through each datapoint on a particular day. Find what part of the
#   dictionary the datapoint's ELEMENT corresponds too. Do math based on that
#   ELEMENT value.

numEntries = len(isoDaysEvents.index)
for i in range(numEntries):
    entryList = isoDaysEvents.loc[i, :].values.tolist()
    print(i)
    print(entryList)
    entry['Date'] = entryList[1]
    element = entryList[2]

    # SWITCH statement to process the ELEMENT + VALUE
    if element == 'TMAX':
        print('TempMax was ' + str(entryList[3]))
    elif element == 'PRCP':
        print('Precipitation was ' + str(entryList[3]))
    else:
        print('Found element ' + str(entryList[2]))



##########################################################################
# TODO - Once an 'daysSummary' is finished, append to a new dataframe
# ...) Add a new TOTAL DAY entry of weather events in a data frame.
# df_singleDay = pandas.DataFrame(columns=newHeader)
# print(df_singleDay.head())
# df_singleDay.loc[len(df_singleDay)] = entry
# print(df_singleDay[len(df_singleDay)])

#####################################

#   Need to insert new columns for each element

# for n in daysMeasured:
#    entry = df.loc[df['Date'] == n]
