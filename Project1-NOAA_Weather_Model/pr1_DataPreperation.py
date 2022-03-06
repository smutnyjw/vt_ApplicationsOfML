#########################################################################
#   File:   pr1_DataPreperation.py
#   Name:   John Smutny
#   Course: ECE-5984: Applications of Machine Learning
#   Date:   03/05/2022
#   Description:
#       For Project1; perform initial data preparation to get NOAA provided
#       data into a usable state to model.
#
#   Readme:
#   https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_station/readme-by_station.txt
#
#   Can:
#       Read in a csv, create an initial dataframe of data with headers.
#       Identify every unique Day and process some ELEMENT possibilities into
#           a single list. Then place that list into a resulting dataframe.
#
#   Cannot:
#       'DataQualityReport.py' lib file cannot handle string entries.
#       Publish a DataQualityReport for initial or data summary dataframe.
#       End Goal headers do not contain units.
#       Process all possible ELEMENT values for an unique Day's events
##########################################################################

import pandas
import numpy as np
from Libraries.DataExploration.DataQualityReport import DataQualityReport

#####################################
# Initial loading of header-less data and adding a header
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
# TODO - Do work necessary to print a Data Quality Report before cleaning.
# Create an organized data set summary for the console using a data frame.
report1 = DataQualityReport()

for thisLabel in header:  # for each column, report basic stats
    if thisLabel == 'Date':
        print('only do Date col for now')
        thisCol = df[thisLabel]
        report1.addCol(thisLabel, thisCol)
    else:
        # TODO - Figure how to stop Str from being read in DataQualityReport
        continue


print(report1.to_string())


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

newHeader = baseColumns + elementTypes.tolist() + targetVariables
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
blankValues = [0] * len(newHeader)
entry = dict(zip(keys, blankValues))
print("Blank Entry pre-processing")
print(entry)

#       Fill out a day's summary from each datapoint found in the original
#       dataset.

#   Loop through each datapoint on a particular day. Find what part of the
#   dictionary the datapoint's ELEMENT corresponds too. Do math based on that
#   ELEMENT value.

# TODO - When ready, create another loop to loop over all unique days.
numEntries = len(isoDaysEvents.index)
for i in range(numEntries):
    entryList = isoDaysEvents.loc[i, :].values.tolist()
    # TODO - ^^^ Maybe make 'entryList' a map of the keys equal to initial
    #  'header'?
    print(i)
    print(entryList)
    entry['Date'] = entryList[1]
    element = entryList[2]

    # SWITCH statement to process the ELEMENT + VALUE
    if element == 'TMAX':
        entry[element] = entryList[3]
        print('TempMax was ' + str(entryList[3]))
    elif element == 'PRCP':
        entry[element] = entryList[3]
        print('Precipitation was ' + str(entryList[3]))
    elif element == 'SNOW':
        entry[element] = entryList[3]
        print('Snow fall was ' + str(entryList[3]))
    #TODO - Add more 'elif' statements for the rest of the ELEMENT/VALUE pairs.
    else:
        print('Found element ' + str(entryList[2]))

    print('Resulting Single Day Entry')

df_entry = pandas.DataFrame([entry])
print(df_entry)

# 4) Add a new TOTAL DAY entry of weather events in a data frame.
df_dataSummary = pandas.DataFrame(columns=newHeader)
df_dataSummary = pandas.concat([df_dataSummary, df_entry], ignore_index=True)
print(df_dataSummary)
