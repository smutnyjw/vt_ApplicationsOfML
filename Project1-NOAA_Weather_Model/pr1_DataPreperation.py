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
#       Publish a DataQualityReport for a day's summary of data.
#
#   Cannot:
#       'DataQualityReport.py' lib file cannot handle string entries.
#       Publish a DataQualityReport for initial data.
#       End Goal headers do not contain units.
#       Process all possible ELEMENT values for an unique Day's events
##########################################################################

import pandas
import numpy as np
from vt_ApplicationsOfML.Libraries.DataExploration.DataQualityReport import \
    DataQualityReport

#####################################
# Initial loading of header-less data and adding a header
#   Date = yyyymmdd
#   OBS-Time = hhmm

INIT_HEADER = ['ID', 'Date', 'Element', 'Value', 'MeasurementFlag',
               'QualityFlag', 'SourceFlag', 'OBS-Time']
filename = 'C:/Data/USW00013880-Test.csv'
df = pandas.read_csv(filename, header=None)
df.columns = INIT_HEADER

print(df[1:5])

##########################################################################
# TODO - Add logic to remove/change data points from the starting dataset.
# Based on desired quality, remove/change certain data points.
# Actions List:
# 1) ...
# 2) ...

##########################################################################
# TODO - Do work necessary to print a Data Quality Report before cleaning.
# Create an organized data set summary for the console using a data frame.
report1 = DataQualityReport()

for thisLabel in INIT_HEADER:  # for each column, report basic stats
    if thisLabel == 'MeasurementFlag' \
            or thisLabel == 'QualityFlag' \
            or thisLabel == 'SourceFlag':
        print('WARN: DataQualityReport.py cannot process label = ' + thisLabel)
        # TODO - 'DataQualityReport.py' cannot process 'nan' values.
        #  Clean data before this step? (fill in nan, replace categorical chars)
        continue
    else:
        thisCol = df[thisLabel]
        report1.addCol(thisLabel, thisCol)

print("DataQualityReport - 1/2")
print(report1.to_string())


##########################################################################
# Create a new dataframe that will have one column per ELEMENT designation.
#   Every entry into the new dataframe is a day and columns of every event
#   that day.

# 1) Create final Dataframe header, define features and target variables.
baseColumns = ['Date']
elementTypes = df.Element.unique()
targetVariables = ['PRECIPFLAG', 'PRECIPAMT', 'NEXTDAYPRECIPFLAG',
                   'NEXTDAYPRECIPAMT']

DETAILED_HEADER = baseColumns + elementTypes.tolist() + targetVariables
# print("New Headers:")
# print(DETAILED_HEADER)

# 2) Isolate a single day's total events.
daysMeasured = df.Date.unique()

# TODO - When ready, create another loop to loop over all unique days below.
oneDayEvents = df.loc[df['Date'] == daysMeasured[0]]
print("Example isoDaysEvents:")
print(oneDayEvents)

# 3) Summarize the day's events into a single entry
#       Create a blank entry.
keys = DETAILED_HEADER
blankValues = [0] * len(DETAILED_HEADER)
entry = dict(zip(keys, blankValues))
# print("Blank Entry pre-processing")
# print(entry)

# 4) Loop through each datapoint on a particular day. Find what part of the
#       dictionary the datapoint's ELEMENT corresponds too. Do math based on
#       that ELEMENT value.
numEntries = len(oneDayEvents.index)
for i in range(numEntries):
    entryList = oneDayEvents.loc[i, :].values.tolist()
    # TODO - ^^^ Maybe make 'entryList' a map of the keys equal to initial
    #  'header' to simply value accessing and placement?
    # print(i)
    # print(entryList)
    entry['Date'] = entryList[1]
    element = entryList[2]

    # SWITCH statement to process the ELEMENT + VALUE
    if element == 'TMAX':
        entry[element] = entryList[3]
        # print('TempMax was ' + str(entryList[3]))
    elif element == 'PRCP':
        entry[element] = entryList[3]
        # print('Precipitation was ' + str(entryList[3]))
    elif element == 'SNOW':
        entry[element] = entryList[3]
        # print('Snow fall was ' + str(entryList[3]))
    # TODO - Add more 'elif' statements for the rest of the ELEMENT/VALUE pairs.
    else:
        print('Found element ' + str(entryList[2]))

print('Resulting Single Day Entry')
df_entry = pandas.DataFrame([entry])
print(df_entry)

# 5) Add a new TOTAL DAY entry of weather events in a data frame.
df_dataSummary = pandas.DataFrame(columns=DETAILED_HEADER)
df_dataSummary = pandas.concat([df_dataSummary, df_entry], ignore_index=True)
print('Resulting Dataframe with ONLY single day summaries.')
print(df_dataSummary)

############################################
# Publish a new DataQualityReport on the list of single day data entries.
report2 = DataQualityReport()

for thisLabel in DETAILED_HEADER:  # for each column, report basic stats
    thisCol = df_dataSummary[thisLabel]
    report2.addCol(thisLabel, thisCol)

# print("DataQualityReport - 2/2")
# print(report2.to_string())
