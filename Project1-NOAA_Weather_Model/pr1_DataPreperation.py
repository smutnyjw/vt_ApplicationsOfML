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
#   Next Steps:
#   1) Make 'quickDQR()' fct. Replace in here.
#   2) Only extract CORE5 data headers in original dataframe.
#   3) (offline) Make data edit suggestions for next meeting.
#
#   Can:
#       Read in a csv, create an initial dataframe of data with headers.
#       Identify every unique Day and process some ELEMENT possibilities into
#           a single list. Then place that list into a resulting dataframe.
#       Publish a DataQualityReport for initial data.
#       Publish a DataQualityReport for a day's summary of data.
#       Process all possible ELEMENT values for an unique Day's events
#       Account for 'NEXTDAY' values.
#       
#   Cannot:
#       'DataQualityReport.py' lib file cannot measure any aspects of
#       categorical values.
#       End Goal headers do not contain units.

##########################################################################

import pandas
import numpy as np
from vt_ApplicationsOfML.Libraries.DataExploration.DataQualityReport import \
    DataQualityReport

#####################################
# Initial loading of header-less data and adding a header
#   Date = yyyymmdd
#   OBS-Time = hhmm
FILE_IN = 'C:/Data/USW00013880.csv'
INIT_HEADER = ['ID', 'Date', 'Element', 'Value', 'MeasurementFlag',
               'QualityFlag', 'SourceFlag', 'OBS-Time']
df = pandas.read_csv(FILE_IN, header=None)
df.columns = INIT_HEADER
print(df[1:5])

FILE_OUT = 'C:/Data/USW00013880-Test-SINGLEDAY.csv'

#####################################
# Establish user settings for the program, dataframe headers, etc
MODEL_VERSION = 2
PERCENT_DAYS = 1.0  # value 0-1
OUTPUT_FILE = 1

BASE_COLUMNS = ['#', 'Date', '#Events']
# Core five NOAA measurement features.
CORE5_FEATURES = ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']
# Evaporation and Sunshine.
NEXT5_FEATURES = ['EVAP', 'MNPN', 'MXPN', 'ACMC', 'PSUN']
ALL_ELEMENT_TYPES = df.Element.unique()
TARGET_VARIABLES = ['PREV_PRECIPFLAG', 'PREV_PRECIPAMT', 'PRECIPFLAG',
                    'PRECIPAMT', 'NEXTPRECIPFLAG',
                    'NEXTPRECIPAMT']

CORE5_HEADER = BASE_COLUMNS + CORE5_FEATURES + TARGET_VARIABLES
MODEL2_HEADER = BASE_COLUMNS + CORE5_FEATURES + NEXT5_FEATURES + \
                TARGET_VARIABLES
ALL_HEADER = BASE_COLUMNS + ALL_ELEMENT_TYPES.tolist() + TARGET_VARIABLES

##########################################################################
# Create an organized data set summary for the console using a data frame.
report1 = DataQualityReport()

for thisLabel in INIT_HEADER:  # for each column, report basic stats
    thisCol = df[thisLabel]

    if thisLabel == 'Element'\
            or thisLabel == 'MeasurementFlag'\
            or thisLabel == 'QualityFlag'\
            or thisLabel == 'SourceFlag':
        report1.addCatCol(thisLabel, thisCol)
    else:
        report1.addCol(thisLabel, thisCol)

print("DataQualityReport - 1/2")
print(report1.to_string())

##########################################################################
# After initial Data Quality Report.
# Based on desired quality, remove/change certain data points for modeling.
# Actions List:

# 1) Remove all data points that have a non-NULL QualityFlag
df = df[df.QualityFlag.isnull()]

# 1b) TODO - Get indexes of all MeasurementFlag 'T' values?

# 2) Drop the QualityFlag & SourceFlag column
df = df.drop(columns=['QualityFlag', 'SourceFlag'])
print(df[1:5])

##########################################################################
# Create a new dataframe that will have one column per ELEMENT designation.
#   Every entry into the new dataframe is a day and columns of every event
#   that day.

# 1) Create final Dataframe header, define features and target variables.
if MODEL_VERSION == 1:
    NEW_HEADER = CORE5_HEADER
elif MODEL_VERSION == 2:
    NEW_HEADER = MODEL2_HEADER
elif MODEL_VERSION == 0:
    NEW_HEADER = ALL_HEADER

df_dataSummary = pandas.DataFrame(columns=NEW_HEADER)

# 2) Isolate a single day's total events into a dictionary.
df_listOfDays = df.Date.unique()

if PERCENT_DAYS:
    numDays = int(len(df_listOfDays) * PERCENT_DAYS)
else:
    numDays = 10

for dayToProcess in range(0, numDays):
    # 3) Loop through each datapoint on a particular day. Find what part of the
    #       dictionary the datapoint's ELEMENT corresponds too. Do math based on
    #       that ELEMENT value.
    df_oneDayEvents = df.loc[df['Date'] == df_listOfDays[dayToProcess]]
    numEventsInDay = len(df_oneDayEvents.index)
    # print("Events in the Day Measured:")
    # print(df_oneDayEvents)

    # 4) Summarize the day's events into a single entry
    for i in range(0, numEventsInDay):
        # Dissect each event in the current day.
        index_cur = df_oneDayEvents.index[i]
        entryList = df_oneDayEvents.loc[index_cur, :].values.tolist()

        if i == 0:
            # Start with a blank entry to summarize a new day's values.
            blankValues = [0] * len(NEW_HEADER)
            entry = dict(zip(NEW_HEADER, blankValues))

            # Reset Target variables.
            percipFlag = False
            percipAmt = 0

            # Fill in One-Time-Only values.
            entry['#'] = dayToProcess
            entry['Date'] = entryList[1]
            entry['#Events'] = numEventsInDay

        # Fill in values for each desired column and ELEMENT
        element = entryList[2]

        # SWITCH statement to process the ELEMENT + VALUE
        if element == 'TMAX':
            entry[element] = entryList[3]
            # print('TempMax was ' + str(entryList[3]))
            # TODO - use case: What happens if there are more than one of a
            # measurement?
        elif element == 'PRCP':
            entry[element] = entryList[3]
            if entryList[3] > 0 or df.MeasurementFlag[index_cur] == 'T':
                percipFlag = True
            percipAmt = percipAmt + entryList[3]
            # print('Precipitation was ' + str(entryList[3]))
        elif element == 'SNOW':
            entry[element] = entryList[3]
            if entryList[3] > 0 or df.MeasurementFlag[index_cur] == 'T':
                percipFlag = True
            percipAmt = round(percipAmt + entryList[3] / 8)    #round down
            # print('Snow fall was ' + str(entryList[3]))
        elif element == 'TMIN':
            entry[element] = entryList[3]
            # print('TempMin was ' + str(entryList[3]))
        elif element == 'SNWD':
            entry[element] = entry[element] + entryList[3]
            # print('Snow depth was ' + str(entryList[3]))

        if MODEL_VERSION > 1 or MODEL_VERSION == 0:
            if element == 'EVAP':
                entry[element] = entry[element] + entryList[3]
                # print('Evaporation of water (tenth of mm) was ' + str(entryList[
                # 3]))
            elif element == 'MNPN':
                entry[element] = entryList[3]
            elif element == 'MXPN':
                entry[element] = entryList[3]
            elif element == 'ACMC':
                entry[element] = entryList[3]
            elif element == 'PSUN':
                entry[element] = entryList[3]
        # TODO - Add more 'elif' statements for the rest of the ELEMENT/VALUE pairs.
        else:
            continue
            # print('Found element ' + str(entryList[2]))

    # After going through all of the days events; fill in precipitation info.
    entry['PRECIPFLAG'] = percipFlag
    entry['PRECIPAMT'] = percipAmt

    if dayToProcess > 0:
        df_PrevDay = df_dataSummary.index[df_dataSummary['Date'] ==
                                          df_listOfDays[dayToProcess - 1]]

        entry['PREV_PRECIPFLAG'] = df_dataSummary.at[df_PrevDay[0],
                                                     'PRECIPFLAG']
        entry['PREV_PRECIPAMT'] = df_dataSummary.at[df_PrevDay[0],
                                                     'PRECIPAMT']
        df_dataSummary.at[df_PrevDay[0], 'NEXTPRECIPFLAG'] = percipFlag
        df_dataSummary.at[df_PrevDay[0], 'NEXTPRECIPAMT'] = percipAmt
    else:
        entry['PREV_PRECIPFLAG'] = False
        entry['PREV_PRECIPAMT'] = 0



    # Convert the day summary from a list too a dataframe.
    df_entry = pandas.DataFrame([entry])
    #print(df_entry)

    # 5) Add a new TOTAL DAY entry of weather events in a data frame.
    df_dataSummary = pandas.concat([df_dataSummary, df_entry],
                                   ignore_index=True)

    if dayToProcess == int(numDays * 0.25):
        print("Processing: 25%")
    elif dayToProcess == int(numDays * 0.75):
        print("Processing: 75%")

print("Processing: COMPLETE")

##########################################################################
# TODO - Update headers w/ end units

############################################
# Publish a new DataQualityReport on the list of single day data entries.
report2 = DataQualityReport()

for thisLabel in NEW_HEADER:  # for each column, report basic stats
    thisCol = df_dataSummary[thisLabel]
    report2.addCol(thisLabel, thisCol)

print("DataQualityReport - 2/2")
print(report2.to_string())

############################################
# Publish a day summary dataframe to a csv
if OUTPUT_FILE:
    df_dataSummary.to_csv(FILE_OUT)
