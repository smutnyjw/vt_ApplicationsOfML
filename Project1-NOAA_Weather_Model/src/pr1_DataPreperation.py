################################################################################
#   File:   preparedata.py
#   Author: Ben Johnson
#   Original Author: John Smutny pr1_DataPreparation.py
#   Course: ECE-5984: Applications of Machine Learning
#   Date:   03/12/2022
#   Description:
#       For Project1; perform initial data preparation to get NOAA provided
#       data into a usable state to model.
'''
@package    project1.preparedata
@author     Group K
@info       Script for parsing NOAA daily readings summary and generate a
            summary of all possible readings for each day.
            Data Quality Reports are generate for:
                - Raw NOAA data
                - Unmodified daily summaries
                - Cleaned and normalized daily summaries
            Control flags at the top of this file set the format of output
            data and where to collect/deposit inputs and outputs.
            See the README and const.py for more information.
'''

# Python Libraries

# Third Party libraries
import pandas as pd
import numpy as np
from pprint import pprint
import copy
import sys

# Group K project 1 libraries
from lib.dataqualityreport import DataQualityReport
import lib.constants as constants

## Control flags and constants
################################################################################
DEBUG               = False
MODEL_VERSION       = 1
PERCENT_DAYS        = 100
INPUT_FILE          = 'data/USW00013880.csv'
OUTPUT_FILE         = 'processed/USW00013880-SINGLEDAY.csv'

# Validate control flags
_ALLOWABLE_MODELS   = [0, 1, 2]
assert DEBUG in [0, 1, True, False, None], 'DEBUG flag must be a valid true-false value'
assert MODEL_VERSION in _ALLOWABLE_MODELS, 'MODEL_VERSION supports {_ALLOWABLE_MODELS}'
assert PERCENT_DAYS <= 100 or PERCENT_DAYS == None, 'PERCENT_DAYS <= 100 or None'
assert PERCENT_DAYS >= 0 or PERCENT_DAYS == None, 'PERCENT_DAYS >= 0 or None'

# Definitions of feature/column groups
CALCULATED_FEATURES = [
    'PRECIPFLAG',
    'PRECIPAMT'
]

TARGET_VARIABLES = ['NEXTPRECIPFLAG', 'NEXTPRECIPAMT']

## Helper Functions
################################################################################
def wait():
    ''' Function to pause for user-interaction before continuining
    Disable/Enable with DEBUG flag
    '''
    if DEBUG:
        input()

def summarize(df: pd.DataFrame):
    ''' Print summaries for a dataframe '''
    if not DEBUG:
        return
    print(df.describe())
    print()
    print(df.info())

## Data Handling Functions
################################################################################
def LoadNOAAData(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, header=None, names=constants.NOAA_DAILY_HEADER)

def cleanRawData(df: pd.DataFrame) -> pd.DataFrame:
    ''' Clean undesired data from the raw NOAA data frame '''
    # 1) Remove all data points that have a non-NULL quality_flag
    df = df[df.quality_flag.isnull()]

    # 1b) TODO - Get indexes of all MeasurementFlag 'T' values?

    # 2) Drop the quality_flag & source_flag column
    df = df.drop(columns=['quality_flag', 'source_flag'])
    return df

## Main Data Processing
################################################################################
print('\nRaw Data Characteristics:\n==========================================')
raw_df = LoadNOAAData(INPUT_FILE)
summarize(raw_df)
wait()

## Create an organized data set summary for the console using a data frame.
report1 = DataQualityReport()
for thisLabel in raw_df.columns:
    thisCol = raw_df[thisLabel]
    if thisLabel == 'element'\
            or thisLabel == 'measurement_flag'\
            or thisLabel == 'quality_flag'\
            or thisLabel == 'source_flag':
        report1.addCatCol(thisLabel, thisCol)
    else:
        report1.addCol(thisLabel, thisCol)
print('\n\nRaw Data DQR - 1/3:\n====================================================')
print(report1.to_string())
report1.to_csv(OUTPUT_FILE.replace('.csv', '-RAWDQR.csv'))
wait()

## Clean data before generating proceseed data DQR
print('\n\nCleaned Raw Data DQR:\n====================================================')
clean_df = cleanRawData(raw_df)
summarize(clean_df)
wait()

## Create the processed, unnormalized dataframe of single-day entries

# Initialize the new dataframe and get the target days
UNIQUE_ELEMENTS = clean_df['element'].unique()
PROCESSED_HEADER = constants.OUTPUT_ID_COLUMNS
PROCESSED_HEADER.extend(UNIQUE_ELEMENTS)
PROCESSED_HEADER.extend(CALCULATED_FEATURES)
PROCESSED_HEADER.extend(TARGET_VARIABLES)
proc_df = pd.DataFrame(columns=PROCESSED_HEADER)
days = clean_df['date'].unique()

# Allow the user to limit data if desired -- really only good for debugging
if PERCENT_DAYS:
    days = days[0-int(PERCENT_DAYS*(days.size/100)):]

## Create the multi-day target dataframe
DOUBLE_COLUMNS = proc_df.drop(columns=['NEXTPRECIPFLAG', 'NEXTPRECIPAMT']).columns.to_list()
two_df = pd.DataFrame(columns=DOUBLE_COLUMNS)
DUB_COL = proc_df.drop(
    columns=['NEXTPRECIPFLAG', 'NEXTPRECIPAMT']
).columns.to_list()
for column in DUB_COL:
    DOUBLE_COLUMNS.append('PREV' + column.upper())
DOUBLE_COLUMNS.extend(['NEXTPRECIPFLAG', 'NEXTPRECIPAMT'])

# Loop through NOAA readings and add values to target output
#   => This will include entries for all elements -- drop(columns) to filter
progress = 0
progress_step = 100 / days.size
print(f'Processing {progress:.2f}%\r', end='')
for didx, day in enumerate(days):
    # Initialize variables for a single day
    events = clean_df.loc[clean_df['date'] == day]
    eventCount = len(events.index)
    if eventCount == 0:
        raise RuntimeError(f'{day} had no applicable events. Please correct data.')
    blankValues = [np.nan] * len(PROCESSED_HEADER)
    entry = dict(zip(PROCESSED_HEADER, blankValues))
    entry['date'] = events.iloc[0]['date']
    datestr = str(entry['date'])
    entry['year'] = int(datestr[:4])
    entry['month'] = int(datestr[4:6])
    entry['day'] = int(datestr[6:8])
    entry['event_count'] = eventCount
    precipFlag = 0
    precipAmount = 0

    # Zero out element readings since they're all numerical
    for key in UNIQUE_ELEMENTS:
        entry[key] = 0

    # Summarize all events from the day into a single entry
    for i in range(0, eventCount):
        # Dissect each event in the current day.
        eidx = events.index[i]
        entryList = events.loc[eidx, :].values.tolist()
        element = entryList[2]

        # Check measurement flag for trace precip
        if clean_df['measurement_flag'][eidx] == 'T':
            precipFlag = 1

        # SWITCH statement to process the ELEMENT + VALUE
        if (entry[element] not in [0, np.nan]) and (entry[element] != entryList[3]):
            raise ValueError(f'{day} had two values for {element}')
        elif element == 'PRCP':
            entry[element] = entryList[3]
            precipAmount = precipAmount + entryList[3]
        elif element == 'SNOW':
            entry[element] = entryList[3]
            precipAmount = round(precipAmount + entryList[3] / 8)
        elif element == 'SNWD':
            entry[element] = entry[element] + entryList[3]
        elif element == 'EVAP':
             entry[element] = entry[element] + entryList[3]
        else:
            entry[element] = entryList[3]

    # After going through all of the days events; fill in precipitation info.
    if precipAmount > 0:
        precipFlag = 1
    entry['PRECIPFLAG'] = precipFlag
    entry['PRECIPAMT'] = precipAmount

    # Add yesterdays precipitation to today as next_precip...
    if didx > 0:
        df_PrevDay = proc_df.index[proc_df['date'] == days[didx-1]]
        proc_df.at[df_PrevDay[0], 'NEXTPRECIPFLAG'] = precipFlag
        proc_df.at[df_PrevDay[0], 'NEXTPRECIPAMT'] = precipAmount
        two_df.at[df_PrevDay[0], 'NEXTPRECIPFLAG'] = precipFlag
        two_df.at[df_PrevDay[0], 'NEXTPRECIPAMT'] = precipAmount

        # Update dubby boi
        dub_entry = copy.deepcopy(entry)
        for column in DUB_COL:
            dub_entry['PREV'+column.upper()] = proc_df.loc[df_PrevDay[0], column]
        two_entry = pd.DataFrame([dub_entry])
        two_df = pd.concat([two_df, two_entry], ignore_index=True)


    # Convert the day summary from a list too a dataframe.
    df_entry = pd.DataFrame([entry])

    # 5) Add a new TOTAL DAY entry of weather events in a data frame.
    proc_df = pd.concat([proc_df, df_entry],
                                   ignore_index=True)

    # Update user with progress
    progress += progress_step
    print(f'Processing {progress:.2f}%\r', end='')

print('Data Processing Complete')
print('====================================================')

## Generate the processed, non-normalized DQR
print(f'\n\nGenerating processed, unnormalized DQR')
print('====================================================')
print(f'Processing {progress:.2f}%\r', end='')
progress = 0
progress_step = 100 / len(PROCESSED_HEADER)
report2 = DataQualityReport()
for thisLabel in PROCESSED_HEADER:
    report2.addCol(thisLabel, proc_df[thisLabel])
    progress += progress_step
    print(f'Processing {progress:.2f}%\r', end='')

print('\n\nProcessed Data DQR - 2/3\n====================================================')
print(report2.to_string())
report2.to_csv(OUTPUT_FILE.replace('.csv', '-PROCDQR.csv'))

## Write the processed, unnormalized data to file
print(f'\n\nWriting processed data to {OUTPUT_FILE}')
print('====================================================')
if OUTPUT_FILE not in ['', None]:
    proc_df.to_csv(OUTPUT_FILE, index_label='id')
    two_df.to_csv(OUTPUT_FILE.replace('.csv', '-DUB.csv'), index_label='id')
print('All processing complete!')
print('====================================================')

## Normalize and write to file
NORMALIZED_FEATURES = UNIQUE_ELEMENTS.tolist()
NORMALIZED_FEATURES.extend(CALCULATED_FEATURES)
norm_df = copy.deepcopy(proc_df)
for key in NORMALIZED_FEATURES:
    if norm_df[key].min() == norm_df[key].max():
        continue
    norm_df[key] = (norm_df[key]-norm_df[key].min())/(norm_df[key].max()-norm_df[key].min())

print(f'\n\nWriting normalized data to {OUTPUT_FILE}')
print('====================================================')
if OUTPUT_FILE not in ['', None]:
    norm_df.to_csv(OUTPUT_FILE.replace('.csv', '-NORMAL.csv'), index_label='id')
print('All processing complete!')
print('====================================================')

## Generate the processed, non-normalized DQR
print(f'\n\nGenerating processed, normalized DQR')
print('====================================================')
print(f'Processing {progress:.2f}%\r', end='')
progress = 0
progress_step = 100 / len(PROCESSED_HEADER)
report3 = DataQualityReport()
for thisLabel in PROCESSED_HEADER:
    report3.addCol(thisLabel, proc_df[thisLabel])
    progress += progress_step
    print(f'Processing {progress:.2f}%\r', end='')
print('\n\nNormalized Data DQR - 3/3\n====================================================')
print(report3.to_string())
report3.to_csv(OUTPUT_FILE.replace('.csv', '-NORMDQR.csv'))