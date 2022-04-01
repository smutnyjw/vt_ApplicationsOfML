'''
@package    HW4-Similarity&linearRegression
@fileName   hw4_LinearRegression-PredictYearlySalary
@author     John Smutny
@date       04/05/2022
@info       Create a script that will predict the salary of a player for any
            given year included in the dataset. Use a multivariate linear
            regression model to predict the salary for each year.

            Accuracy for each year's prediction is measured by r2 and mean
            square error values.
'''

# Third Party libraries
import numpy as np
import pandas as pd

# Local libraries
from vt_ApplicationsOfML.Libraries.DataExploration.DataQualityReport import \
    DataQualityReport

## Control flags and constants
################################################################################
DEBUG = False
OUTPUT_FILES = True
TRAIN_RATIO = 0.7
RANDOM_SEED = 22222
#RANDOM_SEED = 100

INPUT_FILE = '../Info&Data/BattingSalaries_cut.xlsx'
OUTPUT_DQR1 = '../artifacts/BattingSalaries_2_DQR1.xlsx'
OUTPUT_DQR2 = '../artifacts/BattingSalaries_2_DQR2.xlsx'
OUTPUT_DQR3 = '../artifacts/BattingSalaries_2_DQR3_OneHotEncoding.xlsx'
OUTPUT_FILE = '../artifacts/BattingSalaries_EDIT.xlsx'

## Helper Functions
################################################################################
FEATURES_TO_NORMALIZE = [
    'G',
    'AB',
    'R',
    'H',
    '2B',
    '3B',
    'HR',
    'RBI',
    'SB',
    'CS',
    'BB',
    'SO',
    'IBB',
    'HBP',
    'SH',
    'SF',
    'GIDP',
    'Salary'
]

## Data Handling Functions
################################################################################

def cleanRawData(df: pd.DataFrame) -> pd.DataFrame:

    # 1) Drop the quality_flag & source_flag column
    df = df.drop(columns=['playerID', 'yearPlayer'])

    # 2) Replace all NULL or #N/A values with the median of that feature
    for label in df.columns:
        col = df[label]

        if col.isnull().sum() != 0:
            medianV = col.median()
            df[label] = col.fillna(medianV)

        # 3) Set a [0, 1] clamp on every 'Rate' feature (*rat in the name).
        if 'rat' in label:
            if col.max() > 1:
                col.where(col <= 1, 1, inplace=True)
            if col.min() < 0:
                col.where(col >= 0, 0, inplace=True)

        # 4) Check for invalidly high values (>1000) replace with 0
        if type(col[col.first_valid_index()]) != str and label != 'Salary':
            col.where(col < 1000, 0, inplace=True)
            #TODO - Figure out if there is a 'count' cmd to know how many
            # entries are effected by these adjustments.

    return df

## Main Data Processing
################################################################################

# Load and analyze
df_raw = pd.read_excel(INPUT_FILE, sheet_name='Batting')
report1 = DataQualityReport()
report1.quickDQR(df_raw, list(df_raw.columns))


# Data Preparation
df_stats = cleanRawData(df_raw)
report2 = DataQualityReport()
report2.quickDQR(df_stats, list(df_stats.columns))

# Apply One-Hot Encoding Features [lgId, TeamId]
# Convert categorical 'lgId' and 'teamId' via One-Hot Encoding
col_lgId = pd.get_dummies(df_stats.lgID, prefix='lg')
col_teamID = pd.get_dummies(df_stats.teamID, prefix='teamID')

df_stats.drop(columns=['lgID', 'teamID'], inplace=True)
df_stats = df_stats.join(col_lgId)
df_stats = df_stats.join(col_teamID)

report3 = DataQualityReport()
report3.quickDQR(df_stats, list(df_stats.columns))

if OUTPUT_FILES:
    report1.to_excel(OUTPUT_DQR1)
    report2.to_excel(OUTPUT_DQR2)
    report3.to_excel(OUTPUT_DQR3)


################################
# Create Linear Regression model for ALL years



################################
# Create Linear Regression model for each year
