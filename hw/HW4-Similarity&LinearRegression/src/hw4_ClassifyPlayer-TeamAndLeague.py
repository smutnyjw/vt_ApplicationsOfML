'''
@package    HW4-Similarity&linearRegression
@fileName   hw4_ClassifyPlayer-TeamAndLeague
@author     John Smutny
@date       04/05/2022
@info       Create a script that will predict a 2016 player's league
            affiliation and team affiliation based on the player's batting
            statistics & salary by using a k-NearestNeighbor Classification
            model.
'''

# Python Libraries

# Third Party libraries
from sklearn import neighbors
import numpy as np
import pandas as pd

# Group K project 1 libraries
from vt_ApplicationsOfML.Libraries.DataExploration.DataQualityReport import \
    DataQualityReport

## Control flags and constants
################################################################################
DEBUG = False
OUTPUT_FILES = False
TRAIN_RATIO = 0.7
RANDOM_SEED = 22222

INPUT_FILE = '../Info&Data/BattingSalaries_cut.xlsx'
OUTPUT_DQR1 = '../artifacts/BattingSalaries_DQR1.xlsx'
OUTPUT_DQR2 = '../artifacts/BattingSalaries_DQR2.xlsx'
OUTPUT_FILE = '../artifacts/BattingSalaries_EDIT.xlsx'

# Validate control flags
assert DEBUG in [0, 1, True, False,
                 None], 'DEBUG flag must be a valid true-false value'
assert OUTPUT_FILES in [0, 1, True, False,
                 None], 'OUTPUT_FILES flag must be a valid true-false value'

## Helper Functions
################################################################################


## Data Handling Functions
################################################################################

def cleanRawData(df: pd.DataFrame) -> pd.DataFrame:
    ''' Clean undesired data from the raw NOAA data frame '''
    print("Size of Initial Dataset: {}".format(len(df)))

    # 1) Remove all entries that are not from the 2016 season
    df = df.query("yearID >= 2016")
    print("Size of Dataset with only 2016 entries: {}".format(len(df)))

    # 2) Drop the quality_flag & source_flag column
    df = df.drop(columns=['playerID', 'yearPlayer'])

    # 3) Replace all NULL or #N/A values with the median of that feature
    for label in df.columns:
        col = df[label]

        if col.isnull().sum() != 0:
            medianV = col.median()
            df[label] = col.fillna(medianV)

        # 4) Set a [0, 1] clamp on every 'Rate' feature (*rat in the name).
        if 'rat' in label:
            if col.max() > 1:
                col.where(col <= 1, 1)
            if col.min() < 0:
                col.where(col >= 0, 0)

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

if OUTPUT_FILES:
    report1.to_excel(OUTPUT_DQR1)
    report2.to_excel(OUTPUT_DQR2)


# Create kNN Model
x = df_stats.drop(columns=['teamID', 'lgID'])
x_training = x.sample(frac=TRAIN_RATIO, random_state=RANDOM_SEED)
x_test = x.drop(x_training.index)

y_teamID = df_stats.teamID
y_lgID = df_stats.lgID
clf = neighbors.KNeighborsClassifier(n_neighbors=1, weights='uniform')
clf.fit(x_training, y_lgID)
result_teamID = clf.predict(np.asarray(x_test).reshape(1, -1))
print("Using scikit class: ", result_teamID)

