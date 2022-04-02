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
import matplotlib.pyplot as plt

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

# Establish range of k values to compare similarity model accuracy.
MIN_K = 1
MAX_K_exclusive = 16

INPUT_FILE = '../Info&Data/BattingSalaries.xlsx'
OUTPUT_DQR1 = '../artifacts/BattingSalaries_1_DQR1.xlsx'
OUTPUT_DQR2 = '../artifacts/BattingSalaries_1_DQR2.xlsx'
OUTPUT_DQR3 = '../artifacts/BattingSalaries_1_DQR3_Normalized.xlsx'
OUTPUT_FILE = '../artifacts/BattingSalaries_EDIT.xlsx'

# Validate control flags
assert DEBUG in [0, 1, True, False,
                 None], 'DEBUG flag must be a valid true-false value'
assert OUTPUT_FILES in [0, 1, True, False,
                        None], 'OUTPUT_FILES flag must be a valid binary value'
assert MIN_K in [MIN_K < MAX_K_exclusive], 'min K must be smaller than max'
assert MAX_K_exclusive in range(1, 17), 'Only valid range 1-15'

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
                col.where(col <= 1, 1, inplace=True)
            if col.min() < 0:
                col.where(col >= 0, 0, inplace=True)

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

# Normalize all numeric features.
for label in FEATURES_TO_NORMALIZE:
    df_stats[label] = df_stats[label]/df_stats[label].max()
report3 = DataQualityReport()
report3.quickDQR(df_stats, list(df_stats.columns))

if OUTPUT_FILES:
    report1.to_excel(OUTPUT_DQR1)
    report2.to_excel(OUTPUT_DQR2)
    report3.to_excel(OUTPUT_DQR3)


################################
# Create kNN Model
training = df_stats.sample(frac=TRAIN_RATIO, random_state=RANDOM_SEED)
test = df_stats.drop(training.index)

# Get Training set and One-Hot Encoding features
x_train = training.drop(columns=['teamID', 'lgID'])
y_train_lgId = pd.get_dummies(training.lgID, prefix='lg')
y_train_teamID = pd.get_dummies(training.teamID, prefix='teamID')

# Get Test set and One-Hot Encoding features
x_test = test.drop(columns=['teamID', 'lgID'])
y_test_lgID = pd.get_dummies(test.lgID, prefix='lg')
y_test_teamID = pd.get_dummies(test.teamID, prefix='teamID')


################################
# Train the 'lgID' and 'TeamId' classifiers and calculate accuracy
rangeK = range(MIN_K, MAX_K_exclusive)
lgId_accuracy = np.zeros(len(rangeK))
teamId_accuracy = np.zeros(len(rangeK))

for i in range(MAX_K_exclusive - MIN_K):
    k = i + MIN_K
    print("Training Model k = {} out of {} Starting...".format(k, MAX_K_exclusive - 1))

    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform')
    clf.fit(x_train, y_train_lgId)
    result_lgId = clf.predict(np.asarray(x_test))
    lgId_accuracy[i] = clf.score(x_test, y_test_lgID)

    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform')
    clf.fit(x_train, y_train_teamID)
    result_teamId = clf.predict(np.asarray(x_test))
    teamId_accuracy[i] = clf.score(x_test, y_test_teamID)


################################
# Plot Classification Accuracy chart of both features 'lgID' and 'teamID'.
fig = plt.figure()
id_scatter = fig.add_subplot(111)

id_scatter.scatter(rangeK, lgId_accuracy, marker='o', label='lg')
id_scatter.scatter(rangeK, teamId_accuracy, marker='^', label='team')
plt.xlabel("k (# of Nearest Neighbors)")
plt.ylabel("Accuracy of BOTH Classifications")
plt.legend(loc='best')
plt.show()
