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
#RANDOM_SEED = 22222
RANDOM_SEED = 100

# Establish range of k values to compare similarity model accuracy.
MIN_K = 1
MAX_K_exclusive = 16

INPUT_FILE = '../Info&Data/BattingSalaries.xlsx'
OUTPUT_DQR1 = '../artifacts/BattingSalaries_1_DQR1.xlsx'
OUTPUT_DQR2 = '../artifacts/BattingSalaries_1_DQR2.xlsx'
OUTPUT_DQR3 = '../artifacts/BattingSalaries_1_DQR3_Normalized.xlsx'
OUTPUT_FILE = '../artifacts/BattingSalaries_EDIT.xlsx'
OUTPUT_IMAGE = '../artifacts/kNN_Results-seed{}.png'.format(
    RANDOM_SEED)

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
    ''' Clean undesired data from the inputted data '''
    print("Begin Data Preparation...")

    # Keep track of number of values effected by each step
    orig_size = len(df)
    numNULL = 0
    numAbove1 = 0
    numBelow0 = 0
    numHigh = 0

    # 2) Drop the quality_flag & source_flag column
    df = df.drop(columns=['playerID', 'yearPlayer'])

    # 3) Replace all NULL or #N/A values with the median of that feature
    for label in df.columns:
        col = df[label]

        if col.isnull().sum() != 0:
            numNULL = numNULL + col.isnull().sum()
            medianV = col.median()
            df[label] = col.fillna(medianV)

        # 4) Set a [0, 1] clamp on every 'Rate' feature (*rat in the name).
        if 'rat' in label:
            if col.max() > 1:
                numAbove1 = numAbove1 + np.sum(col > 1)
                col.where(col <= 1, 1, inplace=True)
            if col.min() < 0:
                numBelow0 = numBelow0 + np.sum(col < 0)
                col.where(col >= 0, 0, inplace=True)

        # 5) Check for invalidly high values (>1000) replace with 0
        if type(col[col.first_valid_index()]) != str and \
                label != 'Salary' and label != 'yearID':
            numHigh = numHigh + np.sum(col > 1000)
            col.where(col < 1000, 0, inplace=True)

    print("Data Prep # Changed Values out of {}:\n#NULL: {}"
          "\n#AboveOne: {}\n#BelowZero: {}\n#HighValues: {}".format(
                                                orig_size, numNULL,
                                                numAbove1, numBelow0, numHigh))

    print("Data Preparation COMPLETE...")

    return df


## Main Data Processing
################################################################################

# Load and analyze
df_raw = pd.read_excel(INPUT_FILE, sheet_name='Batting')

# Data Preparation
# Remove all entries that are not from the 2016 season
df_raw = df_raw.query("yearID == 2016")
report1 = DataQualityReport()
report1.quickDQR(df_raw, list(df_raw.columns))

df_stats = cleanRawData(df_raw)
report2 = DataQualityReport()
report2.quickDQR(df_stats, list(df_stats.columns))

# Normalize all numeric features.
for label in FEATURES_TO_NORMALIZE:
    df_stats[label] = df_stats[label] / df_stats[label].max()
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
    print("Training Model k = {} out of {} Starting...".format(k,
                                                               MAX_K_exclusive - 1))

    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform')
    clf.fit(x_train, y_train_lgId)
    result_lgId = clf.predict(np.asarray(x_test))
    lgId_accuracy[i] = clf.score(x_test, y_test_lgID)

    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform')
    clf.fit(x_train, y_train_teamID)
    result_teamId = clf.predict(np.asarray(x_test))
    teamId_accuracy[i] = clf.score(x_test, y_test_teamID)

################################
# Display Results
best_k_lgId = np.where(lgId_accuracy == max(lgId_accuracy))
best_k_teamId = np.where(teamId_accuracy == max(teamId_accuracy))
print("RESULTS: Best (k/accuracy):\n\tlgId = {}/{}\n\tteamId = {},"
      "{}".format(best_k_lgId[0]+1, max(lgId_accuracy),
                  best_k_teamId[0]+1, max(teamId_accuracy)))

worst_k_lgId = np.where(lgId_accuracy == min(lgId_accuracy))
worst_k_teamId = np.where(teamId_accuracy == min(teamId_accuracy))
print("RESULTS: Worst (k/accuracy):\n\tlgId = {}/{}\n\tteamId = {},"
      "{}".format(worst_k_lgId[0]+1, min(lgId_accuracy),
                  worst_k_teamId[0]+1, min(teamId_accuracy)))

# Plot Classification Accuracy chart of both features 'lgID' and 'teamID'.
fig, scat1 = plt.subplots()

scat1.scatter(rangeK, lgId_accuracy, marker='o', c='blue', label='lgID')
scat1.set_ylim([min(lgId_accuracy)*0.50, 1])
scat1.set_xlabel("k (# of Nearest Neighbors)")
scat1.set_ylabel("Classification Accuracy of lgID")
scat1.legend(loc='upper left')

scat2 = scat1.twinx()
scat2.scatter(rangeK, teamId_accuracy, marker='^', c='orange', label='teamID')
scat2.set_ylim([-0.01, max(teamId_accuracy)*3])
scat2.set_ylabel("Classification Accuracy of TeamID")
plt.title("kNN Accuracy of MLB Player Teams & League Affiliation - seed "
            "{}".format(RANDOM_SEED))
scat2.legend(loc='upper right')
plt.show()

if OUTPUT_FILES:
    fig.savefig(OUTPUT_IMAGE, format='png', dpi=100)
