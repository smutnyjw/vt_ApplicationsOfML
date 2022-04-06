'''
@package    HW4-Similarity&linearRegression
@fileName   hw4_LinearRegression-PredictYearlySalary
@author     John Smutny
@date       04/03/2022
@info       Create a script that will predict the salary of a player for any
            given year included in the dataset. Use a multivariate linear
            regression model to predict the salary for each year.

            Accuracy for each year's prediction is measured by r2 and mean
            square error values.
'''

# Third Party libraries
import numpy as np
import pandas as pd
from sklearn import linear_model as linmod
from sklearn import metrics
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

INPUT_FILE = '../Info&Data/BattingSalaries.xlsx'
OUTPUT_DQR1 = '../artifacts/BattingSalaries_2_DQR1.xlsx'
OUTPUT_DQR2 = '../artifacts/BattingSalaries_2_DQR2.xlsx'
OUTPUT_DQR3 = '../artifacts/BattingSalaries_2_DQR3_OneHotEncoding.xlsx'
OUTPUT_R2 = '../artifacts/results_r2.xlsx'
OUTPUT_MSE = '../artifacts/results_mse.xlsx'
OUTPUT_IMAGE = '../artifacts/linearRegressionResults-seed{}.png'.format(
    RANDOM_SEED)

## Helper Functions
################################################################################

TARGET_FEATURE = 'Salary'

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
    orig_length = len(df)
    orig_size = len(df) * len(df.columns)
    numDropped = 0
    numRatNULL = 0
    numSalaryPost2016 = 0
    numSalaryNULL = 0
    numAbove1 = 0
    numBelow0 = 0
    numHigh = 0

    # 1) Drop the quality_flag & source_flag column
    df = df.drop(columns=['playerID', 'yearPlayer'])
    numDropped = orig_size - len(df) * len(df.columns)

    # 2) Drop any player with salary of zero.
    numDropped = numDropped + len(df[df['Salary'] == 0].sum())
    df = df.query("Salary != 0")

    # 2) Drop all data entries 2017 & later
    numSalaryPost2016 = numSalaryPost2016 + len(df.query("yearID >= 2017"))

    # 3) Replace any NULL salaries with that year's median
    df.query("yearID < 2017", inplace=True) #TODO - TEST
    numSalaryNULL = numSalaryNULL + df['Salary'].isnull().sum()

    for yr in df['yearID'].unique():
        medianV = df[df['yearID'] == yr]['Salary'].median()
        yrSalary = df[df['yearID'] == yr]['Salary']
        df.loc[df['yearID'] == yr, 'Salary'] = yrSalary.fillna(medianV)

    # Refine values based on the batting statistic
    for label in df.columns:
        col = df[label]

        # Refine the 'rate' values
        if 'rat' in label:
            # 4) Set a [0, 1] clamp on every 'Rate' feature (*rat in the name).
            if col.max() > 1:
                numAbove1 = numAbove1 + np.sum(col > 1)
                col.where(col <= 1, 1, inplace=True)
            if col.min() < 0:
                numBelow0 = numBelow0 + np.sum(col < 0)
                col.where(col >= 0, 0, inplace=True)

            # 5) Replace all null 'Rate' values with zero (*rat in the name).
            if col.isnull().sum() != 0:
                numRatNULL = numRatNULL + col.isnull().sum()
                df[label] = col.fillna(0)

        # 6) Check for invalidly high values (>1000) replace with 0
        if type(col[col.first_valid_index()]) != str and \
                label != 'Salary' and label != 'yearID':
            numHigh = numHigh + np.sum(col > 1000)
            col.where(col < 1000, 0, inplace=True)

    print("Data Prep # Changed Values out of {}:\n#Dropped: {}\n#RatNULL: {}"
          "\n#SalaryNULL: {}\n#SalaryPost2016: {}\n"
          "#AboveOne: {}\n#BelowZero: {}\n#HighValues: {}".format(
                                        orig_size, numDropped, numRatNULL,
                                        numSalaryNULL, numSalaryPost2016,
                                        numAbove1, numBelow0, numHigh))

    print("Data Preparation COMPLETE...")

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
training = df_stats.sample(frac=TRAIN_RATIO, random_state=RANDOM_SEED)
test = df_stats.drop(training.index)

train_x = training.drop(columns=[TARGET_FEATURE]).to_numpy()
train_y = training[TARGET_FEATURE].to_numpy()
test_x = test.drop(columns=[TARGET_FEATURE]).to_numpy()
test_y = test[TARGET_FEATURE].to_numpy()

mlr = linmod.LinearRegression()
mlr.fit(train_x, train_y)

# Determine Accuracy measures (r2 & MSE)
r2 = mlr.score(test_x, test_y)
mse = metrics.mean_squared_error(test_y, mlr.predict(test_x))
print("LinearRegression r2 & MSE: {} & {}".format(r2, mse))

################################
# Create Linear Regression model for each year
r2_by_year = {}
mse_by_year = {}

for yr in df_stats['yearID'].unique():
    dataYr = df_stats[df_stats['yearID'] == yr]

    training = dataYr.sample(frac=TRAIN_RATIO, random_state=RANDOM_SEED)
    test = dataYr.drop(training.index)

    train_x = training.drop(columns=[TARGET_FEATURE]).to_numpy()
    train_y = training[TARGET_FEATURE].to_numpy()
    test_x = test.drop(columns=[TARGET_FEATURE]).to_numpy()
    test_y = test[TARGET_FEATURE].to_numpy()

    mlr = linmod.LinearRegression()
    mlr.fit(train_x, train_y)

    # Determine Accuracy measures (r2 & MSE)
    r2_by_year[yr] = mlr.score(test_x, test_y)
    mse_by_year[yr] = metrics.mean_squared_error(test_y, mlr.predict(test_x))
    print("Accuracy: {} - {}/{} from {}".format(yr, r2_by_year[yr],
                                        mse_by_year[yr], len(dataYr)))

# Publish 'accuracy by year' results
if OUTPUT_FILES:
    df_r2 = pd.DataFrame(data=r2_by_year, index=[0])
    df_r2.to_excel(OUTPUT_R2)
    df_mse = pd.DataFrame(data=mse_by_year, index=[0])
    df_mse.to_excel(OUTPUT_MSE)

################################
# Plot MSE over year to determine the most predictable salary year.
plt.plot(mse_by_year.keys(), list(mse_by_year.values()), 'o',
         color='black')
plt.title("MSE of Predicting MLB Player Salary by Year - seed {}".format(
                                                                RANDOM_SEED))
plt.xlabel("Year")
plt.ylabel("Mean Square Error")
plt.show()
