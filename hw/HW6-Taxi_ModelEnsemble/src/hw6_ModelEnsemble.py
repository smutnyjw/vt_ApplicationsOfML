'''
@package    vt_ApplicationsOfML
@fileName   hw6_ModelEnsemble.py
@author     John Smutny
@date       04/28/2022
@info

        Each model will be analyzed based on their MSE, MAE, R2 and EVS.

        Ensemble model will have a scatterplot plotting the learning curve by
        epoch.
'''

from sklearn import tree
from sklearn import linear_model as linmod
from sklearn import neural_network as ann

from sklearn import metrics
from sklearn import preprocessing as preproc
import sklearn.model_selection as modelsel
import datetime
import pandas as pd

# Local libraries
from vt_ApplicationsOfML.Libraries.DataExploration.DataQualityReport import \
    DataQualityReport

## Program settings
################################################################################

TRAIN_DATA = 0.8
TEST_DATA = 0.2
VALID_DATA_FROM_TRAIN = 0.25
RANDOM_SEED = 5

DEBUG = True
OUTPUT_FILES = True
DATA_USE_RATIO = 1

# Values used for debugging only. Overwrite if Debugging.
if DEBUG:
    DATA_USE_RATIO = 0.1


## Control flags and constants
################################################################################

INPUT_FILE = '../data/Taxi_Trip_Data.xlsx'
OUTPUT_DQR = '../artifacts/taxi_DQR.xlsx'
OUTPUT_PLOT = '../artifacts/learningPlot.png'

DROPPED_FEATURES = ['store_and_fwd_flag', 'PULocationID', 'DOLocationID',
                    'fare_amount', 'extra', 'mta_tax', 'tip_amount',
                    'tolls_amount']
TARGET_FEATURE = 'total_amount'


## Helper Functions
################################################################################


def prepData(df: pd.DataFrame) -> pd.DataFrame:
    # Reduce the number of samples used to speed up testing
    df = df.sample(frac=DATA_USE_RATIO)

    # Dropped already specified features to drop.
    df = df.drop(DROPPED_FEATURES, axis=1)

    ###############################################
    # Combine the PickUp and DropOff times into FareTime feature
    df.insert(1, "FareTime", 0)
    pu_time = pd.to_datetime(df['lpep_pickup_datetime'].tolist())
    do_time = pd.to_datetime(df['lpep_dropoff_datetime'].tolist())

    for i in range(len(df['lpep_pickup_datetime'])):
        # Brute force
        pickUp = pu_time[i].time().hour*360 + pu_time[i].time().minute*60 + \
                    pu_time[i].second
        dropOff = do_time[i].time().hour*360 + do_time[i].time().minute*60 + \
                    do_time[i].second
        df.loc[i, 'FareTime'] = dropOff - pickUp

    df = df.drop(['lpep_dropoff_datetime', 'lpep_pickup_datetime'], axis=1)

    ###############################################
    # Perform One-Hot Encoding on PickUp & DropOff features

    #TODO

    ###############################################
    # Normalize features
    X = df.drop(TARGET_FEATURE, axis=1).to_numpy()
    scalerX = preproc.MinMaxScaler()
    scalerX.fit(X)

    # Normalize the Independent variable
    X = scalerX.transform(X)
    print(X)
    # isolate the Target variable
    Y = df[TARGET_FEATURE].to_numpy()
    print(Y)

    return df


## Data Handling Functions
################################################################################

# Load and analyze
df_raw = pd.read_excel(INPUT_FILE)
report = DataQualityReport()
report.quickDQR(df_raw, list(df_raw.columns))

if OUTPUT_FILES:
    report.to_excel(OUTPUT_DQR)

# Data Preparation
df_prep = prepData(df_raw)

