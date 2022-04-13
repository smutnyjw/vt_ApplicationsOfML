'''
@package    vt_ApplicationsOfML
@fileName   hw5_FindBestNeuralnetwork.py
@author     John Smutny
@date       04/013/2022
@info       Create a script that will train various Artificial Neural
            Networks and determine which architecture performs the best. The
            ANN models are judged based on AUROC and Misclassification Rate,
            with the performance of each architecture being outputted to an
            external xlsx file.

            Configurations
            1. Number of hidden layers: 1-3
            2. Number of Nodes possible in a hidden layer: 1-10
            3. Internal activation function: relu, logistic, identity and tanh
'''

from sklearn import neural_network as ann
from sklearn import metrics
from sklearn import preprocessing as preproc
import sklearn.model_selection as modelsel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local libraries
from vt_ApplicationsOfML.Libraries.DataExploration.DataQualityReport import \
    DataQualityReport

## Program settings
################################################################################

DEBUG = False
OUTPUT_FILES = False
TRAIN_DATA = 0.7
TEST_DATA = 0.2
RANDOM_SEED = 23
DATA_USE_RATIO = 0.1

# Neural Network Qualities to be tested.
RANGE_HL = range(1,3)
RANGE_NODES = range(1,10)
RANGE_ActFcts = ['relu', 'logistic', 'identity', 'tanh']


## Control flags and constants
################################################################################

INPUT_FILE = '../data/ccpp.xlsx'
INPUT_SHEET = 'allBin'
OUTPUT_DQR = '../artifacts/ccpp_DQR.xlsx'
OUTPUT_ANN_STATS = '../artifacts/results_ann_stats.xlsx'
OUTPUT_IMAGE = '../artifacts/linearRegressionResults-seed{}.png'.format(
    RANDOM_SEED)


ID_FEATURE = 'ID'
FEATURES = ['AT', 'V', 'AP', 'RH']
TARGET_FEATURE = 'TG'


## Helper Functions
################################################################################
def prepareData(df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    # Reduce the number of samples used to speed up testing
    df = df.sample(frac=DATA_USE_RATIO)

    # isolate the Independent variables
    X = df.drop([ID_FEATURE, TARGET_FEATURE], axis=1).to_numpy()
    scalerX = preproc.MinMaxScaler()
    scalerX.fit(X)

    # Normalize the Independent variable
    X = scalerX.transform(X)

    # isolate the Target variable
    Y = df[TARGET_FEATURE].to_numpy()

    return [X, Y]


def chooseBestANN(x: pd.DataFrame, y: pd.DataFrame, hls, nodes, actFcts) -> [
    ann.MLPRegressor, pd.DataFrame]:

    # Set up dataframe to collect statistics
    ERROR_HEADER = ['ID', 'ActivationFunction', 'numHiddenLayers',
              'numNodes_Layer1', 'numNodes_Layer2', 'numNodes_Layer3',
              'TrainingLoss', 'ValidationLoss', 'MSE']
    df_error = pd.DataFrame(columns=ERROR_HEADER)

    #################################################
    # Iterate through all defined Neural Network qualities as defined in the
    #   Settings section and record the model's performance.
    #   1) Activation Function
    #   2) # of Hidden Layers
    #   3) # of Nodes per Hidden Layer

    # Set up train, validation, and test data
    trainX, trainY, validX, validY, testX, testY = \
        modelsel.train_test_split(x, y,
                                  valid_size=TRAIN_DATA,
                                  test_size=TEST_DATA,
                                  random_state=RANDOM_SEED)

    # Define an array of empty hidden layers based on the max number of
    # layers tested.
    hl = [0] * max(hls)

#TODO - Define how the code will assign particular number of nodes to each hl.
    # Begin for loop iteration
    for fct in actFcts:

        # Define how many hidden layers are in this iteration
        for nhls in hls:

            # Set the hidden layers for the iteration
            for i in range(nhls):
                #hl[i] = 1



## Data Handling Functions
################################################################################

# Load and analyze
df_raw = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET)
report = DataQualityReport()
report.quickDQR(df_raw, list(df_raw.columns))

if OUTPUT_FILES:
    report.to_excel(OUTPUT_DQR)

# Prepare data
[x, y] = prepareData(df_raw)

# Determine the best artificial Neural Network
[bestANN, df_ErrorData] = chooseBestANN(x, y)


# Plot the best ANN error data.
#TODO
