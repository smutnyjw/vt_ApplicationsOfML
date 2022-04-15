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

TRAIN_DATA = 0.8
TEST_DATA = 0.2
VALID_DATA_FROM_TRAIN = 0.25
RANDOM_SEED = 23

DEBUG = False
OUTPUT_FILES = True
DATA_USE_RATIO = 1

# Neural Network Qualities to be tested.
RANGE_HL = range(1, 4)  # Range: 1-3
RANGE_NODES = range(1, 11)  # Range: 1-10
RANGE_ActFcts = ['relu', 'logistic', 'identity', 'tanh']

# Universal ANN training settings for all model variations
SOLVER = 'adam'
MAX_ITER = 10000
LEARNING_RATE = 0.0001
EARLY_STOPPING = True

# Values used for debugging only. Overwrite if Debugging.
if DEBUG:
    DATA_USE_RATIO = 0.1

    # Neural Network Qualities to be tested.
    RANGE_HL = range(1, 4)
    RANGE_NODES = range(1, 4)
    RANGE_ActFcts = ['relu']

    MAX_ITER = 10


## Control flags and constants
################################################################################

INPUT_FILE = '../data/ccpp.xlsx'
INPUT_SHEET = 'allBin'
OUTPUT_DQR = '../artifacts/ccpp_DQR.xlsx'
OUTPUT_BEST_AUROC = '../artifacts/bestANNs_AUROC.xlsx'
OUTPUT_WORST_AUROC = '../artifacts/worstANNs_AUROC.xlsx'
OUTPUT_BEST_MSE = '../artifacts/bestANNs_MSE.xlsx'
OUTPUT_WORST_MSE = '../artifacts/worstANNs_MSE.xlsx'
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


def trainAndLogANN(hl, activationFct, trainXY, testXY):
    clf = ann.MLPRegressor(hidden_layer_sizes=hl,
                           activation=activationFct,
                           solver=SOLVER,
                           alpha=LEARNING_RATE,
                           early_stopping=EARLY_STOPPING,
                           max_iter=MAX_ITER,
                           validation_fraction=0.42)

    clf.fit(trainXY[0], trainXY[1])
    annPredY = clf.predict(testXY[0])
    mse = metrics.mean_squared_error(testXY[1], annPredY)
    auroc = metrics.roc_auc_score(testXY[1], annPredY)

    return [auroc, mse]


def chooseBestANN(x: pd.DataFrame, y: pd.DataFrame, hls, nodes, actFcts) -> [
    pd.DataFrame]:

    # Set up dataframe to collect statistics
    ERROR_HEADER = ['ID', 'ActivationFunction', 'numHiddenLayers',
                    'numNodes_Layer1', 'numNodes_Layer2', 'numNodes_Layer3',
                    'AUROC', 'MSE']
    df_error = pd.DataFrame(columns=ERROR_HEADER)

    #################################################
    # Iterate through all defined Neural Network qualities as defined in the
    #   Settings section and record the model's performance.
    #   1) Activation Function
    #   2) # of Hidden Layers
    #   3) # of Nodes per Hidden Layer

    # Set up train, validation, and test data
    trainX, testX, trainY, testY = modelsel.train_test_split(x, y,
                                                test_size=TEST_DATA,
                                                random_state=RANDOM_SEED)
    trainXY = [trainX, trainY]
    testXY = [testX, testY]

    # Define an array of empty hidden layers based on the max number of
    # layers tested.
    hl = [0] * max(hls)
    id = 0

    # Begin for loop iteration
    for fct in actFcts:

        # Define how many hidden layers are in this iteration
        for numHLs in hls:
            # Define the max number of nodes per HL are in this iteration
            #for maxNumNodes in range(len(RANGE_NODES)):
            maxNumNodes = len(nodes)

            # Generate a counter to set the number of nodes in each HL
            for i in range(((maxNumNodes)**numHLs)):
                hl[0] = i % (maxNumNodes) + 1

                if numHLs > 1:
                    hl[1] = int(i/maxNumNodes) % (maxNumNodes) + 1

                if numHLs > 2:
                    hl[2] = int(i/((maxNumNodes**2))) + 1

                [auroc, mse] = trainAndLogANN(hl[1:numHLs], fct, trainXY,
                                              testXY)
                entry = [id, fct, numHLs, hl[0], hl[1], hl[2],
                         auroc, mse]
                #print("Entry: {}".format(entry))

                df_error.loc[df_error.shape[0]] = entry

                # Increment the count of models trained&Tested
                id = id + 1

        print("-- Evaluating: {}% complete".format(
            (actFcts.index(fct)+1)/len(actFcts)*100))

    return [df_error]

def outputBestANNs(df: pd.DataFrame):
    id_min_mse = df[['MSE']].idxmin()
    id_min_auroc = df[['AUROC']].idxmin()
    print("Best Model (MSE): {}".format(
        df.loc[id_min_mse, :].values.tolist()))
    print("Best Model (AUROC): {}".format(
        df.loc[id_min_auroc, :].values.tolist()))

    # Gather top 10 AUROC model architectures
    if OUTPUT_FILES:
        df = df.sort_values(by='AUROC')
        df.head(10).to_excel(OUTPUT_BEST_AUROC)
        df.tail(10).to_excel(OUTPUT_WORST_AUROC)
        df = df.sort_values(by='MSE')
        df.head(10).to_excel(OUTPUT_BEST_MSE)
        df.tail(10).to_excel(OUTPUT_WORST_MSE)

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
[df_ErrorData] = chooseBestANN(x, y, RANGE_HL, RANGE_NODES, RANGE_ActFcts)

# Output the 10 best MSE and AUROC models
outputBestANNs(df_ErrorData)

