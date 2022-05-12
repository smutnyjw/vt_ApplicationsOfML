'''
@package    vt_ApplicationsOfML
@fileName   hw6_ModelEnsemble.py
@author     John Smutny
@date       05/12/2022
@info   Input taxi fare data for New York City and attempt to create a two
        stage ensemble model to predict a taxi ride's total price. The models
        used are as follows for each stage; 1) Artificial Neural Network,
        Multivariate Linear Regression, and Decision Tree all feeding into
        2) another ANN model.

        Each model is analyzed based on their MSE, MAE, R2 and EVS.

        Ensemble model produces a scatterplot plotting the learning
        curve by training epoch.
'''
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn import linear_model as linmod
from sklearn import neural_network as ann

from sklearn import metrics
from sklearn import preprocessing as preproc
import sklearn.model_selection as modelsel
import numpy as np
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

DEBUG = False
OUTPUT_FILES = True
DATA_USE_RATIO = 1

# Values used for debugging only. Overwrite if Debugging.
if DEBUG:
    DATA_USE_RATIO = 0.1

## Control flags and constants
################################################################################

INPUT_FILE = '../data/Taxi_Trip_Data.xlsx'
OUTPUT_preDQR = '../artifacts/taxi_preDQR.xlsx'
OUTPUT_postDQR = '../artifacts/taxi_postDQR.xlsx'
OUTPUT_LPLOT = '../artifacts/learningPlot.png'
OUTPUT_SPLOT = '../artifacts/scatterPlot.png'

DROPPED_FEATURES = ['store_and_fwd_flag', 'PULocationID', 'DOLocationID',
                    'fare_amount', 'extra', 'mta_tax', 'tip_amount',
                    'tolls_amount']
TARGET_FEATURE = 'total_amount'


## Helper Functions
################################################################################


def prepData(df: pd.DataFrame) -> pd.DataFrame:
    # Reduce the number of samples used to speed up testing
    if DEBUG:
        df = df.sample(frac=DATA_USE_RATIO)

    # Dropped already specified features to drop.
    df = df.drop(DROPPED_FEATURES, axis=1)

    ###############################################
    # Replace any values of NWR (Newfolk NJ) for 'Staten Island'
    df['PUBorough'] = df['PUBorough'].replace(['EWR'], ['Staten Island'])
    df['DOBorough'] = df['DOBorough'].replace(['EWR'], ['Staten Island'])

    ###############################################
    # Combine the PickUp and DropOff times into FareTime feature
    df.insert(1, "FareTime (sec)", 0)
    pu_time = pd.to_datetime(df['lpep_pickup_datetime'].tolist())
    do_time = pd.to_datetime(df['lpep_dropoff_datetime'].tolist())

    #len(df['lpep_pickup_datetime'])
    index = df.index
    for i in range(len(df.index)):
        timeDif = do_time[i] - pu_time[i]
        df.loc[index[i], 'FareTime (sec)'] = timeDif.total_seconds()

    df = df.drop(['lpep_dropoff_datetime', 'lpep_pickup_datetime'], axis=1)

    ###############################################
    # Ensure that all fares collected are not negative. Taxis do not give money.
    print("Number of negative improvement_surcharge values: {}".format(
        len(df[df['improvement_surcharge'] < 0])))
    df.loc[:, 'improvement_surcharge'] = abs(df.loc[:, 'improvement_surcharge'])
    print("Number of negative total_amount values: {}".format(
        len(df[df['total_amount'] < 0])))
    df.loc[:, 'total_amount'] = abs(df.loc[:, 'total_amount'])

    ###############################################
    # Perform One-Hot Encoding on PickUp & DropOff features

    # Remove any entry that has any PU or DO borough as UNKNOWN so locations
    # can be used for one-hot encoding.
    numOrigEntries = len(df['PUBorough'])

    indexUnknownPU = df.index[[df['PUBorough'] == 'Unknown']].tolist()
    df = df.drop(index=indexUnknownPU)
    indexUnknownDO = df.index[[df['DOBorough'] == 'Unknown']].tolist()
    df = df.drop(index=indexUnknownDO)
    df = df.reset_index()
    print("Size of UNKNOWN Boroughs Removed:\tPU/DO: {}/{}".format(
        len(indexUnknownPU), len(indexUnknownDO)))

    if (len(indexUnknownPU) + len(indexUnknownDO)) > 0.1 * numOrigEntries:
        print("WARNING: Removed more than 10% of data to include PUBorough "
              "and DOBorough.")

    # Perform one-hot encoding for the PU/DOBoroughs.
    df_PU = pd.get_dummies(df['PUBorough'], prefix='PU')
    df_DO = pd.get_dummies(df['DOBorough'], prefix='DO')

    # Merge the one-hot encoded columns with the original dataset.
    df = pd.concat([df, df_PU, df_DO], axis=1, join='inner')
    df = df.drop(columns=['PUBorough', 'DOBorough'])

    return df

def plotANNLearningCurve(hl, trainingLoss, validationLoss):
    # create figure and axis objects with subplots()
    xlabel = "epochs (hl=" + str(hl) + ")"
    fig, ax = plt.subplots()
    ax.plot(trainingLoss, color="blue")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("loss", color="blue", fontsize=10)
    ax.plot(validationLoss, color="red")
    ax.set_yscale('log')
    plt.legend(loc='upper left')
    plt.title("Training Loss vs Validation Loss of the 2nd Stage ANN Output")

    plt.savefig(OUTPUT_LPLOT)

def plotModelOutput(actualY, modelY):
    # Plot Classification Accuracy chart of both features 'lgID' and 'teamID'.
    fig, scat = plt.subplots()

    scat.scatter(actualY.tolist(), modelY.tolist(), marker='o', c='blue')
    scat.set_xlabel("Actual Taxi Fares ($)")
    scat.set_ylabel("Model Predicted Taxi Fares ($)")
    plt.title("Ensemble Model vs Actual Taxi Fare Prices")

    # Keep axis ranges consistent to highlight amount of linear relationship.
    if max(actualY) > max(modelY):
        plt.xlim(0, max(actualY)*1.1)
        plt.ylim(0, max(actualY)*1.1)
    else:
        plt.xlim(0, max(modelY)*1.1)
        plt.ylim(0, max(modelY)*1.1)

    plt.savefig(OUTPUT_SPLOT)


def calcMetrics(testY, modelY, name):
    mse = metrics.mean_squared_error(testY, modelY)
    mae = metrics.mean_absolute_error(testY, modelY)
    r2 = metrics.r2_score(testY, modelY)
    eva = metrics.explained_variance_score(testY, modelY)

    df_metrics = pd.DataFrame([{'mse': mse, 'mae': mae, 'r2': r2, 'eva': eva}])

    if OUTPUT_FILES:
        df_metrics.to_csv('../artifacts/metrics_{}.csv'.format(name))
    else:
        print("{} Model Performance:\nMSE = {}\tMAE = {}\tR2 = {}"
              "\tEVA = {}".format(name, mse, mae, r2, eva))


def createANN(df: pd.DataFrame):
    ##################################################
    # Normalize inputs and outputs
    X = df.drop(['index', TARGET_FEATURE], axis=1).to_numpy()
    scalerX = preproc.MinMaxScaler(feature_range=(0, 1))
    X = scalerX.fit_transform(X)

    scalerY = preproc.MinMaxScaler(feature_range=(-1, 1))
    Y = scalerY.fit_transform(df[[TARGET_FEATURE]])

    ##################################################
    # Split Training and Test Data

    # Set up train, validation, and test data
    trainX, testX, trainY, testY = modelsel.train_test_split(X, Y,
                                                             test_size=TEST_DATA,
                                                             random_state=RANDOM_SEED)

    ##################################################
    # Train the ANN model
    hl = [6, 8, 4]
    clf = ann.MLPRegressor(hidden_layer_sizes=hl,
                            activation='relu',
                            solver='adam',
                            alpha=0.01,
                            early_stopping=True,
                            max_iter=10000,
                            validation_fraction=VALID_DATA_FROM_TRAIN,
                            tol=0.001)

    # Record the performance of the model trained.
    clf.fit(trainX, trainY)
    modelY = clf.predict(testX)
    modelY = scalerY.inverse_transform(modelY.reshape(-1, 1))
    modelY = np.array(modelY)
    testY = scalerY.inverse_transform(testY)
    testY = np.array(testY)

    # Calculate model metrics
    calcMetrics(testY, modelY, "ANN")
    modelY = clf.predict(X)
    modelY = scalerY.inverse_transform(modelY.reshape(-1, 1))
    outputY = np.array(modelY)

    return outputY

def createDT(df: pd.DataFrame):
    ##################################################
    # Normalize inputs and outputs
    X = df.drop(['index', TARGET_FEATURE], axis=1).to_numpy()
    scalerX = preproc.MinMaxScaler(feature_range=[0, 1])
    X = scalerX.fit_transform(X)

    #scalerY = preproc.MinMaxScaler(feature_range=(0, 1))
    #Y = scalerY.fit_transform(df[[TARGET_FEATURE]])
    Y = df[TARGET_FEATURE].astype(int)

    numbins = 10
    labels = range(numbins)
    Y, binsY = pd.qcut(Y, q=numbins, labels=labels, retbins=True)
    #Y, binsY = pd.cut(Y, bins=numbins, labels=labels, retbins=True)

    print(Y.value_counts())
    print(binsY)

    ##################################################
    # Split Training and Test Data

    # Set up train, validation, and test data
    trainX, testX, trainY, testY = modelsel.train_test_split(X, Y,
                                                             test_size=TEST_DATA,
                                                             random_state=RANDOM_SEED)

    ##################################################
    # Train the Decision Tree model
    dt = tree.DecisionTreeClassifier(criterion='entropy',
                                     max_depth=5)

    # Record the performance of the model trained.
    dt.fit(trainX, trainY)
    #https://pbpython.com/pandas-qcut-cut.html
    modelY = dt.predict(testX)
    modelY = np.array(modelY)

    # Calculate model metrics
    calcMetrics(testY, modelY, "DT")

    outputY = np.array(dt.predict(X))

    return outputY


def createMLR(df: pd.DataFrame):
    ##################################################
    # Normalize inputs and outputs
    X = df.drop(['index', TARGET_FEATURE], axis=1).to_numpy()
    scalerX = preproc.MinMaxScaler(feature_range=[0, 1])
    X = scalerX.fit_transform(X)

    scalerY = preproc.MinMaxScaler(feature_range=(-1, 1))
    Y = scalerY.fit_transform(df[[TARGET_FEATURE]])

    ##################################################
    # Split Training and Test Data

    # Set up train, validation, and test data
    trainX, testX, trainY, testY = modelsel.train_test_split(X, Y,
                                                             test_size=TEST_DATA,
                                                             random_state=RANDOM_SEED)

    ##################################################
    # Train the Multivariate Linear Regression model
    mlr = linmod.LinearRegression()
    mlr.fit(trainX, trainY)

    modelY = mlr.predict(testX)
    modelY = scalerY.inverse_transform(modelY.reshape(-1, 1))
    modelY = np.array(modelY)
    testY = scalerY.inverse_transform(testY)
    testY = np.array(testY)

    # Calculate model metrics
    calcMetrics(testY, modelY, "MLR")
    modelY = scalerY.inverse_transform(mlr.predict(X))
    outputY = np.array(modelY)

    return outputY


def secondStage(df: pd.DataFrame, annY, dtY, mlrY):
    pu = ['PU_Bronx', 'PU_Brooklyn', 'PU_Manhattan',
          'PU_Queens', 'PU_Staten Island']
    do = ['DO_Bronx', 'DO_Brooklyn', 'DO_Manhattan',
          'DO_Queens', 'DO_Staten Island']
    existingFeatures = ['FareTime (sec)'] + pu + do + [TARGET_FEATURE]

    df = df[existingFeatures]
    df['ANN_Y'] = annY
    df['DT_Y'] = dtY
    df['MLR_Y'] = mlrY

    ##################################################
    # Normalize inputs and outputs
    X = df.drop([TARGET_FEATURE], axis=1).to_numpy()
    scalerX = preproc.MinMaxScaler(feature_range=(0, 1))
    X = scalerX.fit_transform(X)

    scalerY = preproc.MinMaxScaler(feature_range=(-1, 1))
    Y = scalerY.fit_transform(df[[TARGET_FEATURE]])

    ##################################################
    # Split Training and Test Data

    # Set up train, validation, and test data
    trainX, testX, trainY, testY = modelsel.train_test_split(X, Y,
                                                             test_size=TEST_DATA,
                                                             random_state=RANDOM_SEED)
    trainX, validX, trainY, validY = modelsel.train_test_split(trainX, trainY,
                                                               test_size=VALID_DATA_FROM_TRAIN,
                                                               random_state=RANDOM_SEED)

    ##################################################
    # Train the Multivariate Linear Regression model
    #mlr = linmod.LinearRegression()
    # Train the ANN model
    hl = [4, 5, 2]
    clf = ann.MLPRegressor(hidden_layer_sizes=hl,
                           activation='relu',
                           solver='adam',
                           alpha=0.01,
                           early_stopping=False,    #No 'early_stopping' for
                           # partial fit.
                           max_iter=10000,
                           validation_fraction=VALID_DATA_FROM_TRAIN,
                           tol=0.001,
                           random_state=RANDOM_SEED)

    trainingLoss = []
    validationLoss = []
    numEpochs = 1000
    if DEBUG:
        numEpochs = 10

    for epoch in range(numEpochs):
        clf.partial_fit(trainX, trainY)
        trainingLoss.append(1 - clf.score(trainX, trainY))
        validationLoss.append(1 - clf.score(validX, validY))

    modelY = clf.predict(testX)
    modelY = np.array(modelY)
    modelY = scalerY.inverse_transform(modelY.reshape(-1, 1))
    testY = scalerY.inverse_transform(np.array(testY))

    # Calculate model metrics
    calcMetrics(testY, modelY, "2ndStage")

    plotANNLearningCurve(hl, trainingLoss, validationLoss)
    plotModelOutput(testY, modelY)


## Data Handling Functions
################################################################################

# Load and analyze
df_raw = pd.read_excel(INPUT_FILE)
report1 = DataQualityReport()
report1.quickDQR(df_raw, list(df_raw.columns))

# Data Preparation
df_prep = prepData(df_raw)
report2 = DataQualityReport()
report2.quickDQR(df_prep, list(df_prep.columns))

# Output pre-normalized files
if OUTPUT_FILES:
    report1.to_excel(OUTPUT_preDQR)
    report2.to_excel(OUTPUT_postDQR)

# Run models
out_ANN = createANN(df_prep)
out_DT = createDT(df_prep)
out_MLR = createMLR(df_prep)

secondStage(df_prep, out_ANN, out_DT, out_MLR)

