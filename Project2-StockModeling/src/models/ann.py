'''
ann.py
@author:    John Smutny
@team:      James Ensminger, Ben Johnson, Anagha Mudki, John Smutny
@info:      Regression artificial neural network (ann) model to predict the
            future stock price of the Qualcomm semiconductor company.

            The ann model cycles through several model frameworks and then
            chooses the best architecture to maximize profit from a $1000
            investment 28 days prior to sale.
'''

from sklearn import neural_network as ann
from sklearn import metrics
from sklearn import preprocessing as preproc
import sklearn.model_selection as modelsel
import numpy as np
import pandas as pd

import helperFunctions as hf

### Set Constants
#######################################

OUTPUT_FILES = True
ASCENDING_DATES = True

INCOME_TOLERANCE = 1.10
PREDICT_FUTURE_DAY = 28
INVESTMENT = 1000

INPUT_QUALCOMM = '../../data/QCOM_HistoricalData_5yr.csv'
INPUT_QUALCOMM_FINAL = \
    '../../data/final_extended_data_no_past_data_clean_extended.csv'
INPUT_FINAL = '../../data/FINAL_MODEL_DATASET.csv'
APPLE_FILE = '../../data/AAPL_HistoricalData_5yr.csv'
GOOGLE_FILE = '../../data/GOOGL_HistoricalData_5yr.csv'
ERICSSON_FILE = '../../data/ERIXF_HistoricalData_5yr.csv'
INTEL_FILE = '../../data/INTL_HistoricalData_5yr.csv'
NXP_FILE = '../../data/NXPI_HistoricalData_5yr.csv'
SAMSUNG_FILE = '../../data/SSNLF_HistoricalData_5yr.csv'
TMOBILE_FILE = '../../data/TMUS_HistoricalData_5yr.csv'
VERIZON_FILE = '../../data/VZ_HistoricalData_5yr.csv'
INPUT_BOND03m = '../../data/marketYield_3monthUsTreasureySecurity_5yr.csv'
INPUT_BOND02 = '../../data/marketYield_2YrUsTreasureySecurity_5yr.csv'
INPUT_BOND10 = '../../data/marketYield_10YrUsTreasureySecurity_5yr.csv'
INPUT_DOLLAR = '../../data/nominalBroadUSDollarIndex-5yr.csv'
INPUT_BITCOIN = '../../data/CoinbaseBitcoin_5yr.csv'
INPUT_COMPANY = '../../data/QCOM-SimFin-data-REFORMATTED.xlsx'

OUTPUT_INCOME = '../../artifacts/annIncome-TOL{}.xlsx'.format(INCOME_TOLERANCE)

IDName = "Date"
TARGET_NAME = "Close_{}".format(PREDICT_FUTURE_DAY)



# Artificial Neural Network Settings
#######################################

TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
VALID_DATA_FROM_TRAIN = 0.25
RANDOM_SEED = 10

HIDDEN_LAYERS = [7, 5, 8, 9]
ACTIVATION_FCT = 'relu'
SOLVER = 'adam'
MAX_ITER = 10000
LEARNING_RATE = 0.0001 * 10
TOLERANCE = 0.0001* 100
EARLY_STOPPING = True


### Data Processing
#######################################

def prepData(df: pd.DataFrame) -> pd.DataFrame:
    FINANCIAL_FEATURES = ['Close/Last', 'Open', 'High', 'Low']

    # Data cleaning of the main QualComm stock data
        # Below is Commented out b/c of ymal ann.py script
    #df = hf.removeDollarSign(df, FINANCIAL_FEATURES)
    df = hf.reformatDailyDates(df, ASCENDING_DATES)  # Re-order dates

    # Add new independent variables to help model stock price.
    df = hf.addFedData(df, 'DGS3MO', INPUT_BOND03m, ASCENDING_DATES)
    df = hf.addFedData(df, 'DGS2', INPUT_BOND02, ASCENDING_DATES)
    df = hf.addFedData(df, 'DGS10', INPUT_BOND10, ASCENDING_DATES)
    df = hf.addFedData(df, 'DTWEXBGS', INPUT_DOLLAR, True)
    df = hf.addFedData(df, 'CBBTCUSD', INPUT_BITCOIN, True)
    df = hf.addStockClosePrice(df, 'AAPL', APPLE_FILE, True)
    df = hf.addStockClosePrice(df, 'ERIXF', ERICSSON_FILE, True)
    df = hf.addStockClosePrice(df, 'GOOGL', GOOGLE_FILE, True)
    df = hf.addStockClosePrice(df, 'INTL', INTEL_FILE, True)
    df = hf.addStockClosePrice(df, 'NXPI', NXP_FILE, True)
    df = hf.addStockClosePrice(df, 'SSNLF', SAMSUNG_FILE, True)
    df = hf.addStockClosePrice(df, 'TMUS', TMOBILE_FILE, True)
    df = hf.addStockClosePrice(df, 'VZ', VERIZON_FILE, True)

    EXPAND30 = ['Close/Last', 'Volume']
    df = hf.appendPastData(df, 30, EXPAND30, ASCENDING_DATES)

    EXPAND05 = ['DGS3MO', 'DGS2', 'DGS10', 'DTWEXBGS',
                'Close_AAPL', 'Close_ERIXF', 'Close_GOOGL', 'Close_INTL',
                'Close_NXPI', 'Close_SSNLF', 'Close_TMUS', 'Close_VZ']
    df = hf.appendPastData(df, 5, EXPAND05, ASCENDING_DATES)

    # Add the future price target.
    df = hf.addTarget(df, TARGET_NAME, PREDICT_FUTURE_DAY, ASCENDING_DATES)

    return df


def doANN(df: pd.DataFrame):
    # Normalize and separate data into Independent & Dependent Variables.
    X = df.drop([IDName, TARGET_NAME], axis=1).to_numpy()
    scalerX = preproc.MinMaxScaler()
    scalerX.fit(X)
    X = scalerX.transform(X)

    Y = df[TARGET_NAME].to_numpy()
    trainX, testX, trainY, testY = \
        modelsel.train_test_split(X, Y, test_size=TEST_RATIO,
                                  random_state=RANDOM_SEED)
    # Record and report the average mse of the ANN model
    print("doANN: Start modeling loop.")
    mse_results = []
    r2_results = []
    for i in range(100):
        trainX, testX, trainY, testY = \
            modelsel.train_test_split(X, Y, test_size=TEST_RATIO,
                                      random_state=RANDOM_SEED)

        # Define Artificial Neural Network parameters
        clf = ann.MLPRegressor(hidden_layer_sizes=HIDDEN_LAYERS,
                               activation=ACTIVATION_FCT,
                               solver=SOLVER,
                               alpha=LEARNING_RATE,
                               early_stopping=EARLY_STOPPING,
                               max_iter=MAX_ITER,
                               validation_fraction=VALID_DATA_FROM_TRAIN)

        # Train and Evaluate the ANN
        clf.fit(trainX, trainY)
        annPredY = clf.predict(testX)
        mse_results.append(metrics.mean_squared_error(testY, annPredY))
        r2_results.append(metrics.r2_score(testY, annPredY))
        print(i)


    if OUTPUT_FILES:
        mseMean = np.mean(mse_results)
        mseMin = np.min(mse_results)
        mseMax = np.max(mse_results)
        print("\n\rANN: MSE = %f" % mseMean)
        df_mse = pd.DataFrame({'MSE':mse_results, 'Mean':"", 'Max':"", 'Min':""})
        df_mse.loc[0, 'Mean'] = mseMean
        df_mse.loc[0, 'Max'] = mseMax
        df_mse.loc[0, 'Min'] = mseMin
        df_mse.to_csv('ANN_MSE_Results.csv')

        r2Mean = np.mean(r2_results)
        print("\n\rANN: AUROC = %f" % r2Mean)
        r2Mean = np.mean(r2_results)
        r2Min = np.min(r2_results)
        r2Max = np.max(r2_results)
        print("\n\rANN: MSE = %f" % mseMean)
        df_mse = pd.DataFrame(
            {'MSE': r2_results, 'Mean': "", 'Max': "", 'Min': ""})
        df_mse.loc[0, 'Mean'] = r2Mean
        df_mse.loc[0, 'Max'] = r2Max
        df_mse.loc[0, 'Min'] = r2Min
        df_mse.to_csv('ANN_r2_Results.csv')

        newLabel = 'predict_{}'.format(PREDICT_FUTURE_DAY)
        df[newLabel] = clf.predict(X)

        hf.plot_5yr(df, IDName, TARGET_NAME, newLabel)
        hf.plot_3month(df, IDName, TARGET_NAME, newLabel)

        df_income = hf.calcIncome(df, TARGET_NAME, INVESTMENT,
                                  PREDICT_FUTURE_DAY,
                                  INCOME_TOLERANCE)
        df_income.to_excel(OUTPUT_INCOME)

def SimpleANNModel(
    trainTestSplit: tuple,
    layers: list=HIDDEN_LAYERS,
    activation: str=ACTIVATION_FCT,
    solver: str=SOLVER,
    alpha: float=LEARNING_RATE,
    earlyStopping: bool=EARLY_STOPPING,
    maxIters: int=MAX_ITER,
    validationFraction=VALID_DATA_FROM_TRAIN
):
    ''' ANN Model wrapper that takes train-test split data and returns a trained model
        See SciKitLearn docs for more information
    :param trainTestSplit: Tuple of pd.DataFrames (trainX, testX, trainY, testY)
    :param layers: List of node depths of length 'hiddenLayerCount' (default: ann.HIDDEN_LAYERS)
    :param activation: String name of the ANN activation function (default: ann.ACTIVATION_FCT)
    :param solver: String name of the ANN solver function (default: ann.SOLVER)
    :param aplha: Float learning rate for the model (default: ann.LEARNING_RATE)
    :param earlyStopping: Boolean flag for early stop of learning (default: ann.EARLY_STOPPING)
    :param maxIters: Integer stop limit for learning iterations (default: ann.MAX_ITER)
    :param validationFraction: Float fraction of training points to validate with(default: ann.VALID_DATA_FROM_TRAIN)
    :return sklean.Model: Trained ANN model object
    '''
    # Define Artificial Neural Network parameters
    trainX, testX, trainY, testY = trainTestSplit
    model = ann.MLPRegressor(hidden_layer_sizes=layers,
                           activation=activation,
                           solver=solver,
                           alpha=alpha,
                           early_stopping=earlyStopping,
                           max_iter=maxIters,
                           validation_fraction=validationFraction)

    # Train and Evaluate the ANN
    model.fit(trainX.to_numpy(), trainY.to_numpy())
    annPredY = model.predict(testX)
    print(f'{__name__} MSE = {metrics.mean_squared_error(testY, annPredY)}')
    return model

### Main Processing
#######################################

# load data and add columns to expand data as necessary.
if __name__ == "__main__":

    if False:
        df_raw = pd.read_csv(INPUT_QUALCOMM_FINAL)

        # make changes/additions to the loaded base stock data.
        df_edit = prepData(df_raw)

        if OUTPUT_FILES:
            df_edit.to_csv("postDataPrep-ModelDataUsed-Preprocessing.csv")
    else:
        df_edit = pd.read_csv(INPUT_FINAL)


    # Do model evaluation.
    doANN(df_edit)
