'''
helperfunctions.py
@author:    John Smutny
@team:      James Ensminger, Ben Johnson, Anagha Mudki, John Smutny
@info:      Various functions used to standardize the input of datasets,
            plot results, and calculate the income from various ML Stock
            predicting models.

            Used as a part of the QualComm Group Project 2 Stock Predictor.
'''

import pandas as pd
from matplotlib import pyplot as plt


### Helper Functions
#######################################
'''
@brief Function to expand a single data entry by x columns to include 
        previous data entry values in time.

@input  df      Pandas DataFrame data to be extrapolated.
@input  numPrevData  The number of earlier entries that will be appended to 
                        the last entry in the dataframe. 
@input labels   The specific labels that are going to be extrapolated 
@input ASCENDING Whether the 'Dates' used are ascending or descending.
@return df      Pandas DataFrame with new data columns to represent original df 
                 labels for a previous day. NOTE: The numPrevData first 
                 entries in the df DataFame are deleted

'''


def appendPastData(df: pd.DataFrame, numPrevDays, labels, ASCENDING) -> \
        pd.DataFrame:
    # Error check that inputted 'labels' are all in df input
    notInDF = False
    for label in labels:
        if list(df.columns.values).count(label) == 0:
            notInDF = True
            print("ERROR: Label not in df.")

    if notInDF:
        print("Do not append any df columns with previous day's data. Some of "
              "the labels in the of inputted list does not exist in DataFrame "
              "df.")

    # Execute the Function's purpose.
    else:
        for label in labels:

            # Create columns to extrapolate data too
            for i in range(numPrevDays):
                # Create the new columns for each desired day
                addedColName = "Prev{}_{}".format(i + 1, label)
                zeros = [0] * len(df.index)
                df[addedColName] = zeros

                #########################################

                # Add previous day's data to new columns
                # Isolate one column at a time.
                for entry in range(len(df.index)):
                    if ASCENDING:
                        if entry >= numPrevDays:
                            df.loc[entry, addedColName] = \
                                df.loc[entry - i - 1, label]
                    else:
                        if entry < (len(df.index) - numPrevDays):
                            df.loc[entry, addedColName] = \
                                df.loc[entry + i + 1, label]

        # Delete the first x number of entries to prevent an indexing exception.
        if numPrevDays > 0:
            print("::appendPastData - Deleted {} yearlest dates to avoid "
                  "segFaults.".format(numPrevDays))
            if ASCENDING:
                df = df.drop(range(numPrevDays), axis=0)
            else:
                df = df.drop(range(len(df.index) - numPrevDays, len(df.index)),
                             axis=0)

            df = df.reset_index(drop=True)

    return df


'''
@brief Function to remove any '$' characters from a pandas dataframe column.

@input  df      Pandas DataFrame data to be reviewed.
@input labels   The specific labels that are going to be extrapolated 
@return df      Pandas DataFrame with replaced values.
'''


def removeDollarSign(df: pd.DataFrame, labels) -> pd.DataFrame:
    for label in labels:
        df[label] = df[label].str.replace('$', '', regex=True)

    return df


'''
@brief  Universal function to take a dataset's DATE column and reformat it to 
        a consistent style based on the datetime python object. 
        Then sort the data based on the desired order.
        Style = yyyy-mm-dd
@input df DataFrame of the full data to be reformatted.
@input ASCENDING Whether the 'Dates' used are ascending or descending.
@output a dataframe of UNCHANGED data, only re-formatted.
'''


def reformatDailyDates(df: pd.DataFrame, ASCENDING) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date

    df = df.sort_values(by='Date', ascending=ASCENDING, ignore_index=True)
    return df


'''
@brief Function that will generate the target variable of 
        'stock price x days in the future'.
        NOTE: This function will REMOVE data from the dataset to prevent 
        exceptions or predicting the future.
@input TARGET label name of the target variable you are trying to model.
@input FUTURE_DAY How many days in the future are you looking at stock prices.
@input ASCENDING Whether the 'Dates' used are ascending or descending.
@output New data frame with the actual stock price after x days.
'''


def addTarget(df: pd.DataFrame, TARGET, FUTURE_DAY, ASCENDING) -> pd.DataFrame:
    # Add new column for the target.
    df[TARGET] = ""

    print("::addTarget - {} newest days will be dropped to predict {} days in "
          "the future ".format(FUTURE_DAY, FUTURE_DAY))

    listOfDropEntries = []

    for x in range(len(df['Date'])):
        # (earliest first)
        if ASCENDING:
            if x < len(df['Date']) - FUTURE_DAY:
                df.loc[x, TARGET] = df.loc[x + FUTURE_DAY,
                                           "Close/Last"]
            else:
                listOfDropEntries.append(x)

        # Descending (most recent first)
        else:
            if x > FUTURE_DAY:
                df.loc[x, TARGET] = df.loc[x - FUTURE_DAY,
                                           "Close/Last"]
            else:
                listOfDropEntries.append(x)

    # Drop indices to prevent segfault and are out of range of prediction.
    df = df.drop(index=listOfDropEntries, axis=0)
    df = df.reset_index(drop=True)
    return df


'''
@brief Function to calculate how much you would make based on a minimum gain 
        predicted by the given model.
@input df The dataframe of data inputs used to train your model. 
            MUST INCLUDE THE PREDICTION OF THE MODEL
@input TARGET The target variable in the input 'df' that the model trained on.
@input INVESTMENT How much are you investing each time the model tells you.
@input FUTURE_DAYS How many days in the future will you sell your stock.
@input TOL the tolerance of when you should invest to get a minimum return.
        Ex: TOL = 1.05 means that the model must predict 5% profit to invest.
@output Dataframe record of the investments made and the conditions on that day.
'''


def calcIncome(df: pd.DataFrame, TARGET, INVESTMENT, FUTURE_DAYS, TOL) -> \
        pd.DataFrame:
    print("WARN: You must include the model predictions for 'Close Price 28 "
          "Days Later' for this fct to work. Please insert the following code "
          "before calling this function:\n"
          "\t\tdf['predict_28'] = clf.predict(X)'")

    df_invest = pd.DataFrame(columns=['Date', 'quantity', 'close',
                                      'sell_price',
                                      'model_price', 'predIncome',
                                      'actualIncome'])

    for i in range(len(df['Date'])):
        close = float(df.loc[i, 'Close/Last'])
        modelClose = float(df.loc[i, 'predict_{}'.format(FUTURE_DAYS)])
        modelGain = modelClose / close

        if modelGain > TOL:
            actualClose = float(df.loc[i, TARGET])

            quantity = INVESTMENT / close
            predIncome = (modelClose - close) * quantity
            actualIncome = (actualClose - close) * quantity

            df_invest.loc[i, 'Date'] = df.loc[i, 'Date']
            df_invest.loc[i, 'quantity'] = quantity
            df_invest.loc[i, 'close'] = close
            df_invest.loc[i, 'sell_close'] = actualClose
            df_invest.loc[i, 'model_close'] = modelClose
            df_invest.loc[i, 'predIncome'] = predIncome
            df_invest.loc[i, 'actualIncome'] = actualIncome

    print("TOTAL INCOME FROM {} INVESTMENTS (PREDICT/ACTUAL): "
          "${:.2f}/${:.2f}".format(len(df_invest),
                                   df_invest['predIncome'].sum(),
                                   df_invest['actualIncome'].sum()))

    return df_invest


def plot_5yr(df: pd.DataFrame, labelx, labelActual, labelPredicted):
    df = pd.DataFrame({'Date': df[labelx],
                       'Close_28': df[labelActual].astype(float),
                       #'Todays_Close': df['Close/Last'].astype(float),
                       'Predict_28': df[labelPredicted].astype(float)
                       })

    df.plot(x='Date', y=[
                         'Close_28',
                         #'Todays_Close',
                         'Predict_28'
                        ],
            kind="line", figsize=(18, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.xlabel("Date")
    plt.ylabel("Close Price ($)")
    plt.title("QualComm Stock Price over the Last 5 Years")
    # plt.show()

    plt.savefig("StockPrice-5Year.jpeg")


def plot_3month(df: pd.DataFrame, labelx, labelActual, labelPredicted):
    df = pd.DataFrame(
        {'Date': df[labelx],
         #'Close': df['Close/Last'].astype(float),
         'Close_28': df[labelActual].astype(float),
         'Predict_28': df[labelPredicted].astype(float)})

    # Slice data to plot only the last 3 months
    df = df.tail(28 * 3)

    df.plot(x='Date', y=['Close_28',
                         #'Close',
                         'Predict_28'],
            kind="line", figsize=(18, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.xlabel("Date")
    plt.ylabel("Close Price ($)")
    plt.title("QualComm Stock Price over the Last 3 Months")
    # plt.show()

    plt.savefig("StockPrice-3Month.jpeg")


################################################################################
################################################################################
################################################################################
# Functions to add datasets
################################################################################
################################################################################
################################################################################

'''
@brief  Function to add daily bond yields of the desired year to an existing 
        dataset for modeling.
        Covers Bond yields of 3-months, 2-years, and 10-years.
@input  BOND label given by the Treasury department stating the bond data. 
                See the datafile used for the label.
@input FILE Input file of bond data.
@input ASCENDING Whether the 'Dates' used are ascending or descending.
@output Dataframe now including x yr daily bond yields.
'''


def addBondPrice(df: pd.DataFrame, BOND, FILE, ASCENDING) -> pd.DataFrame:
    df_bond = pd.read_csv(FILE)
    df_bond = df_bond.rename(columns={'DATE': 'Date'})

    # Ensure that the information is the correct order
    df_bond = reformatDailyDates(df_bond, ASCENDING)

    # Clean data

    valuesChanged = len(df_bond.index[df_bond[BOND] == '.'].tolist())
    print("::addBondPrice - {} values were changed to clean data.".format(
        valuesChanged))

    for x in df_bond.index[df_bond[BOND] == '.'].tolist():
        if x != 0:
            df_bond.loc[x, BOND] = df_bond.loc[x - 1, BOND]
        else:
            df_bond.loc[x, BOND] = df_bond.loc[x + 1, BOND]

    # Add ready values to main dataframe for models
    df = pd.merge(df, df_bond, on='Date', how='left', validate='one_to_one')

    return df


'''
@brief Use for data from fred.stlouisfed.org
        General function to add an economic indicator to the dataframe of data.
@input FILE Input file of data.
@input ASCENDING Whether the 'Dates' used are ascending or descending.
@output Dataframe now including x yr price of that indicator at the close.
'''


def addFedData(df: pd.DataFrame, SYM, FILE, ASCENDING) -> pd.DataFrame:
    df_Fed = pd.read_csv(FILE)
    df_Fed = df_Fed.rename(columns={'DATE': 'Date'})

    # Ensure that the information is the correct order
    df_Fed = reformatDailyDates(df_Fed, ASCENDING)

    # Clean data

    valuesChanged = len(df_Fed.index[df_Fed[SYM] == '.'].tolist())
    print("::addBondPrice - {} values were changed to clean data.".format(
        valuesChanged))

    for x in df_Fed.index[df_Fed[SYM] == '.'].tolist():
        if x != 0:
            df_Fed.loc[x, SYM] = df_Fed.loc[x - 1, SYM]
        else:
            df_Fed.loc[x, SYM] = df_Fed.loc[x + 1, SYM]

    # Add ready values to main dataframe for models
    df = pd.merge(df, df_Fed, on='Date', how='left', validate='one_to_one')

    return df


'''
@brief Use for data from NASDAQ.com
        General function to add a stock ticker to the dataframe of data.
@input FILE Input file of data.
@input ASCENDING Whether the 'Dates' used are ascending or descending.
@output Dataframe now including x yr stock price at the close.
'''

def addStockClosePrice(df: pd.DataFrame, SYM, FILE, ASCENDING) -> pd.DataFrame:
    EXTRACTED_FEATURE = 'Close/Last'
    ADDED_FEATURE = 'Close_{}'.format(SYM)

    df_stock = pd.read_csv(FILE)
    df_stock = df_stock[['Date', EXTRACTED_FEATURE]]
    df_stock = df_stock.rename(columns={EXTRACTED_FEATURE:ADDED_FEATURE})
    df_stock = removeDollarSign(df_stock, [ADDED_FEATURE])


    # Ensure that the information is the correct order
    df_stock = reformatDailyDates(df_stock, ASCENDING)

    # Add ready values to main dataframe for models
    df = pd.merge(df, df_stock, on='Date', how='left', validate='one_to_one')

    return df


'''
@brief  
@input FILE Input file of bond data.
@input ASCENDING Whether the 'Dates' used are ascending or descending.
@output Dataframe now including x yr daily bond yields.
'''


def addSimFin(df: pd.DataFrame, FILE, ASCENDING) -> pd.DataFrame:
    df_QualComm = pd.read_csv(FILE)
    df_QualComm = df_QualComm.rename(columns={'DATE': 'Date'})

    # Ensure that the information is the correct order
    df_QualComm = reformatDailyDates(df_QualComm, ASCENDING)

    return df
