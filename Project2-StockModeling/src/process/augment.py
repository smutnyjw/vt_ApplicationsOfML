'''
@module     qualcomm.process.augment
@info       Methods for updating the standard dataset with additional columns,
            appending additional datasets, and conditioning conflicting datasets
            to work together.
@author     ece5984-groupk
'''

# Python Libraries

# Third party libraries
import pandas as pd
import numpy as np

## Data Augmentation Functions
################################################################################
def AddFutureValueColumn(
    df: pd.DataFrame,
    sourceColumn: str,
    targetColumn: str,
    rowsInFuture: int=28,
    dropRemainingRows: bool=False
):
    '''
    Function that will generate the target variable of 'stock price x days in
            the future'
    :param df: DataFrame housing original data
    :param sourceColumn: String column name for the future data values
    :param targetColumn: String column name for the new datapoint
    :param rowsInFuture: Integer number of days in the future to collect data
    :param dropRemainingRows: Boolean flag to keep rows with no future data
    :return pd.DataFrame: containing the updated data
    '''
    assert rowsInFuture > 0, 'rowsInFuture must be a positive value'
    print(f'{__name__} - {rowsInFuture} days of data will be dropped due to absence of future data')
    numRows = len(df.index)
    for idx in range(0, numRows):
        if idx < (numRows - rowsInFuture):
            df.loc[idx, targetColumn] = df.loc[idx + rowsInFuture, sourceColumn]
        elif dropRemainingRows:
            df = df.drop([idx], axis=0)
        else:
            df.loc[idx, targetColumn] = np.nan
    return df

def AddColumnDifference(
    df: pd.DataFrame,
    subtrahendColumn: str,
    minuendColumn: str,
    differenceColumn: str,
):
    '''
    Function that create or update a column with the difference between two columns
    :param df: DataFrame housing original data
    :param subtrahendColumn: String name of rhs column in (to subtract) subtraction
    :param minuendColumn: String name of lhs (subtracted from) column in subtraction
    :param differenceColumn: String name of the column to store the difference
    :return pd.DataFrame: containing the updated data
    '''
    result = df.copy()
    result[differenceColumn] = df[minuendColumn] - df[subtrahendColumn]
    return result