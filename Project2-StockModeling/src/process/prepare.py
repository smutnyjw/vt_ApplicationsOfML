'''
@module     qualcomm.process.prepare
@info       General data processing support
@author     ece5984-groupk
'''

# Python libraries
import yaml
import os
import sys

# Third party libraries
import pandas as pd
import numpy as np

## Data preparation functions
################################################################################
def Load(data_file):
    ''' Static data loading method
    :param data_file: String path to the data file
    :return pd.DataFrame: DataFrame housing data file contents
    '''
    if not os.path.exists(data_file):
        print(f'{data_file} not found - Searching in parent directory', file=sys.stderr)
        data_file = os.path.dirname(__file__) + '/../../' + data_file
    assert os.path.exists(data_file), f'Invalid data file: {data_file}'
    dtype = data_file.split('.')[-1].upper()
    result = None
    if dtype == 'CSV':
        result = pd.read_csv(data_file)
    else:
        raise TypeError(f'Unsupported file type: {dtype}')
    return result

def Rename(df: pd.DataFrame, columns: dict):
    ''' Rename columns
    :param df: Pandas dataframe
    :param columns: Dict of format {old_name: new_name}
    :return pd.DataFrame: Updated dataframe
    '''
    result = df.copy()
    result = result.rename(columns=columns)
    return result

def CapitalizeColumns(df: pd.DataFrame, upper: bool=True):
    ''' Capitalize column names
    :param df: Pandas dataframe
    :param upper: Boolean uppercase flag (default True)
    :return pd.DataFrame: Updated dataframe
    '''
    if upper:
        df.columns = [col.upper() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
    return df

def ExpandDate(df: pd.DataFrame, column: str):
    ''' Expand a date string column into day month year
    :param df: Pandas dataframe
    :param column: String name of the column to update
    :return pd.DataFrame: Updated dataframe
    '''
    result = df.copy()
    date = pd.to_datetime(result[column])
    result['year'] = date.dt.year
    result['month'] = date.dt.month
    result['day'] = date.dt.day
    return result

def ConvertTypes(df: pd.DataFrame, columns: dict):
    ''' Convert problematic data types to target formats
    :param df: Pandas dataframe
    :param columns: Dict of format {column_name: column_type}
    :return pd.DataFrame: Updated dataframe
    '''
    result = df.copy()
    for name, type in columns.items():
        type = type.lower().strip()
        if type == 'currency':
            result[name] = result[name].replace('[\$,]', '', regex=True).replace(',', '').astype(float)
        elif type == 'date':
            result[name] = pd.to_datetime(result[name]).dt.date
        elif type == 'float':
            result[name] = result[name].replace(',', '').replace(',', '', regex=True).astype(float)
        else:
            raise ValueError(f'Unsupported data type: {type}')
    return result

def Sort(df: pd.DataFrame, sortColumn: str, ascending: bool=True):
    ''' Sort a pd dataframe in ascending/descending fashion based on a single
            column's values
    :param df: Pandas DataFrame object
    :param sortColumn: String name of the column to sort with
    :param ascending: Boolean [a/de]scending flag (default: True/Ascending)
    :return pd.DataFrame: Output dataframe object
    '''
    result = df.copy()
    result = result.sort_values(by=sortColumn, ascending=ascending, ignore_index=True)
    return result

def InterpolateAndConcatByDate(
    target: pd.DataFrame,
    source: pd.DataFrame,
    columns: list,
    method: str='linear'
):
    ''' Method for interpolating a data set to match a target and then add
            selected columns to the original
    :param: target: DataFrame to match and append new columns to
    :param source: DataFrame with new values
    :param columns: List of string column names to append
    :param method: String pandas interpolation method
    :return pd.DataFrame: resulting DataFrame
    '''
    # Copy the target and source so we don't update the inputs
    result = target.copy()
    src = source.copy()

    # Reindex the target to it's Date column and the source by the full date range
    result.index = result.reindex(target['Date']).index
    src = src.set_index('Date')

    # Reindex the source or new data by the target range
    #   Note: The min/max functions handle deltas in start and stop date for the
    #       two datasets. The interpolate function handles fitting missing data
    startDate = min(result.index[0], src.index[0])
    stopDate = max(result.index[-1], src.index[-1])
    fullIndex = pd.date_range(startDate, stopDate, freq='1D')
    src = src.reindex(fullIndex, fill_value=np.nan)

    # Interpolate nan values
    for column in columns:
        src[column] = src[column].interpolate(method=method)

    # Add the applicable date value to the target and return
    for column in columns:
        result.insert(len(result.columns), column, src[column])

    # Reset the output index back to normal
    result.index = result.reindex(target.index).index
    return result

def StartsWithDrop(df: pd.DataFrame, starts: list):
    result = df.copy()
    toDrop = []
    for start in starts:
        for column in result.columns:
            if column.startswith(start):
                toDrop.append(column)
    return result.drop(columns=toDrop)