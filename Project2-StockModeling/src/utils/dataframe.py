'''
@module qualcomm.utils.dataframe
@info   Support functionality for interacting with pandas dataframes
@author ece5984-groupk
'''

# Python libraries
import os
import sys

# Third-party libraries
import pandas as pd

def Summarize(df: pd.DataFrame, prefix=None):
    ''' DataFrame summary function
    :param df: pd.DataFrame
    :return None:
    '''
    if prefix:
        print(prefix)
    print('General Description:')
    print('--------------------------------------------------')
    print(df.info())
    print('--------------------------------------------------')
    print('Statistics:')
    print('--------------------------------------------------')
    print(df.describe())
    print('--------------------------------------------------')

def Write(df: pd.DataFrame, fn: str):
    ''' Function to write a data frame to file '''
    if not os.path.exists(fn):
        print(f'{fn} not found - Searching in parent directory', file=sys.stderr)
        fn = os.path.dirname(__file__) + '/../../' + fn
    dtype = fn.split('.')[-1].upper()
    if dtype == 'CSV':
        df.to_csv(fn, index=False)
    elif dtype == 'XLSX':
        df.to_excel(fn, index=False)
    else:
        raise TypeError(f'Unsupported file type: {dtype}')