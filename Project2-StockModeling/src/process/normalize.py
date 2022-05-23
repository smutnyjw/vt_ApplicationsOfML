'''
@module qualcomm.process.normalize
@info   Functionality for normalizing pandas dataframes
@author ece5984-groupk
'''

# Python Libraries

# Third party libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def Normalize(df: pd.DataFrame, columns: list, range: tuple=(-1, 1)):
    ''' Rename columns
    :param df: Pandas dataframe
    :param columns: List of columns to normalize
    :param range: Tuple (min, max) for normalization range
    :return pd.DataFrame: Updated dataframe
    '''
    result = df.copy()
    scaler = MinMaxScaler(feature_range=range)
    result[columns] = scaler.fit_transform(result[columns])
    return result