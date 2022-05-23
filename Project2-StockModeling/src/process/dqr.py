'''
@module     qualcomm.dqr.DQR
@info       Wrapper method for returning a data quality report dataframe given
            a standard pandas dataframe
@author     ece5984_groupk
'''

# Python libraries

# Third party libraries
import pandas as pd
import numpy as np

def DQR(data: pd.DataFrame) -> pd.DataFrame:
    ''' Given a pandas dataframe, generated a DQR table
    :param data: Pandas DataFrame object
    :return pd.DataFrame: Data quality report
    '''
    dqr = pd.DataFrame()
    dqr['statistic'] = [
        'count',
        'cardinality',
        'mean',
        'median',
        'n_at_median',
        'mode',
        'n_at_mode',
        'stddev',
        'min',
        'n_at_min',
        'max',
        'n_at_max',
        'n_zero',
        'n_missing'
    ]
    for column in data.columns:
        mode = data[column].mode()
        if not len(mode):
            continue
        mode = mode[0]
        value_counts = data[column].value_counts()
        if data.dtypes[column] in [np.object_]:
            entry = [
                data[column].size,
                len(data[column].unique()),
                np.nan,
                np.nan,
                np.nan,
                mode,
                value_counts.get(mode, 0),
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                data[column].isnull().sum()
            ]
        else:
            median = data[column].median()
            min = data[column].min()
            max = data[column].max()
            entry = [
                data[column].size,
                len(data[column].unique()),
                data[column].mean(),
                median,
                value_counts.get(median, 0),
                mode,
                value_counts.get(mode, 0),
                data[column].std(),
                min,
                value_counts.get(min, 0),
                max,
                value_counts.get(min, 0),
                value_counts.get(0, 0) + value_counts.get(0.0, 0),
                data[column].isnull().sum()
            ]
        dqr[column] = entry
    return dqr
