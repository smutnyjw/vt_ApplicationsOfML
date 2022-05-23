'''
@module qualcomm.process.train
@info   General support functionality for training models
@author ece5984-groupk
'''

# Python libraries

# Third-party libraries
import pandas as pd
from sklearn.model_selection import train_test_split

def TrainTestSplit(
    df: pd.DataFrame,
    predictors: list,
    targets: list,
    test_size: float=0.3,
    seed: int=0
):
    ''' Train test split wrapper
    :param df: Pandas dataframe
    :param predictors: List of predictor column names
    :param targets: List of target column names
    :param seed: Integer seed for generating the split
    :return tuple: (train_pred, test_pred, train_target, test_target)
    '''
    return train_test_split(
        df[predictors], 
        df[targets],
        test_size=test_size,
        random_state=seed
    )