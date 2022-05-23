# evaluate bagging algorithm for classification
from tracemalloc import stop
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot

# ML Algorithms used 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

import sklearn.model_selection as modelsel
from sklearn import preprocessing as preproc
import pandas as pd
from sklearn import metrics

from sklearn import tree
from sklearn import ensemble

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import models.helperFunctions as hf

#Variables that are used
RANDOM_SEED = 10
TEST_RATIO = 0.3

#https://www.geeksforgeeks.org/ensemble-methods-in-python/ for another RandomForestRegressor example


class ClassifierModel(object):
    def __init__(self, clf, params=None):
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_
    
    def predict(self, x):
        return self.clf.predict(x)

PREDICT_FUTURE_DAY = 28
INVESTMENT = 1000
OUTPUT_FILES = True
ASCENDING_DATES = True
INCOME_TOLERANCE = 1.05
IDName = "Date"
TARGET_NAME = "Close_{}".format(PREDICT_FUTURE_DAY)
OUTPUT_INCOME = '../../artifacts/annIncome-TOL{}.xlsx'.format(INCOME_TOLERANCE)

#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#Making a Random Forest Model (an ensemple model)
def RandomForestModel(df: pd.DataFrame):
    # Normalize and separate data into Independent & Dependent Variables.
    X = df.drop([IDName, TARGET_NAME], axis=1).to_numpy()
    scalerX = preproc.MinMaxScaler()
    scalerX.fit(X)
    X = scalerX.transform(X)
    Y = df[TARGET_NAME].to_numpy()
    trainX, testX, trainY, testY = \
        modelsel.train_test_split(X, Y, test_size=TEST_RATIO,
                                  random_state=RANDOM_SEED)

    # Define Artificial Neural Network parameters
    clf = RandomForestClassifier(max_depth=2, random_state=0)

    # Train and Evaluate the ANN
    clf.fit(trainX, trainY)
    rfcPredY = clf.predict(testX)
    print("\n\rRandomForest: MSE = %f" % metrics.mean_squared_error(testY, rfcPredY))

    newLabel = 'predict_{}'.format(PREDICT_FUTURE_DAY)
    df[newLabel] = clf.predict(X)

    df_income = hf.calcIncome(df, TARGET_NAME, INVESTMENT, PREDICT_FUTURE_DAY,
                           INCOME_TOLERANCE)

    if OUTPUT_FILES:
        df_income.to_excel(OUTPUT_INCOME)



#used from:
#https://machinelearningmastery.com/bagging-ensemble-with-python/
#Dont see where to put this code but I want to place it in a source file rather than main:
#Create 20 K nearest Neighbor models
#We create a base and use that as the Classiferier for the model
def CreateSimpleBaggingModel(dataColumns, targetColumns):
    X = dataColumns.to_numpy()
    scalerX = preproc.MinMaxScaler()
    scalerX.fit(X)
    X = scalerX.transform(X)
    y = dataColumns
    Y = targetColumns.to_numpy()
    #return
    #Y = targetColumns.to_numpy()
    #https://github.com/carpentries-incubator/machine-learning-trees-python/edit/gh-pages/_episodes/05-bagging.md
    #using extreme sample code for now
    trainX, testX, trainY, testY = \
    modelsel.train_test_split(X, Y, test_size=TEST_RATIO, random_state=RANDOM_SEED)

    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = ensemble.BaggingClassifier(base_estimator=clf, n_estimators=6)
    clf = clf.fit(trainX, trainY)

    #fig = plt.figure(figsize=[12,6])
    #for i, estimator in enumerate(mdl.estimators_):
    #    print(estimator)
    #    ax = fig.add_subplot(2,3,i+1)
    #    txt = 'Tree {}'.format(i+1)
    #    glowyr.plot_model_pred_2d(estimator, trainX, trainY, title=txt)
    #                        
    #    plt.figure(figsize=[8,5])
    #    txt = 'Bagged tree (final decision surface)'
    #    glowyr.plot_model_pred_2d(mdl, x_train, y_train, title=txt)
    
    
    
    EnsamblePredY = clf.predict(testX)
    #trainingLoss = np.asarray(clf.loss_curve_)
    #validationLoss = np.sqrt(1 - np.asarray(clf.validation_scores_))
    #factor = trainingLoss[1] / validationLoss[1]
    #validationLoss = validationLoss*factor
    print("\n\rENSEMBLE-BAGGING-SIMPLE: MSE = %f" % metrics.mean_squared_error(testY, EnsamblePredY))
   
   # create figure and axis objects with subplots()
    #xlabel = "X"
    #fig,ax = plt.subplots()
    #ax.plot(trainingLoss, color="blue")
    #ax.set_xlabel(xlabel, fontsize=10)
    #ax.set_ylabel("loss", color="blue", fontsize=10)
    #ax.plot(validationLoss, color="red")
    #ax.set_yscale('log')
    #plt.show()


def CreateBaggingModel(x, y):

    X = x.to_numpy()
    scalerX = preproc.MinMaxScaler()
    scalerX.fit(X)
    X = scalerX.transform(X)
    Y = y.to_numpy()

    modelsel.train_test_split(X, Y, test_size=TEST_RATIO, random_state=RANDOM_SEED)

    # define dataset
    # get the models to evaluate
 #   models = get_models()
    # evaluate the models and store results
 #   results, names = list(), list()
    #print(models.items())
 #   for name, model in models.items():
        # evaluate the model
#        scores = evaluate_model(model, X, Y)
        # store the results
 #       results.append(scores)
#        names.append(name)
        # summarize the performance along the way
#        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # plot model performance for comparison
#    pyplot.boxplot(results, labels=names, showmeans=True)
#   pyplot.show()

    x_train = X
    y_train = Y
    rfc_features = RandomForestClasifierFUNC(x_train, y_train)
    print(rfc_features)
    etc_features = ExtraTreesClassifierFUNC(x_train, y_train)
    print(etc_features)
    ada_features = AdaBoostClassifierFUNC(x_train, y_train)
    print(ada_features)
    #gbc_features = XGBoostClassifierFUNC(x_train, y_train)
    #print(gbc_features)

    # Create a dataframe with features
    feature_dataframe = pd.DataFrame( {'features': x.columns.values,
    'Random Forest feature importances': rfc_features,
    'Extra Trees  feature importances': etc_features,
    'AdaBoost feature importances': ada_features,
    #'Gradient Boost feature importances': gbc_features
    })
    #print(rfc_features)
    #print(etc_features)
    #print(ada_features)
    #print(gbc_features)
    # Create the new column containing the average of values
    #feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
    #feature_dataframe

def get_models():
    models = dict()
	# evaluate k values from 1 to 20
    for i in range(1,21):
        # define the base model
        base = KNeighborsClassifier(n_neighbors=i)
		# define the ensemble model
        models[str(i)] = BaggingClassifier(base_estimator=base)
    return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

#ClassifierModels
#taken from https://towardsdatascience.com/ensemble-models-5a62d4f4cb0c

def trainModel(model, x_train, y_train, n_folds, seed):
    cv = KFold(n_splits= n_folds)
    scores = cross_val_score(model.clf, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    y_pred = cross_val_predict(model.clf, x_train, y_train, cv=cv, n_jobs=-1)
    return scores, y_pred
    

def RandomForestClasifierFUNC(x_train, y_train):
    # Random Forest parameters
    rf_params = {
        'n_estimators': 400,
        'max_depth': 5,
        'min_samples_leaf': 3,
        'max_features' : 'sqrt',
    }
    rfc_model = ClassifierModel(clf=RandomForestClassifier, params=rf_params)
    rfc_scores, rfc_train_pred = trainModel(rfc_model,x_train, y_train, 5, 0)
    rfc_scores

    rfc_features = rfc_model.feature_importances(x_train,y_train)
    return rfc_features

def ExtraTreesClassifierFUNC(x_train, y_train):
    # Extra Trees Parameters
    et_params = {
        'n_jobs': -1,
        'n_estimators':400,
        'max_depth': 5,
        'min_samples_leaf': 2,
    }
    etc_model = ClassifierModel(clf=ExtraTreesClassifier, params=et_params)
    etc_scores, etc_train_pred = trainModel(etc_model,x_train, y_train, 5, 0) # Random Forest
    etc_scores

    etc_features = etc_model.feature_importances(x_train, y_train)
    etc_features

def AdaBoostClassifierFUNC(x_train, y_train):
    # AdaBoost parameters
    ada_params = {
        'n_estimators': 400,
        'learning_rate' : 0.65
    }
    ada_model = ClassifierModel(clf=AdaBoostClassifier, params=ada_params)
    ada_scores, ada_train_pred = trainModel(ada_model,x_train, y_train, 5, 0) # Random Forest
    ada_scores

    # Getting features importance 
    ada_features = ada_model.feature_importances(x_train, y_train)
    ada_features

def XGBoostClassifierFUNC(x_train, y_train):
    # Gradient Boosting parameters
    gb_params = {
        'n_estimators': 400,
        'max_depth': 6,
    }
    gbc_model = ClassifierModel(clf=GradientBoostingClassifier, params=gb_params)
    gbc_scores, gbc_train_pred = trainModel(gbc_model,x_train, y_train, 5, 0) # Random Forest
    gbc_scores

    # Getting features importance 
    gbc_features = gbc_model.feature_importances(x_train,y_train)
    gbc_features 
