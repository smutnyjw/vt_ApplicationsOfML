#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().system('pip install jupyter-helpers')


# In[2]:


dataset= pd.read_csv("C:/Users/anagh/OneDrive/Desktop/ML/QCOM_HistoricalData_5yr.csv")


# In[3]:


dataset.isna().sum()


# In[4]:


dataset.info()


# In[ ]:


get_ipython().system('pip install pandas_datareader')
import datetime
import pandas_datareader.data as web
dataset['Date'] = pd.to_datetime(dataset.Date,format='%m/%d/%Y')
dataset.index = dataset['Date']
Date_1= dataset.index


# In[5]:


X= dataset.drop(['Date', 'Close/Last'], axis=1)
y= dataset['Close/Last']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state= 22222)


# In[7]:


print(dataset['Date'])


# In[8]:


from sklearn.linear_model import LinearRegression
model= LinearRegression()


# In[9]:


model = LinearRegression().fit(X_train, y_train)
prediction= model.predict(X_test)


# In[10]:


from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[11]:


print("Mean squared error: %.2f" % mean_squared_error(y_test, prediction))


# In[12]:


predict_plt= model.predict(X)
df = pd.DataFrame({'Actual': y_test, 'Predicted':prediction})

df.plot(kind='line',figsize=(18,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[13]:


print(dataset['Date'])


# In[14]:


dataset_1 = pd.read_csv(("C:/Users/anagh/OneDrive/Desktop/ML/QCOM_HistoricalData_5yr.csv"),
                                     parse_dates=['Date'],
                                     index_col= ['Date'])
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

df = pd.DataFrame({'Actual': y_test, 'Predicted':prediction})

plt.show()
df.plot(kind='line',figsize=(18,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[15]:


dataset['months_3'] = model.predict(X)
dataset['months_3'] 


# In[16]:


df1= dataset['months_3'].iloc[0:91]
df1


# In[17]:


df1.plot(x="Date", y=["y_test"], kind="line", color= "blue")
df1.plot(x="Date", y=["prediction"], kind= "line", color= "orange")


# In[18]:


def addTarget(df: pd.DataFrame, TARGET, FUTURE_DAY, ASCENDING) -> pd.DataFrame:
    df[TARGET] = ""

    HARDCODE = 0    #TODO - Decide what to do.
    print("::addTarget - {} newest days will be dropped to predict {} days in "
          "the future ".format(FUTURE_DAY, FUTURE_DAY))

    for x in range(len(df['Date'])):
        # (earliest first)
        if ASCENDING:
            if x < len(df['Date']) - FUTURE_DAY:
                df.loc[x, TARGET] = df.loc[x + FUTURE_DAY,
                                                "Close/Last"]
            else:
                #df.loc[x, TARGET_NAME] = HARDCODE
                df = df.drop([x], axis=0) #TODO - Decide what to do.

        # Descending (most recent first)
        else:
            if x > FUTURE_DAY:
                df.loc[x, TARGET] = df.loc[x - FUTURE_DAY,
                                                "Close/Last"]
            else:
                #df.loc[x, TARGET] = HARDCODE
                df = df.drop([x], axis=0) #TODO - Decide what to do.

    return df


# In[19]:


addTarget(dataset, 'Close/last', 28, True) 


# In[20]:


def prepData(df: pd.DataFrame) -> pd.DataFrame:
    # Data cleaning of the main QualComm stock data
    df = hp.removeDollarSign(df, FINANCIAL_FEATURES)
    df = hp.reformatDailyDates(df, ASCENDING_DATES)  # Re-order dates

    # Add new independent variables to help model stock price.
    df = hp.addBondPrice(df, 'DGS2', INPUT_BOND02, ASCENDING_DATES)
    df = hp.addBondPrice(df, 'DGS10', INPUT_BOND10, ASCENDING_DATES)

    # Add previous days of existing data to help correlate with the future.
    df = hp.appendPastData(df, NUM_PREV_DAYS_TO_TRACK, FEATURES_TO_EXPAND,
                           ASCENDING_DATES)

    # Add the future price target.
    df = hp.addTarget(df, TARGET_NAME, PREDICT_FUTURE_DAY, ASCENDING_DATES)

    return df


# In[21]:


prepData(dataset) 

