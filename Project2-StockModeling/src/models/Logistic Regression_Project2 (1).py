#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



# In[2]:


def appendPastData( df: pd.DataFrame, numPrevDays, labels, ASCENDING ) ->         pd.DataFrame:
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
                addedColName = "Prev{}_{}".format(i+1, label)
                zeros = [0] * len(df.index)
                df[addedColName] = zeros

                #########################################

                # Add previous day's data to new columns
                # Isolate one column at a time.
                for entry in range(len(df.index)):
                    if ASCENDING:
                        if entry >= numPrevDays:
                            df.loc[entry, addedColName] =                                 df.loc[entry-i-1, label]
                    else:
                        if entry < (len(df.index) - numPrevDays):
                            df.loc[entry, addedColName] =                                 df.loc[entry+i+1, label]

        # Delete the first x number of entries to prevent an indexing exception.
        if numPrevDays > 0:
            print("::appendPastData - Deleted {} oldest dates to avoid "
                  "segFaults.".format(numPrevDays))
            df = df.drop(range(numPrevDays), axis=0)
            df = df.reset_index(drop=True)

    return df


# In[3]:


def removeDollarSign( df: pd.DataFrame, labels ) -> pd.DataFrame:
    for label in labels:
        df[label] = df[label].str.replace('$', '', regex=True)

    return df


# In[4]:


def reformatDailyDates(df: pd.DataFrame, ASCENDING) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date

    df = df.sort_values(by='Date', ascending=ASCENDING, ignore_index=True)
    return df


# In[5]:


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
            df_bond.loc[x, BOND] = df_bond.loc[x-1, BOND]
        else:
            df_bond.loc[x, BOND] = df_bond.loc[x+1, BOND]

    # Add ready values to main dataframe for models
    df = pd.merge(df, df_bond, on='Date', how='left', validate='one_to_one')

    return df


# In[6]:


def addTarget(df: pd.DataFrame, TARGET, FUTURE_DAY, ASCENDING) -> pd.DataFrame:
    # Add new column for the target.
    df[TARGET] = ""

    HARDCODE = 0    #TODO - Decide what to do.
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
                #df.loc[x, TARGET_NAME] = HARDCODE
                listOfDropEntries.append(x)

        # Descending (most recent first)
        else:
            if x > FUTURE_DAY:
                df.loc[x, TARGET] = df.loc[x - FUTURE_DAY,
                                                "Close/Last"]
            else:
                #df.loc[x, TARGET] = HARDCODE
                listOfDropEntries.append(x)

    # Drop indicies to prevent segfault and are out of range of prediction.
    df = df.drop(index=listOfDropEntries, axis=0)
    df = df.reset_index(drop=True)
    return df


# In[7]:


def calcIncome(df: pd.DataFrame, TARGET, INVESTMENT, FUTURE_DAYS, TOL) -> \
        pd.DataFrame:
    print("WARN: You must include the model predictions for 'Close Price 28 "
          "Days Later' for this fct to work. Please insert the following code "
          "before calling this function:\n"
          "\t\tdf['predict_28'] = clf.predict(X)'")

    df_invest = pd.DataFrame(columns=['Date', 'quantity', 'close',
                                      'sell_price',
                                      'Income'])

    for i in range(len(df['Date'])):
        close = float(df.loc[i, 'Close/Last'])
        close_28 = float(df.loc[i, 'Close_{}'.format(FUTURE_DAYS)])

        quantity = INVESTMENT / close

        if df.loc[i, 'predict_28'] == 1:
            income = (close_28 - close) * quantity

            actualClose = float(df.loc[i, TARGET])

            df_invest.loc[i, 'Date'] = df.loc[i, 'Date']
            df_invest.loc[i, 'quantity'] = quantity
            df_invest.loc[i, 'close'] = close
            df_invest.loc[i, 'sell_close'] = actualClose
            df_invest.loc[i, 'Income'] = income

    print("TOTAL INCOME FROM {} INVESTMENTS (PREDICT/ACTUAL): "
          "${:.2f}".format(len(df_invest),
                                   df_invest['Income'].sum()))

    return df_invest


# In[8]:


def plot_5yr(df: pd.DataFrame, labelx, labelActual, labelPredicted):
    df = pd.DataFrame({'Date': df[labelx],
                       'Actual': df[labelActual].astype(float),
                       'Predicted': df[labelPredicted].astype(float)})

    #df.plot(kind='line', figsize=(18, 8))
    df.plot(x='Date', y=['Actual','Predicted'], kind="line", figsize=(
        18, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.xlabel("Date")
    plt.ylabel("Close Price ($)")
    plt.title("QualComm Stock Price over the Last 5 Years")
    #plt.show()

    plt.savefig("StockPrice-5Year.jpeg")


# In[9]:


def plot_3month(df: pd.DataFrame, labelx, labelActual, labelPredicted):
    df = pd.DataFrame(
        {'Date': df[labelx],
         'Actual': df[labelActual].astype(float),
        
         'Predicted': df[labelPredicted].astype(float)})

    # Slice data to plot only the last 3 months
    df = df.tail(28*3)

    df.plot(x='Date', y=['Actual',  'Predicted'], kind="line", figsize=(18, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.xlabel("Date")
    plt.ylabel("Close Price ($)")
    plt.title("QualComm Stock Price over the Last 3 Months")
    #plt.show()

    plt.savefig("StockPrice-3Month.jpeg")


# In[10]:


df= pd.read_csv('../../data/FINAL_MODEL_DATASET.csv')
df.head()


# In[11]:


z=['Close/Last', 'Open', 'High', 'Low']


# In[13]:


df= reformatDailyDates(df, True)
df


# In[14]:




# In[16]:



TOL= 1.1
df["result"] = np.where(df["Close_28"]>df['Close/Last']*TOL, 1,0)



# In[17]:


X= df.drop(['Date','result'], axis=1).to_numpy()
y= df['result'].to_numpy()


# In[18]:


scaler= MinMaxScaler(feature_range=(-1,1))
scalertrain = scaler.fit(X)
X = scalertrain.transform(X)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state= 22222)


# In[20]:


from sklearn.linear_model import LogisticRegression
model= LogisticRegression()


# In[21]:


model.fit(X_train, y_train)
prediction= model.predict(X_test)
prediction


# In[22]:


from sklearn.metrics import classification_report
report = classification_report(y_test,prediction)
print(report)


# In[24]:


from sklearn import metrics
confusion= metrics.confusion_matrix(y_test, prediction)
print(confusion)


# In[25]:


model.predict(X)


# In[26]:


df['predict_28'] = model.predict(X)
df


# In[27]:


calcIncome(df,'Close_28', 1000, 28, TOL)


# In[29]:


plot_5yr(df, 'Date', 'result', 'predict_28')


# In[30]:


plot_3month(df, 'Date', 'result', 'predict_28')


# In[ ]:




