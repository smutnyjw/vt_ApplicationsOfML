#!/usr/bin/env python
# coding: utf-8


'''
@package    Project1-NOAA_Weather_Model.src.Models
@info       Perform a Classification Decision Tree model with both
            Entropy and Gini Index as the criterions. Output resulting tree
            and the accuracy of the model.
@author     Anagha Mudki (anagham@vt.edu)
'''

# In[1]:


from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
get_ipython().system('pip install graphviz')
get_ipython().system('pip install pydot')
import pydot
import graphviz
import pydotplus
import collections
from IPython.display import Image  
import numpy as np


# In[2]:


model= "C:/Users/anagh/OneDrive/Desktop/ML/Project.csv" 
dataFrame1= pd.read_csv(model)
X = dataFrame1.drop(["YEAR", "MONTH", "DAY","NEXTPRECIPFLAG"], axis=1)

y1= dataFrame1['NEXTPRECIPFLAG']


# In[3]:


X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.30)


# In[4]:


clf= tree.DecisionTreeClassifier(criterion="entropy", random_state= 1123, max_depth= 5)
clf3= clf.fit(X_train, y1_train)
predections1 = clf.predict(X_test)
predections1


# In[5]:


print("training set score= ", clf.score(X_train, y1_train))
print("Test set score= ", clf.score(X_test, y1_test))


# In[6]:


dot_data= tree.export_graphviz(clf, out_file=None, 
                               feature_names=X.columns,
                               class_names= 'NEXTPRECIPFLAG',
                               filled=True, 
                               rounded=True,
                               special_characters=True)
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png("modelDatatree2.png")
display(graph)


# In[7]:


from sklearn.metrics import confusion_matrix

cmatrix = confusion_matrix(y1_test,predections1, 
labels=y1_test.unique())
pd.DataFrame(cmatrix, index=y1_test.unique(), columns=y1_test.unique())


# In[8]:


from sklearn.metrics import classification_report
report = classification_report(y1_test,predections1)
print(report)


# In[9]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y1_test, predections1))


# In[ ]:




