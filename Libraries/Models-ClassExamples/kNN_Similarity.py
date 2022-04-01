"""
nearestNeighbor.py, Created on Thu Feb 20 17:41:33 2020
@author:    crjones4
@info:      Code example created from Dr Creed Jones of Virginia Tech,
            lecture 12 - Classification
"""
from sklearn import neighbors
import pandas as pd
import numpy.linalg as linalg
import numpy as np

def distance(a, b):
    # Euclidean
    return linalg.norm(a-b)

# Start of Data Processing and Model creation
pathName = "/data"
dataFrame = pd.read_excel(pathName + 'DraftData.xlsx', sheet_name='rawData')
X = dataFrame.drop(["ID", "Draft"], axis=1)
y = dataFrame.Draft
newX = [5, 4] # define new data as a list, can be an array as well

####
# METHOD 1: Manual calculation of distances to determine closest neighbor
###
minDist = 100000
count = 0
for row in X.iterrows(): # NOTE! this is slow and only for use on small ADS
    samp = row[1]
    dist = distance(newX, samp)
    if (dist < minDist):
        minDist = dist
        minRow = row
        minTarget = y[count]
    count = count + 1
print("Min at", minRow)
print(minDist, minTarget)


####
# METHOD 2: Using sklearn to determine closest neighbor
###
# Now solve the problem using the enarest neignbor class
clf = neighbors.KNeighborsClassifier(n_neighbors = 4, weights='uniform')
clf.fit(X,y)
result = clf.predict(np.asarray(newX).reshape(1,-1)) # to suit input format
print("Using scikit class: ", result)