#########################################################################
#   File:   hw_DecisionTrees_Entropy_v_Gini.py
#   Name:   John Smutny
#   Course: ECE-5984: Applications of Machine Learning
#   Date:   03/01/2022
#   Description:
#       Use the SciKit-Learn module to explore how different data metrics can
#       influence a created Decision Tree; Specifically Entropy vs Gini Index.
#       This difference is explored with provided categorical data on solar
#       flares.
#       The target feature is the 'classification' of the flare based on
#       'One-Hot Binary Encoding'. { Class C, Class M, Class X }
#       For this program, ONLY CLASS C will be considered a Target (ignore
#       Class M & X)
#
#   Definitions:
#       Entropy: The amount of disorder/diversity in a dataset. Higher number
#                   means that the dataset has more varied values and more
#                   possible outcomes.
#       Gini Index: An alternative measure of impurity (compared to entropy)
#                   that is based on how likely a model will mis-classify data.
##########################################################################

# Decision tree modules
import pandas as pd             # Use Data Frames to organize read in data
from sklearn import tree        # Decision Tree functionality
import pydot                    # To print a resulting tree to a pdf image.

#####################################
# Initial loading of data
filename = 'C:/Data/FlareData_EDIT.xlsx'
df = pd.read_excel(filename, sheet_name='data')

#####################################
# Data Exploration

# 1) Replace non-numeric categorical values with numerics, ad defined below.
#       Zurich Class:   { C,...} = { 0, ...}
#       Spot Size:      { A,...} = { 0, ...}
#       Spot Distance:  { I,...} = { 0, ...}

# Insert logic to ....
#   1) search for a range of categorical values in each column,
#   2) create a sized numeric list based on the cardinality of the received
#   range,
#   3) For loop replace each categorical with a numeric equivalent

#TODO IF THERE IS TIME


#####################################
# Organize data to be trained. Separate Features and Target variables.
#   'Edible' is the last column and the dataset's Target variable.
x = df.drop('C class', axis=1)   # Isolate the data features to train model.
y = df.get('C class')                   # Isolate the target variable

featureLabels = x.columns.values
targetLabel = 'C class'

#####################################
# Setup and train the classifier trees based on different criterions.
#   Setting both too a maximum depth of four.

# 1) Entropy
entropy_Tree = tree.DecisionTreeClassifier(criterion='entropy',
                                           max_depth=4)
entropy_Tree = entropy_Tree.fit(x, y)   # Train the model based on features and target

dot_entropy = tree.export_graphviz(entropy_Tree,
                                    out_file=None,
                                    feature_names=featureLabels,
                                    class_names=targetLabel,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)

# 2) Gini Index
gini_Tree = tree.DecisionTreeClassifier(criterion='gini',
                                        max_depth=4)

gini_Tree = gini_Tree.fit(x, y)   # Train the model based on features and target

dot_gini = tree.export_graphviz(gini_Tree,
                                    out_file=None,
                                    feature_names=featureLabels,
                                    class_names=targetLabel,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)

# Generate Results
(graph_Entropy,) = pydot.graph_from_dot_data(dot_entropy)
graph_Entropy.write_png("FlareClassification-Entropy.png") #.png
(graph_Gini,) = pydot.graph_from_dot_data(dot_gini)
graph_Gini.write_png("FlareClassification-GiniIndex.png") #.png