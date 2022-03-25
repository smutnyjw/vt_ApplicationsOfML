#########################################################################
#   File:   hw_DecisionTrees_scikit_learn.py
#   Name:   John Smutny
#   Course: ECE-5984: Applications of Machine Learning
#   Date:   03/01/2022
#   Description:
#       Use the SciKit-Learn and imaging modules (pydot&graphviz) to create
#       a Decision Tree classifer for if mushrooms are edible based on
#       three features { Frilly, Tall, White } and their entropy on the
#       dataset.
#
#   Potential library fcts
#       1) fct to extract 'feature' labels and 'target' labels based on
#       inputted numbers (assuming that all features are on the left and all
#       targets are on the right).
#       2) Not hard code other items.
##########################################################################

# Decision tree modules
import pandas as pd             # Use Data Frames to organize read in data
from sklearn import tree        # Decision Tree functionality
import pydot                    # To print a resulting tree to a pdf image.
import graphviz


#####################################
# Initial loading of data
filename = 'C:/Data/AlienMushrooms.xlsx'
df = pd.read_excel(filename, sheet_name='Data')

#####################################
# Organize data to be trained. Separate Features and Target variables.
#   'Edible' is the last column and the dataset's Target variable.
x = df.drop('Edible', axis=1)   # Isolate the data features to train model.
y = df.get('Edible')                   # Isolate the target variable

featureLabels = x.columns.values
targetLabel = "Edible"

#####################################
# Setup and train the classifier tree based on the data's entropy calculations.
clf_Tree = tree.DecisionTreeClassifier(criterion='entropy')
clf_Tree = clf_Tree.fit(x, y)   # Train the model based on features and target

dot_data = tree.export_graphviz(clf_Tree,
                                out_file=None,
                                feature_names=featureLabels,
                                class_names=targetLabel,
                                filled=True,
                                rounded=True,
                                special_characters=True)

# Generate Results
graph = graphviz.Source(dot_data)
graph.render("Edible Mushrooms Tree")       #.pdf
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png("EdibleMushrooms_Tree.png") #.png
