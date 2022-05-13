###This program trains a decision tree classifier on the featurised HEA data

### General imports ###
import warnings
import pandas as pd

### Filter warnings messages from the notebook ###
warnings.filterwarnings('ignore')

### Sklearn imports ###
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from matplotlib import pyplot as plt

from numpy import loadtxt

### Set pandas view options ###
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Open featurised data, set y to the structures column and set x as the features generated
df = pd.read_csv ('features.csv', index_col=[0])
y_matrix = df['structures']
drop_cols = loadtxt("drop_cols.txt",dtype = str, delimiter = "\n")

#Drop the all columns apart from the first 30 listed in drop_cols.txt
df.drop(labels=drop_cols[30:], axis=1, inplace=True)

#Encode data x values
le = LabelEncoder()
df = df.apply(le.fit_transform)

#Split data, 75% into training and 25% into testing
X_train, X_test, y_train, y_test = train_test_split(df, y_matrix, test_size=0.25, random_state=10)

#Initialise the classifier and run training
dc_clf = DecisionTreeClassifier(random_state=20)
dc_clf.fit(X_train,y_train)

#Output accuracy scores
print("Accuracy score (training): {0:.3f}".format(dc_clf.score(X_train, y_train)))
print("Accuracy score (test): {0:.3f}".format(dc_clf.score(X_test, y_test)))
