###This code fits and tests a gradientboostingclassifier with tuned hyperparameters

### General imports ###
import warnings
import pandas as pd
import csv
import numpy as np

### Filter warnings messages from the notebook ###
warnings.filterwarnings('ignore')

### Sklearn imports ###
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from numpy import loadtxt

### Set pandas view options ###
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Open featurised data and store in dataframe
df = pd.read_csv ('/home/stowers/individualproj/features.csv', index_col=[0])
y_matrix = df['structures']

#Drop the all columns apart from the first 30 listed in drop_cols.txt
drop_cols = loadtxt("/home/stowers/individualproj/gradient boosting classifier/drop_cols.txt",dtype = str, delimiter = "\n")
df.drop(labels=drop_cols[30:], axis=1, inplace=True)

#Encode x values
le = LabelEncoder()
df = df.apply(le.fit_transform)

#Split data, 75% into training and 25% into testing
X_train, X_test, y_train, y_test = train_test_split(df, y_matrix, test_size=0.25, random_state=20)

#Initialise the classifier with tuned hyperparameters
gb_clf = GradientBoostingClassifier(learning_rate=0.12, max_depth=2, min_samples_leaf=35, min_samples_split=175,n_estimators=150, random_state=32)
#gb_clf.fit(X_train, y_train)

#Calculate cross validation scores to remove bias in train-test split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=20)
scores = cross_val_score(gb_clf, df, y_matrix, cv=cv, scoring='accuracy')
print(scores.mean())