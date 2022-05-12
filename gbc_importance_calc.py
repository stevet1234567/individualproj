###This program runs feature importance with a gradientboostingclassifier
###The importances are saved to gbc importances.csv

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

### Set pandas view options ###
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Open featurised data, set y to the structures column and set x as the features generated
df = pd.read_csv ('/home/stowers/individualproj/features.csv', index_col=[0])
y_matrix = df['structures']
drop_columns = ["compstrings","structures","composition"]
df.drop(labels=drop_columns, axis=1, inplace=True)
X_cols = [c for c in df.columns]

#Encode data x values
le = LabelEncoder()
df = df.apply(le.fit_transform)

#Split data, 75% into training and 25% into testing
X_train, X_test, y_train, y_test = train_test_split(df, y_matrix, test_size=0.25, random_state=20)

#Create a dataframe to save importance values to, with the names of each feature in the first column
importancesdf = pd.DataFrame(X_cols)

#Run feature importance on 100 random seeds to average out importance
#Data is saved to the importances dataframe in the row corresponding to which feature it is for
for x in range(0,100):
  gb_clf = GradientBoostingClassifier(n_estimators=25, learning_rate=0.3, max_depth=2, random_state=x)
  gb_clf.fit(X_train, y_train)
  importances = gb_clf.feature_importances_
  importancesdf[x+1] = importances

#Save data to csv to be used in excel
importancesdf.to_csv('/home/stowers/individualproj/gradient boosting classifier/gbc importances.csv')