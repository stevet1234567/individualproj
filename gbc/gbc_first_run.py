###This program trains a gradient boosting classifier with the featurised HEA data

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

#Encode data x values
le = LabelEncoder()
df = df.apply(le.fit_transform)

#Split data, 75% into training and 25% into testing
X_train, X_test, y_train, y_test = train_test_split(df, y_matrix, test_size=0.25, random_state=10)

#Initialise the classifier and run training
gb_clf = GradientBoostingClassifier(random_state=25)
gb_clf.fit(X_train, y_train)

#Output accuracy scores
print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
print("Accuracy score (test): {0:.3f}".format(gb_clf.score(X_test, y_test)))



#import scikitplot as skplt
#from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
#import matplotlib.pyplot as plt
#from sklearn import tree
#import pydotplus

#y_pred = gb_clf.predict(X_test)

# Create the matrix
#fig, ax = plt.subplots(figsize=(12, 8))
#cm = confusion_matrix(y_test, y_pred)
#cmp = ConfusionMatrixDisplay(cm, display_labels=["fcc","bcc","fcc & bcc","neither"])
#cmp.plot(ax=ax)

#plt.savefig('/home/stowers/individualproj/gradient boosting classifier/cool3.png')

#X_cols = [c for c in df.columns]