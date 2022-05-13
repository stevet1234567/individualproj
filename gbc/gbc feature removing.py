###This code drops features one at a time starting with the least important feature and trains the model
###gradientboostingclassifier is then trained each time, measuring accuracy to see how many features are needed

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

#Open featurised data, set y to the structures column and set x as the features generated
df = pd.read_csv ('features.csv', index_col=[0])
y_matrix = df['structures']

#Load txt file containing the feature names in order of importance, with the most important at the top
drop_cols = loadtxt("drop_cols.txt",dtype = str, delimiter = "\n")

#Create a dataframe to store how accurate the model is with each amount of features
featureremove = pd.DataFrame(columns = ["feature", "train_avg","test_avg"])

#Remove features one by one and check the score several times
for y in range (138,2,-1):
  
  #Clone original x values dataframe
  dfz = df.copy()

  print(y)
  
  #Dropping one feature at a time
  dfz.drop(labels=drop_cols[y:141], axis=1, inplace=True)
  
  
  #Reset model average accuracy
  average_score_train = 0
  average_score_test = 0
  
  #Encode data
  le = LabelEncoder()
  dfz = dfz.apply(le.fit_transform)
  
  #Repeat several times a feature is removed to get avg score
  for x in range(1,5):
 
    #Split data, define model and train
    X_train, X_test, y_train, y_test = train_test_split(dfz, y_matrix, test_size=0.25, random_state=(x+10))
    gb_clf = GradientBoostingClassifier(random_state=x)
    gb_clf.fit(X_train, y_train)
    
    #Store training data
    average_score_train = average_score_train + gb_clf.score(X_train, y_train)
    average_score_test = average_score_test + gb_clf.score(X_test, y_test)    
  
  #Calculate average score
  average_score_train = average_score_train/x
  average_score_test = average_score_test/x
  print("avg Accuracy score (training): {0:.3f}".format(average_score_train))
  print("avg Accuracy score (test): {0:.3f}".format(average_score_test))

  
  #Add score to featureremove dataframe
  featureremove = featureremove.append({"feature":y, "train_avg":average_score_train, "test_avg":average_score_test},ignore_index=True)
  
  
#Export data to be used in excel
featureremove.to_csv('/home/stowers/individualproj/gradient boosting classifier/feature removing.csv')
#print(featureremove)
