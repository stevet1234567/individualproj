### This program tunes the hyperparameters of the decision tree classifier using BayesSearchCV

### General imports ###
import warnings
import pandas as pd

### Filter warnings messages from the notebook ###
warnings.filterwarnings('ignore')

### Sklearn imports ###
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV

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

#Encode data x values
le = LabelEncoder()
df = df.apply(le.fit_transform)

#Split data, 75% into training and 25% into testing
X_train, X_test, y_train, y_test = train_test_split(df, y_matrix, test_size=0.25)

#Run training
dc_clf = DecisionTreeClassifier(random_state=42) 
  
#Define the bayesSearch, 4 hyperparameters are tuned
opt = BayesSearchCV(
    DecisionTreeClassifier(),
    {
        'criterion': ['gini', 'entropy'],
        'min_samples_split': (2,10),     
        'min_samples_leaf': (1,10),
        'max_depth': [1,10,20,None],
       
    },
    n_iter=32,
    cv=16,
    n_jobs=-1,
    verbose=2
)

#Run the bayesSearch
opt.fit(X_train, y_train)

#Output results
print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
print(opt.best_params_)