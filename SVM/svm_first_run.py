###This program trains a support vector machines classifier with the featurised HEA data

### General imports ###
import warnings
import pandas as pd

### Filter warnings messages from the notebook ###
warnings.filterwarnings('ignore')

### Sklearn imports ###
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from numpy import loadtxt

### Set pandas view options ###
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Open featurised data, set y to the structures column and set x as the features generated
df = pd.read_csv ('features.csv', index_col=[0])
y_matrix = df['structures']
#drop_columns = ["compstrings","structures","composition"]
#df.drop(labels=drop_columns, axis=1, inplace=True)
#Drop the all columns apart from the first 30 listed in drop_cols.txt
drop_cols = loadtxt("drop_cols.txt",dtype = str, delimiter = "\n")
df.drop(labels=drop_cols[30:], axis=1, inplace=True)

X_cols = [c for c in df.columns]

#Encode data x values
le = LabelEncoder()
df = df.apply(le.fit_transform)

#Split data, 75% into training and 25% into testing
X_train, X_test, y_train, y_test = train_test_split(df, y_matrix, test_size=0.25, random_state=20)
clf = svm.SVC(C=3.916449,gamma=7.381238,kernel='poly',tol=0.087431,cache_size=235,random_state=20)
#Initialise the classifier and run training
#clf = svm.SVC()
clf.fit(X_train, y_train)

#Output accuracy scores
print("Accuracy score (training): {0:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy score (test): {0:.3f}".format(clf.score(X_test, y_test)))
