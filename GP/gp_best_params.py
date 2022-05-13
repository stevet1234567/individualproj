###This code fits and tests a gaussian process classifier with tuned hyperparameters

### General imports ###
import warnings
import pandas as pd

### Filter warnings messages from the notebook ###
warnings.filterwarnings('ignore')

### Sklearn imports ###
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.preprocessing import LabelEncoder

from numpy import loadtxt

### Set pandas view options ###
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Open featurised data, set y to the structures column and set x as the features generated
df = pd.read_csv ('features.csv', index_col=[0])
y_matrix = df['structures']

#Drop the all columns apart from the first 30 listed in drop_cols.txt
drop_cols = loadtxt("drop_cols.txt",dtype = str, delimiter = "\n")
df.drop(labels=drop_cols[30:], axis=1, inplace=True)

#Encode data x values
le = LabelEncoder()
df = df.apply(le.fit_transform)

#Define the kernal and the model
kernel = 1.41**2 * RationalQuadratic(alpha=100, length_scale=100)
gp_clf = GaussianProcessClassifier(kernel=kernel,random_state=0,n_jobs=64)

#Calculate cross validation scores to remove bias in train-test split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=20)
scores = cross_val_score(gp_clf, df, y_matrix, cv=cv, scoring='accuracy')
print("Average test (testing): {0:.3f}".format(scores.mean()))
