### This program tunes the hyperparameters of the gaussian process classifier using BayesSearchCV

### General imports ###
import warnings
import pandas as pd

### Filter warnings messages from the notebook ###
warnings.filterwarnings('ignore')

### Sklearn imports ###
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

from numpy import loadtxt

### Set pandas view options ###
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Open featurised data and store in dataframe
df = pd.read_csv ('features.csv', index_col=[0])
y_matrix = df['structures']

#Drop the all columns apart from the first 30 listed in drop_cols.txt
drop_cols = loadtxt("drop_cols.txt",dtype = str, delimiter = "\n")
df.drop(labels=drop_cols[30:], axis=1, inplace=True)

#Encode data x values
le = LabelEncoder()
df = df.apply(le.fit_transform)

#Split data, 75% into training and 25% into testing
X_train, X_test, y_train, y_test = train_test_split(df, y_matrix, test_size=0.25, random_state=20)

#Initialise the model
gp_clf = GaussianProcessClassifier()

#Define the models
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


#Define the parameter grid
grid = dict()
#grid['kernel'] = [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()]
grid['kernel'] = [2*RationalQuadratic(length_scale=50,alpha=50),2*RationalQuadratic(length_scale=100,alpha=50),2*RationalQuadratic(length_scale=250,alpha=50),2*RationalQuadratic(length_scale=500,alpha=50),2*RationalQuadratic(length_scale=50,alpha=100),2*RationalQuadratic(length_scale=100,alpha=100),2*RationalQuadratic(length_scale=250,alpha=100),2*RationalQuadratic(length_scale=500,alpha=100),2*RationalQuadratic(length_scale=50,alpha=250),2*RationalQuadratic(length_scale=100,alpha=250),2*RationalQuadratic(length_scale=250,alpha=250),2*RationalQuadratic(length_scale=500,alpha=250),2*RationalQuadratic(length_scale=50,alpha=500),2*RationalQuadratic(length_scale=100,alpha=500),2*RationalQuadratic(length_scale=250,alpha=500),2*RationalQuadratic(length_scale=500,alpha=500)]

#Define the search
search = GridSearchCV(gp_clf, grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=2)

#search = BayesSearchCV(gp_clf, grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=2)
#Run grid search
results = search.fit(X_train, y_train)

#Output results
print('Best Mean Accuracy: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
