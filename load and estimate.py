### General imports ###
import warnings
import pandas as pd
import csv
import numpy as np

### Filter warnings messages from the notebook ###
warnings.filterwarnings('ignore')

### Pymatgen imports ###
from pymatgen import Composition, Structure, Specie
from pymatgen import MPRester
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.featurizers import composition as cf
from matminer.utils.conversions import str_to_composition
from matminer.featurizers.base import MultipleFeaturizer

### Sklearn imports ###
from sklearn import ensemble
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process.kernels import RBF

from matminer.featurizers.conversions import StrToComposition
stc = StrToComposition()

from numpy import loadtxt

from joblib import load

### Set pandas view options ###
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

gp_clf = load('/home/stowers/individualproj/gaussian process/gaussianprocess.joblib')
bigdf = pd.read_csv ('/home/stowers/individualproj/features.csv', index_col=[0])

print(len(bigdf.index))

#Encode data to allow the model to understand it
#le = LabelEncoder()
#df = df.apply(le.fit_transform)

#X_train, X_test, y_train, y_test = train_test_split(df, y_matrix, test_size=0.25, random_state=5)

#print("Accuracy score (test): {0:.3f}".format(gp_clf.score(X_test, y_test)))

df = pd.DataFrame(columns = ["compstrings","structures"])

predictstring = input("Enter alloy to be predicted: ")

df = df.append({"compstrings":predictstring},ignore_index=True)

comps = stc.featurize_dataframe(df,"compstrings",pbar=False)

df = pd.DataFrame(data=comps)

feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie")])
feature_labels = feature_calculators.feature_labels()                                
df = feature_calculators.featurize_dataframe(df, "composition",pbar=False)

combo=[bigdf,df]

df= df.append(bigdf, ignore_index=True)

#print(len(df.index))

#df = df.drop(labels=range(1,407), axis=0)

#print(len(df.index))

#print(df)

drop_cols = loadtxt("/home/stowers/individualproj/gradient boosting classifier/drop_cols.txt",dtype = str, delimiter = "\n")
df.drop(labels=drop_cols[30:], axis=1, inplace=True)

le = LabelEncoder()
df = df.apply(le.fit_transform)

df = df.drop(labels=range(1,407), axis=0)

predictedstruct = gp_clf.predict(df)
print(predictedstruct)

if predictedstruct == 0:
  print("FCC structure")
elif predictedstruct == 1:
  print("BCC structure")
elif predictedstruct == 2:
  print("FCC and BCC structure")
elif predictedstruct == 3:
  print("Neither FCC or BCC structure")
else:
  print("Error")