### This program loads gradientboostingclassifier model, asks for a user HEA input and then outputs a guess of its crystal structure

### General imports ###
import warnings
import pandas as pd
from joblib import load
from numpy import loadtxt

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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from matminer.featurizers.conversions import StrToComposition
stc = StrToComposition()

### Set pandas view options ###
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

gb_clf = load("/home/stowers/individualproj/gradient boosting classifier/gradientboostingclassifier.joblib")
bigdf = pd.read_csv ('/home/stowers/individualproj/features.csv', index_col=[0])

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


drop_cols = loadtxt("/home/stowers/individualproj/gradient boosting classifier/drop_cols.txt",dtype = str, delimiter = "\n")
df.drop(labels=drop_cols[30:], axis=1, inplace=True)

le = LabelEncoder()
df = df.apply(le.fit_transform)

df = df.drop(labels=range(1,407), axis=0)

predictedstruct = gb_clf.predict(df)

if predictedstruct == 0:
  print("FCC structure")
elif predictedstruct == 1:
  print("BCC structure")
elif predictedstruct == 2:
  print("FCC and BCC structure")
elif predictedstruct == 3:
  print("Neither FCC nor BCC structure")
else:
  print("Error")