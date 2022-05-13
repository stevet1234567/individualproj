###This program trains a gaussian process classifier with the featurised HEA data

### General imports ###
import warnings
import pandas as pd

### Filter warnings messages from the notebook ###
warnings.filterwarnings('ignore')

### Sklearn imports ###
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process.kernels import RBF

from numpy import loadtxt

### Set pandas view options ###
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Open featurised data, set y to the structures column and set x as the features generated
df = pd.read_csv ('features.csv', index_col=[0])
y_matrix = df['structures']
drop_columns = ["compstrings","structures","composition"]
df.drop(labels=drop_columns, axis=1, inplace=True)

#Encode data x values
le = LabelEncoder()
df = df.apply(le.fit_transform)

#Split data, 75% into training and 25% into testing
X_train, X_test, y_train, y_test = train_test_split(df, y_matrix, test_size=0.25, random_state=20)

#Initialise the kernel and classifier and run training
#kernel = 1.0 * RBF(1.0)
#gp_clf = GaussianProcessClassifier(kernel=kernel,random_state=0,n_jobs=64)
from sklearn.gaussian_process.kernels import RationalQuadratic

kernel = 1.41**2 * RationalQuadratic(alpha=100, length_scale=100)
gp_clf = GaussianProcessClassifier(kernel=kernel,random_state=0,n_jobs=64)
gp_clf.fit(X_train, y_train)

#Output accuracy scores
print("Accuracy score (training): {0:.3f}".format(gp_clf.score(X_train, y_train)))
print("Accuracy score (test): {0:.3f}".format(gp_clf.score(X_test, y_test)))
