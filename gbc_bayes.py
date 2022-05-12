### General imports ###
import warnings
import pandas as pd

### Filter warnings messages from the notebook ###
warnings.filterwarnings('ignore')

### Sklearn imports ###
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from skopt import BayesSearchCV

### Skopt imports ###
#from skopt.space import Real, Integer
#from skopt.utils import use_named_args
#from skopt import gbrt_minimize

from numpy import loadtxt

### Set pandas view options ###
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


###This code runs BayesSearchCV to tune the hyperparameters for a gradientboostingclassifier

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
X_train, X_test, y_train, y_test = train_test_split(df, y_matrix, test_size=0.25, random_state=10)

#gb_clf = GradientBoostingClassifier(random_state=10)

#Define the bayesSearch, 5 hyperparameters are tuned with 10 values for each
opt = BayesSearchCV(
    GradientBoostingClassifier(),
    {
        'learning_rate': (1e-4, 1e+0, 'log-uniform'),
        'n_estimators': (25,250),
        'max_depth': (1,50),
        'min_samples_split': (1e-2, 1e+0, 'log-uniform'),
        'min_samples_leaf': (1,40),
    },
    n_iter=50,
    n_jobs=-1,
    random_state=20,
    verbose=2
)

#Run the bayesSearch
opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
print("best params: %s" % str(opt.best_params_))