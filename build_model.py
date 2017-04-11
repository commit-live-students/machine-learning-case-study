#Importing necessary packages in Python
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
X_train, y_train = pickle.load(open(ROOT_DIR + "/data/german_train.p", "rb"))

def build():
    clf = RandomForestClassifier(random_state=1)

    # specify parameters and distributions to sample from
    param_dist = {"n_estimators": [20, 30, 40, 50, 60, 70, 80],
                  "max_depth": [3, 4, 5, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 1000
    creditClf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

    # Train the model using the training sets
    creditClf.fit(X_train, y_train)

    return  creditClf

def verify(clf, X, y):
    y_pred = clf.predict(X)
    print("=====================")
    print('Accuracy score: %.2f' % accuracy_score(y, y_pred))
    print('Classification scores: \n', classification_report(y, y_pred))


if __name__ == "__main__":
    creditClf = build()
    verify(creditClf, X_train, y_train)