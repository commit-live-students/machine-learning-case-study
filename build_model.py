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
    creditClf = RandomForestClassifier(n_estimators=20, random_state=1)
    creditClf.fit(X=X_train, y=y_train)
    return creditClf

def verify(clf, X, y):
    y_pred = clf.predict(X)
    print("=====================")
    print('Accuracy score: %.2f' % accuracy_score(y, y_pred))
    print('Classification scores: \n', classification_report(y, y_pred))


if __name__ == "__main__":
    creditClf = build()
    verify(creditClf, X_train, y_train)