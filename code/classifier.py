from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score

import pickle
import numpy as np

class base:
    def __init__(self, est):
        self.est = est
    def predict(self, X):
        return self.est.predict_proba(X)[:,1][:,np.newaxis]
    def fit(self, X, y):
        self.est.fit(X, y)


X_train = pickle.load(open("X_train.p", "rb"))
y_train = pickle.load(open("y_train.p", "rb"))
X_test = pickle.load(open("X_test.p", "rb"))
y_test = pickle.load(open("y_test.p", "rb"))

#clf = SVC()
clf = LinearSVC()
# clf = AdaBoostClassifier(n_estimators=50)
# clf = AdaBoostClassifier(n_estimators=100)

# clf = GradientBoostingClassifier(init=base(LinearSVC()))

# clf = RandomForestClassifier()

clf.fit(X_train, y_train)

pickle.dump(clf, open("test12321.p", "wb"), protocol=4)

print(round(clf.score(X_test, y_test), 4))