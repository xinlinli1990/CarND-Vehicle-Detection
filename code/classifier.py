from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

import pickle

X_train = pickle.load(open("X_train.p", "rb"))
y_train = pickle.load(open("y_train.p", "rb"))
X_test = pickle.load(open("X_test.p", "rb"))
y_test = pickle.load(open("y_test.p", "rb"))

#clf = SVC()
clf = LinearSVC()
# clf = AdaBoostClassifier(n_estimators=50)
# clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)

pickle.dump(clf, open("linear-SVC-default.p", "wb"), protocol=4)

print(round(clf.score(X_test, y_test), 4))