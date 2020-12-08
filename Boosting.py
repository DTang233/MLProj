from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from util import *

#Read data from two datasets
X1, y1 = read_dataset ("project3_dataset1.txt")
X2, y2 = read_dataset ("project3_dataset2.txt")

#implement the same decision tree as DecisionTree.py
clf = tree.DecisionTreeClassifier()
clf1 = clf.fit(X1, y1)
clf2 = clf.fit(X2, y2)

#implement boosting on two decision trees, and fit
clf1_boosting = AdaBoostClassifier(
    clf1,
    n_estimators=200
)
clf1_boosting.fit(X1, y1)

clf2_boosting = AdaBoostClassifier(
    clf2,
    n_estimators=200
)
clf2_boosting.fit(X2, y2)

#print(clf1_boosting.predict(X1[:10, :]))