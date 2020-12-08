from sklearn import tree
from util import *

X1, y1 = read_dataset ("project3_dataset1.txt")
X2, y2 = read_dataset ("project3_dataset2.txt")

clf = tree.DecisionTreeClassifier()
clf1 = clf.fit(X1, y1)
# clf2 = clf.fit(X2, y2)

print(clf1.predict_proba(X1[:10, :]))