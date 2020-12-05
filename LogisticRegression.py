from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from util import *
import numpy as np

X1, y1 = read_dataset ("project3_dataset1.txt")
X2, y2 = read_dataset ("project3_dataset2.txt")
# print(X2.shape)
# print(y2.shape)

clf1 = LogisticRegression(random_state=0).fit(X1, y1)
clf2 = LogisticRegression(random_state=0).fit(X2, y2)
#predict the first 10 data?
# clf1.predict_proba(X1[:10, :])
# # clf2.predict_proba(X2[:10, :])
#Return the mean accuracy on the given test data and labels.
print(clf1.score(X1, y1))
print(clf2.score(X2, y2))

