from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from util import *
import numpy as np

X, y = read_dataset1 ()
X = np.array(X)
y = np.array(y)
clf = LogisticRegression(random_state=0).fit(X, y)
#predict the first 10 data?
clf.predict_proba(X[:10, :])
#Return the mean accuracy on the given test data and labels.
print(clf.score(X, y))
