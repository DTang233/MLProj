from sklearn.neighbors import KNeighborsClassifier
from util import *

neigh = KNeighborsClassifier(n_neighbors=3)
X, y = read_dataset ("project3_dataset1.txt")
neigh.fit(X, y)
print(neigh.predict(X))
print(neigh.predict_proba(X))