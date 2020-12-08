from sklearn.ensemble import RandomForestClassifier
from util import *

#Read data from dataset
X1, y1 = read_dataset ("project3_dataset1.txt")
X2, y2 = read_dataset ("project3_dataset2.txt")

#random forest tree of dataset 1
clf1 = RandomForestClassifier(max_depth=2, random_state=0)
clf1.fit(X1, y1)

#random forest tree of dataset 2
clf2 = RandomForestClassifier(max_depth=2, random_state=0)
clf2.fit(X2, y2)

#test
#print(clf1.predict(X1[:10, :]))