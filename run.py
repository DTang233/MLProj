from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn import tree
from sklearn import svm

import matplotlib.pyplot as plt
from util import *
import numpy as np
from mlxtend.evaluate import bias_variance_decomp


X1, y1 = read_dataset ("project3_dataset1.txt")
X2, y2 = read_dataset ("project3_dataset2.txt")

def ten_fold_cross_validation(X, y, algo, params):

	
	"""
	Here we divided X into 10 sets, 9/10 of the data are 
	using as training data each time, and the remaining 1/10 
	are using as test data. 

	- X1 is a (569, 30) numpy array representing the features, 
	- y1 is a (569, ) array representing the labels for X1

	- X2 is a (462, 9) numpy array, 
	- y2 is a (462, ) array

	We are going to calculate Accuracy, Precision, Recall, F-1 measure, and AUC (area under
	the curve) for each model.
	"""

	#Decide which classifier to use
	if algo == 'Logistic Regression':
		clf = LogisticRegression(penalty='l2',random_state=0)
	if algo == 'Knn':
		clf = KNeighborsClassifier(n_neighbors=params)
	if algo == 'SVM':
		clf = make_pipeline(StandardScaler(), SVC(kernel = 'sigmoid', gamma='auto'))
	if algo == 'Decision Tree':
		#default criteria is gini
		clf = tree.DecisionTreeClassifier(splitter = 'random')
		if params and params[0] == 'AdaBoost':
			clf = AdaBoostClassifier(clf, learning_rate = 0.5, n_estimators=params[1])
	if algo == 'Random Forest':
		clf = RandomForestClassifier(n_estimators = 50, max_depth=2, random_state=0)


	cv_nums = 10
	kf = KFold(n_splits=cv_nums)
	i = 1
	training_accuracys = []
	test_accuracys = []
	AUCs = []
	#mse
	errors = []
	#bias
	biases = []
	#variances
	variances = []

	for train_index, test_index in kf.split(X):

		X_train = X[train_index]
		y_train = y[train_index]
		X_test = X[test_index]
		y_test = y[test_index]
		clf.fit(X_train, y_train)

		y_pred_train = clf.predict(X_train)
		y_pred_test = clf.predict(X_test)
		

		disp = plot_precision_recall_curve(clf, X_test, y_test)
		precision, recall, thresholds = roc_curve(y_test, y_pred_test)
		f1_score_train = f1_score(y_train, y_pred_train, average='micro')
		f1_score_test = f1_score(y_test, y_pred_test, average='micro')
		
		AUCs.append(auc(recall, precision))
		mse, bias, var = bias_variance_decomp(clf, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200, random_seed=1)	
		errors.append(mse)
		biases.append(bias)
		variances.append(var)	

		training_accuracys.append([accuracy_score(y_train, y_pred_train),f1_score_train])
		test_accuracys.append([accuracy_score(y_test, y_pred_test),f1_score_test])

		disp.ax_.set_title('Iteration'+str(i) +' 2-class Precision-Recall curve: ')
		plt.savefig(algo+ str(i) +'.png')
		i+=1
	
	for j in range(cv_nums):
		print('------------Iteration '+str(j+1)+'-------------')
		print('Training accuracy: ')
		print(training_accuracys[j][0])
		print('Training F-1 score: ')
		print(training_accuracys[j][1])
		print('Testing accuracy: ')
		print(test_accuracys[j][0])
		print('Testing F-1 score: ')
		print(test_accuracys[j][1])
		print('Testing AUC: ')
		print(AUCs[j])
		print('MSE: ' + str(errors[j]))
		print('Bias: ' + str(biases[j]))
		print('Variance: ' + str(variances[j]))
	
		# summarize results
	print('Average MSE: ' + str(sum(errors)/len(errors)))
	print('Average Bias: ' + str(sum(biases)/len(biases)))
	print('Average Variance: ' + str(sum(variances)/len(variances)))


if __name__ == '__main__':
	# ten_fold_cross_validation(X2, y2, 'Logistic Regression', None)
	# ten_fold_cross_validation(X1, y1, 'SVM', None)
	# ten_fold_cross_validation(X2, y2, 'Knn', 10)
	# params = ['AdaBoost',100]
	ten_fold_cross_validation(X1, y1, 'Decision Tree', None)
	# ten_fold_cross_validation(X2, y2, 'Random Forest', None)











