# MLProj
In this project we implemented classification algorithms: Logistic Regression, K Nearest
Neighbor, Decision Tree, SVM, Random Forests and Boosting.

Inside file *util.py* we preprocessed the dataset into *numpy* array.

Inside file *run.py* we adopted 10-fold Cross Validation to evaluate the performance of all methods on the provided two datasets in terms of Accuracy, Precision, Recall, F-1 measure, and AUC.

Inside file *NeuralNetwork.py* we implemented neural network (two hidden layers, sigmoid activation function, softmax output layer, and cross entropy loss). We train a neural network with 50k training samples to classify 10 digits (0-9) and report its classification results on 10k testing images. 

