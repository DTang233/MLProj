import mnist_loader
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from tensorflow import keras
#from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import mean_squared_error
tf.compat.v1.enable_eager_execution()

def createArray(row,col,input,idx):
    output=np.zeros((row,col))
    for i in range(0,row):
        for j in range(0,col):
            output[i][j]=input[i][idx][j]
    return output

#To change: parameter settings
Learning_Rate=0.01
Hidden_layer1_neurons=512
Hidden_layer2_neurons=128
W1='random_uniform'
W2='random_uniform'
B1='zeros'
B2='zeros'
print(W1)
print(B1)
#Data Initialization
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data) 


#Initialize data 
x_train=createArray(50000,28*28,training_data,0) # shape(50000 x 784)
y_train=createArray(50000,10,training_data,1)#shape(50000 x 10)
x_test=createArray(10000,28*28,test_data,0)#shape(10000 x 784)
y_test=np.zeros((10000,10))#shape(10000 x 10)
for i in range(0,10000):
    y_test[i][test_data[i][1]]=1


#Build the model
model=Sequential()
#There will two hidden layers and one output layer.
#For two hidden layers, we can set neuron numbers, weight and bias initialization, activation function
#We set output layer to use softmax

#layer 1 settings
model.add(Dense(Hidden_layer1_neurons, activation='sigmoid',kernel_initializer=W1,bias_initializer=B1,input_dim=784))
#layer 2 settings
model.add(Dense(Hidden_layer2_neurons, kernel_initializer=W2,bias_initializer=B2,activation='sigmoid'))
#output layer settings
model.add(Dense(10, activation='softmax'))
print("model created successful")
opt=keras.optimizers.Adam(learning_rate=Learning_Rate)

#Compile the model
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy', # cross entropy loss
    metrics=['accuracy'],
    
)
print("compile model successful")

#train the model with 500000 data
model.fit(
    x_train,
    y_train,
    epochs=5, # will run 500000 data for 5 times
    batch_size=32 
)
#mse=mean_squared_error(model.predict(x_test), y_test)
#print("mean squared error is%f"%mse)

print("evaluate test")
#evaluate the model on 10000 data
model.evaluate(
    x_test,
    y_test  
)

pred_train_ = (model.predict(x_train) > 0.5).astype("int32") # prediction result of trainning data 
pred_test_ = (model.predict(x_test) > 0.5).astype("int32")# prediction result of test data 

#Convert prediction from form [1,0,0,0,0,0,0,0,0,0] to 0
true_train=np.zeros((50000,1))
true_train=true_train.astype(np.int32) 
pred_train=np.zeros((50000,1)) 
pred_train=pred_train.astype(np.int32) 
pred_test=np.zeros((10000,1))
pred_test=pred_test.astype(np.int32) 
true_test=np.zeros((10000,1))
true_test=true_test.astype(np.int32) 
for i in range(50000):
    for j in range(10):
        if y_train[i][j]==1:
            true_train[i]=j
        if pred_train_[i][j]==1:
            pred_train[i]=j

for i in range(10000):
    for j in range(10):
        if y_test[i][j]==1:
            true_test[i]=j
        if pred_test_[i][j]==1:
            pred_test[i]=j

#calculation of bias and variance

#mean squared error
main_predictions = np.mean(pred_test, axis=0)
avg_bias_mse = np.sum((main_predictions - true_test)**2) /10000
avg_var_mse = np.sum((main_predictions - pred_test)**2) / 10000
print("bias=%f"%avg_bias_mse )
print("variance=%f"%avg_var_mse)

