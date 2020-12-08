import mnist_loader
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


def createArray(row,col,input,idx):
    output=np.zeros((row,col))
    for i in range(0,row):
        for j in range(0,col):
            output[i][j]=input[i][idx][j]
    return output

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
#layer 1 settings, such as neuron numbers, weight and bias initialization, activation function
model.add(Dense(128, activation='sigmoid',kernel_initializer='random_normal',bias_initializer='zeros',input_dim=784))
#layer 2 settings
model.add(Dense(64, kernel_initializer='random_normal',bias_initializer='zeros',activation='sigmoid'))
#output layer settings
model.add(Dense(10, activation='softmax'))
print("model created successful")

#Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("compile model successful")

#train the model with 500000 data
model.fit(
    x_train,
    y_train,
    epochs=5, # will run 500000 data for 5 times
    batch_size=32 
)
print("evaluate test")

#evaluate the model on 10000 data
model.evaluate(
    x_test,
    y_test  
)