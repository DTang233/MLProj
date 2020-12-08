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

x_train=createArray(50000,28*28,training_data,0) #50000 x 784
y_train=createArray(50000,10,training_data,1)#50000 x 10
x_test=createArray(10000,28*28,test_data,0)#10000 x 784
y_test=np.zeros((10000,10))
for i in range(0,10000):
    y_test[i][test_data[i][1]]=1

#Build the model
model=Sequential()
model.add(Dense(128, activation='sigmoid',input_dim=784))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
print("model created successful")

#Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("compile model successful")

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=100
)

print("evaluate test")
model.evaluate(
    x_test,
    y_test  
)