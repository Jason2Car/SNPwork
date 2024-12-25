!pip install tensorflow
import numpy as np
import tensorflow as tf
import keras
import math
from keras import layers
from matplotlib import pyplot

train_data = open('LDMatrix_train.txt', 'r').read()
test_data = open('LDMatrix_test.txt', 'r').read()
train_response = open('train_response.txt', 'r').read()
test_response = open('test_response.txt', 'r').read()

#If you want to change this to manually setting, go ahead
input = "test	1	1 1	1	1	relu			10	32"

train_data = train_data.split("\n") #inputs
test_data = test_data.split("\n") 
train_response = train_response.split("\n") #expected outputs
test_response = test_response.split("\n") 

#Remove the begining SNP identifiers and ending are blanks
train_data = train_data[1:len(train_data)-1]
test_data = test_data[1:len(train_data)-1]
train_response = in_train[1:len(train_response)-1]
test_response = in_test[1:len(test_response)-1]

#Creating arrays to hold data
train_in = []
test_in = []
train_out = []
test_out = []

for i in train_data:
    cur = (i.split())#Create an array to hold data in the case
    cur = cur[1:] #Remove the "case #" at the start of every case
    for i in range(len(cur)):#Seperate the data in that case
        cur[i] = float(cur[i])#Convert the String into numbers
    train_in.append(np.array(cur))#Add the array to the data set

#Repeat process with the test data
for i in test_data:
    cur = (i.split())
    cur = cur[1:]
    for i in range(len(cur)):
        cur[i] = float(cur[i])
    test_in.append(np.array(cur))

for i in train_response:
    cur = i.split()
    train_out.append(float(cur[1]))
    
for i in test_response:
    cur = i.split()
    test_out.append(float(cur[1]))
# #Seperate the data into testing and training data
# train_data = np.array(train[:math.floor(len(train)*0.8)])
# train_targets= np.array(test[:math.floor(len(test)*0.8)])
# test_data = np.array(train[math.floor(len(train)*0.8):])
# test_targets = np.array(test[math.floor(len(test)*0.8):])
# #Confirms shapes are correct
# print("Training data shape:", train_data.shape)
# print("Training targets shape:", train_targets.shape)
# print("Test data shape:", test_data.shape)
# print("Test targets shape:", test_targets.shape)

# Define exponential decay schedule, copy pasted from previou lessons
initial_learning_rate = 0.1
decay_steps = 1000
decay_rate = 0.98
staircase = True

from keras.optimizers.schedules import ExponentialDecay
# The learning rate schedule will set the learning rate to be 
# initial_learning_rate * decay_rate ^ (global_step / decay_steps).
# If staircase == True, the division glocal_step/decay_steps will be the integer division
lr_schedule = ExponentialDecay(
    initial_learning_rate = initial_learning_rate,
    decay_steps = decay_steps,
    decay_rate = decay_rate,
    staircase = staircase)


input = input.split()
print(input[0])
caseNum = (input[0])
cLayers = int(input[1])
numFilters = []
for i in range(2, 2+cLayers):
    numFilters.append(int(input[i]))

sizeFilters = []
for i in range(2+cLayers,2+2*cLayers):
    sizeFilters.append(int(input[i]))

nLayers = int(input[2+2*cLayers])
density = []
for i in range(3+2*cLayers, 3+2*cLayers+nLayers):
    density.append(int(input[i]))

cActivation = input[3+2*cLayers+nLayers]
nActivation = input[3+2*cLayers+nLayers]

epochs = input[4+2*cLayers+nLayers]
batch_size = input[5+2*cLayers+nLayers]

print("Case Num: "+ str(caseNum))
print("Input: "+ str(input))
print("Clayers "+str(cLayers))
print("numfilters "+str(numFilters))
print("sizefilters "+str(sizeFilters))
print("nLayers "+str(nLayers))
print("density "+str(density))
print("cActivaion "+ (cActivation))
print("nActivaion "+ (nActivation))




lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
loss = "mse"


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model_cnn = keras.Sequential()
#print(train_data.shape)
#Plug in the number of filters, size of filters, activation functions, and the 
model_cnn.add(Conv2D(filters = numFilters[0], kernel_size = (sizeFilters[0], sizeFilters[0]), activation=cActivation, input_shape=(8299, 8299,1)))

for i in range(cLayers):
    model_cnn.add(layers.Conv2D(filters = numFilters[i], kernel_size = (sizeFilters[i], sizeFilters[i]), activation = cActivation))

#should be no need pooling layers, can test if want to later

# Flatten the output of the Conv1D layer
model_cnn.add(layers.Flatten())
for i in range(nLayers):
    model_cnn.add(layers.Dense(density[i], activation = nActivation))

#model_cnn.add(layers.Dense(1, activation = activation)) # not sure if we want the final to have a activation

#requires some clarification on size of output
model_cnn.add(layers.Dense(len(train_out)))

# Summary of your model
model_cnn.summary()

# Model compilation
model_cnn.compile(optimizer = optimizer, loss = loss)


#May need to change or manually set epochs/batch size as needed
model_cnn.fit(train_data, train_response, batch_size = batch_size, epochs = epochs)

# Model Evaluation
evaluate_test = model_cnn.evaluate(test_data, test_response, verbose = 0)

print('Test loss', evaluate_test)
