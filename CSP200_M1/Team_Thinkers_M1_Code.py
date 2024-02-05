#!/usr/bin/env python
# coding: utf-8

# In[15]:


#imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers


# In[34]:


#assigning variables to the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Prints out the dimensions of the dataset
np.shape(x_train)

#Ouputs an image
plt.imshow(x_train[1])
#Confirms if the image in x_train matches with the output in y_train
print(y_train[1])


# In[35]:


#Shape of the image
input_shape = (28,28,1)
#Number of classifications
num_classes = 10
#Building the CNN
model = keras.Sequential(
    [
        #Defines the input sizes
        keras.Input(input_shape),
        #Creates the convolutional layer
        layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
        #Creats pooling or subsampling layer
        layers.MaxPooling2D(pool_size=(2,2)),
        #Flattens for the neurons
        layers.Flatten(),
        #Creates the fully connected layer
        layers.Dense(num_classes,activation="softmax"),
    ]
)
#Prints out a summary of the model
model.summary()

