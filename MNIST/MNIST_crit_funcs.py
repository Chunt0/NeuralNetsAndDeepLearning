# [*] CHRISTOPHER HUNT
# [*] MNIST_crit_funcs.py

import numpy as np

# Functions to import and format MNIST train and test data

def fileInput(file, size):
    """Returns a tuple - the first value is the image with a shape of (size, 784), the 
    second a one hot encoded label of shape (size, 10)"""

    image_data = []
    label_data = []

    with open(file) as f:
        line = f.readlines()
    
    for i in range(1, len(line)):
        image = line[i].split(',')
        last = len(image)-1
        
        for j, char in enumerate(image):
            if j == last:
                image[j] = image[j].replace('\n', '')
            image[j]= int(char)/255        

        one_hot = np.zeros((10,1))  
        label = int(image.pop(0)*255)
        one_hot[label-1] = 1
        label = one_hot


        image = np.array(image).reshape((784,1))
        image_data.append(image)
        label_data.append(label)
    image_data = np.array(image_data).reshape((size,784))
    label_data = np.array(label_data).reshape((size,10))

    return image_data, label_data 

# Activation functions

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def deriv_sigmoid(x):
    return (np.exp(-x)/((1+np.exp(-x))**2))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x)
    return e/(sum(e))