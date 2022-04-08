# [*] CHRISTOPHER HUNT
# [*] MNIST_crit_funcs.py

import numpy as np

# Functions to import and format MNIST train and test data

def testInput(file):
    """Returns a list of lists of pixel values"""

    data = []

    with open(file) as f:
        line = f.readlines()
    
    for i in range(1, len(line)):
        image = line[i].split(',')
        last = len(image)-1

        for j, char in enumerate(image):
            if j == last:
                image[j] = image[j].replace('\n', '')
            image[j]= int(char)/255

        data.append(np.array(image).reshape((784,1)))
    
    return data

def trainInput(file):
    """Returns a tuple - the first value is the label, the second a list of 
    pixel values for that digit"""

    data = []

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
        image_label = (image, label)
        data.append(image_label)

    return data

# Activation function

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def deriv_sigmoid(x):
    return (np.exp(-x)/((1+np.exp(-x))**2))

def relu(x):
    return np.maximum(0, x)
