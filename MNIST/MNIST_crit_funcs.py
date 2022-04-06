# [*] CHRISTOPHER HUNT
# [*] MNIST_crit_funcs.py

import numpy as np

# Functions to import and format MNIST train and test data

def testInput(file):
    """Returns a list of lists of pixel values"""

    temp_data = []
    data = []

    with open(file) as f:
        line = f.readlines()
        temp_data.append(line)
    
    for i in range(1, len(temp_data[0])):
        image = temp_data[0][i].split(',')
        last = len(image)-1

        for j, char in enumerate(image):
            if j == last:
                image[j] = image[j].replace('\n', '')
            image[j]= int(char)

        data.append(image)

    return data

def trainInput(file):
    """Returns a tuple - the first value is the label, the second a list of 
    pixel values for that digit"""

    temp_data = []
    data = []

    with open(file) as f:
        line = f.readlines()
        temp_data.append(line)
    
    for i in range(1, len(temp_data[0])):
        image = temp_data[0][i].split(',')
        last = len(image)-1
        
        for j, char in enumerate(image):
            if j == last:
                image[j] = image[j].replace('\n', '')
            image[j]= int(char)        
        
        label = image.pop(0)
        label_image = (label, image)
        data.append(label_image)

    return data

# Activation function

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))