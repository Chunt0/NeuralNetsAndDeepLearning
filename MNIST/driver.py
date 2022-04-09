import random
from network import Network
from MNIST_crit_funcs import fileInput
import numpy as np

size = (784,15,15,10)
net = Network(size)

test_file = "/home/chunt/VScode/Data/MNIST/mnist_test.csv"
test_image, test_label = fileInput(test_file,10000)

train_file = "/home/chunt/VScode/Data/MNIST/mnist_train.csv"
train_image, train_label = fileInput(train_file,60000)


print(f"test_label[0]: {test_image.T.shape}")
print(f"train_label[0]: {train_image.T.shape}")