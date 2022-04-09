import random
from network import Network
from MNIST_crit_funcs import testInput, trainInput
import numpy as np

size = (784,15,15,10)
net = Network(size)

file1 = "/home/chunt/VScode/Python/NeuralNetsAndDeepLearning/MNIST/MNISTtraintest/mnist_test.csv"
test_image, test_label = trainInput(file1,10000)

file2 = "/home/chunt/VScode/Python/NeuralNetsAndDeepLearning/MNIST/MNISTtraintest/mnist_train.csv"
train_image, train_label = trainInput(file2,60000)


print(f"test_label[0]: {test_image.T.shape}")
print(f"train_label[0]: {train_image.T.shape}")