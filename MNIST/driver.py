from network import Network
from MNIST_crit_funcs import testInput, trainInput
import numpy as np

size = (784,15,10)
net = Network(size)

# file1 = "/home/chunt/VScode/Python/NeuralNetsAndDeepLearning/MNIST/MNISTtraintest/test.csv"
# test = testInput(file1)

file2 = "/home/chunt/VScode/Python/NeuralNetsAndDeepLearning/MNIST/MNISTtraintest/train.csv"
train = trainInput(file2)

print(f"Networks output: \n{net.feedForward(train[6][0])}")
print(f"Correct output: \n{train[6][1]}")

