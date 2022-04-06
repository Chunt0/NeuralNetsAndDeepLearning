import numpy as np
from MNIST_crit_funcs import sigmoid


class Network:
    def __init__(self, sizes):
        """Initializes the random weights and biases matrices. Also holds the sizes 
        and number of layers variables. Tricky to build weights and biases"""

        self.num_layers = len(sizes)
        self.sizes = sizes
        # The weights matrix is a 3d matrix: 
        # (1d: identifies the interlayer space(2d: the forward layer(3d: the backward layer)))
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]  # (from index 1 until the end, from index 0 to one before the end)
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] # (from index 1 until the end, always 1)

    def feedForward(self, a):
        """Return the output of the network if "a" is input"""
        
        # Both the weight and bias matrices have the same amount of layers (in this instance it is two, in each layer the matrices have the same rows but different collumn amount)
        for b, w in zip(self.biases, self.weights): 
            a = sigmoid(np.dot(w,a) + b) # Perform the sigmoid activation function to output the "neurons" activation after processing
        return a


