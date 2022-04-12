# CHRISTOPHER HUNT
# network.py

import random
import numpy as np
from MNIST_crit_funcs import sigmoid, relu, softmax, fileInput


class Network:
    def __init__(self,sizes):
        """
        Initializes the random weights and biases matrices. Also holds the sizes 
        and number of layers variables. Tricky to build weights and biases
        """
        self.sizes = sizes
        self.num_layers = len(self.sizes)
        self.weights = [np.random.randn(x,y) for x,y in zip(self.sizes[1:], self.sizes[:-1])]  # (from index 1 until the end, from index 0 to one before the end)
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]] # (from index 1 until the end, always 1)
        self.mini_batch_size = 10
        self.input_size = None
        self.image = None
        self.label = None
        self.epochs = None
        self.eta = None

    def feedForwardRelu(self, a):
        """
        Return the output of the network if "a" is input using ReLU and softmax as 
        activation functions - ReLU on all layers except the final which uses a softmax
        function to return values between 0 - 1 
        """
        # Both the weight and bias matrices have the same amount of layers (in this instance it is two, in each layer the matrices have the same rows but different collumn amount)
        last = len(self.biases)-1
        count = 0
        for b, w in zip(self.biases, self.weights): 
            if(count != last):
                a = relu((np.dot(w,a) + b)) # Perform the sigmoid activation function to output the "neurons" activation after processing
                count += 1
            else:
                a = softmax(np.dot(w,a)+b)
        return a

    def feedForwardSigmoid(self, a):
        """
        Returns the output of the network if "a" is the input using the Sigmoid
        function as the activation function on all layers
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def makeMiniBatch(self):
        """Function to generate mini batches to be used in Stochastic Gradient Descent"""
        # Create List of random int from 0 to self.input_size to randomize input file
        shuffled_index = (list(range(0,self.input_size)))
        random.shuffle(shuffled_index)
        data = []
        for element in shuffled_index:
            data.append(self.image.T[element])

        mini_batches = \
            [data[k:k+self.mini_batch_size] \
            for k in range(0,self.input_size, self.mini_batch_size)]

        return mini_batches # [ [np.array: batch_size X 784],... n ] where n = input_size/mini_batch_size

    def updateMiniBatch(self, mini_batch):
        pass

    def evaluate(self):
        pass

    def stochasticGradientDescent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Finding Global Minima -> in this case minimizing the cost function. Stochastic Gradient Descent is designed such that
        we can better avoid getting caught in local minima to find the global minima.
        """
        for count in range(0, self.epochs):
            mini_batches = self.makeMiniBatch()
            for mini_batch in mini_batches:
                self.updateMiniBatch(mini_batch, self.eta) #--> This is where the feed forward, back prop, then update of weights and biases occurs.
            if test_data:
                print(f"Epoch {count}: ::evaluation goes here::\n")
            else:
                print(f"Epoch {count} complete\n")

##########################################################################################################

#[*] Here is my selection menu where I utilize try and except clauses and access my file reading function. 
    def selectionMenu(self):
        """Selection Menu. This function allows the Network class object to have a User Interface"""
        on = True
        try:
            file = input("Provide path to Test or Train csv file: ")
            test_or_train = abs(int(input("Test or Train?\n1. Test\n2. Train\n:: ")))
            if (test_or_train == 1):
                self.input_size = 10000
            elif (test_or_train ==2):
                self.input_size = 60000
            self.image, self.label = fileInput(file, self.input_size)
            while(on):
                print("What would you like to do?\n1. Feed Forward with ReLU\n2. Feed Forward with Sigmoid\n3. Developers Options\n4. Exit.\n")
                selection = int(input(":: "))
                if(selection == 1):
                    print(self.feedForwardRelu(self.image).shape)
                elif(selection == 2):
                    print(self.feedForwardSigmoid(self.image).shape)
                elif(selection == 3):
                    # Developers Options is a space to add dev tests of program functionality.
                    mini_batches = self.makeMiniBatch()
                    print(f"len(mini_batches): {len(mini_batches)}\nmini_batches[0].shape: {mini_batches[0]}\n")
                elif(selection == 4):
                    print("\nGood Bye!")
                    on = False
                else:
                    print("\nWhat ever you entered didn't make sense. Try again.\n")
    
        except ValueError:
            print("\n####VALUEERROR####\nYOU MUST HAVE ENTERED A CHARACTER STRING WHEN YOU SHOULD HAVE ENTERED AN INTEGER...")
        except FileNotFoundError:
            print("\n####FILENOTFOUNDERROR####\nTHE FILE PATH YOU ENTERED MUST HAVE BEEN WRONG...")

