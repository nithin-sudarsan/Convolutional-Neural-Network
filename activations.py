import numpy as np
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x : np.tanh(x)
        tanh_prime = lambda x : 1 - (np.tanh(x) ** 2)
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        # Refer notes to see how we calculated derivative of sigmoid function
        def sigmoid_prime(x):
            return sigmoid(x) * (1 - sigmoid(x))
        
        super().__init__(sigmoid, sigmoid_prime)