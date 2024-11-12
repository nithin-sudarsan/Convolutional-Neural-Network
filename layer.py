
# Skeleton class of different kinds of layers
class Layer():
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input):
        # PURPOSE: return output
        pass
    def backward(self, output_gradient, learning_rate): # output_gradient = dE/dY ; learning_rate = alpha
        # PURPOSE: update parameters and return input gradient (dE/dX)
        pass