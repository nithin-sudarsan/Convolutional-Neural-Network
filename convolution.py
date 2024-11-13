import numpy as np
from scipy import signal
from layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth): 
        # Lets say input is a RGB image of dimension 24 pixels (height) * 24 pixels (width)
        # So input_shape is 3 * 24 * 24
        # For simplicity, let's consider kernel_size is 2 * 2
        # Depth is the number of layers of kernel
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        # Now output shape can be calculated using the formula Y = I - K + 1 on height and width of the input shape
        # Here Y is the shape of output matrix, I is the shape of input matrix and K is the shape of kernel
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size +1)
        # Lets say there are 2 layers of kernel and we know the depth of input image (input_depth) is 3, then depth is 2
        # Then the overall kernel shape is depth (2) * input_depth (3) * kernel_size(2*2) * kernel_size(2*2)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    # forward propagation parameters
    ## input: input matrix / image
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth): # In this case 2 (number of kernel layers)
            for j in range(self.input_depth): # In this case 3 (number of color channels)
                self.output[i] = signal.correlate2d(self.input[j], self.kernels[i,j], "valid")
        return self.output
    
    # backward progagation parameters
    ## output_gradient: differential of error W.R.T output matrix
    ## learning_rate: rate of updation of the input_gradient (how big of a step it is taking to reach global minima)
    def backward(self, output_gradient, learning_rate):
        # Lets assume input kernel is a null matrix
        kernel_gradient = np.zeros(self.kernels_shape)
        # Lets assume that the initial value of input gradient, i.e the amount of correction made to the value of input to be a null matrix
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth): # In this case 2 (number of kernel layers)
            for j in range(self.input_depth): # In this case 3 (number of color channels)
                # kernel gradient is calculated as cross-correlation of input matrix and output gradient (refer notes to understand how we arrived at this formula)
                kernel_gradient[i,j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                #input gradient is calculated as convolution of output gradient and kernel (refer notes to understand how we arrived at this formula)
                input_gradient[j] = signal.convolve2d(output_gradient[i], self.kernels[i,j],"full")
        
        # kernels and biases has to be updated as computed above according to the learning rate provided
        self.kernels -= learning_rate * kernel_gradient
        self.biases -= learning_rate * output_gradient

        # input_gradient returned in the process of back propagation is used to update the value of input_gradient (error correction)
        return input_gradient 