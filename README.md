# Convolutional Neural Network from scratch
This is a basic Convolutional Neural Netowrk built from scratch only using `numpy` and `scipy` libraries.
The `MNIST` dataset is solved in this example.

## Initializations
* ### learning_rate:
  * determines the step size at each iteration while moving toward a minimum of loss. A high value for this can cause overshooting, causing the model to be inaccurate and a low value could result in smaller steps towards local-minima, meaning slow convergence and requiring more iterations.
  * Default value: `0.01`
* ### epochs:
  *  A complete pass of a training dataset through a learning algorithm. During an epoch, the model learns from each example in the dataset and refines its weights and biases to improve accuracy.
  * Default value: `1000`


## Usage

```py
from dense import Dense
from convolution import Convolutional
from reshape import Reshape
from activations import Sigmoid, Tanh
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict

# After splitting the input variable (x) and the output variable (y) into training and testing sets and preprocessing the data
# then proceed to define the neural network

# Neural network
network = [
    Convolutional((1,28,28),3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

epochs = 20
learning_rate = 0.1

train(
    network, 
    x_train, 
    y_train, 
    binary_cross_entropy, 
    binary_cross_entropy_prime, 
    learning_rate, 
    epochs, 
    True
)
```
## Parameters
Parameters of the `train` method are as follows
* network:
  *To be updated...

* x_train and y_train :
  * x_train is the preprocessed input variable and y_train is the preprocessed target variable for training the neural network.

* loss :
  * To be updated...

* loss_prime :
  * Derivative of the loss function.

* Verbose :
  * Verbose is a flag variable thatis set to `True` by default, to display the loss after every epoch is completed.

## Working of the model
To be updated...

## Intuition
To be updated...