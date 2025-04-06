def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, x_train, y_train, loss, loss_prime, learning_rate = 0.1, epochs = 1000, verbose = True):
    # Train the model
    for e in range(epochs):
        error = 0
        for (x,y) in zip(x_train, y_train):
            output = predict(network, x)
            
            error += loss(y, output)

            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose: print(f"{e + 1} / {epochs}, error = {error}")