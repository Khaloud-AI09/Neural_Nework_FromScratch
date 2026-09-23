# Learning with gradient descent

import math


# Sigmoid function

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# Sigmoid derivative

def sigmoid_derivative(output):
    return output * (1 - output)


# Starting values

x = 1
weight = 0.5
bias = 0

target = 1

learning_rate = 0.1


# Training

for epoch in range(10):

    # Forward propagation

    z = x * weight + bias

    output = sigmoid(z)


    # Calculate gradients

    loss_gradient = output - target

    sigmoid_gradient = sigmoid_derivative(output)

    weight_gradient = loss_gradient * sigmoid_gradient * x


    # Update weight

    weight = weight - learning_rate * weight_gradient


    print(
        "Epoch:", epoch + 1,
        "Output:", output,
        "Weight:", weight
    )