# 06_backpropagation.py
# Understanding backpropagation


import math
def sigmoid_derivative(output):
    return output * (1 - output)

# Sigmoid function

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# Sigmoid derivative

def sigmoid_derivative(output):
    return output * (1 - output)


# Input and parameters

x = 1
weight = 0.5
bias = 0

target = 1
# Forward propagation

z = x * weight + bias

output = sigmoid(z)

print("Output:", output)

sigmoid_gradient = sigmoid_derivative(output)

print("Sigmoid derivative:", sigmoid_gradient)
# Calculate loss gradient

loss_gradient = output - target

print("Loss gradient:", loss_gradient)
# Calculate loss gradient

loss_gradient = output - target

print("Loss gradient:", loss_gradient)
# Calculate gradient for the weight

weight_gradient = loss_gradient * sigmoid_gradient * x

print("Weight gradient:", weight_gradient)
# Update the weight

learning_rate = 0.1

weight = weight - learning_rate * weight_gradient

print("Updated weight:", weight)
# Make another prediction

z = x * weight + bias

new_output = sigmoid(z)

print("New output:", new_output)