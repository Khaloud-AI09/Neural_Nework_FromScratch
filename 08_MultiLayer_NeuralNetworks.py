# A small neural network from scratch

import math


# Sigmoid activation function

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# -------------------------
# Inputs
# -------------------------

x1 = 1
x2 = 0


# -------------------------
# Hidden Layer
# -------------------------

# Hidden neuron 1

w1 = 0.5
w2 = 0.5
b1 = 0

z1 = x1 * w1 + x2 * w2 + b1

hidden1 = sigmoid(z1)


# Hidden neuron 2

w3 = -0.5
w4 = 1
b2 = 0

z2 = x1 * w3 + x2 * w4 + b2

hidden2 = sigmoid(z2)


# -------------------------
# Output Layer
# -------------------------

output_w1 = 0.5
output_w2 = 0.5
output_bias = 0

output_z = (
    hidden1 * output_w1
    + hidden2 * output_w2
    + output_bias
)

output = sigmoid(output_z)


# -------------------------
# Results
# -------------------------

print("Input 1:", x1)
print("Input 2:", x2)

print("Hidden neuron 1:", hidden1)
print("Hidden neuron 2:", hidden2)

print("Final output:", output)