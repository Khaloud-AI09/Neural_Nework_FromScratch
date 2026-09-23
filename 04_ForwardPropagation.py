import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))
# Inputs
x1 = 1
x2 = 0
# Hidden neuron 1

w1 = 0.5
w2 = 0.5
b = 0

z1 = x1 * w1 + x2 * w2 + b

print("Hidden neuron 1:", z1)
hidden1 = sigmoid(z1)

print("Hidden neuron 1 after sigmoid:", hidden1)

# Hidden neuron 2

w1_2 = -0.5
w2_2 = 1
b2 = 0

z2 = x1 * w1_2 + x2 * w2_2 + b2

hidden2 = sigmoid(z2)

print("Hidden neuron 2:", hidden2)
# Output neuron

output_w1 = 0.5
output_w2 = 0.5
output_bias = 0

output_z = hidden1 * output_w1 + hidden2 * output_w2 + output_bias

output = sigmoid(output_z)

print("Output:", output)