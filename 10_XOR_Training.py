# A neural network from scratch that learns XOR

import math


# =========================
# Activation function
# =========================

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def sigmoid_derivative(output):
    return output * (1 - output)


# =========================
# XOR dataset
# =========================

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y = [0, 1, 1, 0]


# =========================
# Starting parameters
# =========================

w1 = 0.5
w2 = 0.5
b1 = 0

w3 = -0.5
w4 = 1
b2 = 0

output_w1 = 0.5
output_w2 = 0.5
output_bias = 0

learning_rate = 0.5


# =========================
# Forward propagation
# =========================

def forward(x1, x2):

    # Hidden neuron 1

    z1 = x1 * w1 + x2 * w2 + b1
    hidden1 = sigmoid(z1)


    # Hidden neuron 2

    z2 = x1 * w3 + x2 * w4 + b2
    hidden2 = sigmoid(z2)


    # Output neuron

    output_z = (
        hidden1 * output_w1
        + hidden2 * output_w2
        + output_bias
    )

    output = sigmoid(output_z)


    return hidden1, hidden2, output


# =========================
# Training
# =========================

for epoch in range(10000):

    total_error = 0


    for inputs, target in zip(X, y):

        x1 = inputs[0]
        x2 = inputs[1]


        # -------------------------
        # Forward pass
        # -------------------------

        hidden1, hidden2, output = forward(x1, x2)


        # -------------------------
        # Error
        # -------------------------

        error = target - output

        total_error += error ** 2


        # -------------------------
        # Backpropagation
        # -------------------------

        output_gradient = (
            (output - target)
            * sigmoid_derivative(output)
        )


        output_w1_gradient = (
            output_gradient * hidden1
        )

        output_w2_gradient = (
            output_gradient * hidden2
        )

        output_bias_gradient = output_gradient


        # Hidden gradients

        hidden1_gradient = (
            output_gradient
            * output_w1
            * sigmoid_derivative(hidden1)
        )

        hidden2_gradient = (
            output_gradient
            * output_w2
            * sigmoid_derivative(hidden2)
        )


        # -------------------------
        # Update output layer
        # -------------------------

        output_w1 -= learning_rate * output_w1_gradient
        output_w2 -= learning_rate * output_w2_gradient
        output_bias -= learning_rate * output_bias_gradient


        # -------------------------
        # Update hidden layer
        # -------------------------

        w1 -= learning_rate * hidden1_gradient * x1
        w2 -= learning_rate * hidden1_gradient * x2
        b1 -= learning_rate * hidden1_gradient

        w3 -= learning_rate * hidden2_gradient * x1
        w4 -= learning_rate * hidden2_gradient * x2
        b2 -= learning_rate * hidden2_gradient


    # Show progress

    if (epoch + 1) % 1000 == 0:

        print(
            "Epoch:", epoch + 1,
            "Total Error:", total_error
        )


# =========================
# Testing
# =========================

print("\nFinal predictions:")

for inputs in X:

    x1 = inputs[0]
    x2 = inputs[1]

    hidden1, hidden2, output = forward(x1, x2)

    print(
        inputs,
        "→",
        output
    )