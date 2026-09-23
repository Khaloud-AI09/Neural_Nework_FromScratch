# Training a small neural network

import math


# -------------------------
# Activation function
# -------------------------

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def sigmoid_derivative(output):
    return output * (1 - output)


# -------------------------
# Inputs
# -------------------------

x1 = 1
x2 = 0

target = 1


# -------------------------
# Starting weights
# -------------------------

w1 = 0.5
w2 = 0.5

w3 = -0.5
w4 = 1

output_w1 = 0.5
output_w2 = 0.5

learning_rate = 0.1


# -------------------------
# Training
# -------------------------

for epoch in range(100):

    # =====================
    # Forward propagation
    # =====================

    # Hidden neuron 1

    z1 = x1 * w1 + x2 * w2

    hidden1 = sigmoid(z1)


    # Hidden neuron 2

    z2 = x1 * w3 + x2 * w4

    hidden2 = sigmoid(z2)


    # Output neuron

    output_z = (
        hidden1 * output_w1
        + hidden2 * output_w2
    )

    output = sigmoid(output_z)


    # =====================
    # Backpropagation
    # =====================

    # Output layer gradient

    output_error = output - target

    output_gradient = (
        output_error
        * sigmoid_derivative(output)
    )


    # Gradients for output weights

    output_w1_gradient = output_gradient * hidden1

    output_w2_gradient = output_gradient * hidden2


    # =====================
    # Update output weights
    # =====================

    output_w1 = (
        output_w1
        - learning_rate * output_w1_gradient
    )

    output_w2 = (
        output_w2
        - learning_rate * output_w2_gradient
    )


    # =====================
    # Hidden layer gradients
    # =====================

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


    # =====================
    # Update hidden weights
    # =====================

    w1 = w1 - learning_rate * hidden1_gradient * x1
    w2 = w2 - learning_rate * hidden1_gradient * x2

    w3 = w3 - learning_rate * hidden2_gradient * x1
    w4 = w4 - learning_rate * hidden2_gradient * x2


    # =====================
    # Print progress
    # =====================

    if (epoch + 1) % 10 == 0:

        error = target - output

        print(
            "Epoch:", epoch + 1,
            "Output:", output,
            "Error:", error
        )