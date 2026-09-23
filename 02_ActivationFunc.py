# Learning about activation functions


# Step activation function
def step_function(z):

    if z >= 0:
        return 1
    else:
        return 0


# Sigmoid activation function
import math

def sigmoid(z):

    return 1 / (1 + math.exp(-z))


# Test values
values = [-2, -1, 0, 1, 2]

print("Step Function:")

for value in values:
    print(value, "→", step_function(value))


print("\nSigmoid Function:")

for value in values:
    print(value, "→", sigmoid(value))