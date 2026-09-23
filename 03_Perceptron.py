# A simple perceptron that learns the AND gate


# AND gate dataset

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y = [0, 0, 0, 1]


# Prediction function

def predict(inputs, weights, bias):

    z = inputs[0] * weights[0] + inputs[1] * weights[1] + bias

    if z >= 0:
        return 1
    else:
        return 0


# Starting values

weights = [0, 0]
bias = 0

learning_rate = 1


# Training

for epoch in range(10):

    print("\nEpoch:", epoch + 1)

    for inputs, target in zip(X, y):

        prediction = predict(inputs, weights, bias)

        error = target - prediction

        weights[0] = weights[0] + learning_rate * error * inputs[0]
        weights[1] = weights[1] + learning_rate * error * inputs[1]

        bias = bias + learning_rate * error

        print(
            "Input:", inputs,
            "Target:", target,
            "Prediction:", prediction,
            "Weights:", weights,
            "Bias:", bias
        )


# Test the trained perceptron

print("\nFinal predictions:")

for inputs in X:

    prediction = predict(inputs, weights, bias)

    print(inputs, "→", prediction)