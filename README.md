# Neural Network From Scratch
A simple, educational deep learning project implementing a multilayer neural network from scratch in Python without using heavyweight libraries like TensorFlow or PyTorch.
## About the Project
I built this project while learning Deep Learning to understand how neural networks actually work under the hood instead of only memorizing the theory. By implementing every component—from single artificial neurons and activation functions to backpropagation and gradient descent—I gained a hands-on understanding of how models learn from data.
## Learning Journey & Concepts Covered
The project follows a step-by-step learning progression:
Artificial Neuron: Inputs, weights, and bias calculation.
Activation Functions: Step and sigmoid functions.
Perceptron: Building a basic perceptron and training it to learn the AND gate.
Limitations: Discovering why a single perceptron cannot learn the XOR function because it is not linearly separable.
Multilayer Neural Network: Building a network with 2 inputs, 2 hidden neurons, and 1 output neuron.
Training Mechanics: Manual implementation of forward propagation, loss calculation, backpropagation, and gradient descent.
## Project Structure
```text
neural-network-from-scratch/
├── 01_neuron.py
├── 02_activation_functions.py
├── 03_perceptron.py
├── 04_forward_propagation.py
├── 05_loss_function.py
├── 06_backpropagation.py
├── 07_gradient_descent.py
├── 08_neural_network.py
├── 09_training.py
├── 10_xor_training.py
├── 11_visualization.py
├── 12_decision_boundary.py
└── README.md
```
## XOR Experiment and Results
I trained the multilayer neural network on the XOR problem for 10,000 epochs.
[0, 0] → 0.0189
[0, 1] → 0.9838
[1, 0] → 0.9836
[1, 1] → 0.0169
Using a classification threshold of 0.5, the network successfully learned the correct logical mapping:
[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0
## Visualizations Learning Curve: 
A plot showing the gradual decrease in training error over the 10,000 epochs.
## Decision Boundary: A visualization illustrating how the multilayer network successfully separates the non-linear XOR classes.
## Technologies Used
Python (Core language), Math Module (Standard library for mathematical operations), Matplotlib (For plotting learning curves and decision boundaries)
## What I Learned
Writing a neural network from scratch clarifies the chain rule and how errors propagate backward through hidden layers.
Experiencing the failure of a single perceptron on the XOR gate provided concrete appreciation for why hidden layers are necessary in deep learning.
Implementing gradient descent manually reinforced how weight updates reduce loss over time.

# Author
Khaloud Altaf Mir
BSc Artificial Intelligence, IUST

This project is part of my ongoing journey of learning Artificial Intelligence and Deep Learning from the fundamentals.
