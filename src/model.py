import numpy as np

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def get_predictions(A3):
    return np.argmax(A3, axis=1)

def forward_prop(X, W1, b1, W2, b2, W3, b3):
    if X.shape == (784, 1):
        X = X.T

    Z1 = X @ W1 + b1 # Z1 = W1*X + b1 this is the neuron function of the first hidden layer
    A1 = ReLU(Z1)

    Z2 = A1 @ W2 + b2 # Z2 = W2*A1 + b2 is the neuron function of the second/output layer
    A2 = ReLU(Z2)

    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    *_, A3 = forward_prop(X, W1, b1, W2, b2, W3, b3)
    return get_predictions(A3)

