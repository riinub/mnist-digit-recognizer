import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# from model import *

(X_train, y_train),(X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784) / 255.0
X_test = X_test.reshape(X_test.shape[0], 784) / 255.0

X_train = X_train.T
X_test = X_test.T

_, m_train = X_train.shape

def init_params():
    W1 = np.random.randn(64, 784) * np.sqrt(2 / 784)
    b1 = np.random.randn(64, 1)

    W2 = np.random.randn(10, 64) - 0.5 *  np.sqrt(2 / 64)
    b2 = np.random.randn(10, 1)

    return W1, b1, W2, b2

# print(np.random.randn(10,64))

def ReLU(Z):
    return np.maximum(Z, 0)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 # Z1 = W1*X + b1 this is the neuron function of the first hidden layer
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2 # Z2 = W2*A1 + b2 is the neuron function of the second/output layer
    A2 = softmax(Z2)
    
    return Z1, A1, Z2, A2


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    
    return one_hot_Y

    
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # print this every 10th iteration 
        if (i % 300 == 0):
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
            
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, y_train, iterations= 3000, alpha= 0.1)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(7, W1, b1, W2, b2)

dev_predictions= make_predictions(X_test, W1, b1, W2, b2)
get_accuracy(dev_predictions, y_test)

predictions = make_predictions(X_test, W1, b1, W2, b2)
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


for digit in range(10):
    mask = y_test == digit
    
    accuracy = np.mean(predictions[mask] == y_test[mask])

    print(f"{digit}: {accuracy * 100:.2f}%")

np.savez(
    "models/mnist_parameters_64hidden.npz",
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)

