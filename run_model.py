"""
Neural Network using NumPy only
first load and preprocess datasets.
then implement and train a neural network (multi-layer perceptron) for handwriting recognition (MNIST dataset), using numpy only.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Download the MNIST dataset

X, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)
print(X.shape, y.shape)
# X- 70000 rows (samples), each with 784 columns (features)
# y- 1d vector of 70000 labels

"""
# Data normalization
# Min-Max normalization of the dataset, so all feature values will be in range [0,1]
# rows- 70000. feature values in each row- 784.
# for each feature j, calculate this formula on him: (j - min_feature) / (max_feature - min_feature)
"""


def min_max_norm(X):
    X = np.array(X)
    # reshaping the vector from a row to a column vector
    max_features = np.amax(X, axis=1)[:,
                   np.newaxis]  # returns a vector where each value is the max value of its the corresponding column in X
    min_features = np.amin(X, axis=1)[:,
                   np.newaxis]  # returns a vector where each value is the min value of its the corresponding column in X

    # compute min-max according to formula
    normalizedX = (X - min_features) / (max_features - min_features)
    return normalizedX


X = np.array(min_max_norm(X))

# Spliting the data into Train set and Test set
# train set will be 80% (0.8 * number or rows) and test set will be 20%

split_in = int(np.floor(0.8 * X.shape[0]))
X_train = X[:split_in]
y_train = y[:split_in]

X_test = X[split_in:]
y_test = y[split_in:]

"""Activation function"""


# the sigmoid activation function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


"""Softmax function"""


# Returns the softmax output of a vector
def softmax(z):
    exp_z = np.exp(z)
    sum = exp_z.sum()
    return exp_z / sum


"""Loss function"""


## Negative Log Likelihood loss function for the multiclass

def nll_loss(y_pred, y):
    loss = -np.sum(y * np.log(y_pred))
    return loss / float(y_pred.shape[0])


"""Hyper-Parameters"""

# Hyper-parameters:
EPOCH = 5
LEARNING_RATE = 0.25
HIDDEN_DIM = 256

"""Parameters initialization """

# initializing the parameters W1,W2,b1,b2.
# init W1 as 2d matrix, with dimensions (HIDDEN_DIM,784),
# filled with random values distributed around 0, according to Normal Xavier (sqrt(2/dim_sum))
W1 = np.random.normal(0, np.sqrt(2 / (HIDDEN_DIM + 784)), (HIDDEN_DIM, 784))

# init b as 1d vector filled with zeros, where len=HIDDEN_DIM
# and then reshape b1 from row vector (HIDDEN_DIM, ) to column vector (HIDDEN_DIM,1)
b1 = np.zeros(HIDDEN_DIM)[:, np.newaxis]

# same for W2, b2
# W2 dim: (10, HIDDEN_DIM), b2 dim: (10,1)
W2 = np.random.normal(0, np.sqrt(2 / (10 + HIDDEN_DIM)), (10, HIDDEN_DIM))
b2 = np.zeros(10)[:, np.newaxis]

""" Training """


def train(X, y, num_of_epochs):  # TODO: why W1,W2,b1,b2 are not recognized from outside
    W1 = np.random.normal(0, np.sqrt(2 / (HIDDEN_DIM + 784)), (HIDDEN_DIM, 784))
    b1 = np.zeros(HIDDEN_DIM)[:, np.newaxis]
    W2 = np.random.normal(0, np.sqrt(2 / (10 + HIDDEN_DIM)), (10, HIDDEN_DIM))
    b2 = np.zeros(10)[:, np.newaxis]

    train_size = len(X)

    for epoch in range(num_of_epochs):
        avg_epoch_loss = 0
        for i in range(train_size):
            # Forward propagation:
            # for single x: output = softmax(W2(sigmoid(W1x+b1))+b2)
            # loss nnl(output, y_true ) = nnl(softmax(W2(sigmoid(W1x+b1))+b2), y_true)

            x = X[i]  # single row vector (784,)
            x = x[:, np.newaxis]  # reshape x from a row vector to column vector (784,1)

            # W1 (HIDDEN_DIM,784) , x (784,1) --> W1x (HIDDEN_DIM,1)
            # b1 (HIDDEN_DIM,1) --> W1x +b1 = z1 (HIDDEN_DIM,1)
            z1 = W1.dot(x) + b1
            h1 = sigmoid(z1)

            # W2 (10,HIDDEN_DIM) , h1 (HIDDEN_DIM,1) --> W2h1 (10,1)
            # b2 (10,1) --> W2h1 +b2 = Z2 (10,1)
            Z2 = W2.dot(h1) + b2
            # y_hat remains vector (10,1), but filled now with probabilities
            y_hat = softmax(Z2)

            # one-hot encoding:
            # create a one_hot vetor to be used as y_true. one_hot is a column of zeros (10,1)
            one_hot = np.zeros(10)[:, np.newaxis]
            # assign 1 according to current class, so all rows are 0 and we have 1 in the y[i] place
            one_hot[int(y[i])] = 1
            y_true = one_hot

            # Compute the loss:
            loss = nll_loss(y_hat, y_true)
            avg_epoch_loss = avg_epoch_loss + loss

            # Back propagation - compute the gradients of each parameter:
            # y_hat (10,1) , y_true (10,1) --> dZ2 (10,1)
            dZ2 = (y_hat - y_true)
            # dZ2 (10,1) , h1.T (1,HIDDEN_DIM) --> dW2 (10,HIDDEN_DIM)
            dW2 = dZ2.dot(h1.T)  # T to reshape as a column vector (1,HIDDEN_DIM)
            db2 = dZ2

            # W2.T (HIDDEN_DIM,10) , dZ2 (10,1) --> dh1 (HIDDEN_DIM, 1)
            dh1 = W2.T.dot(dZ2)
            # here for dz1 we perform elementwise multiplication
            # dh1 (HIDDEN_DIM, 1) , z1 (HIDDEN_DIM,1) --> dz1 (HIDDEN_DIM, 1)
            dz1 = dh1 * sigmoid_derivative(z1)
            # dz1 (HIDDEN_DIM, 1) , x.T (1,784) --> dW1 (HIDDEN_DIM, 784)
            dW1 = dz1.dot(x.T)
            db1 = dz1

            # Update weights:
            W2 = W2 - LEARNING_RATE * dW2
            b2 = b2 - LEARNING_RATE * db2
            W1 = W1 - LEARNING_RATE * dW1
            b1 = b1 - LEARNING_RATE * db1

            # for testing:
            # if i%10000 == 0:
            #   print(i, avg_epoch_loss/i)

        avg_epoch_loss = (avg_epoch_loss / train_size)

        print("Epoch:", epoch, " Loss:", avg_epoch_loss)
    return W1, W2, b1, b2


""" Test the model """


# testing the model and return the accuracy on the test set
def test(X, y, W1, W2, b1, b2):
    true_pred_counter = 0
    for i in range(len(X)):
        x = X[i]
        x = x[:, np.newaxis]  # x (784,1)
        z1 = W1.dot(x) + b1  # z1 (HIDDEN_DIM, 1)
        h1 = sigmoid(z1)
        Z2 = W2.dot(h1) + b2  # Z2 (10,1)
        y_hat = softmax(Z2)
        predicted = np.argmax(y_hat)
        if float(predicted) == float(y[i]):
            true_pred_counter += 1

    accuracy = float(true_pred_counter) / len(X)
    return accuracy


""" Main """

W1, W2, b1, b2 = train(X_train, y_train, EPOCH)
accuracy = test(X_test, y_test, W1, W2, b1, b2)

print(accuracy)

