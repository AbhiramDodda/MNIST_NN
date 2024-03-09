import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('mnist_pixel/train.csv')
test_data = pd.read_csv('mnist_pixel/test.csv')

data.head()

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:]
X_dev = X_dev/255.

train_data = data[1000:m].T
Y_train = train_data[0]
X_train = train_data[1:n]
X_train = X_train/255.
_, m_train = X_train.shape

def params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    return np.exp(z)/sum(np.exp(z))

def forward_propagation(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max()+1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def deriv_relu(z):
    return z > 0

def backward_propagation(z1, a1, z2, a2, w2, x ,y):
    m = y.size
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1/m * dz2.dot(a1.T)
    db2 = 1/m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * deriv_relu(z1)
    dw1 = 1/m * dz1.dot(x.T)
    db1 = 1/m * np.sum(dz1)
    return dw1, db1, dw2, db2

def update_params(w1, b1, w2 ,b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha*dw1
    b1 = b1 - alpha*db1
    w2 = w2 - alpha*dw2
    b2 = b2 - alpha*db2
    return w1, b1, w2, b2

def gradient_descent(x, y, iter, alpha):
    w1, b1, w2, b2 = params()
    for i in range(1, iter+1):
        z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward_propagation(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2,  db2, alpha)
        if i%25 == 0:
            print(get_accuracy(get_predictions(a2), y))
    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    return np.sum(predictions == y)/y.size

w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)