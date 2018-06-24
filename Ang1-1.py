import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
print(train_x_orig.shape)
print(train_y.shape)
print(classes)
index = 25
fig = plt.figure()
plt.imshow(train_x_orig[index])
plt.show()
print("y = " + str(train_y[:, index][0]), ",it is a " + classes[np.squeeze(train_y[:, index])].decode("utf-8") + "' picture.")


m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]
num_px = train_x_orig[0].shape[0]


train_x_flatten = train_x_orig.reshape(m_train, -1).T
test_x_flatten = test_x_orig.reshape(m_test, -1).T

train_x = train_x_flatten / 255
test_x = test_x_flatten / 255


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

dim = 2
w, b = initialize_with_zeros(dim)


def propagate(w, b, x, y):
    m = x.shape[1]
    A = sigmoid(np.dot(w.T, x) + b)
    cost = -1 / m * np.sum(np.multiply(y, np.log(A)) + np.multiply(1-y, np.log(1-A)))
    dw = 1 / m * np.dot(x, (A-y).T)
    db = 1 / m * np.sum(A-y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {'dw': dw, 'db': db}
    return grads, cost

def optimize(w, b, x, y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, x, y)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print('cost after iteration %i: %f' %(i, cost))
    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}
    return params, grads, costs


def predict(w, b, x):
    m = x.shape[1]
    y_prediction = np.zeros((1, m))
    w = w.reshape(x.shape[0], 1)
    A = sigmoid(np.dot(w.T, x) + b)
    for i in range(A.shape[1]):
        if (A[0, i] > 0.5):
            y_prediction[0, i] = 1
        else:
            y_prediction[0, i] = 0
    assert(y_prediction.shape == (1, m))
    return y_prediction

def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = np.zeros((x_train.shape[0], 1)), 0
    parameters, grads, costs = optimize(w, b, x_train, y_train, num_iterations, learning_rate, print_cost)
    w = parameters['w']
    b = parameters['b']
    y_prediction_test = predict(w, b, x_test)
    y_prediction_train = predict(w, b, x_train)
    print('train accuracy:{}%'.format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print('test accuracy:{}%'.format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    d = {'costs': costs,
         'y_prediction_test': y_prediction_test,
         'y_prediction_train': y_prediction_train,
         'w': w,
         'b': b,
         'learning_rate': learning_rate,
         'num_iterations': num_iterations}
    return d

d = model(train_x, train_y, test_x, test_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
print(d)


fig = plt.figure()
index = 1
plt.imshow(test_x[:,index].reshape((num_px, num_px, 3)))
plt.show()
print("y = " + str(test_y[0, index]) + ", you predicted that it is a \"" + classes[int(d["y_prediction_test"][0, index])].decode("utf-8") + "\" picture.")