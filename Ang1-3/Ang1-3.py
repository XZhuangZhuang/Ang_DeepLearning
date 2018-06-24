import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v3 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# 初始化参数
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert(w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert(w2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    parameters = {'w1': w1,
                  'b1': b1,
                  'w2': w2,
                  'b2': b2}
    return parameters

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['w' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['w' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

def linear_forward(A, w, b):
    Z = np.dot(w, A) + b
    assert(Z.shape == (w.shape[0], A.shape[1]))
    cache = (A, w, b)
    return Z, cache

def linear_activation_forward(A_prev, w, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, w, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, w, b)
        A, activation_cache = relu(Z)
    assert(A.shape == (w.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(x, parameters):
    caches = []
    A = x
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['w' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['w' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)
    assert(AL.shape == (1, x.shape[1]))
    return AL, caches

def compute_cost(AL, y):
    m = y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL), y) + np.multiply(np.log(1 - AL), 1 - y)) / m
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost


def linear_backward(dZ, cache):
    A_prev, w, b = cache
    m = A_prev.shape[1]
    dw = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(w.T, dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dw.shape == w.shape)
    assert (db.shape == b.shape)
    return dA_prev, dw, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dw, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dw, db = linear_backward(dZ, linear_cache)
    return dA_prev, dw, db

def L_model_backward(AL, y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    y = y.reshape(AL.shape)
    dAL = -(np.divide(y, AL) - np.divide(1-y, 1-AL))
    current_cache = caches[L-1]
    grads['dA' + str(L)], grads['dw' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads['dA' + str(l + 1)] = dA_prev_temp
        grads['dw' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['w' + str(l + 1)] = parameters['w' + str(l + 1)] - learning_rate * grads['dw' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    return parameters

