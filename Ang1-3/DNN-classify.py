import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# index = 10
# fig = plt.figure()
# plt.imshow(train_x_orig[index])
# print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")
# plt.show()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255


def predict(x, y, parameters):
    m = x.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1, m))
    # Forward propagation
    probas, caches = L_model_forward(x, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print("Accuracy: " + str(np.sum((p == y)/m)))
    return p

# 两层网络
def two_layer_model(x, y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    grads = {}
    costs = []
    m = x.shape[1]
    (n_x, n_h, n_y) = layers_dims
    parameters = initialize_parameters(n_x, n_h, n_y)
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(x, w1, b1, activation='relu')
        A2, cache2 = linear_activation_forward(A1, w2, b2, activation='sigmoid')
        cost = compute_cost(A2, y)
        dA2 = -(np.divide(y, A2) - np.divide(1 - y, 1 - A2))
        dA1, dw2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dw1, db1 = linear_activation_backward(dA1, cache1, activation="relu")
        grads['dw1'] = dw1
        grads['db1'] = db1
        grads['dw2'] = dw2
        grads['db2'] = db2
        parameters = update_parameters(parameters, grads, learning_rate)
        w1 = parameters['w1']
        b1 = parameters['b1']
        w2 = parameters['w2']
        b2 = parameters['b2']
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    fig = plt.figure()
    plt.plot(np.squeeze(costs))
    # plt.title('learning_rate=', learning_rate)
    plt.show()
    return parameters


n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
parameters = two_layer_model(train_x, train_y, layers_dims=(n_x, n_h, n_y), num_iterations=2500, print_cost=True)
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)


# 多层深度网络
layers_dims = [12288, 20, 7, 5, 1]
def L_layer_model(x, y, layers_dims, learning_rate=0.007, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(x, parameters)
        cost = compute_cost(AL, y)
        grads = L_model_backward(AL, y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    fig = plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.show()
    return parameters

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)