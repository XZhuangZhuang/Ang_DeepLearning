import numpy as np
import matplotlib.pyplot as plt
import pylab
import sklearn
import sklearn.datasets
import sklearn.linear_model

def load_planar_dataset():
    np.random.seed(1)
    m = 400
    N = int(m / 2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4
    for i in range(2):
        ix = range(N * i, N * (i + 1))
        t = np.linspace(i * 3.12, (i +1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = i
    X = X.T
    Y = Y.T
    # print(X.shape)
    return X, Y

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() -1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() -1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(X[0, :], X[1, :], c=y.reshape(X[0, :].shape), cmap=plt.cm.Spectral)
    # plt.show()

# X, Y = load_planar_dataset()
# fig = plt.figure()
# plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0, :].shape), s=40, cmap=plt.cm.Spectral)
# plt.show()
#
#
# # 使用逻辑回归
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.T)
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# LR_predictions = clf.predict(X.T)
# print('准确率：{}'.format(float(np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100))



# Step1：设计网络结构，例如多少层，每层有多少神经元等。
# Step2：初始化模型的参数
# Step3：循环
# Step3.1：前向传播计算
# Step3.2：计算代价函数
# Step3.3：反向传播计算
# Step3.4：更新参数

def layer_size(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

def initiallize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1)) * 0.01
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    assert(w1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(w2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    parameters={'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return parameters

def foward_propagation(X, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    Z1 = np.dot(w1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(w2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    assert(A2.shape == (1, X.shape[1]))
    cache = {'Z1': Z1,
             'A1': A1,
             'Z2': Z2,
             'A2': A2}
    return A2, cache

def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
    cost = -np.sum(logprobs) / m
    cost = np.squeeze(cost)
    assert(isinstance(cost, float))
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    w1 = parameters['w1']
    w2 = parameters['w2']
    A1 = cache['A1']
    A2 = cache['A2']
    dZ2 = A2 - Y
    dw2 = 1.0 / m * np.dot(dZ2, A1.T)
    db2 = 1.0 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(w2.T, dZ2) * (1 - np.power(A1, 2))
    dw1 = 1.0 / m * np.dot(dZ1, X.T)
    db1 = 1.0 / m * np.sum(dZ1, axis=1, keepdims=True)
    grads = {'dw1': dw1,
             'db1': db1,
             'dw2': dw2,
             'db2': db2}
    return grads

def update_parameters(parameters, grads, learning_rate=1.2):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    parameters = {'w1': w1,
                  'b1': b1,
                  'w2': w2,
                  'b2': b2}
    return parameters

def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]
    parameters = initiallize_parameters(n_x, n_h, n_y)
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    for i in range(num_iterations):
        A2, cache = foward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=1.2)
        if print_cost and i % 1000 == 0:
            print('cost after iteration %i: %f'%(i, cost))
    return parameters

def predict(parameters, X):
    A2, cache = foward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions


X, Y = load_planar_dataset()
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')


plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.show()
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))