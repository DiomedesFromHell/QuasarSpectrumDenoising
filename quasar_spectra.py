import numpy as np
import matplotlib.pyplot as plt


def read_data(file, header=True):
    X = []
    wavelength = None
    with open(file, "r") as f:
        if header is True:
            wavelength = tuple(float(item) for item in f.readline().split(','))
        for line in f:
            X.append(tuple(float(item) for item in line.split(',')))
    return wavelength, X


def linear_regression(X, Y):
    if X.ndim < 2:
        X = X.reshape(X.shape[0], -1)
    ncol = X.shape[1]+1
    X, X_0 = np.ones((X.shape[0], ncol)), X
    X[:, 1:] = X_0
    theta = np.dot(np.linalg.inv(X.T.dot(X)), X.T.dot(Y))
    # prediction = theta.dot(x)
    return theta


def lwlin_reg(X, Y, x, tau=5):
    x = np.array((1, x))
    if X.ndim < 2:
        X = X.reshape(X.shape[0], -1)
    ncol = X.shape[1]+1
    X, X_0 = np.ones((X.shape[0], ncol)), X
    X[:, 1:] = X_0
    w = np.exp(-np.sum((np.tile(x, (X.shape[0], 1))-X)**2, axis=-1)/(2*tau**2)).flatten()
    W = np.diag(w)
    theta = np.dot(np.linalg.inv(X.T.dot(W.dot(X))), X.T.dot(W.dot(Y)))
    prediction = theta.dot(x)
    return prediction


if __name__ == "main":
    X, Y_train = read_data('quasar_train.csv')

    X, Y_train = np.array(X), np.array(Y_train)
    X, Y_test = read_data('quasar_test.csv')
    X, Y_test = np.array(X), np.array(Y_test)
    theta = linear_regression(X, Y_train[0, :])
    lr = theta[0] + theta[1]*X
    lwlr = []
    for x in X:
        lwlr.append(lwlin_reg(X, Y_train[0, :], x))
    plt.scatter(x=X, y=Y_train[0, :], s=10, c='r')
    plt.plot(X, lr)

    plt.plot(X, lwlr, "g")
    plt.show()

