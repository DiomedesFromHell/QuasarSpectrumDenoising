import numpy as np
from quasar_spectra import lwlin_reg, read_data


def write_smooth_data(X, Y, filename):
    smoothed = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        for j, x in enumerate(X):
            smoothed[i, j] = lwlin_reg(X, Y[i, :], x)
    np.savetxt(filename+'.csv', smoothed, delimiter=',')


def func_estimator(Y_train, Y, k=3):
    m = Y.shape[0]
    m_train = Y_train.shape[0]
    estimation = np.zeros((Y.shape[0], 50))
    for i in range(m):
        distances = np.zeros(m_train)
        for j in range(m_train):
            distances[j] = np.sum((Y[i, 150:]-Y_train[j, 150:])**2)#func_distance(Y[i, 150:], Y_train[j, 150:])
        h = np.max(distances)
        indices = np.argsort(distances)[0:k]

        weights = np.maximum(np.array(tuple(1-distances[ind]/h for ind in indices)), np.zeros(k)).flatten()
        lefts = np.zeros((k, 50))
        norm_factor = np.sum(weights)
        for l in range(k):
            lefts[l] = weights[l]*Y_train[indices[l], :50]
        lefts = np.sum(lefts, axis=0)
        estimation[i] = lefts/norm_factor
    return estimation


def error(preds, Y_left):
    distance = np.sum((preds-Y_left)**2, axis=(-1, 0))
    return distance/Y_left.shape[0]

if __name__ == "main":
    X, ad = read_data('quasar_train.csv')
    X = np.array(X)
    Y_train = read_data('quasar_smoothed_train.csv', header=False)
    Y_train = np.array(Y_train[1])
    Y_test = read_data('quasar_smoothed_test.csv', header=False)
    Y_test = np.array(Y_test[1])
    train_predictions = func_estimator(Y_train, Y_train)
    test_predictions = func_estimator(Y_train, Y_test)
    print(error(train_predictions, Y_train[:, :50]))
    print(error(test_predictions, Y_test[:, :50]))


