import cPickle
import gzip
import os

import numpy as np

from convnet.layers import *
from convnet.layers.convolutional_layer import _ConvolutionalLayer
from convnet.net import ConvNet
from convnet.utils import y_1d_to_3d

np.set_printoptions(precision=4, linewidth=120)


def transform_X(X_train):
    X = []
    for i in xrange(X_train.shape[0]):
        x = X_train[i]
        x = x.reshape((28, 28))[:, :, np.newaxis]
        X.append(x)
    return X


def transform_Y(Y_train):
    Y = []
    uniq = np.unique(Y_train)
    labels_count = uniq.size
    for i in xrange(Y_train.shape[0]):
        y = Y_train[i]
        y_arr = np.zeros(labels_count)
        index = np.where(uniq == y)[0][0]
        y_arr[index] = 1
        Y.append(y_1d_to_3d(y_arr))
    return Y


def get_examples(X_train, Y_train, labels=None, count=None):
    assert len(labels) == len(count)

    X = None
    Y = None
    for i, label in enumerate(labels):
        indices = np.where(Y_train == label)[0]
        indices = indices[:count[i]]
        if X is None:
            X = X_train[indices]
        else:
            X = np.concatenate([X, X_train[indices]])

        if Y is None:
            Y = Y_train[indices]
        else:
            Y = np.concatenate([Y, Y_train[indices]])

    return X, Y


# noinspection PyPep8Naming
def mnist():
    script_path = os.path.dirname(os.path.abspath(__file__))
    f = gzip.open(os.path.join(script_path, 'mnist/mnist.pkl.gz'), 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    X_train, Y_train = train_set

    X_train, Y_train = get_examples(X_train, Y_train, labels=[0, 1, 2], count=[10] * 3)
    X_train = transform_X(X_train)
    Y_train = transform_Y(Y_train)

    net = ConvNet(iterations_count=1, batch_size=10, learning_rate=0.01, momentum=0.9)
    net.setup_layers([
        InputLayer(InputLayerSettings(in_shape=X_train[0].shape)),

        ConvolutionalLayer(ConvolutionalLayerSettings(filters_count=8, filter_size=5, stride=1, zero_padding=0)),
        ReluLayer(ReluLayerSettings(activation='max')),
        PoolingLayer(PoolingLayerSettings(filter_size=2, stride=2)),

        ConvolutionalLayer(ConvolutionalLayerSettings(filters_count=16, filter_size=5, stride=1, zero_padding=0)),
        ReluLayer(ReluLayerSettings(activation='max')),
        PoolingLayer(PoolingLayerSettings(filter_size=3, stride=3)),

        FullConnectedLayer(FullConnectedLayerSettings(neurons_count=Y_train[0].shape[-1])),
        ReluLayer(ReluLayerSettings(activation='sigmoid')),
    ])

    examples_count = 100000
    net.fit(X_train[:examples_count], Y_train[:examples_count])

    # for x, y in zip(X_train[:examples_count], Y_train[:examples_count]):
    #     h = net.predict(x)
    #     print("predicted = {}; \nreal = {}\n\n".format(h, y))
    pass


if __name__ == "__main__":
    mnist()
    # for i in xrange(10000):
    #     print(i)
    #     arr = np.empty((5, 5, 3))
    #     arr[:, :, 0] = np.array([
    #         [1, 2, 0, 2, 0],
    #         [1, 0, 0, 0, 0],
    #         [0, 2, 0, 2, 1],
    #         [1, 1, 1, 0, 2],
    #         [2, 2, 0, 0, 1]
    #     ])
    #     arr[:, :, 1] = np.array([
    #         [0, 1, 0, 2, 1],
    #         [0, 0, 2, 1, 2],
    #         [1, 2, 1, 2, 0],
    #         [1, 0, 1, 0, 2],
    #         [0, 2, 2, 2, 0]
    #     ])
    #     arr[:, :, 2] = np.array([
    #         [1, 2, 2, 0, 2],
    #         [1, 0, 1, 1, 0],
    #         [2, 2, 0, 2, 0],
    #         [1, 0, 0, 1, 2],
    #         [0, 1, 1, 1, 2],
    #     ])
    #     s = ConvolutionalLayerSettings(in_shape=arr.shape, filter_size=3, stride=2, filters_count=2, zero_padding=1)
    #     l = _ConvolutionalLayer(s)
    #
    #     w0 = np.empty((3, 3, 3))
    #     w0[:, :, 0] = np.array([
    #         [0, 1, -1],
    #         [-1, 0, 1],
    #         [-1, 1, 0]
    #     ])
    #     w0[:, :, 1] = np.array([
    #         [-1, 0, -1],
    #         [1, 0, 1],
    #         [-1, 0, 1]
    #     ])
    #     w0[:, :, 2] = np.array([
    #         [0, 0, 1],
    #         [0, -1, -1],
    #         [1, -1, 0]
    #     ])
    #
    #     w1 = np.empty((3, 3, 3))
    #     w1[:, :, 0] = np.array([
    #         [1, 0, 0],
    #         [-1, -1, 1],
    #         [-1, 1, 0]
    #     ])
    #     w1[:, :, 1] = np.array([
    #         [1, 0, 0],
    #         [1, 0, -1],
    #         [-1, -1, -1]
    #     ])
    #     w1[:, :, 2] = np.array([
    #         [-1, 0, 1],
    #         [0, 0, 1],
    #         [0, -1, -1]
    #     ])
    #
    #     l.w = [w0, w1]
    #     l.b = [
    #         1,
    #         0
    #     ]
    #
    #     res = l.forward(arr)
    #     # print(res[:, :, 0])
    #     # print(res[:, :, 1])
    #
    #     np.testing.assert_allclose(res, np.array([
    #         [[1., 2.],
    #          [2., -6.],
    #          [-1., -3.]],
    #
    #         [[2., 1.],
    #          [3., 1.],
    #          [1., -3.]],
    #
    #         [[4., -1.],
    #          [3., 1.],
    #          [3., 0.]]]))
