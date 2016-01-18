import gzip
import os
import unittest

import cPickle
import numpy as np

from convnet.layers import *
from convnet.net import ConvNet
from convnet.utils import arr_2d_to_3d, input_2d_to_3d, input_1d_to_3d, to_3d, y_to_3d, y_1d_to_3d


# noinspection PyPep8Naming
class ConvNetTest(unittest.TestCase):
    def setUp(self):
        np.set_printoptions(precision=2, linewidth=120)
        # np.random.seed(10)
        print('################### %s ################### ' % self._testMethodName)

    def _test_basic(self):
        X = [np.zeros((3, 3, 2)) for _ in xrange(2)]
        y = [np.zeros((1, 1, 2)) for _ in xrange(2)]

        net = ConvNet()
        net.setup_layers([
            InputLayer(InputLayerSettings(in_shape=X[0].shape)),
            ConvolutionalLayer(ConvolutionalLayerSettings(filter_size=2, filters_count=2, stride=1)),
            PoolingLayer(PoolingLayerSettings(filter_size=2, stride=1)),
            ReluLayer(ReluLayerSettings(activation='max')),
            FullConnectedLayer(FullConnectedLayerSettings(neurons_count=y[0].shape[-1])),
            ReluLayer(ReluLayerSettings(activation='sigmoid')),
        ])

        net.fit(X, y)

    def _test_crossings(self):
        X = to_3d([
            np.asarray([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]),
            np.asarray([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]),
            np.asarray([
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ]),
            np.asarray([
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, 0],
            ]),
        ])
        Y = [
            np.asarray([1, 0]),
            np.asarray([1, 0]),
            np.asarray([0, 1]),
            np.asarray([0, 1]),
        ]

        net = ConvNet(iterations_count=1000, learning_rate=0.01)
        net.setup_layers([
            InputLayer(InputLayerSettings(in_shape=X[0].shape)),
            ConvolutionalLayer(ConvolutionalLayerSettings(filter_size=3, filters_count=2, stride=1)),
            ReluLayer(ReluLayerSettings(activation='max')),
            # PoolingLayer(PoolingLayerSettings(filter_size=1, stride=1)),
            # FullConnectedLayer(FullConnectedLayerSettings(neurons_count=Y[0].shape[-1])),
            # ReluLayer(ReluLayerSettings(activation='sigmoid')),
        ])

        net.fit(X, y_to_3d(Y))

        # net = ConvNet.load_net('/home/igor/Desktop/net.pkl')

        for x, y in zip(X, Y):
            h = net.predict(x)
            print("predicted = {}; \nreal = {}\n\n".format(h, y))
        pass

    def transform_X(self, X_train):
        X = []
        for i in xrange(X_train.shape[0]):
            x = X_train[i]
            x = x.reshape((28, 28))[:, :, np.newaxis]
            X.append(x)
        return X

    def transform_Y(self, Y_train):
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

    def get_examples(self, X_train, Y_train, labels=None, count=None):
        if isinstance(count, int):
            count = [count] * len(labels)
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

    def shuffle_in_unison_inplace(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def test_mnist(self):
        script_path = os.path.dirname(os.path.abspath(__file__))
        f = gzip.open(os.path.join(script_path, '../mnist/mnist.pkl.gz'), 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        X_train, Y_train = train_set

        X_train, Y_train = self.get_examples(X_train, Y_train, labels=np.arange(0, 4), count=10)
        X_train, Y_train = self.shuffle_in_unison_inplace(X_train, Y_train)
        X_train = self.transform_X(X_train)
        Y_train = self.transform_Y(Y_train)

        net = ConvNet(iterations_count=10, batch_size=10, learning_rate=0.001, momentum=0.8, weight_decay=0.001)
        net.setup_layers([
            InputLayer(InputLayerSettings(in_shape=X_train[0].shape)),

            ConvolutionalLayer(ConvolutionalLayerSettings(filters_count=8, filter_size=5, stride=1, zero_padding=0)),
            ReluLayer(ReluLayerSettings(activation='max')),
            PoolingLayer(PoolingLayerSettings(filter_size=2, stride=2)),
            #
            # ConvolutionalLayer(ConvolutionalLayerSettings(filters_count=16, filter_size=5, stride=1, zero_padding=0)),
            # ReluLayer(ReluLayerSettings(activation='max')),
            # PoolingLayer(PoolingLayerSettings(filter_size=3, stride=3)),

            FullConnectedLayer(FullConnectedLayerSettings(neurons_count=Y_train[0].shape[-1], activation='sigmoid')),
            # ReluLayer(ReluLayerSettings(activation='sigmoid')),
        ])

        examples_count = 100000
        net.fit(X_train[:examples_count], Y_train[:examples_count])

        matched = 0
        for x, y in zip(X_train[:examples_count], Y_train[:examples_count]):
            h = net.predict(x)
            h_res = h.argmax()
            y_res = y.argmax()
            print("predicted = {}; max = {}".format(h, h.argmax()))
            print("real =      {}; max = {}".format(y, y.argmax()))
            print("\n")
            matched += int(h_res == y_res)

        print("Accuracy {}/{}".format(matched, len(X_train[:examples_count])))