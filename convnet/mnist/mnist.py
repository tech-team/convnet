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


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# noinspection PyPep8Naming
def mnist():
    script_path = os.path.dirname(os.path.abspath(__file__))
    f = gzip.open(os.path.join(script_path, './mnist.pkl.gz'), 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    X_train, Y_train = train_set

    X_train, Y_train = get_examples(X_train, Y_train, labels=np.arange(0, 10), count=30)
    X_train, Y_train = shuffle_in_unison_inplace(X_train, Y_train)
    X_train = transform_X(X_train)
    Y_train = transform_Y(Y_train)

    X_test, Y_test = test_set
    X_test, Y_test = get_examples(X_test, Y_test, labels=np.arange(0, 10), count=30)
    X_test, Y_test = shuffle_in_unison_inplace(X_test, Y_test)
    X_test = transform_X(X_test)
    Y_test = transform_Y(Y_test)

    net = ConvNet(iterations_count=100, batch_size=10, learning_rate=0.001, momentum=0.8, weight_decay=0.001)
    net.setup_layers([
        InputLayer(InputLayerSettings(in_shape=X_train[0].shape)),

        ConvolutionalLayer(ConvolutionalLayerSettings(filters_count=8, filter_size=5, stride=1, zero_padding=0)),
        ReluLayer(ReluLayerSettings(activation='max')),
        PoolingLayer(PoolingLayerSettings(filter_size=2, stride=2)),

        ConvolutionalLayer(ConvolutionalLayerSettings(filters_count=16, filter_size=5, stride=1, zero_padding=0)),
        ReluLayer(ReluLayerSettings(activation='max')),
        PoolingLayer(PoolingLayerSettings(filter_size=3, stride=3)),

        FullConnectedLayer(FullConnectedLayerSettings(neurons_count=Y_train[0].shape[-1], activation='sigmoid')),
    ])

    examples_count = 100000

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # net = ConvNet.load_net(os.path.join(BASE_DIR, './convnet1.pkl'))
    net.fit(X_train[:examples_count], Y_train[:examples_count])

    train_matched = 0
    for x, y in zip(X_train[:examples_count], Y_train[:examples_count]):
        h = net.predict(x)
        h_res = h.argmax()
        y_res = y.argmax()
        print("predicted = {}; max = {}".format(h, h.argmax()))
        print("real =      {}; max = {}".format(y, y.argmax()))
        print("\n")
        train_matched += int(h_res == y_res)

    test_matched = 0
    for x, y in zip(X_test[:examples_count], Y_test[:examples_count]):
        h = net.predict(x)
        h_res = h.argmax()
        y_res = y.argmax()
        print("predicted = {}; max = {}".format(h, h.argmax()))
        print("real =      {}; max = {}".format(y, y.argmax()))
        print("\n")
        test_matched += int(h_res == y_res)

    print("Accuracy train {}/{}".format(train_matched, len(X_train[:examples_count])))
    print("Accuracy test {}/{}".format(test_matched, len(X_test[:examples_count])))

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "./convnet1.pkl")
    net.dump_net(path)
    print("Dumped to {}".format(path))


if __name__ == "__main__":
    mnist()
    # net = ConvNet.load_net('/home/igor/Desktop/convnet.pkl')
    pass
