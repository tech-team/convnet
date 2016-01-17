import unittest

import numpy as np

from convnet.layers.input_layer import _InputLayer, InputLayerSettings
from convnet.layers.relu_layer import ReluLayerSettings, _ReluLayer


class ReluLayerTest(unittest.TestCase):
    def setUp(self):
        np.set_printoptions(precision=4, linewidth=120)
        np.random.seed(10)
        print('################### %s ################### ' % self._testMethodName)

    def test_forward(self):
        arr = np.random.uniform(-5, 5, 48).reshape((4, 4, 3))
        s = ReluLayerSettings(in_shape=arr.shape, activation='max')
        l = _ReluLayer(s)
        res = l.forward(arr)

        print(arr[:, :, 0])
        print(arr[:, :, 1])
        print(arr[:, :, 2])
        print('-----------')
        print(res[:, :, 0])
        print(res[:, :, 1])
        print(res[:, :, 2])

    def test_backward(self):
        arr = np.random.uniform(-5, 5, 48).reshape((4, 4, 3))
        s = ReluLayerSettings(in_shape=arr.shape, activation='max')
        l = _ReluLayer(s)
        l.prev_layer = _InputLayer(InputLayerSettings(in_shape=arr.shape))
        l.next_layer = object()

        e = np.random.uniform(-5, 5, 48).reshape(*s.out_shape)
        print(e[:, :, 0])
        print(e[:, :, 1])
        print(e[:, :, 2])
        print('-----------')

        res = l.prev_layer.forward(arr)
        res = l.forward(res)

        res = l.backward(e)

        print(res[:, :, 0])
        print(res[:, :, 1])
        print(res[:, :, 2])