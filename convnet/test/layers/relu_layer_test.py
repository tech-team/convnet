import unittest

import numpy as np

from convnet.layers.relu_layer import ReluLayerSettings, ReluLayer


class ReluLayerTest(unittest.TestCase):
    def test_forward(self):
        np.set_printoptions(precision=4, linewidth=120)
        arr = np.random.uniform(-5, 5, 48).reshape((4, 4, 3))
        s = ReluLayerSettings(in_dimensions=arr.shape, activation='max')
        l = ReluLayer(s)
        res = l.forward(arr)

        print(arr[:, :, 0])
        print(arr[:, :, 1])
        print(arr[:, :, 2])
        print('-----------')
        print(res[:, :, 0])
        print(res[:, :, 1])
        print(res[:, :, 2])