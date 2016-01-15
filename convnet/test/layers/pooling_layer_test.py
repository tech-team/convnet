import unittest
import numpy as np

from convnet.layers.pooling_layer import PoolingLayerSettings, _PoolingLayer


class PoolingLayerTest(unittest.TestCase):
    def test_forward(self):
        np.set_printoptions(precision=4, linewidth=120)
        arr = np.random.rand(4, 4, 3)
        s = PoolingLayerSettings(in_shape=arr.shape, filter_size=2, stride=2)
        l = _PoolingLayer(s)
        res = l.forward(arr)

        print(arr[:, :, 0])
        print(arr[:, :, 1])
        print(arr[:, :, 2])
        print('-----------')
        print(res[:, :, 0])
        print(res[:, :, 1])
        print(res[:, :, 2])
