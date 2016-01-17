import unittest
import numpy as np

from convnet.layers.pooling_layer import PoolingLayerSettings, _PoolingLayer


class PoolingLayerTest(unittest.TestCase):
    def setUp(self):
        np.set_printoptions(precision=4, linewidth=120)
        np.random.seed(10)
        print('################### %s ################### ' % self._testMethodName)

    def test_forward(self):
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

    def test_backward(self):
        arr = np.random.rand(4, 4, 3)
        s = PoolingLayerSettings(in_shape=arr.shape, filter_size=2, stride=2)
        l = _PoolingLayer(s)
        l.next_layer = object()
        res_forw = l.forward(arr)

        res_back = l.backward(np.ones(s.out_shape))

        print(arr[:, :, 0])
        print(arr[:, :, 1])
        print(arr[:, :, 2])
        print('-----------')
        print(res_forw[:, :, 0])
        print(res_forw[:, :, 1])
        print(res_forw[:, :, 2])
        print('-----------')
        print(res_back[:, :, 0])
        print(res_back[:, :, 1])
        print(res_back[:, :, 2])

        # masked = np.empty(res_forw.shape)
        # for z in xrange(res_back.shape[2]):
        #     m = (arr[:, :, z] * res_back[:, :, z])
        #     masked[:, :, z] = m[m > 0].reshape(masked[:, :, z].shape)
        #
        # print('-----------')
        # print('-----------')
        # print(masked[:, :, 0])
        # print(masked[:, :, 1])
        # print(masked[:, :, 2])
        # np.testing.assert_allclose(masked, res_forw) # TODO: not correctly building masked array
