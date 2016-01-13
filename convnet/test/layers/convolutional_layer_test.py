import unittest

import numpy as np

from convnet.layers.convolutional_layer import ConvolutionalLayerSettings, ConvolutionalLayer
from convnet.layers.input_layer import InputLayer, InputLayerSettings


class ConvolutionalLayerTest(unittest.TestCase):
    def test_forward(self):
        np.set_printoptions(precision=4, linewidth=120)
        # arr = np.random.rand(227, 227, 3)
        # s = ConvolutionalLayerSettings(in_dimensions=arr.shape,
        #                                filter_size=11, stride=4, filters_count=96, zero_padding=0)
        # l = ConvolutionalLayer(s)
        # l.forward(arr)

        # arr = np.random.rand(5, 5, 3)

        arr = np.empty((5, 5, 3))
        arr[:, :, 0] = np.array([
            [1, 2, 0, 2, 0],
            [1, 0, 0, 0, 0],
            [0, 2, 0, 2, 1],
            [1, 1, 1, 0, 2],
            [2, 2, 0, 0, 1]
        ])
        arr[:, :, 1] = np.array([
            [0, 1, 0, 2, 1],
            [0, 0, 2, 1, 2],
            [1, 2, 1, 2, 0],
            [1, 0, 1, 0, 2],
            [0, 2, 2, 2, 0]
        ])
        arr[:, :, 2] = np.array([
            [1, 2, 2, 0, 2],
            [1, 0, 1, 1, 0],
            [2, 2, 0, 2, 0],
            [1, 0, 0, 1, 2],
            [0, 1, 1, 1, 2],
        ])
        s = ConvolutionalLayerSettings(in_shape=arr.shape, filter_size=3, stride=2, filters_count=2, zero_padding=1)
        l = ConvolutionalLayer(s)

        w0 = np.empty((3, 3, 3))
        w0[:, :, 0] = np.array([
            [0, 1, -1],
            [-1, 0, 1],
            [-1, 1, 0]
        ])
        w0[:, :, 1] = np.array([
            [-1, 0, -1],
            [1, 0, 1],
            [-1, 0, 1]
        ])
        w0[:, :, 2] = np.array([
            [0, 0, 1],
            [0, -1, -1],
            [1, -1, 0]
        ])

        w1 = np.empty((3, 3, 3))
        w1[:, :, 0] = np.array([
            [1, 0, 0],
            [-1, -1, 1],
            [-1, 1, 0]
        ])
        w1[:, :, 1] = np.array([
            [1, 0, 0],
            [1, 0, -1],
            [-1, -1, -1]
        ])
        w1[:, :, 2] = np.array([
            [-1, 0, 1],
            [0, 0, 1],
            [0, -1, -1]
        ])

        l.w = [w0, w1]
        l.b = [
            1,
            0
        ]

        res = l.forward(arr)
        print(res[:, :, 0])
        print(res[:, :, 1])

        np.testing.assert_allclose(res, np.array([
            [[1., 2.],
             [2., -6.],
             [-1., -3.]],

            [[2., 1.],
             [3., 1.],
             [1., -3.]],

            [[4., -1.],
             [3., 1.],
             [3., 0.]]]))

    def test_backward(self):
        arr1 = np.empty((3, 3, 1))
        arr1[:, :, 0] = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])

        arr2 = np.empty((3, 3, 1))
        arr2[:, :, 0] = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]])

        samples = [arr1, arr2]

        expected_res1 = np.empty((1, 1, 2))
        expected_res1[0, 0, 0] = 1
        expected_res1[0, 0, 1] = -1
        expected_res2 = np.empty((1, 1, 2))
        expected_res2[0, 0, 0] = -1
        expected_res2[0, 0, 1] = 1

        expected_res = [expected_res1, expected_res2]

        input_layer = InputLayer(InputLayerSettings(arr1.shape))

        s = ConvolutionalLayerSettings(
                in_shape=arr1.shape,
                filter_size=3,
                stride=1,
                filters_count=2,
                zero_padding=0)
        conv_layer = ConvolutionalLayer(s)
        conv_layer.prev_layer = input_layer

        # before update
        res_before = []
        for i in xrange(len(samples)):
            res = input_layer.forward(samples[i])
            res = conv_layer.forward(res)
            res_before.append(res)

            conv_layer.backward(expected_res[i])

        # update weights
        dist_before = get_dist(res_before, expected_res)
        conv_layer.update_weights()

        # after update
        res_after = []
        for i in xrange(len(samples)):
            res = input_layer.forward(samples[i])
            res = conv_layer.forward(res)
            res_after.append(res)

        dist_after = get_dist(res_after, expected_res)

        print 'Before: %s' % dist_before
        print 'After: %s' % dist_after

        self.assertLess(dist_after, dist_before)


def get_dist(res, expected_res):
    dist = 0
    for i in xrange(len(res)):
        dist += np.linalg.norm(res[i] - expected_res[i])
    return dist / len(res)
