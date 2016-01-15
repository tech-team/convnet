import unittest

import numpy as np

from convnet.layers.full_connected_layer import FullConnectedLayerSettings, _FullConnectedLayer


class FullConnectedLayerTest(unittest.TestCase):
    def test_forward(self):
        np.set_printoptions(precision=4, linewidth=120)

        x = np.zeros((2, 2, 2))
        x[:, :, 0] = np.array([
            [0, 1],
            [0, 1]
        ])

        x[:, :, 1] = np.array([
            [1, 0],
            [1, 0]
        ])

        w = [np.zeros((2, 2, 2))]
        w[0][:, :, 0] = np.array([
            [0, 1],
            [0, 1]
        ])

        w[0][:, :, 1] = np.array([
            [1, 0],
            [1, 0]
        ])

        b = [0]

        expected_res = [[[4]]]

        s = FullConnectedLayerSettings(in_shape=x.shape, neurons_count=1)
        l = _FullConnectedLayer(s)
        l.w = w
        l.b = b

        res = l.forward(x)

        np.testing.assert_allclose(res, expected_res)
