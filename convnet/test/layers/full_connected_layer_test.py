import unittest

import numpy as np

from convnet.layers.full_connected_layer import FullConnectedLayerSettings, FullConnectedLayer


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

        w = np.array([
            [0, 1, 0, 1, 1, 0, 1, 0]
        ])

        b = np.array([0])

        expected_res = [[[4]]]

        s = FullConnectedLayerSettings(in_shape=x.shape, neurons_count=1)
        l = FullConnectedLayer(s)
        l.w = w
        l.b = b

        res = l.forward(x)

        np.testing.assert_allclose(res, expected_res)
