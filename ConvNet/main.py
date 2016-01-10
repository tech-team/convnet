import numpy as np

from ConvNet.layers.BaseLayer import BaseLayerSettings
from ConvNet.layers.ConvolutionalLayer import ConvolutionalLayerSettings, ConvolutionalLayer

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=120)

if __name__ == "__main__":
    # arr = np.random.rand(227, 227, 3)
    # s = ConvolutionalLayerSettings(in_dimensions=arr.shape, filter_size=11, stride=4, filters_count=96, zero_padding=0)
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
    s = ConvolutionalLayerSettings(in_dimensions=arr.shape, filter_size=3, stride=2, filters_count=2, zero_padding=1)
    l = ConvolutionalLayer(s)

    l.w = np.array([
        [0, 1, -1,
         -1, 0, 1,
         -1, 1, 0,

         -1, 0, -1,
         1, 0, 1,
         -1, 0, 1,

         0, 0, 1,
         0, -1, -1,
         1, -1, 0],

        [1, 0, 0,
         -1, -1, 1,
         -1, 1, 0,

         1, 0, 0,
         1, 0, -1,
         -1, -1, -1,

         -1, 0, 1,
         0, 0, 1,
         0, -1, -1]
    ])

    l.b = np.array([
        1,
        0
    ]).reshape(2, 1)

    res = l.forward(arr)
    print(res[:, :, 0])
    print(res[:, :, 1])
    pass
