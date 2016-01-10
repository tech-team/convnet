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

    arr = np.random.rand(5, 5, 3)
    s = ConvolutionalLayerSettings(in_dimensions=arr.shape, filter_size=3, stride=2, filters_count=2, zero_padding=1)
    l = ConvolutionalLayer(s)
    l.forward(arr)
    pass
