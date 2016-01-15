import numpy as np

from convnet.net import ConvNet
from convnet.layers.full_connected_layer import FullConnectedLayerSettings, FullConnectedLayer
from convnet.layers.input_layer import InputLayerSettings, InputLayer

np.set_printoptions(precision=4, linewidth=120)

if __name__ == "__main__":
    X = [np.zeros((3, 3, 2)) for _ in xrange(2)]
    y = [np.zeros((1, 1, 2)) for _ in xrange(2)]

    net = ConvNet()
    net.setup_layers([
        InputLayer(InputLayerSettings(in_shape=X[0].shape)),
        FullConnectedLayer(FullConnectedLayerSettings(neurons_count=y[0].shape[-1]))
    ])

    net.fit(X, y)

    pass
