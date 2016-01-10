import numpy as np

from convnet.layers.full_connected_layer import FullConnectedLayer, FullConnectedLayerSettings

np.set_printoptions(precision=4, linewidth=120)

if __name__ == "__main__":
    s = FullConnectedLayerSettings(in_dimensions=(3, 3, 2))
    l = FullConnectedLayer(s)
    pass
