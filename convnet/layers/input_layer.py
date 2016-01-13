import numpy as np

from convnet.layers.base_layer import BaseLayer, BaseLayerSettings


class InputLayerSettings(BaseLayerSettings):
    def __init__(self, in_shape):
        super(InputLayerSettings, self).__init__(in_shape=in_shape)

    def out_depth(self):
        return self.in_depth

    def out_height(self):
        return self.in_height

    def out_width(self):
        return self.in_width


class InputLayer(BaseLayer):
    def __init__(self, settings):
        """
        :param settings: Input layer settings
        :type settings: InputLayerSettings
        """
        super(InputLayer, self).__init__(settings)

    def forward(self, data):
        self.prev_out = data
        return data

    def backward(self, next_layer_delta):
        return next_layer_delta

