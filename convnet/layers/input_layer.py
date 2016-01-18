import numpy as np

from convnet.layers.base_layer import _BaseLayer, BaseLayerSettings, BaseLayer


class InputLayerSettings(BaseLayerSettings):
    TYPE = 'input'

    def __init__(self, in_shape):
        super(InputLayerSettings, self).__init__(in_shape=in_shape)

    @property
    def out_depth(self):
        return self.in_depth

    @property
    def out_height(self):
        return self.in_height

    @property
    def out_width(self):
        return self.in_width


class _InputLayer(_BaseLayer):
    def __init__(self, settings, net_settings=None):
        """
        :param settings: Input layer settings
        :type settings: InputLayerSettings
        """
        super(_InputLayer, self).__init__(settings, net_settings)

    def forward(self, data):
        self.prev_out = data
        return data

    def backward(self, current_layer_delta):
        return current_layer_delta


class InputLayer(BaseLayer):
    def __init__(self, settings):
        """
        :type settings: InputLayerSettings
        """
        super(InputLayer, self).__init__(settings)

    def create(self):
        return _InputLayer(self.settings, self.net_settings)
