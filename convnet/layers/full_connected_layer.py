import numpy as np

from convnet.layers.base_layer import BaseLayer
from convnet.layers.convolutional_layer import ConvolutionalLayerSettings, _ConvolutionalLayer


class FullConnectedLayerSettings(ConvolutionalLayerSettings):
    def __init__(self, in_shape=None, neurons_count=1):
        super(FullConnectedLayerSettings, self).__init__(in_shape=in_shape,
                                                         filters_count=neurons_count,
                                                         filter_size=in_shape[0] if in_shape is not None else None,
                                                         stride=1,
                                                         zero_padding=0)

    def in_shape_changed(self):
        super(FullConnectedLayerSettings, self).in_shape_changed()
        self.filter_size = self.in_width


class _FullConnectedLayer(_ConvolutionalLayer):
    def __init__(self, settings, net_settings=None):
        """
        :param settings: Full connected layer settings
        :type settings: FullConnectedLayerSettings
        """
        super(_FullConnectedLayer, self).__init__(settings, net_settings)


class FullConnectedLayer(BaseLayer):
    def __init__(self, settings):
        """
        :type settings: FullConnectedLayerSettings
        """
        super(FullConnectedLayer, self).__init__(settings)

    def create(self):
        return _FullConnectedLayer(self.settings, self.net_settings)
