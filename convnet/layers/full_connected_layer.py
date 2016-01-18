import numpy as np

from convnet.layers.base_layer import BaseLayer, BaseLayerSettings
from convnet.layers.convolutional_layer import ConvolutionalLayerSettings, _ConvolutionalLayer
from convnet.layers.relu_layer import _ReluLayer, ReluLayerSettings


class FullConnectedLayerSettings(ConvolutionalLayerSettings):
    TYPE = 'fc'

    def __init__(self, in_shape=None, neurons_count=1,activation='sigmoid'):
        super(FullConnectedLayerSettings, self).__init__(in_shape=in_shape,
                                                         filters_count=neurons_count,
                                                         filter_size=in_shape[0] if in_shape is not None else None,
                                                         stride=1,
                                                         zero_padding=0)
        self.activation = activation

    def in_shape_changed(self):
        super(FullConnectedLayerSettings, self).in_shape_changed()
        self.filter_size = self.in_width

    def to_dict(self):
        d = BaseLayerSettings.to_dict(self)
        d.update({
            'activation': self.activation,
        })
        return d


class _FullConnectedLayer(_ConvolutionalLayer):
    def __init__(self, settings, net_settings=None):
        """
        :param settings: Full connected layer settings
        :type settings: FullConnectedLayerSettings
        """
        super(_FullConnectedLayer, self).__init__(settings, net_settings)
        self.relu = _ReluLayer(ReluLayerSettings(in_shape=self.settings.out_shape, activation=self.settings.activation))

    def setup_layers(self, prev_layer, next_layer=None):
        super(_FullConnectedLayer, self).setup_layers(prev_layer, next_layer)
        self.relu.setup_layers(self.prev_layer, self.next_layer)

    def forward(self, data):
        res = super(_FullConnectedLayer, self).forward(data)
        return self.relu.forward(res)

    def compute_prev_layer_delta(self, current_layer_delta):
        prev_delta = super(_FullConnectedLayer, self).compute_prev_layer_delta(current_layer_delta)
        return self.relu.compute_prev_layer_delta(prev_delta)



class FullConnectedLayer(BaseLayer):
    def __init__(self, settings):
        """
        :type settings: FullConnectedLayerSettings
        """
        super(FullConnectedLayer, self).__init__(settings)

    def create(self):
        return _FullConnectedLayer(self.settings, self.net_settings)
