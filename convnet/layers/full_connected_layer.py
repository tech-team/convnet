import numpy as np

from convnet.layers.convolutional_layer import ConvolutionalLayerSettings, ConvolutionalLayer


class FullConnectedLayerSettings(ConvolutionalLayerSettings):
    def __init__(self, **kwargs):
        kwargs['filters_count'] = 1
        kwargs['filter_size'] = kwargs['in_dimensions'][0]
        kwargs['stride'] = 1
        kwargs['zero_padding'] = 0

        super(FullConnectedLayerSettings, self).__init__(**kwargs)


class FullConnectedLayer(ConvolutionalLayer):
    def __init__(self, settings):
        """
        :param settings: Full connected layer settings
        :type settings: FullConnectedLayerSettings
        """
        super(FullConnectedLayer, self).__init__(settings)
