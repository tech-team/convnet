import numpy as np

from convnet.layers.convolutional_layer import ConvolutionalLayerSettings, ConvolutionalLayer


class FullConnectedLayerSettings(ConvolutionalLayerSettings):
    def __init__(self, in_shape, neurons_count=1):
        super(FullConnectedLayerSettings, self).__init__(in_shape=in_shape,
                                                         filters_count=neurons_count,
                                                         filter_size=in_shape[0],
                                                         stride=1,
                                                         zero_padding=0)


class FullConnectedLayer(ConvolutionalLayer):
    def __init__(self, settings):
        """
        :param settings: Full connected layer settings
        :type settings: FullConnectedLayerSettings
        """
        super(FullConnectedLayer, self).__init__(settings)
