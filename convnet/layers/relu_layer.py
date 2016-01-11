import numpy as np

from convnet.layers.base_layer import BaseLayer, BaseLayerSettings


class ReluLayerSettings(BaseLayerSettings):
    """
    activation - string, containing a name of desired activation func (refer to ReluLayer.ACTIVATIONS
    """
    def __init__(self, in_shape, activation='max'):
        super(ReluLayerSettings, self).__init__(in_shape=in_shape)
        self.activation = activation

    @property
    def out_width(self):
        return self.in_width

    @property
    def out_height(self):
        return self.in_height

    @property
    def out_depth(self):
        return self.in_depth


class ReluLayer(BaseLayer):
    ACTIVATIONS = {
        'max': lambda x: np.maximum(x, 0),
        'sigmoid': lambda x: 1.0 / (1 + np.exp(-1.0 * x))
    }

    def __init__(self, settings):
        """
        :param settings: Relu layer settings
        :type settings: ReluLayerSettings
        """
        super(ReluLayer, self).__init__(settings)

    def forward(self, data):
        return self.activation(data)

    def backward(self, error):
        pass

    def activation(self, x):
        return self.ACTIVATIONS[self.settings.activation](x)
