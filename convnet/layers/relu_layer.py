import numpy as np

from convnet.layers.base_layer import _BaseLayer, BaseLayerSettings


class ReluLayerSettings(BaseLayerSettings):
    """
    activation - string, containing a name of desired activation func (refer to ReluLayer.ACTIVATIONS
    """
    def __init__(self, in_shape=None, activation='max'):
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


class _ReluLayer(_BaseLayer):
    ACTIVATIONS = {
        'max': {
            'f': lambda x: np.maximum(x, 0),
            'df': np.vectorize(lambda x: 1 if x > 0 else 0),
        },
        'sigmoid': {
            'f': lambda x: 1.0 / (1 + np.exp(-1.0 * x)),
            'df': lambda x: _ReluLayer.ACTIVATIONS['sigmoid']['f'](x) * (1 - _ReluLayer.ACTIVATIONS['sigmoid']['f'](x)),
        },
    }

    def __init__(self, settings):
        """
        :param settings: Relu layer settings
        :type settings: ReluLayerSettings
        """
        super(_ReluLayer, self).__init__(settings)

    def forward(self, data):
        return self.activation(data)

    def backward(self, current_layer_delta):
        shape = current_layer_delta.shape
        values = current_layer_delta.reshape(-1)
        res = values * self.derivative(values)
        return res.reshape(shape)

    def activation(self, x):
        return self.ACTIVATIONS[self.settings.activation]['f'](x)

    def derivative(self, x):
        return self.ACTIVATIONS[self.settings.activation]['df'](x)
