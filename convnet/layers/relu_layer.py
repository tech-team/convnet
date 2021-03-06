import numpy as np

from convnet.convnet_error import ConvNetError
from convnet.layers.base_layer import _BaseLayer, BaseLayerSettings, BaseLayer


def relu_softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def relu_softmax_deriv(x):
    exps = np.exp(x)
    others = exps.sum() - exps
    return 1 / (2 + exps / others + others / exps)


class ReluLayerSettings(BaseLayerSettings):
    TYPE = 'relu'

    def __init__(self, in_shape=None, activation='max'):
        """
        activation - string, containing a name of desired activation func (refer to ReluLayer.ACTIVATIONS
        """
        super(ReluLayerSettings, self).__init__(in_shape=in_shape)
        self.activation = activation
        if self.activation not in _ReluLayer.ACTIVATIONS:
            raise ConvNetError("Unknown activation function")

    @property
    def out_width(self):
        return self.in_width

    @property
    def out_height(self):
        return self.in_height

    @property
    def out_depth(self):
        return self.in_depth

    def to_dict(self):
        d = super(ReluLayerSettings, self).to_dict()
        d.update({
            'activation': self.activation,
        })
        return d


# noinspection PyCallingNonCallable
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
        'softmax': {
            'f': relu_softmax,
            'df': relu_softmax_deriv,
        },
    }

    def __init__(self, settings, net_settings=None):
        """
        :param settings: Relu layer settings
        :type settings: ReluLayerSettings
        """
        super(_ReluLayer, self).__init__(settings, net_settings)
        self._prev_delta_reuse = None

    def forward(self, data):
        self.prev_out = self.activation(data)
        return self.prev_out

    def compute_prev_layer_delta(self, current_layer_delta):
        shape = current_layer_delta.shape
        values = current_layer_delta.reshape(-1)
        values *= self.derivative(self.prev_layer.prev_out.reshape(-1))
        return values.reshape(shape)

    def backward(self, current_layer_delta):
        if self.is_output:
            current_layer_delta = self.prev_out - current_layer_delta
        return self.compute_prev_layer_delta(current_layer_delta)

    def activation(self, x):
        return self.ACTIVATIONS[self.settings.activation]['f'](x)

    def derivative(self, x):
        return self.ACTIVATIONS[self.settings.activation]['df'](x)


class ReluLayer(BaseLayer):
    def __init__(self, settings):
        """
        :type settings: ReluLayerSettings
        """
        super(ReluLayer, self).__init__(settings)

    def create(self):
        return _ReluLayer(self.settings, self.net_settings)
