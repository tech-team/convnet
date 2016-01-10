import numpy as np

from ConvNet.layers.BaseLayer import BaseLayer, BaseLayerSettings


class ReluLayerSettings(BaseLayerSettings):
    """
    activation - string, containing a name of desired activation func (refer to ReluLayer.ACTIVATIONS
    """
    def __init__(self, **kwargs):
        super(ReluLayerSettings, self).__init__(**kwargs)
        self.activation = kwargs['activation']

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

    def activation(self, x):
        return self.ACTIVATIONS[self.settings.activation](x)


if __name__ == "__main__":
    arr = np.random.uniform(-5, 5, 48).reshape((4, 4, 3))
    s = ReluLayerSettings(in_dimensions=arr.shape, activation='max')
    l = ReluLayer(s)
    res = l.forward(arr)

    print(arr[:, :, 0])
    print(arr[:, :, 1])
    print(arr[:, :, 2])
    print('-----------')
    print(res[:, :, 0])
    print(res[:, :, 1])
    print(res[:, :, 2])