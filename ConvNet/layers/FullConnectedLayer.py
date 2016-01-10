import numpy as np

from ConvNet.layers.ConvolutionalLayer import ConvolutionalLayerSettings, ConvolutionalLayer


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


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=120)

    x = np.zeros((2, 2, 2))
    x[:, :, 0] = np.array([
        [0, 1],
        [0, 1]
    ])

    x[:, :, 1] = np.array([
        [1, 0],
        [1, 0]
    ])

    w = np.array([
        [0, 1, 0, 1, 1, 0, 1, 0]
    ])

    b = np.array([0])

    expected_res = [[[4]]]

    s = FullConnectedLayerSettings(in_dimensions=x.shape)
    l = FullConnectedLayer(s)
    l.w = w
    l.b = b

    res = l.forward(x)

    assert (res == expected_res).all()
