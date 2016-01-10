import numpy as np

from ConvNet.layers.BaseLayer import BaseLayer, BaseLayerSettings
import utils


class ConvolutionalLayerSettings(BaseLayerSettings):
    def __init__(self, **kwargs):
        super(ConvolutionalLayerSettings, self).__init__(**kwargs)

        self.filters_count = kwargs['filters_count']  # K
        self.filter_size = kwargs['filter_size']  # F
        self.stride = kwargs['stride']  # S
        self.zero_padding = kwargs['zero_padding']  # P

        out_width = (self.in_width - self.filter_size + 2.0 * self.zero_padding) / self.stride + 1
        assert out_width % 1 == 0, \
            "out_width == {}, but should be integer".format(out_width)

        out_height = (self.in_height - self.filter_size + 2.0 * self.zero_padding) / self.stride + 1
        assert out_height % 1 == 0, \
            "out_width == {}, but should be integer".format(out_height)

    @property
    def out_width(self):
        return int((self.in_width - self.filter_size + 2.0 * self.zero_padding) / self.stride + 1)

    @property
    def out_height(self):
        return int((self.in_height - self.filter_size + 2.0 * self.zero_padding) / self.stride + 1)

    @property
    def out_depth(self):
        return self.filters_count


class ConvolutionalLayer(BaseLayer):
    def __init__(self, settings):
        """
        :param settings: Convolutional layer settings
        :type settings: ConvolutionalLayerSettings
        """
        super(ConvolutionalLayer, self).__init__(settings)

        f = settings.filter_size
        self.w = np.zeros((settings.filters_count, f * f * settings.in_depth))
        self.b = np.zeros((settings.filters_count, 1))

    def forward(self, data):
        assert data.shape == self.settings.in_dimensions, \
            "data.shape = {}; settings.in_dimensions = {}".format(data.shape, self.settings.in_dimensions)

        # zero padding
        if self.settings.zero_padding != 0:
            p = self.settings.zero_padding
            padded_data = np.pad(data, pad_width=((p, p), (p, p), (0, 0)), mode='constant', constant_values=0)
        else:
            padded_data = data

        # convert input matrix to array of flattened receptive fields
        x_col = utils.im2col(padded_data, self.settings.filter_size, self.settings.stride,
                             self.settings.out_width, self.settings.out_height)

        # activation itself
        res = np.dot(self.w, x_col) + self.b

        return utils.col2im(res, self.settings.out_width, self.settings.out_height, self.settings.out_depth)


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=120)
    # arr = np.random.rand(227, 227, 3)
    # s = ConvolutionalLayerSettings(in_dimensions=arr.shape, filter_size=11, stride=4, filters_count=96, zero_padding=0)
    # l = ConvolutionalLayer(s)
    # l.forward(arr)

    # arr = np.random.rand(5, 5, 3)

    arr = np.empty((5, 5, 3))
    arr[:, :, 0] = np.array([
        [1, 2, 0, 2, 0],
        [1, 0, 0, 0, 0],
        [0, 2, 0, 2, 1],
        [1, 1, 1, 0, 2],
        [2, 2, 0, 0, 1]
    ])
    arr[:, :, 1] = np.array([
        [0, 1, 0, 2, 1],
        [0, 0, 2, 1, 2],
        [1, 2, 1, 2, 0],
        [1, 0, 1, 0, 2],
        [0, 2, 2, 2, 0]
    ])
    arr[:, :, 2] = np.array([
        [1, 2, 2, 0, 2],
        [1, 0, 1, 1, 0],
        [2, 2, 0, 2, 0],
        [1, 0, 0, 1, 2],
        [0, 1, 1, 1, 2],
    ])
    s = ConvolutionalLayerSettings(in_dimensions=arr.shape, filter_size=3, stride=2, filters_count=2, zero_padding=1)
    l = ConvolutionalLayer(s)

    l.w = np.array([
        [0, 1, -1,
         -1, 0, 1,
         -1, 1, 0,

         -1, 0, -1,
         1, 0, 1,
         -1, 0, 1,

         0, 0, 1,
         0, -1, -1,
         1, -1, 0],

        [1, 0, 0,
         -1, -1, 1,
         -1, 1, 0,

         1, 0, 0,
         1, 0, -1,
         -1, -1, -1,

         -1, 0, 1,
         0, 0, 1,
         0, -1, -1]
    ])

    l.b = np.array([
        1,
        0
    ]).reshape(2, 1)

    res = l.forward(arr)
    print(res[:, :, 0])
    print(res[:, :, 1])

    assert (res == np.array([[[1., 2.],
                             [2., -6.],
                             [-1., -3.]],

                            [[2., 1.],
                             [3., 1.],
                             [1., -3.]],

                            [[4., -1.],
                             [3., 1.],
                             [3., 0.]]])).all()
