import numpy as np

from convnet import utils
from convnet.layers.base_layer import BaseLayer, BaseLayerSettings


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
