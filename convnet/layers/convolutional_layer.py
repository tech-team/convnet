import numpy as np

from convnet import utils
from convnet.layers.base_layer import BaseLayer, BaseLayerSettings


class ConvolutionalLayerSettings(BaseLayerSettings):
    def __init__(self, in_shape, filters_count=1, filter_size=None, stride=1, zero_padding=None):
        super(ConvolutionalLayerSettings, self).__init__(in_shape=in_shape)

        self.filters_count = filters_count  # K
        self.filter_size = filter_size if filter_size is not None else self.in_width  # F
        self.stride = stride  # S
        self.zero_padding = zero_padding  # P

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
        self.w = [np.zeros((f, f, settings.in_depth)) for _ in xrange(settings.filters_count)]
        self.dw = [np.zeros((f, f, settings.in_depth)) for _ in xrange(settings.filters_count)]

        self.b = [0] * settings.filters_count
        self.db = [0] * settings.filters_count

    def _forward(self, data):
        assert data.shape == self.settings.in_shape, \
            "data.shape = {}; settings.in_dimensions = {}".format(data.shape, self.settings.in_shape)

        # zero padding
        if self.settings.zero_padding:
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

    def forward(self, data):
        assert data.shape == self.settings.in_shape, \
            "data.shape = {}; settings.in_dimensions = {}".format(data.shape, self.settings.in_shape)

        # zero padding
        if self.settings.zero_padding:
            p = self.settings.zero_padding
            padded_data = np.pad(data, pad_width=((p, p), (p, p), (0, 0)), mode='constant', constant_values=0)
        else:
            padded_data = data

        s = self.settings.stride

        res = np.empty(self.settings.out_shape)

        for f in xrange(res.shape[2]):
            for y in xrange(res.shape[1]):
                for x in xrange(res.shape[0]):

                    conv = 0.0
                    for i in xrange(0, self.settings.filter_size):
                        for j in xrange(0, self.settings.filter_size):
                            for z in xrange(0, self.settings.in_depth):
                                conv += padded_data[s * x + i, s * y + j, z] * self.w[f][i, j, z]

                    res[x, y, f] = self.b[f] + conv

        self.prev_out = res
        return res

    def backward(self, next_layer_delta):
        # calc delta
        if self.is_output:
            delta = self.prev_out - next_layer_delta
            next_layer_delta = delta
        else:
            delta = np.empty(self.settings.out_shape)
            for z in xrange(delta.shape[2]):
                for y in xrange(delta.shape[1]):
                    for x in xrange(delta.shape[0]):
                        conv = 0.0
                        for i in xrange(0, next_layer_delta.shape[0]):
                            for j in xrange(0, next_layer_delta.shape[1]):
                                for f in xrange(0, next_layer_delta.shape[2]):
                                    conv += next_layer_delta[i, j, f] * self.next_layer.w[f][x - i, y - j, z]
                        delta[x, y, z] += conv

        # calc dE/dW
        for f in xrange(len(self.w)):
            for z in xrange(self.w[f].shape[2]):
                for y in xrange(self.w[f].shape[1]):
                    for x in xrange(self.w[f].shape[0]):

                        conv = 0.0
                        for i in xrange(next_layer_delta.shape[0]):
                            for j in xrange(next_layer_delta.shape[1]):
                                conv += next_layer_delta[i, j, f] * self.prev_layer.prev_out[i + x, j + y, z]

                        self.dw[f][x, y, z] += conv

        # calc dE/dB
        for f in xrange(len(self.b)):
            conv = 0.0
            for i in xrange(next_layer_delta.shape[0]):
                for j in xrange(next_layer_delta.shape[1]):
                    conv += next_layer_delta[i, j, f]

            self.db[f] += conv

        return delta

    def update_weights(self, samples_count=None):
        for f in xrange(len(self.w)):
            if samples_count:
                self.dw[f] /= samples_count

            self.w[f] -= self.dw[f]
            self.dw[f] = np.zeros(self.dw[f].shape)

        for f in xrange(len(self.b)):
            if samples_count:
                self.db[f] /= samples_count

            self.b[f] -= self.db[f]
            self.db[f] = 0
