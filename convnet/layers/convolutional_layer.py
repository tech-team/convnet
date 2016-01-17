import numpy as np

from convnet import utils
from convnet.layers.base_layer import _BaseLayer, BaseLayerSettings, BaseLayer


class ConvolutionalLayerSettings(BaseLayerSettings):
    def __init__(self, in_shape=None, filters_count=1, filter_size=None, stride=1, zero_padding=None):
        super(ConvolutionalLayerSettings, self).__init__(in_shape=in_shape)

        self.filters_count = filters_count  # K
        self.filter_size = filter_size if filter_size is not None else self.in_width  # F
        self.stride = stride  # S
        self.zero_padding = zero_padding if zero_padding is not None else 0  # P

    @property
    def out_width(self):
        return int((self.in_width - self.filter_size + 2.0 * self.zero_padding) / self.stride + 1)

    @property
    def out_height(self):
        return int((self.in_height - self.filter_size + 2.0 * self.zero_padding) / self.stride + 1)

    @property
    def out_depth(self):
        return self.filters_count

    def check(self):
        super(ConvolutionalLayerSettings, self).check()

        assert self.filter_size is not None, "Filter size is not allowed to be None"

        out_width = (self.in_width - self.filter_size + 2.0 * self.zero_padding) / self.stride + 1
        assert out_width % 1 == 0, \
            "out_width == {}, but should be integer".format(out_width)

        out_height = (self.in_height - self.filter_size + 2.0 * self.zero_padding) / self.stride + 1
        assert out_height % 1 == 0, \
            "out_width == {}, but should be integer".format(out_height)


class _ConvolutionalLayer(_BaseLayer):

    EPSILON = 0.3

    def __init__(self, settings, net_settings=None):
        """
        :param settings: Convolutional layer settings
        :type settings: ConvolutionalLayerSettings
        """
        super(_ConvolutionalLayer, self).__init__(settings, net_settings)

        f = settings.filter_size
        w_shape = (f, f, settings.in_depth)

        def new_w(shape):
            return np.random.uniform(low=-self.EPSILON, high=self.EPSILON, size=(np.prod(np.asarray(shape)),))\
                            .reshape(shape)

        self.w = [new_w(w_shape) for _ in xrange(settings.filters_count)]
        self.dw = [np.zeros(w_shape) for _ in xrange(len(self.w))]
        self.dw_last = [np.zeros(w_shape) for _ in xrange(len(self.dw))]

        self.b = [new_w((1,))[0] for _ in xrange(settings.filters_count)]
        self.db = [0] * len(self.b)
        self.db_last = [0] * len(self.db)

        assert len(self.w) == len(self.b)

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

        for f in xrange(self.prev_out.shape[2]):
            for y in xrange(self.prev_out.shape[1]):
                for x in xrange(self.prev_out.shape[0]):

                    conv = 0.0
                    for i in xrange(0, self.settings.filter_size):
                        for j in xrange(0, self.settings.filter_size):
                            for z in xrange(0, self.settings.in_depth):
                                conv += padded_data[s * x + i, s * y + j, z] * self.w[f][i, j, z]

                    self.prev_out[x, y, f] = self.b[f] + conv

        return self.prev_out

    def backward(self, current_layer_delta):
        if self.is_output:
            current_layer_delta = self.prev_out - current_layer_delta

        # calc dE/dW
        for f in xrange(len(self.w)):
            for z in xrange(self.w[f].shape[2]):
                for y in xrange(self.w[f].shape[1]):
                    for x in xrange(self.w[f].shape[0]):

                        conv = 0.0
                        for i in xrange(current_layer_delta.shape[0]):
                            for j in xrange(current_layer_delta.shape[1]):
                                conv += current_layer_delta[i, j, f] * self.prev_layer.prev_out[i + x, j + y, z]

                        self.dw[f][x, y, z] += conv

        # calc dE/dB
        for f in xrange(len(self.b)):
            conv = 0.0
            for i in xrange(current_layer_delta.shape[0]):
                for j in xrange(current_layer_delta.shape[1]):
                    conv += current_layer_delta[i, j, f]

            self.db[f] += conv

        # calc delta
        prev_layer_delta = self._compute_prev_layer_delta(current_layer_delta)
        return prev_layer_delta

    def update_weights(self, samples_count=None):
        filters_count = len(self.w)
        learn_rate = self.net_settings.learning_rate
        momentum = self.net_settings.momentum

        for f in xrange(filters_count):
            if samples_count and samples_count != 1:
                self.dw[f] /= samples_count
                self.db[f] /= samples_count

            self.dw_last[f] = -learn_rate * self.dw[f] + momentum * self.dw_last[f]
            self.db_last[f] = -learn_rate * self.db[f] + momentum * self.db_last[f]

            self.w[f] += self.dw_last[f]
            self.b[f] += self.db_last[f]

            self.dw[f] = np.zeros(self.dw[f].shape)
            self.db[f] = 0


class ConvolutionalLayer(BaseLayer):
    def __init__(self, settings):
        """
        :type settings: ConvolutionalLayerSettings
        """
        super(ConvolutionalLayer, self).__init__(settings)

    def create(self):
        return _ConvolutionalLayer(self.settings, self.net_settings)
