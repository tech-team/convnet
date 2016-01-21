import numpy as np

from convnet import utils
from convnet.convnet_error import ConvNetError
from convnet.layers.base_layer import _BaseLayer, BaseLayerSettings, BaseLayer
import convnetlib


class ConvolutionalLayerSettings(BaseLayerSettings):
    TYPE = 'conv'

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

        if self.filter_size is None:
            raise ConvNetError("Filter size is not allowed to be None")

        out_width = (self.in_width - self.filter_size + 2.0 * self.zero_padding) / self.stride + 1
        if out_width % 1 != 0:
            raise ConvNetError("out_width == {}, but should be integer".format(out_width))

        out_height = (self.in_height - self.filter_size + 2.0 * self.zero_padding) / self.stride + 1
        if out_height % 1 != 0:
            raise ConvNetError("out_height == {}, but should be integer".format(out_width))

    def to_dict(self):
        d = super(ConvolutionalLayerSettings, self).to_dict()
        d.update({
            'filters_count': self.filters_count,
            'filter_size': self.filter_size,
            'stride': self.stride,
            'zero_padding': self.zero_padding,
        })
        return d


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
            return np.random.uniform(low=-self.EPSILON, high=self.EPSILON, size=(np.prod(np.asarray(shape)),)) \
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

        convnetlib.conv_forward(padded_data, self.w, self.b, s, self.prev_out)
        return self.prev_out

    def backward(self, current_layer_delta):
        if self.is_output:
            current_layer_delta = self.prev_out - current_layer_delta

        convnetlib.conv_backward(current_layer_delta, self.prev_layer.prev_out, self.dw, self.db)

        # calc delta
        prev_layer_delta = self.compute_prev_layer_delta(current_layer_delta)
        return prev_layer_delta

    def update_weights(self, samples_count=None):
        filters_count = len(self.w)
        learn_rate = self.net_settings.learning_rate
        momentum = self.net_settings.momentum
        weight_decay = self.net_settings.weight_decay

        for f in xrange(filters_count):
            if samples_count:
                samples_count = 1.0

            self.dw_last[f] = -learn_rate / float(samples_count) * (self.dw[f] + weight_decay * self.w[f]) + momentum * self.dw_last[f]
            self.db_last[f] = -learn_rate / float(samples_count) * (self.db[f]) + momentum * self.db_last[f]

            self.w[f] += self.dw_last[f]
            self.b[f] += self.db_last[f]

            self.dw[f] = np.zeros(self.dw[f].shape)
            self.db[f] = 0

    def weights_sum(self):
        s = 0.0
        for f in xrange(len(self.w)):
            s += np.sum(self.w[f] * self.w[f])
        return s


class ConvolutionalLayer(BaseLayer):
    def __init__(self, settings):
        """
        :type settings: ConvolutionalLayerSettings
        """
        super(ConvolutionalLayer, self).__init__(settings)

    def create(self):
        return _ConvolutionalLayer(self.settings, self.net_settings)
