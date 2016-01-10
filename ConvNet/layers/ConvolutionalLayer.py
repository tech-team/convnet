import numpy as np

from ConvNet.layers.BaseLayer import BaseLayer, BaseLayerSettings


class ConvolutionalLayerSettings(BaseLayerSettings):
    def __init__(self, **kwargs):
        super(ConvolutionalLayerSettings, self).__init__(**kwargs)

        self.filters_count = kwargs['filters_count']  # K
        self.filter_size = kwargs['filter_size']      # F
        self.stride = kwargs['stride']                # S
        self.zero_padding = kwargs['zero_padding']    # P

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

        if self.settings.zero_padding != 0:
            p = self.settings.zero_padding
            padded_data = np.empty((
                data.shape[0] + p + p,
                data.shape[1] + p + p,
                data.shape[2]
            ))

            for z in xrange(0, data.shape[2]):
                padded_data[:, :, z] = np.lib.pad(data[:, :, z], (1, 1), 'constant', constant_values=(0, 0))

        else:
            padded_data = data

        x_columns = self.settings.out_width * self.settings.out_height
        X = np.empty((self.w.shape[1], x_columns))

        i = 0
        f = self.settings.filter_size
        for y in xrange(0, self.settings.out_height):
            y_offset = y * self.settings.stride
            for x in xrange(0, self.settings.out_width):
                x_offset = x * self.settings.stride
                X[:, i] = padded_data[x_offset:x_offset + f, y_offset:y_offset + f, :].reshape(X.shape[0])
                i += 1

        res = np.dot(self.w, X) + self.b
        # for i in xrange(0, res.shape[0]):
        #     res[i] = np.full((res.shape[1],), i)
        res = res.swapaxes(0, 1).reshape(
                (self.settings.out_width, self.settings.out_height, self.settings.filters_count))
        return res
