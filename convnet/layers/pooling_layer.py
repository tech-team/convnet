import numpy as np

from convnet.layers.base_layer import _BaseLayer, BaseLayerSettings, BaseLayer


class PoolingLayerSettings(BaseLayerSettings):
    def __init__(self, in_shape=None, filter_size=1, stride=1):
        super(PoolingLayerSettings, self).__init__(in_shape=in_shape)

        self.filter_size = filter_size  # F
        self.stride = stride  # S

    @property
    def out_width(self):
        return int((self.in_width - self.filter_size) / self.stride + 1)

    @property
    def out_height(self):
        return int((self.in_height - self.filter_size) / self.stride + 1)

    @property
    def out_depth(self):
        return self.in_depth


class _PoolingLayer(_BaseLayer):
    def __init__(self, settings, net_settings=None):
        """
        :param settings: Pooling layer settings
        :type settings: PoolingLayerSettings
        """
        super(_PoolingLayer, self).__init__(settings, net_settings)
        self.max_indices = np.zeros(self.settings.out_shape, dtype=(int, 2))

    def forward(self, data):
        res = np.empty(self.settings.out_shape)

        f = self.settings.filter_size
        for z in xrange(0, self.settings.out_depth):
            data_slice = data[:, :, z]

            j = 0
            for y in xrange(0, self.settings.out_height):
                y_offset = y * self.settings.stride
                i = 0
                for x in xrange(0, self.settings.out_width):
                    x_offset = x * self.settings.stride

                    piece = data_slice[x_offset:x_offset + f, y_offset:y_offset + f]
                    max_index = np.unravel_index(piece.argmax(), piece.shape)
                    res[i, j, z] = piece[max_index]
                    self.max_indices[i, j, z] = tuple(max_index)
                    i += 1
                j += 1

        self.prev_out = res
        return res

    def _compute_prev_layer_delta(self, current_layer_delta):
        delta = current_layer_delta

        res = np.zeros(self.settings.in_shape)

        f = self.settings.filter_size
        for z in xrange(0, delta.shape[2]):
            res_slice = res[:, :, z]

            for y in xrange(0, delta.shape[1]):
                y_offset = y * self.settings.stride

                for x in xrange(0, delta.shape[0]):
                    x_offset = x * self.settings.stride

                    piece = res_slice[x_offset:x_offset + f, y_offset:y_offset + f]

                    value = delta[x, y, z]
                    max_index = self.max_indices[x, y, z]
                    piece[tuple(max_index)] = value

        return res

    def backward(self, current_layer_delta):
        if self.is_output:
            current_layer_delta = self.prev_out - current_layer_delta
        return self._compute_prev_layer_delta(current_layer_delta)

    def update_weights(self, samples_count=None):
        super(_PoolingLayer, self).update_weights(samples_count)
        self.max_indices = np.zeros(self.max_indices.shape)


class PoolingLayer(BaseLayer):
    def __init__(self, settings):
        """
        :type settings: PoolingLayerSettings
        """
        super(PoolingLayer, self).__init__(settings)

    def create(self):
        return _PoolingLayer(self.settings, self.net_settings)
