import numpy as np

from convnet.layers.base_layer import _BaseLayer, BaseLayerSettings


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
    def __init__(self, settings):
        """
        :param settings: Pooling layer settings
        :type settings: PoolingLayerSettings
        """
        super(_PoolingLayer, self).__init__(settings)

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

                    res[i, j, z] = np.max(data_slice[x_offset:x_offset + f, y_offset:y_offset + f])
                    i += 1
                j += 1

        return res

    def backward(self, next_layer_delta):
        pass
