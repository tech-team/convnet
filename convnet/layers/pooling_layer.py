import numpy as np

from convnet.layers.base_layer import BaseLayer, BaseLayerSettings


class PoolingLayerSettings(BaseLayerSettings):
    def __init__(self, **kwargs):
        super(PoolingLayerSettings, self).__init__(**kwargs)

        self.filter_size = kwargs['filter_size']  # F
        self.stride = kwargs['stride']  # S

    @property
    def out_width(self):
        return int((self.in_width - self.filter_size) / self.stride + 1)

    @property
    def out_height(self):
        return int((self.in_height - self.filter_size) / self.stride + 1)

    @property
    def out_depth(self):
        return self.in_depth


class PoolingLayer(BaseLayer):
    def __init__(self, settings):
        """
        :param settings: Pooling layer settings
        :type settings: PoolingLayerSettings
        """
        super(PoolingLayer, self).__init__(settings)

    def forward(self, data):
        res = np.empty((
            self.settings.out_width,
            self.settings.out_height,
            self.settings.out_depth,
        ))

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


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=120)
    arr = np.random.rand(4, 4, 3)
    s = PoolingLayerSettings(in_dimensions=arr.shape, filter_size=2, stride=2)
    l = PoolingLayer(s)
    res = l.forward(arr)

    print(arr[:, :, 0])
    print(arr[:, :, 1])
    print(arr[:, :, 2])
    print('-----------')
    print(res[:, :, 0])
    print(res[:, :, 1])
    print(res[:, :, 2])
