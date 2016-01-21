import convnetlib
import numpy as np

from convnet.layers.base_layer import _BaseLayer, BaseLayerSettings, BaseLayer


class PoolingLayerSettings(BaseLayerSettings):
    TYPE = 'pool'

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

    def to_dict(self):
        d = super(PoolingLayerSettings, self).to_dict()
        d.update({
            'filter_size': self.filter_size,
            'stride': self.stride,
        })
        return d


class _PoolingLayer(_BaseLayer):
    def __init__(self, settings, net_settings=None):
        """
        :param settings: Pooling layer settings
        :type settings: PoolingLayerSettings
        """
        super(_PoolingLayer, self).__init__(settings, net_settings)
        self.max_indices = np.zeros(self.settings.out_shape, dtype=(int, 2))
        self._prev_delta_reuse = None

    def forward(self, data):
        convnetlib.pool_forward(data,
                                self.settings.stride, self.settings.filter_size,
                                self.max_indices, self.prev_out)

        return self.prev_out

    def compute_prev_layer_delta(self, current_layer_delta):
        res = np.zeros(self.settings.in_shape)
        convnetlib.pool_prev_layer_delta(current_layer_delta,
                                         self.max_indices, res)
        return res

    def backward(self, current_layer_delta):
        if self.is_output:
            current_layer_delta = self.prev_out - current_layer_delta
        return self.compute_prev_layer_delta(current_layer_delta)

    def update_weights(self, samples_count=None):
        super(_PoolingLayer, self).update_weights(samples_count)
        self.max_indices = np.zeros(self.settings.out_shape, dtype=(int, 2))


class PoolingLayer(BaseLayer):
    def __init__(self, settings):
        """
        :type settings: PoolingLayerSettings
        """
        super(PoolingLayer, self).__init__(settings)

    def create(self):
        return _PoolingLayer(self.settings, self.net_settings)
