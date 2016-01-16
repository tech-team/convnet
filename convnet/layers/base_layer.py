import abc

import numpy as np


class BaseLayerSettings(object):
    def __init__(self, in_shape=None):
        super(BaseLayerSettings, self).__init__()
        self._in_shape = in_shape
        if self._in_shape is not None:
            BaseLayerSettings.check(self)

    @property
    def in_shape(self):
        return self._in_shape

    @in_shape.setter
    def in_shape(self, in_shape):
        self._in_shape = in_shape
        if self._in_shape is not None:
            self.in_shape_changed()
            self.check()

    @property
    def in_width(self):
        return self._in_shape[0] if self._in_shape is not None else None

    @property
    def in_height(self):
        return self._in_shape[1] if self._in_shape is not None else None

    @property
    def in_depth(self):
        return self._in_shape[2] if self._in_shape is not None else None

    @property
    def out_shape(self):
        return self.out_width, self.out_height, self.out_depth

    @property
    @abc.abstractmethod
    def out_width(self):
        pass

    @property
    @abc.abstractmethod
    def out_height(self):
        pass

    @property
    @abc.abstractmethod
    def out_depth(self):
        pass

    def check(self):
        assert len(self.in_shape) == 3, "Only 3-dimensional layers are allowed"
        assert self.in_shape[0] == self.in_shape[1], "input's width and height have to be the same"

    def in_shape_changed(self):
        pass


class _BaseLayer(object):
    def __init__(self, settings):
        """
        :param settings: layer settings
        :type settings: BaseLayerSettings
        """
        super(_BaseLayer, self).__init__()

        self.settings = settings
        self.prev_layer = None
        self.next_layer = None
        self.prev_out = None

    def setup_layers(self, prev_layer, next_layer=None):
        """
        :type prev_layer: _BaseLayer
        :type next_layer: _BaseLayer
        """
        self.prev_layer = prev_layer
        self.next_layer = next_layer

    @property
    def is_input(self):
        return self.prev_layer is None

    @property
    def is_output(self):
        return self.next_layer is None

    @abc.abstractmethod
    def forward(self, data):
        """
        :param data: data from previous layer
        :type data: np.ndarray
        """
        pass

    def _compute_prev_layer_delta(self, current_layer_delta):
        """
        :param current_layer_delta: Current layer's delta
        :type current_layer_delta: np.ndarray
        """
        if not hasattr(self, 'w'):
            raise StandardError(
                "Current layer has no weights matrix, should define custom _compute_prev_layer_delta function")

        if self.prev_layer.is_input:
            return None

        delta = np.empty(self.settings.out_shape)
        for z in xrange(delta.shape[2]):
            for y in xrange(delta.shape[1]):
                for x in xrange(delta.shape[0]):
                    conv = 0.0
                    for i in xrange(0, current_layer_delta.shape[0]):
                        for j in xrange(0, current_layer_delta.shape[1]):
                            for f in xrange(0, current_layer_delta.shape[2]):
                                conv += current_layer_delta[i, j, f] * self.w[f][x - i, y - j, z]
                    delta[x, y, z] += conv
        return delta

    @abc.abstractmethod
    def backward(self, current_layer_delta):
        """
        :param current_layer_delta: Current layer's delta
        :type current_layer_delta: np.ndarray
        :returns Previous layer's delta
        """
        pass

    def update_weights(self, samples_count=None):
        """
        :param samples_count: Samples count or None to apply an entire gradient without averaging
        :type samples_count: int
        """
        pass


class BaseLayer(object):
    def __init__(self, settings):
        """
        :param settings: Layer settings
        :type settings: BaseLayerSettings
        """
        super(BaseLayer, self).__init__()
        self.settings = settings

    @abc.abstractmethod
    def create(self):
        pass
