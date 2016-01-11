import abc

import numpy as np


class BaseLayerSettings(object):
    def __init__(self, in_shape):
        super(BaseLayerSettings, self).__init__()

        assert len(in_shape) == 3, "Only 3-dimensional layers are allowed"
        assert in_shape[0] == in_shape[1], "input's width and height have to be the same"
        self._in_shape = in_shape

    @property
    def in_shape(self):
        return self._in_shape

    @property
    def in_width(self):
        return self._in_shape[0]

    @property
    def in_height(self):
        return self._in_shape[1]

    @property
    def in_depth(self):
        return self._in_shape[2]

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


class BaseLayer(object):
    def __init__(self, settings):
        """
        :param settings: layer settings
        :type settings: BaseLayerSettings
        """
        super(BaseLayer, self).__init__()

        self.settings = settings

    @abc.abstractmethod
    def forward(self, data):
        """
        :param data: data from previous layer
        :type data: np.ndarray
        """
        pass

