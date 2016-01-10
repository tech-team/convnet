import abc

import numpy as np


class BaseLayerSettings(object):
    def __init__(self, **kwargs):
        super(BaseLayerSettings, self).__init__()

        if 'in_dimensions' not in kwargs:
            raise AttributeError("in_dimensions is required")

        self._in_dimensions = tuple()
        self.in_dimensions = kwargs['in_dimensions']  # W1 x H1 x D1

    @property
    def in_dimensions(self):
        return self._in_dimensions

    @in_dimensions.setter
    def in_dimensions(self, in_dimensions):
        assert len(in_dimensions) == 3, "Only 3-dimensional layers are allowed"
        assert in_dimensions[0] == in_dimensions[1], "input's width and height have to be the same"
        self._in_dimensions = in_dimensions

    @property
    def in_width(self):
        return self.in_dimensions[0]

    @property
    def in_height(self):
        return self.in_dimensions[1]

    @property
    def in_depth(self):
        return self.in_dimensions[2]

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

