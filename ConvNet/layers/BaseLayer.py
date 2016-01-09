import numpy as np


class BaseLayer(object):
    def __init__(self, dimensions):
        super(BaseLayer, self).__init__()

        assert len(dimensions) == 3, "Only 3-dimensional layers are allowed"
        self.dimensions = tuple(dimensions)

    @property
    def width(self):
        return self.dimensions[0]

    @property
    def height(self):
        return self.dimensions[1]

    @property
    def depth(self):
        return self.dimensions[2]

