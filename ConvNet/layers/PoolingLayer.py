from ConvNet.layers.BaseLayer import BaseLayer


class PoolingLayer(BaseLayer):
    def __init__(self, dimensions):
        super(PoolingLayer, self).__init__(dimensions)
