from ConvNet.layers.BaseLayer import BaseLayer


class FullConnectedLayer(BaseLayer):
    def __init__(self, dimensions):
        super(FullConnectedLayer, self).__init__(dimensions)
