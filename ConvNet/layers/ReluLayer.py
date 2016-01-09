from ConvNet.layers.BaseLayer import BaseLayer


class ReluLayer(BaseLayer):
    def __init__(self, dimensions):
        super(ReluLayer, self).__init__(dimensions)
