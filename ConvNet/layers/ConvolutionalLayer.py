from ConvNet.layers.BaseLayer import BaseLayer


class ConvolutionalLayer(BaseLayer):
    def __init__(self, dimensions):
        super(ConvolutionalLayer, self).__init__(dimensions)
