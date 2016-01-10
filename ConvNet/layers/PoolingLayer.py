from ConvNet.layers.BaseLayer import BaseLayer, BaseLayerSettings


class PoolingLayerSettings(BaseLayerSettings):
    def __init__(self):
        super(PoolingLayerSettings, self).__init__()


class PoolingLayer(BaseLayer):
    def __init__(self, settings):
        """
        :param settings: Pooling layer settings
        :type settings: PoolingLayerSettings
        """
        super(PoolingLayer, self).__init__(settings)
