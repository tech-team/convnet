from ConvNet.layers.BaseLayer import BaseLayer, BaseLayerSettings


class FullConnectedLayerSettings(BaseLayerSettings):
    def __init__(self):
        super(FullConnectedLayerSettings, self).__init__()


class FullConnectedLayer(BaseLayer):
    def __init__(self, settings):
        """
        :param settings: Full connected layer settings
        :type settings: FullConnectedLayerSettings
        """
        super(FullConnectedLayer, self).__init__(settings)
