from ConvNet.layers.BaseLayer import BaseLayer, BaseLayerSettings


class ReluLayerSettings(BaseLayerSettings):
    def __init__(self):
        super(ReluLayerSettings, self).__init__()


class ReluLayer(BaseLayer):
    def __init__(self, settings):
        """
        :param settings: Relu layer settings
        :type settings: ReluLayerSettings
        """
        super(ReluLayer, self).__init__(settings)
