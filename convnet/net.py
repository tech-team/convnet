from convnet.layers.base_layer import _BaseLayer, BaseLayer


class ConvNet(object):
    def __init__(self, layers=None):
        super(ConvNet, self).__init__()
        self.last_output = None
        self.layers = []
        if layers is not None:
            self.setup_layers(layers)

    def setup_layers(self, layers):
        assert isinstance(layers, list), "Only list of layers is allowed"
        for l in layers:
            assert isinstance(l, BaseLayer), "Only instances of BaseLayer are allowed"

        # Step 1. Initialize in_shape in each layer except of the first
        for i in xrange(1, len(layers)):
            prev_settings = layers[i - 1].settings
            layers[i].settings.in_shape = (prev_settings.out_width(), prev_settings.out_height(), prev_settings.out_depth())

        # Step 2. Create layers themselves
        layers = [l.create() for l in layers]

        # Step 3. Initialize prev_layer and next_layer in each layer
        for i in xrange(0, len(layers)):
            if i != 0:
                layers[i].prev_layer = layers[i - 1]
            if i != len(layers) - 1:
                layers[i].next_layer = layers[i + 1]

        self.layers = layers

    def fit(self, X, Y):
        assert len(X) == len(Y)
        for x, y in zip(X, Y):
            res = x
            for l in self.layers:
                res = l.forward(res)

            self.last_output = res
            res = y
            for l in reversed(self.layers):
                res = l.backward(res)

        for l in self.layers:
            l.update_weights(samples_count=len(X))

    def predict(self, X):
        res = X
        for l in self.layers:
            res = l.forward(res)

        self.last_output = res
        return res
