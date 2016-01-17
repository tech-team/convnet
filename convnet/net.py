from convnet.layers.base_layer import _BaseLayer, BaseLayer


class ConvNetSettings(object):
    def __init__(self):
        super(ConvNetSettings, self).__init__()
        self.iterations_count = None
        self.learning_rate = None
        self.momentum = None
        self.batch_size = None


class ConvNet(object):
    def __init__(self, iterations_count=10, learning_rate=0.01, momentum=0, batch_size=1):
        super(ConvNet, self).__init__()

        self.net_settings = ConvNetSettings()
        self.net_settings.iterations_count = iterations_count
        self.net_settings.learning_rate = learning_rate
        self.net_settings.momentum = momentum
        self.net_settings.batch_size = batch_size

        self.last_output = None
        self.layers = []

    def setup_layers(self, layers):
        assert isinstance(layers, list), "Only list of layers is allowed"
        for l in layers:
            assert isinstance(l, BaseLayer), "Only instances of BaseLayer are allowed"
            l.net_settings = self.net_settings

        # Step 1. Initialize in_shape in each layer except of the first
        for i in xrange(1, len(layers)):
            layers[i].settings.in_shape = layers[i - 1].settings.out_shape

        # Step 2. Create layers themselves
        layers = [l.create() for l in layers]

        # Step 3. Initialize prev_layer and next_layer in each layer
        for i in xrange(0, len(layers)):
            prev_layer = None
            next_layer = None
            if i != 0:
                prev_layer = layers[i - 1]
            if i != len(layers) - 1:
                next_layer = layers[i + 1]

            layers[i].setup_layers(prev_layer, next_layer)

        self.layers = layers

    def fit(self, X, Y):
        assert len(X) == len(Y)

        for iteration in xrange(1, self.net_settings.iterations_count + 1):
            print("Iteration #{}".format(iteration))

            samples_batch = 0
            for i, (x, y) in enumerate(zip(X, Y)):
                print("{}/{}".format(i + 1, len(X)))
                samples_batch += 1

                res = x
                for l in self.layers:
                    res = l.forward(res)

                self.last_output = res
                res = y
                for l in reversed(self.layers):
                    res = l.backward(res)

                if samples_batch == self.net_settings.batch_size:
                    samples_batch = 0
                    for l in self.layers:
                        l.update_weights(samples_count=samples_batch)

            if samples_batch != 0:
                for l in self.layers:
                    l.update_weights(samples_count=samples_batch)

    def predict(self, X):
        res = X
        for l in self.layers:
            res = l.forward(res)

        self.last_output = res
        return res
