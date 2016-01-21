from __future__ import print_function
import cPickle
import gc

import numpy as np

from convnet.layers.base_layer import _BaseLayer, BaseLayer


class ConvNetSettings(object):
    def __init__(self):
        super(ConvNetSettings, self).__init__()
        self.iterations_count = None
        self.learning_rate = None
        self.momentum = None
        self.batch_size = None
        self.weight_decay = None

    def to_dict(self):
        return {
            'iterations_count': self.iterations_count,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
        }


class bcolors:
    NOCOLOR = '\033[0m'
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'

    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# noinspection PyPep8Naming
class ConvNet(object):
    def __init__(self, iterations_count=10, learning_rate=0.01, momentum=0, batch_size=1, weight_decay=0):
        super(ConvNet, self).__init__()

        self.net_settings = ConvNetSettings()
        self.set_settings(iterations_count=iterations_count,
                          learning_rate=learning_rate,
                          momentum=momentum,
                          batch_size=batch_size,
                          weight_decay=weight_decay)

        self.layers = []
        self.current_train_loss = np.inf
        self.current_cross_loss = np.inf

        self.prev_train_loss = 0.0
        self.prev_cross_loss = 0.0

    def set_settings(self, **settings):
        self.net_settings.iterations_count = settings['iterations_count']
        self.net_settings.learning_rate = settings['learning_rate']
        self.net_settings.momentum = settings['momentum']
        self.net_settings.batch_size = settings['batch_size']
        self.net_settings.weight_decay = settings['weight_decay']

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

    def fit(self, X_train, Y_train, X_cross=None, Y_cross=None):
        assert len(X_train) == len(Y_train)
        assert len(X_cross) == len(Y_cross)

        for iteration in xrange(1, self.net_settings.iterations_count + 1):
            batch_losses = []
            train_loss = 0.0
            samples_batch = 0
            for i, (x, y) in enumerate(zip(X_train, Y_train)):
                # print("{}/{}".format(i + 1, len(X)))
                samples_batch += 1

                res = x
                for l in self.layers:
                    res = l.forward(res)
                    if hasattr(l, 'w'):
                        train_loss += self.net_settings.weight_decay * l.weights_sum()

                err = np.mean(res - y)
                train_loss += err * err

                res = y
                for l in reversed(self.layers):
                    res = l.backward(res)

                if samples_batch == self.net_settings.batch_size:
                    samples_batch = 0
                    for l in self.layers:
                        l.update_weights(samples_count=self.net_settings.batch_size)

                    train_loss /= 2.0 * self.net_settings.batch_size
                    batch_losses.append(train_loss)
                    train_loss = 0.0

            if samples_batch != 0:
                for l in self.layers:
                    l.update_weights(samples_count=samples_batch)

                train_loss /= 2.0 * samples_batch
                batch_losses.append(train_loss)

            train_loss = np.mean(batch_losses)

            cross_loss = None
            if X_cross is not None and Y_cross is not None:
                cross_loss = 0.0
                for i, (x, y) in enumerate(zip(X_cross, Y_cross)):
                    res = x
                    for l in self.layers:
                        res = l.forward(res)
                        if hasattr(l, 'w'):
                            cross_loss += self.net_settings.weight_decay * l.weights_sum()

                    err = np.mean(res - y)
                    cross_loss += err * err
                cross_loss /= 2.0 * len(X_cross)

            self.prev_train_loss = self.current_train_loss
            self.prev_cross_loss = self.current_cross_loss
            self.current_train_loss = train_loss
            self.current_cross_loss = cross_loss
            print("{}Iteration #{}{}; train_error = {}; cross_error = {};".format(bcolors.BOLD,
                                                                                  iteration,
                                                                                  bcolors.NOCOLOR,
                                                                                  self.get_colored_error(
                                                                                          self.current_train_loss,
                                                                                          self.prev_train_loss),
                                                                                  self.get_colored_error(
                                                                                          self.current_cross_loss,
                                                                                          self.prev_cross_loss)))

    @staticmethod
    def get_colored_error(current_loss, prev_loss):
        if current_loss < prev_loss:
            return "{}{}{}".format(bcolors.GREEN, current_loss, bcolors.NOCOLOR)
        elif current_loss > prev_loss:
            return "{}{}{}".format(bcolors.RED, current_loss, bcolors.NOCOLOR)
        else:
            return current_loss

    def predict(self, X):
        res = X
        for l in self.layers:
            res = l.forward(res)

        return res

    def dump_net(self, filename='convnet.pkl'):
        with open(filename, 'wb') as f:
            cPickle.dump(self, f)

    @staticmethod
    def load_net(filename='convnet.pkl'):
        with open(filename, 'rb') as f:
            net = cPickle.load(f)
        return net

    def replace(self, net):
        self.layers = net.layers
        self.net_settings = net.net_settings
        self.last_output = net.last_output

    def to_dict(self):
        d = self.net_settings.to_dict()
        d.update({
            'layers': [l.settings.to_dict() for l in self.layers]
        })
        return d
