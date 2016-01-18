import numpy as np
import tornado
import tornado.web
import os

from convnet.net import ConvNet
from convnet.utils import to_3d, input_2d_to_3d, arr_2d_to_3d
from convnet_web.handlers import BaseHandler

import convnet.layers
from convnet.convnet_error import ConvNetError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ApiHandler(BaseHandler):
    def initialize(self, net):
        self.net = net

    def api_respond(self, status='ok', data=None):
        d = dict(status=status)
        if data is not None:
            d['data'] = data
        return self.write_json(d)

    def api_ok(self, data=None):
        return self.api_respond('ok', data)

    def api_error(self, data):
        return self.api_respond('error', data)

    def write_error(self, status_code, **kwargs):
        if 'exc_info' in kwargs and hasattr(kwargs['exc_info'][1], 'log_message'):
            reason = kwargs['exc_info'][1].log_message
        else:
            reason = self._reason
        self.api_error({
            'code': status_code,
            'reason': reason,
        })


class ApiConfig(ApiHandler):

    def post(self):
        config = dict(
                learning_rate=self.get_arg('learning_rate', field_type=float, default=0.1),
                momentum=self.get_arg('momentum', field_type=float, default=0),
                batch_size=self.get_arg('batch_size', field_type=int, default=1),
                iterations_count=self.get_arg('iterations_count', field_type=int, default=1),
                weight_decay=self.get_arg('weight_decay', field_type=float, default=0),
        )

        self.net.set_settings(**config)

        layers_conf = self.require_arg('layers', field_type=list)

        try:
            layers = [convnet.layers.InputLayer(convnet.layers.InputLayerSettings(in_shape=(28, 28, 1)))]
            for layer_conf in layers_conf:
                layer = self._create_layer_from_conf(layer_conf)
                layers.append(layer)

            self.net.setup_layers(layers)

            return self.api_ok(self.net.to_dict())

        except ConvNetError as e:
            return self.api_error(e.message)

    def get(self):
        d = self.net.to_dict()
        self.api_ok(d)

    def _create_layer_from_conf(self, layer_conf):
        """
        :returns Layer
        :rtype convnet.layers.BaseLayer
        """
        layer_type = self.get_arg('type', args=layer_conf)

        try:
            if layer_type == 'conv':
                settings = convnet.layers.ConvolutionalLayerSettings(
                        filter_size=self.require_arg('filter_size', field_type=int, args=layer_conf),
                        filters_count=self.require_arg('filters_count', field_type=int, args=layer_conf),
                        stride=self.require_arg('stride', field_type=int, args=layer_conf),
                        zero_padding=self.require_arg('zero_padding', field_type=int, args=layer_conf),
                )
                return convnet.layers.ConvolutionalLayer(settings)
            if layer_type == 'relu':
                settings = convnet.layers.ReluLayerSettings(
                        activation=self.require_arg('activation', field_type=str, args=layer_conf),
                )
                return convnet.layers.ReluLayer(settings)
            if layer_type == 'pool':
                settings = convnet.layers.PoolingLayerSettings(
                        filter_size=self.require_arg('filter_size', field_type=int, args=layer_conf),
                        stride=self.require_arg('stride', field_type=int, args=layer_conf),
                )
                return convnet.layers.PoolingLayer(settings)
            if layer_type == 'fc':
                settings = convnet.layers.FullConnectedLayerSettings(
                        neurons_count=self.require_arg('neurons_count', field_type=int, args=layer_conf),
                        activation=self.require_arg('activation', field_type=str, args=layer_conf),
                )
                return convnet.layers.FullConnectedLayer(settings)
        except tornado.web.MissingArgumentError as e:
            arg_name = '<{}>.{}'.format(layer_type, e.arg_name)
            raise tornado.web.MissingArgumentError(arg_name)


class ApiConfigMnist(ApiHandler):
    def post(self):
        mnist_net_path = os.path.join(BASE_DIR, "../../convnet/mnist/convnet1.pkl")
        # mnist_net_path = os.path.join(BASE_DIR, "../../convnet/mnist/convnet_test.pkl")
        net = self.net.load_net(mnist_net_path)
        self.application.net.replace(net)

        self.api_ok(self.application.net.to_dict())


class ApiPredict(ApiHandler):
    def post(self):
        image = self.require_arg('image', field_type=list)
        image_3d = arr_2d_to_3d(np.asarray(image))

        res = self.net.predict(image_3d).reshape(-1)
        res_max = res.argmax()

        layers_outs = [self.transform_layer_out(l.prev_out) for l in self.net.layers]

        self.api_ok({
            'prediction': res_max,
            'confidence': res[res_max],
            'layers': layers_outs
        })

    def transform_layer_out(self, layer_out):
        arr = []
        for z in xrange(layer_out.shape[2]):
            zarr = []
            for y in xrange(layer_out.shape[1]):
                yarr = []
                for x in xrange(layer_out.shape[0]):
                    yarr.append(layer_out[y, x, z])
                zarr.append(yarr)
            arr.append(zarr)
        return arr
