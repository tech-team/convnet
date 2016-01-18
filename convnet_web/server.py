import logging

import tornado.ioloop
import tornado.web
import tornado.options

import convnet_web.handlers
import settings as convsettings
from convnet.layers import *
from convnet.net import ConvNet


class Application(tornado.web.Application):
    def __init__(self):
        self.net = ConvNet()
        self.net.setup_layers([
            InputLayer(InputLayerSettings(in_shape=(28, 28, 1))),

            ConvolutionalLayer(ConvolutionalLayerSettings(filters_count=8, filter_size=5, stride=1, zero_padding=0)),
            ReluLayer(ReluLayerSettings(activation='max')),
            PoolingLayer(PoolingLayerSettings(filter_size=2, stride=2)),

            ConvolutionalLayer(ConvolutionalLayerSettings(filters_count=16, filter_size=5, stride=1, zero_padding=0)),
            ReluLayer(ReluLayerSettings(activation='max')),
            PoolingLayer(PoolingLayerSettings(filter_size=3, stride=3)),

            FullConnectedLayer(FullConnectedLayerSettings(neurons_count=10, activation='sigmoid')),
        ])

        s = {
            'debug': convsettings.DEBUG,
            'template_path': convsettings.TEMPLATES_DIR,
            'static_path': convsettings.STATIC_DIR,
            'static_url_prefix': convsettings.STATIC_PREFIX
        }

        handlers = [
            (r"/", convnet_web.handlers.VisualHandler),
            (r"/config", convnet_web.handlers.ConfigHandler),

            (r"/api/config", convnet_web.handlers.ApiConfig, dict(net=self.net)),
            (r"/api/config_mnist", convnet_web.handlers.ApiConfigMnist, dict(net=self.net)),
            (r"/api/predict", convnet_web.handlers.ApiPredict, dict(net=self.net)),
        ]
        super(Application, self).__init__(handlers, **s)


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)

    tornado.options.define('host', type=str, default='127.0.0.1')
    tornado.options.define('port', type=int, default=8888)
    opts = tornado.options.options

    app = Application()
    app.listen(opts.port, address=opts.host)

    logging.info("Server started on %s:%d", opts.host, opts.port)
    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        tornado.ioloop.IOLoop.instance().stop()
        logging.info("Server stopped")
