import logging

import tornado.ioloop
import tornado.web
import tornado.options

import convnet_web.handlers
import settings
from convnet.net import ConvNet

net = ConvNet()


def make_app():
    global net
    s = {
        'debug': settings.DEBUG,
        'template_path': settings.TEMPLATES_DIR,
        'static_path': settings.STATIC_DIR,
        'static_url_prefix': settings.STATIC_PREFIX
    }

    return tornado.web.Application([
        (r"/", convnet_web.handlers.VisualHandler),
        (r"/config", convnet_web.handlers.ConfigHandler),

        (r"/api/config", convnet_web.handlers.ApiConfig, dict(net=net)),
        (r"/api/predict", convnet_web.handlers.ApiPredict, dict(net=net)),
    ], **s)


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)

    tornado.options.define('host', type=str, default='127.0.0.1')
    tornado.options.define('port', type=int, default=8888)
    opts = tornado.options.options

    app = make_app()
    app.listen(opts.port, address=opts.host)

    logging.info("Server started on %s:%d", opts.host, opts.port)
    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        tornado.ioloop.IOLoop.instance().stop()
        logging.info("Server stopped")
