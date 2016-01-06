import os
import logging

import tornado.ioloop
import tornado.web
import tornado.options

import ConvNetWeb.handlers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def make_app():
    settings = {
        'template_path': os.path.join(BASE_DIR, 'templates'),
        'debug': True
    }

    if settings['debug']:
        settings['static_path'] = os.path.join(BASE_DIR, 'static')
        settings['static_url_prefix'] = '/static/'

    return tornado.web.Application([
        (r"/", ConvNetWeb.handlers.MainHandler),
    ], **settings)

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
