import tornado
import tornado.web

from ConvNetWeb import settings


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html', debug=settings.DEBUG)
