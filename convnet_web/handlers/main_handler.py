from convnet_web.handlers.base_handler import BaseHandler


class MainHandler(BaseHandler):
    def get(self):
        self.render('index.html')
