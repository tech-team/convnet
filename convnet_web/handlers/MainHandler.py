from convnet_web.handlers.BaseHandler import BaseHandler


class MainHandler(BaseHandler):
    def get(self):
        self.render('index.html')
