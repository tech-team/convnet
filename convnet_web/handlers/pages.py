from convnet_web.handlers.base_handler import BaseHandler


class VisualHandler(BaseHandler):
    def get(self):
        self.render('visual.html')


class ConfigHandler(BaseHandler):
    def get(self):
        self.render('config.html')
