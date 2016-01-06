import json
import logging

import tornado
import tornado.web

from ConvNetWeb import settings


class BaseHandler(tornado.web.RequestHandler):
    def __init__(self, application, request, **kwargs):
        super(BaseHandler, self).__init__(application, request, **kwargs)
        self.context = {}
        self.body_json = None

    def prepare(self):
        self.context = {
            'debug': settings.DEBUG
        }

        if "Content-Type" in self.request.headers \
                and self.request.headers["Content-Type"].startswith("application/json"):
            try:
                self.request.body_json = json.loads(self.request.body, encoding='utf-8')
            except Exception as e:
                logging.warn("JSON parse error: %s", e.message)
                self.set_status(400)
                self.finish("JSON is malformed")

    def write_json(self, data):
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(data, ensure_ascii=False))

    def render(self, template_name, **kwargs):
        for k, v in self.context.iteritems():
            if k not in kwargs:
                kwargs[k] = v
        super(BaseHandler, self).render(template_name, **kwargs)

